//! Prony / Reed–Solomon syndrome sketch — an MDS alternative to `mse.rs`.
//!
//! Same contract as [`crate::mse::MseEncoding`]: an additively homomorphic
//! sketch over the KAHE plaintext modulus, packed into `KahePoly` slots, whose
//! coefficient-wise sum decodes to the multiset union. Different mechanism:
//! instead of hashing each element into `γ` of `δ` buckets and peeling, each
//! contributor picks a random evaluation point `z ∈ F_p^*` and writes a
//! Vandermonde row.
//!
//! ```text
//!   z ← F_p^*                              (per-element evaluation point)
//!   S[j]    += z^j                         (j ∈ [0, cols])
//!   W[s][j] += x_s · z^j                   (s ∈ [ξ], j ∈ [0, cols))
//! ```
//!
//! Summed over the round, `S[j] = Σ_i z_i^j` are the power sums of the
//! contributors' points and `W[s][j] = Σ_i x_{i,s} z_i^j`. Decoding:
//!
//! 1. `k := S[0]` is the number of contributors (a zero sketch — cover
//!    traffic — adds nothing, exactly as `MseEncoding::cover`).
//! 2. Newton's identities turn `S[1..=k]` into the elementary symmetric
//!    polynomials, i.e. the coefficients of `Λ(X) = Π_i (X − z_i)`.
//! 3. The `z_i` are the roots of `Λ` (Cantor–Zassenhaus over `F_p`).
//! 4. With the points known, each payload symbol is one transposed-Vandermonde
//!    solve: `x_{i,s} = (Σ_m b_{i,m} W[s][m]) / Λ'(z_i)` where
//!    `Λ(X)/(X − z_i) = Σ_m b_{i,m} X^m`.
//!
//! Cost of the swap: **`p` must be prime.** `T_MODULUS_DEFAULT = 2^36` is not,
//! and Newton's identities divide by `1..k`. [`PRONY_PRIME`] is a 36-bit prime,
//! so nothing in the KAHE budget `t·8σ_e·√ρ + ρ·t/2 < q_kahe/2` moves and the
//! codec's 32-bit symbols still fit. `KaheParams::t_modulus` is already a free
//! `u64`.

use chipmunk_code::{KahePoly, N};
use rand::Rng;
use rayon::prelude::*;

/// Plaintext modulus, FFT-friendly in the sense Rabbit-Mix's root-finder needs — `q = M·2^m + 1`
pub const PRONY_PRIME: u64 = (65_535u64 << 20) + 1;

// ---------------------------------------------------------------
// F_p scalars
// ---------------------------------------------------------------

/// A modulus together with its Barrett reciprocal, so a modular multiply costs
/// two `u128` multiplies instead of a `u128` division.
#[derive(Clone, Copy, Debug)]
struct Fp {
    p: u64,
    /// `⌊2^{2k}/p⌋`.
    mu: u128,
    /// `⌈log₂ p⌉`.
    k: u32,
}

impl Fp {
    fn new(p: u64) -> Self {
        assert!(p >= 3, "modulus must be an odd prime");
        let k = 64 - p.leading_zeros();
        Fp {
            p,
            mu: (1u128 << (2 * k)) / p as u128,
            k,
        }
    }

    /// `x mod p` for `x < p²` (HAC 14.42 — the estimate leaves a residue
    /// below `3p`, hence at most two corrections).
    #[inline]
    fn reduce(self, x: u128) -> u64 {
        let q = ((x >> (self.k - 1)) * self.mu) >> (self.k + 1);
        let mut r = (x - q * self.p as u128) as u64;
        while r >= self.p {
            r -= self.p;
        }
        r
    }

    /// `x mod p` for any `x`. Only called once per dot-product output, where
    /// the deferred accumulator has grown past `p²`.
    #[inline]
    fn reduce_wide(self, x: u128) -> u64 {
        (x % self.p as u128) as u64
    }

    #[inline]
    fn add(self, a: u64, b: u64) -> u64 {
        let s = a + b;
        if s >= self.p {
            s - self.p
        } else {
            s
        }
    }

    #[inline]
    fn sub(self, a: u64, b: u64) -> u64 {
        if a >= b {
            a - b
        } else {
            a + self.p - b
        }
    }

    #[inline]
    fn mul(self, a: u64, b: u64) -> u64 {
        self.reduce(a as u128 * b as u128)
    }

    fn pow(self, mut a: u64, mut e: u64) -> u64 {
        let mut r = 1u64;
        a %= self.p;
        while e > 0 {
            if e & 1 == 1 {
                r = self.mul(r, a);
            }
            a = self.mul(a, a);
            e >>= 1;
        }
        r
    }

    fn inv(self, a: u64) -> u64 {
        if a == 1 {
            return 1;
        }
        self.pow(a, self.p - 2)
    }

    /// Shoup companion `⌊w·2^64/p⌋` for a multiplier reused across a loop.
    #[inline]
    fn shoup(self, w: u64) -> u64 {
        (((w as u128) << 64) / self.p as u128) as u64
    }

    /// `a·w mod p` given `w`'s precomputed companion. Two multiplies and a
    /// correction, against Barrett's three.
    #[inline]
    fn mul_shoup(self, a: u64, w: u64, w_shoup: u64) -> u64 {
        let q = ((a as u128 * w_shoup as u128) >> 64) as u64;
        let r = a.wrapping_mul(w).wrapping_sub(q.wrapping_mul(self.p));
        if r >= self.p {
            r - self.p
        } else {
            r
        }
    }
}

/// Centered representative, matching `kahe::reduce_centered` so that
/// pack → encrypt → decrypt → unpack is the identity on `F_p`.
#[inline]
fn center(x: u64, p: u64) -> i64 {
    if x >= p / 2 {
        x as i64 - p as i64
    } else {
        x as i64
    }
}

#[inline]
fn canon(x: i64, p: u64) -> u64 {
    x.rem_euclid(p as i64) as u64
}

// ---------------------------------------------------------------
// F_p[X], coefficients little-endian by degree, trailing zeros trimmed
// ---------------------------------------------------------------

fn trim(mut v: Vec<u64>) -> Vec<u64> {
    while v.last() == Some(&0) {
        v.pop();
    }
    v
}

fn poly_sub(a: &[u64], b: &[u64], f: Fp) -> Vec<u64> {
    let mut out = vec![0u64; a.len().max(b.len())];
    out[..a.len()].copy_from_slice(a);
    for (o, &x) in out.iter_mut().zip(b.iter()) {
        *o = f.sub(*o, x);
    }
    trim(out)
}

/// Schoolbook, accumulating raw products and reducing once per output
/// coefficient. Each accumulator sums at most `min(|a|,|b|)` terms below
/// `p² < 2^72`, so `u128` has room for degrees far past any capacity here.
fn poly_mul(a: &[u64], b: &[u64], f: Fp) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut acc = vec![0u128; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        if x == 0 {
            continue;
        }
        let xw = x as u128;
        for (j, &y) in b.iter().enumerate() {
            acc[i + j] += xw * y as u128;
        }
    }
    trim(acc.into_iter().map(|v| f.reduce_wide(v)).collect())
}

/// `(quotient, remainder)` of `a / b`. `b` must be nonzero.
fn poly_divmod(a: &[u64], b: &[u64], f: Fp) -> (Vec<u64>, Vec<u64>) {
    debug_assert!(!b.is_empty());
    if a.len() < b.len() {
        return (Vec::new(), a.to_vec());
    }
    let inv_lead = f.inv(*b.last().unwrap());
    let mut rem = a.to_vec();
    let mut q = vec![0u64; a.len() - b.len() + 1];
    for shift in (0..q.len()).rev() {
        let c = f.mul(rem[shift + b.len() - 1], inv_lead);
        if c == 0 {
            continue;
        }
        q[shift] = c;
        // `c` is fixed across the row, so it earns a Shoup companion.
        let cs = f.shoup(c);
        for (i, &bi) in b.iter().enumerate() {
            rem[shift + i] = f.sub(rem[shift + i], f.mul_shoup(bi, c, cs));
        }
    }
    (trim(q), trim(rem))
}

fn poly_rem(a: &[u64], b: &[u64], f: Fp) -> Vec<u64> {
    poly_divmod(a, b, f).1
}

/// Monic gcd, or the zero polynomial if both inputs are zero.
fn poly_gcd(a: &[u64], b: &[u64], f: Fp) -> Vec<u64> {
    let (mut x, mut y) = (a.to_vec(), b.to_vec());
    while !y.is_empty() {
        let r = poly_rem(&x, &y, f);
        x = y;
        y = r;
    }
    if x.is_empty() {
        return x;
    }
    let inv = f.inv(*x.last().unwrap());
    trim(x.iter().map(|&c| f.mul(c, inv)).collect())
}

fn poly_powmod(base: &[u64], mut e: u64, m: &[u64], f: Fp) -> Vec<u64> {
    let mut r = vec![1u64];
    let mut b = poly_rem(base, m, f);
    while e > 0 {
        if e & 1 == 1 {
            r = poly_rem(&poly_mul(&r, &b, f), m, f);
        }
        b = poly_rem(&poly_mul(&b, &b, f), m, f);
        e >>= 1;
    }
    trim(r)
}

/// All roots of a monic `lam`, or `None` unless `lam` splits into *distinct*
/// linear factors over `F_p` (which is exactly `lam | X^p − X`).
///
/// Equal-degree splitting with `d = 1`: for random `a`, `gcd((X+a)^{(p−1)/2} −
/// 1, f)` collects the roots `z` with `z + a` a quadratic residue, splitting
/// `f` with probability ≈ 1/2 per try. `a` walks a fixed sequence so decoding
/// stays deterministic.
fn roots_of_split(lam: &[u64], f: Fp) -> Option<Vec<u64>> {
    let x = vec![0u64, 1u64];
    if poly_powmod(&x, f.p, lam, f) != poly_rem(&x, lam, f) {
        return None;
    }
    let mut out = Vec::with_capacity(lam.len() - 1);
    let mut stack = vec![lam.to_vec()];
    let mut a: u64 = 0;
    while let Some(fac) = stack.pop() {
        match fac.len() {
            0 | 1 => continue,
            2 => {
                // monic X + c
                out.push(f.sub(0, fac[0]));
                continue;
            }
            _ => {}
        }
        loop {
            a = (a + 1) % f.p;
            let h = poly_sub(&poly_powmod(&[a, 1], (f.p - 1) / 2, &fac, f), &[1], f);
            let g = poly_gcd(&h, &fac, f);
            if g.len() > 1 && g.len() < fac.len() {
                let (cof, _) = poly_divmod(&fac, &g, f);
                stack.push(g);
                stack.push(cof);
                break;
            }
        }
    }
    out.sort_unstable();
    Some(out)
}

// ---------------------------------------------------------------
// Payload solve kernel
// ---------------------------------------------------------------
//
// The solve is one GEMM, `X = D·Wᵀ`, with `D` the k×k Lagrange-dual matrix and
// `W` the ξ×k payload block. Two structural points make it cheap:
//
//   * Reduction is deferred to one per output. A dot product of `k ≤ 2^20`
//     terms below `p² < 2^72` stays under `2^92`, so nothing needs reducing
//     inside the loop.
//   * Every operand splits into two 18-bit limbs (`p < 2^36`). Accumulating
//     the three limb-product classes separately keeps each accumulator below
//     `2·k·2^36`, which for `k < 2^26` fits a `u64` lane with no carries — so
//     `_mm256_mul_epu32` (32×32→64, the widest AVX2 integer multiply) can run
//     the inner loop with no shifts, masks, or corrections.
//
// `D` is stored pre-split as limb planes so the inner loop does no masking,
// and symbols are processed in tiles so `D` is streamed once per tile rather
// than once per symbol.

const LIMB_BITS: u32 = 18;
const LIMB_MASK: u64 = (1 << LIMB_BITS) - 1;
/// Symbols per tile. `D` is k²·8 B (720 KB at k=300) and is re-read per tile,
/// so the tile wants to be wide enough to amortise that against L2.
const TILE: usize = 24;

/// `D` as two 18-bit limb planes, row-major, `k` columns per row.
struct DualLimbs {
    lo: Vec<u32>,
    hi: Vec<u32>,
    k: usize,
}

impl DualLimbs {
    fn new(dual: &[Vec<u64>]) -> Self {
        let k = dual.len();
        assert!(k < 1 << 26, "limb accumulators assume k < 2^26");
        let mut lo = Vec::with_capacity(k * k);
        let mut hi = Vec::with_capacity(k * k);
        for row in dual {
            debug_assert_eq!(row.len(), k);
            for &c in row {
                lo.push((c & LIMB_MASK) as u32);
                hi.push((c >> LIMB_BITS) as u32);
            }
        }
        Self { lo, hi, k }
    }
}

/// `Σ_m D[i][m]·w[m]` for one contributor row, as the three limb sums
/// `(A00, A01, A11)`. Caller recombines and reduces.
#[inline]
fn dot_limbs_scalar(d_lo: &[u32], d_hi: &[u32], w_lo: &[u32], w_hi: &[u32]) -> (u64, u64, u64) {
    let (mut a00, mut a01, mut a11) = (0u64, 0u64, 0u64);
    for i in 0..d_lo.len() {
        let (dl, dh) = (d_lo[i] as u64, d_hi[i] as u64);
        let (wl, wh) = (w_lo[i] as u64, w_hi[i] as u64);
        a00 += dl * wl;
        a01 += dl * wh + dh * wl;
        a11 += dh * wh;
    }
    (a00, a01, a11)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_limbs_avx2(
    d_lo: &[u32],
    d_hi: &[u32],
    w_lo: &[u32],
    w_hi: &[u32],
) -> (u64, u64, u64) {
    use std::arch::x86_64::*;
    let n = d_lo.len();
    let (mut v00, mut v01, mut v11) = (
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
    );
    // `mul_epu32` reads the low 32 bits of each 64-bit lane, so four u32 limbs
    // are widened to four 64-bit lanes per step.
    let mut i = 0usize;
    while i + 4 <= n {
        let dl = _mm256_cvtepu32_epi64(_mm_loadu_si128(d_lo.as_ptr().add(i) as *const __m128i));
        let dh = _mm256_cvtepu32_epi64(_mm_loadu_si128(d_hi.as_ptr().add(i) as *const __m128i));
        let wl = _mm256_cvtepu32_epi64(_mm_loadu_si128(w_lo.as_ptr().add(i) as *const __m128i));
        let wh = _mm256_cvtepu32_epi64(_mm_loadu_si128(w_hi.as_ptr().add(i) as *const __m128i));
        v00 = _mm256_add_epi64(v00, _mm256_mul_epu32(dl, wl));
        v01 = _mm256_add_epi64(
            v01,
            _mm256_add_epi64(_mm256_mul_epu32(dl, wh), _mm256_mul_epu32(dh, wl)),
        );
        v11 = _mm256_add_epi64(v11, _mm256_mul_epu32(dh, wh));
        i += 4;
    }
    let horiz = |v: __m256i| -> u64 {
        let mut buf = [0u64; 4];
        _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, v);
        buf[0] + buf[1] + buf[2] + buf[3]
    };
    let (mut a00, mut a01, mut a11) = (horiz(v00), horiz(v01), horiz(v11));
    let (t00, t01, t11) = dot_limbs_scalar(&d_lo[i..], &d_hi[i..], &w_lo[i..], &w_hi[i..]);
    a00 += t00;
    a01 += t01;
    a11 += t11;
    (a00, a01, a11)
}

#[inline]
fn dot_limbs(d_lo: &[u32], d_hi: &[u32], w_lo: &[u32], w_hi: &[u32]) -> (u64, u64, u64) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_limbs_avx2(d_lo, d_hi, w_lo, w_hi) };
        }
    }
    dot_limbs_scalar(d_lo, d_hi, w_lo, w_hi)
}

#[inline]
fn recombine(a00: u64, a01: u64, a11: u64, f: Fp) -> u64 {
    f.reduce_wide(a00 as u128 + ((a01 as u128) << LIMB_BITS) + ((a11 as u128) << (2 * LIMB_BITS)))
}

/// Payloads for a tile of symbols: `x[s][i] = Σ_m D[i][m]·W[s][m] mod p`.
///
/// Row-outer / symbol-inner, so each row of `D` is read once per tile instead
/// of once per symbol.
fn solve_tile(d: &DualLimbs, ws: &[Vec<u64>], f: Fp) -> Vec<Vec<u64>> {
    let (t, k) = (ws.len(), d.k);
    let mut w_lo = vec![0u32; t * k];
    let mut w_hi = vec![0u32; t * k];
    for (s, w) in ws.iter().enumerate() {
        for (m, &c) in w.iter().enumerate() {
            w_lo[s * k + m] = (c & LIMB_MASK) as u32;
            w_hi[s * k + m] = (c >> LIMB_BITS) as u32;
        }
    }
    let mut out = vec![vec![0u64; k]; t];
    for i in 0..k {
        let (d_lo, d_hi) = (&d.lo[i * k..(i + 1) * k], &d.hi[i * k..(i + 1) * k]);
        for s in 0..t {
            let (a00, a01, a11) = dot_limbs(
                d_lo,
                d_hi,
                &w_lo[s * k..(s + 1) * k],
                &w_hi[s * k..(s + 1) * k],
            );
            out[s][i] = recombine(a00, a01, a11, f);
        }
    }
    out
}

// ---------------------------------------------------------------
// Sketch
// ---------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct PronyParams {
    /// Maximum number of contributors the sketch can resolve.
    pub capacity: usize,
    /// Symbols per element; total payload is `payload_symbols · log₂ p` bits.
    pub payload_symbols: usize,
    /// Extra Vandermonde columns beyond `capacity`, checked against the
    /// recovered solution. Turns malformed contributions and capacity
    /// overflow into `CheckFailed` instead of silent garbage.
    pub slack: usize,
    pub p: u64,
}

impl PronyParams {
    pub fn new(capacity: usize, payload_symbols: usize) -> Self {
        Self::with_slack(capacity, payload_symbols, 1, PRONY_PRIME)
    }

    pub fn with_slack(capacity: usize, payload_symbols: usize, slack: usize, p: u64) -> Self {
        assert!(capacity >= 1, "capacity must be ≥ 1");
        assert!(payload_symbols >= 1, "payload_symbols must be ≥ 1");
        assert!(p > 2, "p must be an odd prime");
        assert!(
            (capacity as u64) < p,
            "Newton's identities divide by 1..capacity",
        );
        Self {
            capacity,
            payload_symbols,
            slack,
            p,
        }
    }

    /// Bits of payload per symbol: every `⌊log₂ p⌋`-bit pattern is a distinct
    /// residue. 35 at `PRONY_PRIME`, against `mse::BITS_PER_SYMBOL = 36`.
    pub fn bits_per_symbol(&self) -> usize {
        self.p.ilog2() as usize
    }

    /// Vandermonde columns: `capacity + slack`.
    pub fn cols(&self) -> usize {
        self.capacity + self.slack
    }

    /// `(cols + 1)` point syndromes plus `payload_symbols · cols` payload cells.
    pub fn total_scalars(&self) -> usize {
        (self.cols() + 1) + self.payload_symbols * self.cols()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PronySketch {
    pub params: PronyParams,
    /// `s[j] = Σ_i z_i^j`, `j ∈ [0, cols]`. `s[0]` is the contributor count.
    pub s: Vec<i64>,
    /// `w[sym][j] = Σ_i x_{i,sym} · z_i^j`, `j ∈ [0, cols)`.
    pub w: Vec<Vec<i64>>,
}

#[derive(Debug, PartialEq)]
pub enum PronyError {
    /// `s[0]` exceeds `capacity` — more contributors than the sketch resolves.
    CapacityExceeded { count: usize, capacity: usize },
    /// `Λ` does not split into distinct linear factors: two contributors drew
    /// the same `z`, or a contribution was malformed.
    NotSplit,
    /// A `slack` column disagrees with the recovered solution.
    CheckFailed,
    /// `params` of two sketches disagree under `add_assign`.
    ParamsMismatch,
    /// `payload` slice length disagrees with `params.payload_symbols`.
    PayloadArity,
}

impl PronySketch {
    pub fn new(params: PronyParams) -> Self {
        let cols = params.cols();
        let w = (0..params.payload_symbols)
            .map(|_| vec![0i64; cols])
            .collect();
        Self {
            s: vec![0i64; cols + 1],
            w,
            params,
        }
    }

    /// Insert one element under a fresh random evaluation point.
    pub fn insert<R: Rng>(&mut self, rng: &mut R, payload: &[i64]) {
        let p = self.params.p;
        let z = rng.gen_range(1..p);
        self.insert_with_z(payload, z);
    }

    /// Insert with a caller-supplied point — useful for deterministic tests.
    /// `z` must be a nonzero residue.
    pub fn insert_with_z(&mut self, payload: &[i64], z: u64) {
        assert_eq!(
            payload.len(),
            self.params.payload_symbols,
            "payload arity must match params.payload_symbols",
        );
        let p = self.params.p;
        assert!(z > 0 && z < p, "z must be a nonzero residue in [1, p)");
        let f = Fp::new(p);
        let cols = self.params.cols();

        let mut pows = Vec::with_capacity(cols + 1);
        let mut cur = 1u64;
        for _ in 0..=cols {
            pows.push(cur);
            cur = f.mul(cur, z);
        }

        for (slot, &pw) in self.s.iter_mut().zip(pows.iter()) {
            *slot = center(f.add(canon(*slot, p), pw), p);
        }
        self.w
            .par_iter_mut()
            .zip(payload.par_iter())
            .for_each(|(sym, &x)| {
                let xc = canon(x, p);
                for (slot, &pw) in sym.iter_mut().zip(pows.iter()) {
                    *slot = center(f.add(canon(*slot, p), f.mul(xc, pw)), p);
                }
            });
    }

    /// Pointwise add another sketch into self. Both must share `params`.
    pub fn add_assign(&mut self, other: &Self) -> Result<(), PronyError> {
        if self.params != other.params {
            return Err(PronyError::ParamsMismatch);
        }
        let p = self.params.p;
        let f = Fp::new(p);
        for (a, &b) in self.s.iter_mut().zip(other.s.iter()) {
            *a = center(f.add(canon(*a, p), canon(b, p)), p);
        }
        for (sa, sb) in self.w.iter_mut().zip(other.w.iter()) {
            for (a, &b) in sa.iter_mut().zip(sb.iter()) {
                *a = center(f.add(canon(*a, p), canon(b, p)), p);
            }
        }
        Ok(())
    }

    /// Recover `(z_i, payload_i)` for every contributor, sorted by `z_i`.
    pub fn decode_with_points(&self) -> Result<Vec<(u64, Vec<i64>)>, PronyError> {
        let p = self.params.p;
        let f = Fp::new(p);
        let cols = self.params.cols();
        let count = canon(self.s[0], p);
        if count > self.params.capacity as u64 {
            return Err(PronyError::CapacityExceeded {
                count: count as usize,
                capacity: self.params.capacity,
            });
        }
        let k = count as usize;
        if k == 0 {
            let clean = self.s.iter().all(|&x| x == 0)
                && self.w.iter().all(|sym| sym.iter().all(|&x| x == 0));
            return if clean {
                Ok(Vec::new())
            } else {
                Err(PronyError::CheckFailed)
            };
        }

        // Newton's identities: e_m = (1/m) Σ_{i=1..m} (−1)^{i−1} e_{m−i} P_i.
        // The alternating sum is split into two deferred accumulators so the
        // inner loop reduces nothing.
        let mut e = vec![0u64; k + 1];
        e[0] = 1;
        let syn: Vec<u64> = (0..=cols).map(|i| canon(self.s[i], p)).collect();
        for m in 1..=k {
            let (mut pos, mut neg) = (0u128, 0u128);
            for i in 1..=m {
                let term = e[m - i] as u128 * syn[i] as u128;
                if i % 2 == 1 {
                    pos += term;
                } else {
                    neg += term;
                }
            }
            let acc = f.sub(f.reduce_wide(pos), f.reduce_wide(neg));
            e[m] = f.mul(acc, f.inv(m as u64 % p));
        }
        // Λ(X) = Σ_m (−1)^m e_m X^{k−m}, monic of degree k.
        let mut lam = vec![0u64; k + 1];
        for m in 0..=k {
            lam[k - m] = if m % 2 == 0 { e[m] } else { f.sub(0, e[m]) };
        }

        let z = roots_of_split(&lam, f).ok_or(PronyError::NotSplit)?;
        if z.len() != k || z.contains(&0) {
            return Err(PronyError::NotSplit);
        }

        // Unused point syndromes must match the recovered points.
        let mut zp: Vec<u64> = z.iter().map(|&zi| f.pow(zi, k as u64 + 1)).collect();
        for j in (k + 1)..=cols {
            let got = zp.iter().fold(0u64, |a, &v| f.add(a, v));
            if got != syn[j] {
                return Err(PronyError::CheckFailed);
            }
            for (v, &zi) in zp.iter_mut().zip(z.iter()) {
                *v = f.mul(*v, zi);
            }
        }

        // Lagrange duals: b_i = Λ/(X − z_i) by synthetic division, scaled by
        // 1/Λ'(z_i) = 1/b_i(z_i). Then x_{i,s} = Σ_m b_{i,m}·W[s][m].
        let dual: Vec<Vec<u64>> = z
            .iter()
            .map(|&zi| {
                let mut b = vec![0u64; k];
                b[k - 1] = 1;
                for m in (1..k).rev() {
                    b[m - 1] = f.add(lam[m], f.mul(zi, b[m]));
                }
                let mut den = 0u64;
                for &c in b.iter().rev() {
                    den = f.add(f.mul(den, zi), c);
                }
                let inv = f.inv(den);
                b.iter().map(|&c| f.mul(c, inv)).collect()
            })
            .collect();
        let dual_limbs = DualLimbs::new(&dual);

        let xs: Vec<Vec<u64>> = self
            .w
            .par_chunks(TILE)
            .flat_map_iter(|tile| {
                let ws: Vec<Vec<u64>> = tile
                    .iter()
                    .map(|sym| sym[..k].iter().map(|&v| canon(v, p)).collect())
                    .collect();
                solve_tile(&dual_limbs, &ws, f)
            })
            .collect();

        // Unused payload columns must match the recovered payloads. The power
        // ladder depends only on the points, so it is built once here rather
        // than per symbol.
        let zpow: Vec<Vec<u64>> = (k..cols)
            .scan(
                z.iter()
                    .map(|&zi| f.pow(zi, k as u64))
                    .collect::<Vec<u64>>(),
                |cur, _| {
                    let this = cur.clone();
                    for (v, &zi) in cur.iter_mut().zip(z.iter()) {
                        *v = f.mul(*v, zi);
                    }
                    Some(this)
                },
            )
            .collect();
        let bad = self.w.par_iter().zip(xs.par_iter()).any(|(sym, xrow)| {
            zpow.iter().enumerate().any(|(idx, zj)| {
                let acc = xrow
                    .iter()
                    .zip(zj.iter())
                    .fold(0u128, |a, (&x, &v)| a + x as u128 * v as u128);
                f.reduce_wide(acc) != canon(sym[k + idx], p)
            })
        });
        if bad {
            return Err(PronyError::CheckFailed);
        }

        Ok(z.iter()
            .enumerate()
            .map(|(i, &zi)| (zi, xs.iter().map(|row| row[i] as i64).collect()))
            .collect())
    }

    /// Recover the multiset of payloads, sorted lexicographically —
    /// signature-compatible with `MseEncoding::decode`.
    pub fn decode(&self) -> Result<Vec<Vec<i64>>, PronyError> {
        let mut out: Vec<Vec<i64>> = self
            .decode_with_points()?
            .into_iter()
            .map(|(_, x)| x)
            .collect();
        out.sort();
        Ok(out)
    }

    /// Number of `KahePoly`s required to pack this sketch.
    pub fn n_polys(params: &PronyParams) -> usize {
        params.total_scalars().div_ceil(N)
    }

    /// Cover traffic: zero polys, same count as `pack()`. Contributes nothing
    /// to the sum, so it does not consume capacity.
    pub fn cover(params: &PronyParams) -> Vec<KahePoly> {
        Self::new(params.clone()).pack()
    }

    /// Flatten `S` then `W_0 … W_{ξ−1}` into `KahePoly` coefficient slots.
    pub fn pack(&self) -> Vec<KahePoly> {
        let total = self.params.total_scalars();
        let n_polys = total.div_ceil(N);
        let mut polys = Vec::with_capacity(n_polys);
        let mut buf = [0i64; N];
        let mut written = 0usize;
        let mut feed = |val: i64, polys: &mut Vec<KahePoly>| {
            buf[written % N] = val;
            written += 1;
            if written.is_multiple_of(N) {
                polys.push(KahePoly::from_coeffs(buf));
                buf = [0i64; N];
            }
        };
        for &x in &self.s {
            feed(x, &mut polys);
        }
        for sym in &self.w {
            for &x in sym {
                feed(x, &mut polys);
            }
        }
        if !written.is_multiple_of(N) {
            polys.push(KahePoly::from_coeffs(buf));
        }
        debug_assert_eq!(polys.len(), n_polys);
        polys
    }

    /// Inverse of `pack`. `polys.len()` must equal `n_polys(params)`.
    pub fn unpack(params: &PronyParams, polys: &[KahePoly]) -> Self {
        assert_eq!(polys.len(), params.total_scalars().div_ceil(N));
        let mut flat: Vec<i64> = Vec::with_capacity(polys.len() * N);
        for p in polys {
            let mut q = *p;
            q.normalize();
            flat.extend_from_slice(q.coeffs());
        }
        let cols = params.cols();
        let s = flat[..cols + 1].to_vec();
        let base = cols + 1;
        let w = (0..params.payload_symbols)
            .map(|i| flat[base + i * cols..base + (i + 1) * cols].to_vec())
            .collect();
        Self {
            params: params.clone(),
            s,
            w,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    const SMALL_P: u64 = 1_048_583; // smallest prime > 2^20

    #[test]
    fn field_and_poly_sanity() {
        let f = Fp::new(SMALL_P);
        assert_eq!(f.mul(f.inv(12345), 12345), 1);
        // (X−2)(X−3)(X−5) = X³ − 10X² + 31X − 30
        let poly = poly_mul(
            &poly_mul(&[f.sub(0, 2), 1], &[f.sub(0, 3), 1], f),
            &[f.sub(0, 5), 1],
            f,
        );
        assert_eq!(poly, vec![f.sub(0, 30), 31, f.sub(0, 10), 1]);
        assert_eq!(roots_of_split(&poly, f), Some(vec![2, 3, 5]));
        // A repeated root does not divide X^p − X.
        let g = poly_mul(&[f.sub(0, 2), 1], &poly, f);
        assert_eq!(roots_of_split(&g, f), None);
    }

    /// Barrett must agree with the reference remainder everywhere it is used:
    /// `reduce` over `[0, p²)` and `reduce_wide` over the deferred-accumulator
    /// range.
    #[test]
    fn barrett_matches_reference() {
        let mut rng = ChaCha20Rng::from_seed([41u8; 32]);
        for &p in &[SMALL_P, PRONY_PRIME, 3, 5] {
            let f = Fp::new(p);
            let pp = p as u128 * p as u128;
            for x in [0u128, 1, p as u128 - 1, p as u128, pp - 1] {
                assert_eq!(f.reduce(x), (x % p as u128) as u64, "reduce {x} mod {p}");
            }
            for _ in 0..20_000 {
                let x = rng.gen::<u128>() % pp;
                assert_eq!(f.reduce(x), (x % p as u128) as u64);
                let (a, w) = (x as u64 % p, (x >> 64) as u64 % p);
                assert_eq!(
                    f.mul_shoup(a, w, f.shoup(w)),
                    ((a as u128 * w as u128) % p as u128) as u64,
                    "shoup {a}·{w} mod {p}",
                );
                assert_eq!(f.mul(x as u64 % p, (x >> 64) as u64 % p), {
                    let a = (x as u64 % p) as u128;
                    let b = ((x >> 64) as u64 % p) as u128;
                    ((a * b) % p as u128) as u64
                });
            }
            // Deferred accumulators reach ~2^92 at the capacities we allow.
            for _ in 0..5_000 {
                let x = rng.gen::<u128>() >> 36;
                assert_eq!(f.reduce_wide(x), (x % p as u128) as u64);
            }
        }
    }

    /// The vector kernel must be bit-identical to the scalar one — nothing
    /// else distinguishes a correct AVX2 path from a silently wrong one.
    #[test]
    fn solve_kernel_avx2_matches_scalar() {
        let mut rng = ChaCha20Rng::from_seed([43u8; 32]);
        let f = Fp::new(PRONY_PRIME);
        for k in [1usize, 3, 4, 7, 8, 17, 64, 301] {
            let dual: Vec<Vec<u64>> = (0..k)
                .map(|_| (0..k).map(|_| rng.gen_range(0..f.p)).collect())
                .collect();
            let d = DualLimbs::new(&dual);
            let w: Vec<u64> = (0..k).map(|_| rng.gen_range(0..f.p)).collect();
            let (w_lo, w_hi): (Vec<u32>, Vec<u32>) = w
                .iter()
                .map(|&c| ((c & LIMB_MASK) as u32, (c >> LIMB_BITS) as u32))
                .unzip();

            for i in 0..k {
                let (d_lo, d_hi) = (&d.lo[i * k..(i + 1) * k], &d.hi[i * k..(i + 1) * k]);
                let want = dual[i]
                    .iter()
                    .zip(w.iter())
                    .fold(0u128, |a, (&x, &y)| a + x as u128 * y as u128);
                let want = f.reduce_wide(want);

                let s = dot_limbs_scalar(d_lo, d_hi, &w_lo, &w_hi);
                assert_eq!(recombine(s.0, s.1, s.2, f), want, "scalar k={k} row={i}");

                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("avx2") {
                    let v = unsafe { dot_limbs_avx2(d_lo, d_hi, &w_lo, &w_hi) };
                    assert_eq!(v, s, "avx2 limb sums k={k} row={i}");
                }
            }
        }
    }

    /// Exercises the ρ² solve path under `cargo test`, not only the example.
    #[test]
    fn capacity_300_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([47u8; 32]);
        let cap = 300;
        let mut sk = PronySketch::new(PronyParams::new(cap, 8));
        let mut msgs: Vec<Vec<i64>> = (0..cap)
            .map(|i| {
                (0..8)
                    .map(|s| ((i * 31 + s * 7) % 100_000) as i64)
                    .collect()
            })
            .collect();
        for m in &msgs {
            sk.insert(&mut rng, m);
        }
        let mut got = sk.decode().unwrap();
        msgs.sort();
        got.sort();
        assert_eq!(got, msgs);
    }

    #[test]
    fn single_element_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        let pp = PronyParams::new(4, 1);
        let mut sk = PronySketch::new(pp);
        sk.insert(&mut rng, &[42]);
        assert_eq!(sk.decode().unwrap(), vec![vec![42]]);
    }

    #[test]
    fn multi_element_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        let mut sk = PronySketch::new(PronyParams::new(8, 1));
        let mut elements = vec![1, 7, 13, 19, 100, 200, 300, 400];
        for &x in &elements {
            sk.insert(&mut rng, &[x]);
        }
        let mut got: Vec<i64> = sk.decode().unwrap().into_iter().map(|t| t[0]).collect();
        elements.sort();
        got.sort();
        assert_eq!(got, elements);
    }

    /// Capacity is a hard bound, unlike peeling's probabilistic one: exactly
    /// `capacity` contributors must decode.
    #[test]
    fn full_capacity_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([9u8; 32]);
        let cap = 32;
        let mut sk = PronySketch::new(PronyParams::new(cap, 3));
        let mut msgs: Vec<Vec<i64>> = (0..cap)
            .map(|i| vec![i as i64 * 7 + 1, i as i64 * 11, 1 << 31])
            .collect();
        for m in &msgs {
            sk.insert(&mut rng, m);
        }
        let mut got = sk.decode().unwrap();
        msgs.sort();
        got.sort();
        assert_eq!(got, msgs);
    }

    #[test]
    fn union_via_pointwise_add() {
        let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
        let pp = PronyParams::new(8, 1);
        let mut a = PronySketch::new(pp.clone());
        let mut b = PronySketch::new(pp);
        for x in [1, 5, 9, 17] {
            a.insert(&mut rng, &[x]);
        }
        for x in [2, 6, 11, 23] {
            b.insert(&mut rng, &[x]);
        }
        a.add_assign(&b).unwrap();
        let mut got: Vec<i64> = a.decode().unwrap().into_iter().map(|t| t[0]).collect();
        got.sort();
        assert_eq!(got, vec![1, 2, 5, 6, 9, 11, 17, 23]);
    }

    #[test]
    fn cover_adds_nothing() {
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        let pp = PronyParams::new(8, 2);
        let mut a = PronySketch::new(pp.clone());
        a.insert(&mut rng, &[5, 6]);
        let cover = PronySketch::unpack(&pp, &PronySketch::cover(&pp));
        a.add_assign(&cover).unwrap();
        assert_eq!(a.decode().unwrap(), vec![vec![5, 6]]);
        assert_eq!(
            PronySketch::new(pp).decode().unwrap(),
            Vec::<Vec<i64>>::new()
        );
    }

    #[test]
    fn pack_sum_unpacks_to_union() {
        let mut rng = ChaCha20Rng::from_seed([6u8; 32]);
        let pp = PronyParams::new(8, 2);
        let mut a = PronySketch::new(pp.clone());
        let mut b = PronySketch::new(pp.clone());
        for x in [100i64, 200, 300] {
            a.insert(&mut rng, &[x, x + 1]);
        }
        for x in [400i64, 500, 600] {
            b.insert(&mut rng, &[x, x + 1]);
        }
        let (pa, pb) = (a.pack(), b.pack());
        assert_eq!(pa.len(), PronySketch::n_polys(&pp));
        let summed: Vec<KahePoly> = pa.iter().zip(pb.iter()).map(|(x, y)| *x + *y).collect();
        let mut got = PronySketch::unpack(&pp, &summed).decode().unwrap();
        got.sort();
        assert_eq!(
            got,
            vec![
                vec![100, 101],
                vec![200, 201],
                vec![300, 301],
                vec![400, 401],
                vec![500, 501],
                vec![600, 601]
            ]
        );
    }

    /// Packed cells stay centered in `[−p/2, p/2)`, which the KAHE noise
    /// budget's `ρ·t/2` term assumes.
    #[test]
    fn packed_cells_are_centered() {
        let mut rng = ChaCha20Rng::from_seed([8u8; 32]);
        let pp = PronyParams::new(16, 2);
        let mut sk = PronySketch::new(pp.clone());
        for i in 0..16 {
            sk.insert(&mut rng, &[i, i * 3]);
        }
        let half = (pp.p / 2) as i64;
        for poly in sk.pack() {
            let mut q = poly;
            q.normalize();
            assert!(q.coeffs().iter().all(|&c| c >= -half - 1 && c < half + 1));
        }
    }

    #[test]
    fn overflowing_capacity_is_detected() {
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        let pp = PronyParams::new(4, 1);
        let mut sk = PronySketch::new(pp);
        for i in 0..6 {
            sk.insert(&mut rng, &[i]);
        }
        assert_eq!(
            sk.decode(),
            Err(PronyError::CapacityExceeded {
                count: 6,
                capacity: 4
            })
        );
    }

    /// Two contributors on the same point: `Λ` is not squarefree. Detected,
    /// not returned as garbage.
    #[test]
    fn duplicate_point_is_detected() {
        let pp = PronyParams::new(8, 1);
        let mut sk = PronySketch::new(pp);
        sk.insert_with_z(&[11], 12345);
        sk.insert_with_z(&[22], 12345);
        sk.insert_with_z(&[33], 999);
        assert_eq!(sk.decode(), Err(PronyError::NotSplit));
    }

    /// A contribution that is not a Vandermonde row fails the slack check.
    #[test]
    fn malformed_contribution_is_detected() {
        let mut rng = ChaCha20Rng::from_seed([13u8; 32]);
        let pp = PronyParams::with_slack(8, 1, 1, PRONY_PRIME);
        let mut sk = PronySketch::new(pp);
        for x in [3i64, 4, 5] {
            sk.insert(&mut rng, &[x]);
        }
        sk.w[0][2] += 7;
        assert_eq!(sk.decode(), Err(PronyError::CheckFailed));
    }

    #[test]
    #[should_panic(expected = "payload arity must match")]
    fn payload_arity_mismatch_panics() {
        let mut rng = ChaCha20Rng::from_seed([29u8; 32]);
        let mut sk = PronySketch::new(PronyParams::new(8, 3));
        sk.insert(&mut rng, &[1, 2]);
    }
}
