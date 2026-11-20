//! Prony / Reed–Solomon syndrome sketch — an MDS alternative to `mse.rs`.
//!
//! Same contract as [`crate::mse::MseEncoding`]: an additively homomorphic
//! sketch over the KAHE plaintext modulus, packed into `KahePoly` slots, whose
//! coefficient-wise sum decodes to the multiset union. Different mechanism:
//! instead of hashing each element into `γ` of `δ` buckets and peeling, each
//! contributor picks a random key `r ∈ F^*` and writes a Vandermonde row.
//!
//! The field is `F = F_{t²} = F_t[Z]/(Z² − 3)`, with `t` the KAHE plaintext
//! modulus. As an additive group `F ≅ Z_t²`, so an element of `F` takes two
//! KAHE plaintext slots and sketches add coordinate-wise. A payload of `ξ`
//! symbols of `Z_t` becomes `ξ_F = ⌈ξ/2⌉` elements of `F`, the symbols in
//! pairs, the last one padded with zero.
//!
//! ```text
//!   r ← F^*                                (per-element key)
//!   K[j]    += r^j                         (j ∈ [1, C])
//!   V[ℓ][j] += x_ℓ · r^j                   (ℓ ∈ [ξ_F], j ∈ [0, C))
//! ```
//!
//! That is `2C(1 + ξ_F)` scalars for capacity `C`. Summed over the round,
//! `K[j] = Σ_i r_i^j` are the power sums of the contributors' keys and
//! `V[ℓ][j] = Σ_i x_{i,ℓ} r_i^j`. Decoding:
//!
//! 1. Newton's identities turn `K[1..=C]` into the elementary symmetric
//!    polynomials `e_1, …, e_C` of the keys. The number of contributors `ρ̂`
//!    is the largest `m` with `e_m ≠ 0`; a zero sketch (cover traffic) has
//!    `ρ̂ = 0` and decodes to nothing.
//! 2. The `r_i` are the roots of `Λ(X) = Π_i (X − r_i)`, whose coefficients
//!    are `e_0, …, e_ρ̂` (Cantor–Zassenhaus over `F`). Decoding fails unless
//!    `Λ` has `ρ̂` distinct nonzero roots in `F`.
//! 3. With the keys known, each payload element is one transposed-Vandermonde
//!    solve: `x_{i,ℓ} = (Σ_m b_{i,m} V[ℓ][m]) / Λ'(r_i)` where
//!    `Λ(X)/(X − r_i) = Σ_m b_{i,m} X^m`. For `ρ̂ < C`, the payload columns
//!    `ρ̂..C` must agree with the solution.
//!
//! Newton's identities divide by `1..C`, so the characteristic `t` must exceed
//! the capacity.

use crate::kahe::T_MODULUS_DEFAULT;
use crate::rings::center_canonical_half_open_i64;
use crate::{KahePoly, N};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use sha3::digest::{ExtendableOutput as _, Update as _, XofReader as _};
use sha3::Shake256;

/// The prime `t` underneath the sketch's field: the KAHE plaintext modulus.
pub const PRONY_PRIME: u64 = T_MODULUS_DEFAULT;

/// The extension degree `ω` of the sketch's field `F = F_{t^ω}`. Two of `C`
/// contributors draw the same key, and the round fails to decode, with
/// probability at most `C²/t^ω`: about `2^-51.5` at `C = 300`.
pub const PRONY_OMEGA: usize = 2;

// An element's coordinates never straddle two packed polynomials.
const _: () = assert!(N.is_multiple_of(PRONY_OMEGA));

/// `F_{t²} = F_t[Z]/(Z² − NONRESIDUE)`; `Z² − 3` is irreducible since 3 is
/// not a square mod `t`.
const NONRESIDUE: u64 = 3;

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

    /// `x mod p` for `x < 2^{2k}` (HAC 14.42 — the estimate leaves a residue
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

    /// `x mod p` for any `x`.
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

    /// `acc + a·b`, unreduced, for residues `a, b < p`. Each term is below
    /// `p² < 2^72`, so a `u128` holds the sum of more than `2^55` of them;
    /// overflow panics in debug builds.
    #[inline]
    fn mul_acc(self, acc: Wide, a: u64, b: u64) -> Wide {
        Wide(acc.0 + a as u128 * b as u128)
    }

    /// `acc mod p`: Barrett while the sum is below `2^{2k}`, a division
    /// beyond.
    #[inline]
    fn reduce_acc(self, acc: Wide) -> u64 {
        if acc.0 >> (2 * self.k) == 0 {
            self.reduce(acc.0)
        } else {
            self.reduce_wide(acc.0)
        }
    }

    /// `(a·b + c·d) mod p`, with a single reduction.
    #[inline]
    fn mul2(self, a: u64, b: u64, c: u64, d: u64) -> u64 {
        self.reduce_acc(self.mul_acc(self.mul_acc(Wide::ZERO, a, b), c, d))
    }
}

/// A sum of products of residues mod `p`, not yet reduced. Built with
/// [`Fp::mul_acc`] and reduced with [`Fp::reduce_acc`], so that a sum of
/// products costs one reduction instead of one per product.
#[derive(Clone, Copy, Debug, Default)]
struct Wide(u128);

impl Wide {
    const ZERO: Wide = Wide(0);
}

// ---------------------------------------------------------------
// F = F_{t²}
// ---------------------------------------------------------------

/// `a + b·Z`, both coordinates canonical in `[0, p)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct F2 {
    a: u64,
    b: u64,
}

impl F2 {
    const ZERO: F2 = F2 { a: 0, b: 0 };
    const ONE: F2 = F2 { a: 1, b: 0 };

    fn is_zero(self) -> bool {
        self == Self::ZERO
    }
}

/// A sum of products in `F`, not yet reduced, as its coordinate sums
/// `Σ x_a y_a`, `Σ x_b y_b` and `Σ (x_a y_b + x_b y_a)`; see [`Ext::mul_acc`].
#[derive(Clone, Copy, Debug, Default)]
struct WideF2 {
    aa: Wide,
    bb: Wide,
    ab: Wide,
}

impl WideF2 {
    /// `x` itself, as the start of a sum.
    fn of(x: F2) -> Self {
        WideF2 {
            aa: Wide(x.a as u128),
            bb: Wide::ZERO,
            ab: Wide(x.b as u128),
        }
    }
}

/// Arithmetic in `F_{t²}`, on top of the Barrett context of `F_t`.
#[derive(Clone, Copy, Debug)]
struct Ext {
    f: Fp,
}

impl Ext {
    fn new(p: u64) -> Self {
        Ext { f: Fp::new(p) }
    }

    /// `|F| = p²`.
    fn order(self) -> u128 {
        self.f.p as u128 * self.f.p as u128
    }

    #[inline]
    fn add(self, x: F2, y: F2) -> F2 {
        F2 {
            a: self.f.add(x.a, y.a),
            b: self.f.add(x.b, y.b),
        }
    }

    #[inline]
    fn sub(self, x: F2, y: F2) -> F2 {
        F2 {
            a: self.f.sub(x.a, y.a),
            b: self.f.sub(x.b, y.b),
        }
    }

    #[inline]
    fn neg(self, x: F2) -> F2 {
        self.sub(F2::ZERO, x)
    }

    /// `(x_a + x_b Z)(y_a + y_b Z) = x_a y_a + 3 x_b y_b + (x_a y_b + x_b y_a) Z`.
    #[inline]
    fn mul(self, x: F2, y: F2) -> F2 {
        let f = self.f;
        F2 {
            a: f.add(f.mul(x.a, y.a), f.mul(NONRESIDUE, f.mul(x.b, y.b))),
            b: f.mul2(x.a, y.b, x.b, y.a),
        }
    }

    /// `acc + x·y`, unreduced.
    #[inline]
    fn mul_acc(self, acc: WideF2, x: F2, y: F2) -> WideF2 {
        let f = self.f;
        WideF2 {
            aa: f.mul_acc(acc.aa, x.a, y.a),
            bb: f.mul_acc(acc.bb, x.b, y.b),
            ab: f.mul_acc(f.mul_acc(acc.ab, x.a, y.b), x.b, y.a),
        }
    }

    /// `acc` as an element of `F`.
    #[inline]
    fn reduce_acc(self, acc: WideF2) -> F2 {
        let f = self.f;
        F2 {
            a: f.add(
                f.reduce_acc(acc.aa),
                f.mul(NONRESIDUE, f.reduce_acc(acc.bb)),
            ),
            b: f.reduce_acc(acc.ab),
        }
    }

    /// `x · c` for `c ∈ F_t`.
    #[inline]
    fn scale(self, x: F2, c: u64) -> F2 {
        F2 {
            a: self.f.mul(x.a, c),
            b: self.f.mul(x.b, c),
        }
    }

    /// `(a + bZ)⁻¹ = (a − bZ)/(a² − 3b²)`; the norm `a² − 3b²` is nonzero for
    /// `x ≠ 0` because `Z² − 3` is irreducible.
    fn inv(self, x: F2) -> F2 {
        let f = self.f;
        let norm = f.sub(f.mul(x.a, x.a), f.mul(NONRESIDUE, f.mul(x.b, x.b)));
        let n = f.inv(norm);
        F2 {
            a: f.mul(x.a, n),
            b: f.mul(f.sub(0, x.b), n),
        }
    }

    fn random<R: Rng>(self, rng: &mut R) -> F2 {
        F2 {
            a: rng.gen_range(0..self.f.p),
            b: rng.gen_range(0..self.f.p),
        }
    }
}

/// Centered representative, matching `kahe::reduce_centered` so that
/// pack → encrypt → decrypt → unpack is the identity on `F_p`.
#[inline]
fn center(x: u64, p: u64) -> i64 {
    center_canonical_half_open_i64(x as i64, p as i64)
}

#[inline]
fn canon(x: i64, p: u64) -> u64 {
    x.rem_euclid(p as i64) as u64
}

// ---------------------------------------------------------------
// F[X], coefficients little-endian by degree, trailing zeros trimmed
// ---------------------------------------------------------------

fn trim(mut v: Vec<F2>) -> Vec<F2> {
    while v.last().is_some_and(|c| c.is_zero()) {
        v.pop();
    }
    v
}

fn poly_sub(a: &[F2], b: &[F2], ext: Ext) -> Vec<F2> {
    let mut out = vec![F2::ZERO; a.len().max(b.len())];
    out[..a.len()].copy_from_slice(a);
    for (o, &x) in out.iter_mut().zip(b.iter()) {
        *o = ext.sub(*o, x);
    }
    trim(out)
}

/// Schoolbook, with one reduction per output coefficient.
fn poly_mul(a: &[F2], b: &[F2], ext: Ext) -> Vec<F2> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut acc = vec![WideF2::default(); a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        if x.is_zero() {
            continue;
        }
        for (j, &y) in b.iter().enumerate() {
            acc[i + j] = ext.mul_acc(acc[i + j], x, y);
        }
    }
    trim(acc.into_iter().map(|w| ext.reduce_acc(w)).collect())
}

/// `(quotient, remainder)` of `a / b`. `b` must be nonzero.
///
/// Long division with deferred reduction: each coefficient of `a` keeps the
/// products subtracted from it unreduced and is reduced only when it becomes
/// the leading coefficient or part of the remainder. Subtracting `c·b` is
/// adding `(−c)·b`, so the sums only grow.
fn poly_divmod(a: &[F2], b: &[F2], ext: Ext) -> (Vec<F2>, Vec<F2>) {
    debug_assert!(!b.is_empty());
    if a.len() < b.len() {
        return (Vec::new(), a.to_vec());
    }
    let mut acc: Vec<WideF2> = a.iter().map(|&x| WideF2::of(x)).collect();
    let inv_lead = ext.inv(*b.last().unwrap());
    let top = b.len() - 1;
    let mut q = vec![F2::ZERO; a.len() - top];
    for shift in (0..q.len()).rev() {
        let c = ext.mul(ext.reduce_acc(acc[shift + top]), inv_lead);
        if c.is_zero() {
            continue;
        }
        q[shift] = c;
        let nc = ext.neg(c);
        for (i, &y) in b[..top].iter().enumerate() {
            acc[shift + i] = ext.mul_acc(acc[shift + i], y, nc);
        }
    }
    let rem = acc[..top].iter().map(|&w| ext.reduce_acc(w)).collect();
    (trim(q), trim(rem))
}

fn poly_rem(a: &[F2], b: &[F2], ext: Ext) -> Vec<F2> {
    poly_divmod(a, b, ext).1
}

/// Monic gcd, or the zero polynomial if both inputs are zero.
fn poly_gcd(a: &[F2], b: &[F2], ext: Ext) -> Vec<F2> {
    let (mut x, mut y) = (a.to_vec(), b.to_vec());
    while !y.is_empty() {
        let r = poly_rem(&x, &y, ext);
        x = y;
        y = r;
    }
    if x.is_empty() {
        return x;
    }
    let inv = ext.inv(*x.last().unwrap());
    trim(x.iter().map(|&c| ext.mul(c, inv)).collect())
}

/// `(X + c)^e mod m`, `deg m ≥ 1`. Each multiplication by the linear factor is
/// `X·g + c·g` followed by a single division step, not a full product.
fn powmod_linear(c: F2, e: u128, m: &[F2], ext: Ext) -> Vec<F2> {
    let mut r = vec![F2::ONE];
    for bit in (0..u128::BITS - e.leading_zeros()).rev() {
        r = poly_rem(&poly_mul(&r, &r, ext), m, ext);
        if (e >> bit) & 1 == 1 {
            let mut h = vec![F2::ZERO; r.len() + 1];
            for (i, &g) in r.iter().enumerate() {
                h[i + 1] = ext.add(h[i + 1], g);
                h[i] = ext.add(h[i], ext.mul(c, g));
            }
            r = poly_rem(&trim(h), m, ext);
        }
    }
    trim(r)
}

/// The seed of the root finder's shifts: a hash of the polynomial.
fn split_seed(lam: &[F2]) -> [u8; 32] {
    let mut hasher = Shake256::default();
    hasher.update(b"panetiere/prony-roots/v1");
    for c in lam {
        hasher.update(&c.a.to_le_bytes());
        hasher.update(&c.b.to_le_bytes());
    }
    let mut seed = [0u8; 32];
    hasher.finalize_xof().read(&mut seed);
    seed
}

/// All roots of a monic `lam` of degree ≥ 1, or `None` unless `lam` splits
/// into distinct linear factors over `F` (which is exactly `lam | X^q − X`
/// for `q = |F|`).
fn roots_of_split(lam: &[F2], ext: Ext) -> Option<Vec<F2>> {
    let q = ext.order();
    let x = vec![F2::ZERO, F2::ONE];
    if powmod_linear(F2::ZERO, q, lam, ext) != poly_rem(&x, lam, ext) {
        return None;
    }
    let mut out = split_roots(lam.to_vec(), ext);
    out.sort_unstable();
    Some(out)
}

/// The roots of a monic `fac` that splits into distinct linear factors.
///
/// Equal-degree splitting with `d = 1` (Cantor–Zassenhaus): for a random
/// `c ∈ F`, `gcd((X + c)^{(q−1)/2} − 1, fac)` collects the roots `x` with
/// `x + c` a square, splitting `fac` with probability about 1/2 per try. The
/// two factors are split in parallel. Each factor's shifts come from a
/// ChaCha20 stream seeded with a hash of that factor, so the result is a
/// function of `fac` alone, however the work is scheduled; the shifts only
/// affect how long the splitting takes, never which roots it returns.
fn split_roots(fac: Vec<F2>, ext: Ext) -> Vec<F2> {
    match fac.len() {
        0 | 1 => return Vec::new(),
        // monic X + c
        2 => return vec![ext.neg(fac[0])],
        _ => {}
    }
    let half = (ext.order() - 1) / 2;
    let mut rng = ChaCha20Rng::from_seed(split_seed(&fac));
    loop {
        let c = ext.random(&mut rng);
        let h = poly_sub(&powmod_linear(c, half, &fac, ext), &[F2::ONE], ext);
        let g = poly_gcd(&h, &fac, ext);
        if g.len() > 1 && g.len() < fac.len() {
            let (cof, _) = poly_divmod(&fac, &g, ext);
            let (mut a, b) = rayon::join(|| split_roots(g, ext), || split_roots(cof, ext));
            a.extend(b);
            return a;
        }
    }
}

// ---------------------------------------------------------------
// Payload solve kernel
// ---------------------------------------------------------------
//
// The solve is one GEMM, `X = D·Vᵀ`, with `D` the k×k Lagrange-dual matrix and
// `V` the ξ×k payload block. Two structural points make it cheap:
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
// `D` is stored pre-split as limb planes so the inner loop does no masking.

const LIMB_BITS: u32 = 18;
const LIMB_MASK: u64 = (1 << LIMB_BITS) - 1;
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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_limbs_neon(
    d_lo: &[u32],
    d_hi: &[u32],
    w_lo: &[u32],
    w_hi: &[u32],
) -> (u64, u64, u64) {
    use std::arch::aarch64::*;
    let n = d_lo.len();
    let (mut v00, mut v01, mut v11) = (vdupq_n_u64(0), vdupq_n_u64(0), vdupq_n_u64(0));
    let mut i = 0usize;
    while i + 4 <= n {
        let dl = vld1q_u32(d_lo.as_ptr().add(i));
        let dh = vld1q_u32(d_hi.as_ptr().add(i));
        let wl = vld1q_u32(w_lo.as_ptr().add(i));
        let wh = vld1q_u32(w_hi.as_ptr().add(i));

        for (dl, dh, wl, wh) in [
            (
                vget_low_u32(dl),
                vget_low_u32(dh),
                vget_low_u32(wl),
                vget_low_u32(wh),
            ),
            (
                vget_high_u32(dl),
                vget_high_u32(dh),
                vget_high_u32(wl),
                vget_high_u32(wh),
            ),
        ] {
            v00 = vaddq_u64(v00, vmull_u32(dl, wl));
            v01 = vaddq_u64(v01, vaddq_u64(vmull_u32(dl, wh), vmull_u32(dh, wl)));
            v11 = vaddq_u64(v11, vmull_u32(dh, wh));
        }
        i += 4;
    }

    let (mut a00, mut a01, mut a11) = (vaddvq_u64(v00), vaddvq_u64(v01), vaddvq_u64(v11));
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
    #[cfg(target_arch = "aarch64")]
    return unsafe { dot_limbs_neon(d_lo, d_hi, w_lo, w_hi) };
    #[cfg(not(target_arch = "aarch64"))]
    dot_limbs_scalar(d_lo, d_hi, w_lo, w_hi)
}

#[inline]
fn recombine(a00: u64, a01: u64, a11: u64, f: Fp) -> u64 {
    f.reduce_wide(a00 as u128 + ((a01 as u128) << LIMB_BITS) + ((a11 as u128) << (2 * LIMB_BITS)))
}

fn solve_symbol(d: &DualLimbs, w: &[u64], f: Fp) -> Vec<u64> {
    let (w_lo, w_hi): (Vec<u32>, Vec<u32>) = w
        .iter()
        .map(|&c| ((c & LIMB_MASK) as u32, (c >> LIMB_BITS) as u32))
        .unzip();
    (0..d.k)
        .map(|i| {
            let row = i * d.k..(i + 1) * d.k;
            let (a00, a01, a11) = dot_limbs(&d.lo[row.clone()], &d.hi[row], &w_lo, &w_hi);
            recombine(a00, a01, a11, f)
        })
        .collect()
}

// ---------------------------------------------------------------
// Sketch
// ---------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct PronyParams {
    /// Maximum number of contributors the sketch can resolve.
    pub(crate) capacity: usize,
    /// Symbols of `Z_t` per element; total payload is `payload_symbols · log₂ p`
    /// bits.
    pub(crate) payload_symbols: usize,
    pub(crate) p: u64,
}

impl PronyParams {
    pub fn new(capacity: usize, payload_symbols: usize) -> Self {
        let p = PRONY_PRIME;
        assert!(capacity >= 1, "capacity must be ≥ 1");
        assert!(payload_symbols >= 1, "payload_symbols must be ≥ 1");
        assert!(
            (capacity as u64) < p,
            "Newton's identities divide by 1..capacity",
        );
        Self {
            capacity,
            payload_symbols,
            p,
        }
    }

    /// Bits of payload per symbol: every `⌊log₂ p⌋`-bit pattern is a distinct
    /// residue. Equal to `mse::BITS_PER_SYMBOL`, since both encodings use `t`.
    pub(crate) fn bits_per_symbol(&self) -> usize {
        self.p.ilog2() as usize
    }

    /// `ξ_F`: elements of `F` per payload, `ω` symbols each.
    fn payload_elements(&self) -> usize {
        self.payload_symbols.div_ceil(PRONY_OMEGA)
    }

    /// `capacity` power sums plus `ξ_F · capacity` payload cells, `ω` scalars
    /// each.
    fn total_scalars(&self) -> usize {
        PRONY_OMEGA * self.capacity * (1 + self.payload_elements())
    }
}

/// A round's summed sketch, for decoding. Clients don't build one: each
/// encodes its single element with [`PronySketch::encode`], and summing the
/// encodings, as KAHE does, gives the sketch of the union.
#[derive(Clone, Debug, PartialEq)]
pub struct PronySketch {
    params: PronyParams,
    /// `k[j − 1] = Σ_i r_i^j`, `j ∈ [1, capacity]`.
    k: Vec<F2>,
    /// `V[ℓ][j] = Σ_i x_{i,ℓ} · r_i^j`, `j ∈ [0, C)`, one row of `C` per
    /// payload element `ℓ`, as the summed plaintext holds them: coordinates
    /// in turn, each the centered representative mod `q`. The decoder
    /// reduces mod `t` only the cells it reads; see [`PronySketch::cell`].
    v: Vec<i64>,
}

#[derive(Debug, PartialEq)]
pub enum PronyError {
    /// `Λ` does not split into distinct nonzero linear factors: two
    /// contributors drew the same key, there were more contributors than the
    /// capacity, or a contribution was malformed.
    NotSplit,
    /// A payload column the solve did not use disagrees with the recovered
    /// solution.
    CheckFailed,
}

impl PronySketch {
    /// The encoding of one element under a fresh random key, packed into
    /// `n_polys(params)` `KahePoly`s.
    pub fn encode<R: Rng>(params: &PronyParams, rng: &mut R, payload: &[i64]) -> Vec<KahePoly> {
        let ext = Ext::new(params.p);
        let r = loop {
            let r = ext.random(rng);
            if !r.is_zero() {
                break r;
            }
        };
        Self::encode_with_key(params, payload, r)
    }

    /// The encoding of one element under the key `r`, a nonzero element of
    /// `F`: `r^j` for `j ∈ [1, C]`, then `x_ℓ · r^j` for `j ∈ [0, C)` per payload
    /// element, each element of `F` as its two coordinates.
    fn encode_with_key(params: &PronyParams, payload: &[i64], r: F2) -> Vec<KahePoly> {
        assert_eq!(
            payload.len(),
            params.payload_symbols,
            "payload arity must match params.payload_symbols",
        );
        let p = params.p;
        assert!(
            !r.is_zero() && r.a < p && r.b < p,
            "r must be a nonzero element of F"
        );
        let ext = Ext::new(p);
        let f = ext.f;
        let c = params.capacity;

        // r^0, …, r^C: the power sums take r^1..r^C, the payload r^0..r^{C−1}.
        let mut pows = Vec::with_capacity(c + 1);
        let mut cur = F2::ONE;
        for _ in 0..=c {
            pows.push(cur);
            cur = ext.mul(cur, r);
        }

        // The payload's symbols in pairs, the last one padded with zero, with
        // `3·x_b` precomputed: with `x` fixed per row, each coordinate of
        // `x·r^j` is then a sum of two products.
        let xs: Vec<[u64; 3]> = payload
            .chunks(PRONY_OMEGA)
            .map(|s| {
                let xa = canon(s[0], p);
                let xb = s.get(1).map_or(0, |&y| canon(y, p));
                [xa, xb, f.mul(NONRESIDUE, xb)]
            })
            .collect();
        let element = |e: usize| -> F2 {
            if e < c {
                return pows[e + 1];
            }
            let (row, j) = ((e - c) / c, (e - c) % c);
            let ([xa, xb, xb3], pw) = (xs[row], pows[j]);
            F2 {
                a: f.mul2(xa, pw.a, xb3, pw.b),
                b: f.mul2(xa, pw.b, xb, pw.a),
            }
        };
        // Each polynomial holds `N / ω` consecutive elements; the last one is
        // padded with zeros.
        let per_poly = N / PRONY_OMEGA;
        let n_elements = c * (1 + xs.len());
        (0..Self::n_polys(params))
            .into_par_iter()
            .map(|i| {
                let mut coeffs = [0i64; N];
                let start = i * per_poly;
                let end = (start + per_poly).min(n_elements);
                for (slot, e) in coeffs.chunks_exact_mut(PRONY_OMEGA).zip(start..end) {
                    let x = element(e);
                    slot[0] = center(x.a, p);
                    slot[1] = center(x.b, p);
                }
                KahePoly::from_coeffs(coeffs)
            })
            .collect()
    }

    /// Recover `(r_i, payload_i)` for every contributor, sorted by `r_i`.
    fn decode_with_keys(&self) -> Result<Vec<(F2, Vec<i64>)>, PronyError> {
        let p = self.params.p;
        let ext = Ext::new(p);
        let f = ext.f;
        let c = self.params.capacity;

        // Newton's identities: e_m = (1/m) Σ_{i=1..m} (−1)^{i−1} e_{m−i} K[i].
        let mut e = vec![F2::ZERO; c + 1];
        e[0] = F2::ONE;
        // The terms with odd and even i are summed separately, unreduced.
        for m in 1..=c {
            let mut odd = WideF2::default();
            let mut even = WideF2::default();
            for i in 1..=m {
                let acc = if i % 2 == 1 { &mut odd } else { &mut even };
                *acc = ext.mul_acc(*acc, e[m - i], self.k[i - 1]);
            }
            let sum = ext.sub(ext.reduce_acc(odd), ext.reduce_acc(even));
            e[m] = ext.scale(sum, f.inv(m as u64 % p));
        }
        // The number of contributors. Since `e_m = 0` for every `m > ρ̂`, the
        // power sums are exactly those of the roots of `Λ` once it splits, so
        // they need no separate check.
        let rho_hat = (0..=c).rev().find(|&m| !e[m].is_zero()).unwrap_or(0);
        if rho_hat == 0 {
            return if self.v.iter().all(|&x| x == 0) {
                Ok(Vec::new())
            } else {
                Err(PronyError::CheckFailed)
            };
        }
        // Λ(X) = Σ_m (−1)^m e_m X^{ρ̂−m}, monic of degree ρ̂.
        let mut lam = vec![F2::ZERO; rho_hat + 1];
        for m in 0..=rho_hat {
            lam[rho_hat - m] = if m % 2 == 0 { e[m] } else { ext.neg(e[m]) };
        }

        let r = roots_of_split(&lam, ext).ok_or(PronyError::NotSplit)?;
        if r.len() != rho_hat || r.iter().any(|x| x.is_zero()) {
            return Err(PronyError::NotSplit);
        }

        // Lagrange duals: b_i = Λ/(X − r_i) by synthetic division, scaled by
        // 1/Λ'(r_i) = 1/b_i(r_i). Then x_{i,ℓ} = Σ_m b_{i,m}·V[ℓ][m].
        let dual: Vec<Vec<F2>> = r
            .par_iter()
            .map(|&ri| {
                let mut b = vec![F2::ZERO; rho_hat];
                b[rho_hat - 1] = F2::ONE;
                for m in (1..rho_hat).rev() {
                    b[m - 1] = ext.add(lam[m], ext.mul(ri, b[m]));
                }
                let mut den = F2::ZERO;
                for &c in b.iter().rev() {
                    den = ext.add(ext.mul(den, ri), c);
                }
                let inv = ext.inv(den);
                b.iter().map(|&c| ext.mul(c, inv)).collect()
            })
            .collect();
        // The solve runs on the coordinates, Karatsuba-style: with
        // D = D_a + D_b Z and V = V_a + V_b Z,
        // D·V = (D_a V_a + 3 D_b V_b) + ((D_a + D_b)(V_a + V_b) − D_a V_a − D_b V_b) Z,
        // three products over F_t.
        let plane = |coord: &(dyn Fn(&F2) -> u64 + Sync)| {
            DualLimbs::new(
                &dual
                    .iter()
                    .map(|row| row.iter().map(coord).collect())
                    .collect::<Vec<Vec<u64>>>(),
            )
        };
        let d_a = plane(&|x| x.a);
        let d_b = plane(&|x| x.b);
        let d_s = plane(&|x| f.add(x.a, x.b));
        let xs: Vec<Vec<F2>> = (0..self.params.payload_elements())
            .into_par_iter()
            .map(|l| {
                let w: Vec<F2> = (0..rho_hat).map(|j| self.cell(l * c + j)).collect();
                let w_a: Vec<u64> = w.iter().map(|x| x.a).collect();
                let w_b: Vec<u64> = w.iter().map(|x| x.b).collect();
                let w_s: Vec<u64> = w.iter().map(|x| f.add(x.a, x.b)).collect();
                let aa = solve_symbol(&d_a, &w_a, f);
                let bb = solve_symbol(&d_b, &w_b, f);
                let ss = solve_symbol(&d_s, &w_s, f);
                (0..rho_hat)
                    .map(|i| F2 {
                        a: f.add(aa[i], f.mul(NONRESIDUE, bb[i])),
                        b: f.sub(ss[i], f.add(aa[i], bb[i])),
                    })
                    .collect()
            })
            .collect();

        // Unused payload columns must match the recovered payloads. The power
        // ladder depends only on the keys, so it is built once here rather
        // than per element.
        let rpow: Vec<Vec<F2>> = (rho_hat..c)
            .scan(
                r.iter()
                    .map(|&ri| (0..rho_hat).fold(F2::ONE, |acc, _| ext.mul(acc, ri)))
                    .collect::<Vec<F2>>(),
                |cur, _| {
                    let this = cur.clone();
                    for (pw, &ri) in cur.iter_mut().zip(r.iter()) {
                        *pw = ext.mul(*pw, ri);
                    }
                    Some(this)
                },
            )
            .collect();
        let bad = xs.par_iter().enumerate().any(|(l, xrow)| {
            rpow.iter().enumerate().any(|(idx, rj)| {
                let acc = xrow
                    .iter()
                    .zip(rj.iter())
                    .fold(WideF2::default(), |acc, (&x, &pw)| ext.mul_acc(acc, x, pw));
                ext.reduce_acc(acc) != self.cell(l * c + rho_hat + idx)
            })
        });
        if bad {
            return Err(PronyError::CheckFailed);
        }

        let symbols = self.params.payload_symbols;
        Ok(r.iter()
            .enumerate()
            .map(|(i, &ri)| {
                let mut payload: Vec<i64> = xs
                    .iter()
                    .flat_map(|row| [row[i].a as i64, row[i].b as i64])
                    .collect();
                payload.truncate(symbols);
                (ri, payload)
            })
            .collect())
    }

    /// Recover the multiset of payloads, sorted lexicographically —
    /// signature-compatible with `MseEncoding::decode`.
    pub fn decode(&self) -> Result<Vec<Vec<i64>>, PronyError> {
        let mut out: Vec<Vec<i64>> = self
            .decode_with_keys()?
            .into_iter()
            .map(|(_, x)| x)
            .collect();
        out.sort();
        Ok(out)
    }

    /// Number of `KahePoly`s an encoding takes.
    pub fn n_polys(params: &PronyParams) -> usize {
        params.total_scalars().div_ceil(N)
    }

    /// Cover traffic: the encoding of nothing, all zero. Contributes nothing
    /// to the sum, so it does not consume capacity.
    pub fn cover(params: &PronyParams) -> Vec<KahePoly> {
        vec![KahePoly::default(); Self::n_polys(params)]
    }

    /// The sketch in a round's summed plaintext. `polys.len()` must equal
    /// `n_polys(params)`.
    pub fn unpack(params: &PronyParams, polys: &[KahePoly]) -> Self {
        assert_eq!(polys.len(), Self::n_polys(params));
        let p = params.p;
        let mut v: Vec<i64> = polys
            .par_iter()
            .map(|poly| {
                let mut q = *poly;
                q.normalize();
                *q.coeffs()
            })
            .collect::<Vec<_>>()
            .into_flattened();
        v.truncate(params.total_scalars());
        let c = params.capacity;
        let k = v
            .drain(..PRONY_OMEGA * c)
            .collect::<Vec<i64>>()
            .chunks_exact(PRONY_OMEGA)
            .map(|s| F2 {
                a: canon(s[0], p),
                b: canon(s[1], p),
            })
            .collect();
        Self {
            params: params.clone(),
            k,
            v,
        }
    }

    /// `V[ℓ][j]` for `idx = ℓ·C + j`, as an element of `F`.
    fn cell(&self, idx: usize) -> F2 {
        let p = self.params.p;
        F2 {
            a: canon(self.v[PRONY_OMEGA * idx], p),
            b: canon(self.v[PRONY_OMEGA * idx + 1], p),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    const SMALL_P: u64 = 1_048_583; // smallest prime > 2^20

    fn rand_f2(rng: &mut ChaCha20Rng) -> F2 {
        Ext::new(PRONY_PRIME).random(rng)
    }

    /// 3 is not a square mod `t`, so `Z² − 3` is irreducible and `F_t[Z]/(Z² − 3)`
    /// is a field.
    #[test]
    fn nonresidue_is_not_a_square() {
        let f = Fp::new(PRONY_PRIME);
        assert_eq!(f.pow(NONRESIDUE, (PRONY_PRIME - 1) / 2), PRONY_PRIME - 1);
    }

    #[test]
    fn extension_field_arithmetic() {
        let mut rng = ChaCha20Rng::from_seed([40u8; 32]);
        let ext = Ext::new(PRONY_PRIME);
        let f = ext.f;
        let z = F2 { a: 0, b: 1 };
        assert_eq!(
            ext.mul(z, z),
            F2 {
                a: NONRESIDUE,
                b: 0
            }
        );
        for _ in 0..1000 {
            let (x, y, w) = (rand_f2(&mut rng), rand_f2(&mut rng), rand_f2(&mut rng));
            assert_eq!(ext.mul(x, y), ext.mul(y, x));
            assert_eq!(ext.mul(ext.mul(x, y), w), ext.mul(x, ext.mul(y, w)));
            assert_eq!(
                ext.mul(x, ext.add(y, w)),
                ext.add(ext.mul(x, y), ext.mul(x, w))
            );
            assert_eq!(ext.scale(x, w.a), ext.mul(x, F2 { a: w.a, b: 0 }));
            if !x.is_zero() {
                assert_eq!(ext.mul(x, ext.inv(x)), F2::ONE);
            }
        }
        // Frobenius: x^t = x_a − x_b Z, since Z^t = Z · 3^{(t−1)/2} = −Z.
        let x = rand_f2(&mut rng);
        let mut xt = F2::ONE;
        let mut base = x;
        let mut e = PRONY_PRIME;
        while e > 0 {
            if e & 1 == 1 {
                xt = ext.mul(xt, base);
            }
            base = ext.mul(base, base);
            e >>= 1;
        }
        assert_eq!(
            xt,
            F2 {
                a: x.a,
                b: f.sub(0, x.b)
            }
        );
    }

    #[test]
    fn roots_in_the_extension_field() {
        let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
        let ext = Ext::new(PRONY_PRIME);
        let mut want: Vec<F2> = (0..12).map(|_| rand_f2(&mut rng)).collect();
        let lam = want.iter().fold(vec![F2::ONE], |acc, &r| {
            poly_mul(&acc, &[ext.neg(r), F2::ONE], ext)
        });
        want.sort_unstable();
        assert_eq!(roots_of_split(&lam, ext), Some(want.clone()));
        // A repeated root does not divide X^q − X.
        let g = poly_mul(&lam, &[ext.neg(want[3]), F2::ONE], ext);
        assert_eq!(roots_of_split(&g, ext), None);
        // Neither does a factor without roots in F: X² − Z is irreducible
        // unless Z is a square, and Z's norm −3 decides that.
        let z = F2 { a: 0, b: 1 };
        let quad = poly_mul(&lam, &[ext.neg(z), F2::ZERO, F2::ONE], ext);
        let norm_is_square = ext.f.pow(PRONY_PRIME - NONRESIDUE, (PRONY_PRIME - 1) / 2) == 1;
        assert_eq!(roots_of_split(&quad, ext).is_some(), norm_is_square);
    }

    /// Barrett must agree with the reference remainder everywhere it is used:
    /// `reduce` over `[0, p²)`, and `reduce_acc` and `reduce_wide` over the
    /// range of the deferred accumulators.
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
                let y = rng.gen::<u128>() % (3 * pp);
                assert_eq!(
                    f.reduce_acc(Wide(y)),
                    (y % p as u128) as u64,
                    "reduce_acc {y} mod {p}"
                );
                let [a, b, c, d] = [0, 1, 2, 3].map(|_| rng.gen::<u64>() % p);
                let want = (a as u128 * b as u128 + c as u128 * d as u128) % p as u128;
                assert_eq!(f.mul2(a, b, c, d), want as u64, "mul2 mod {p}");
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
                assert_eq!(f.reduce_acc(Wide(x)), (x % p as u128) as u64);
            }
        }
    }

    /// The vector kernels must be bit-identical to the scalar one.
    #[test]
    fn solve_kernel_matches_scalar() {
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
                #[cfg(target_arch = "aarch64")]
                {
                    let v = unsafe { dot_limbs_neon(d_lo, d_hi, &w_lo, &w_hi) };
                    assert_eq!(v, s, "neon limb sums k={k} row={i}");
                }
            }
        }
    }

    /// Exercises the ρ² solve path under `cargo test`, not only the example.
    #[test]
    fn capacity_300_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([47u8; 32]);
        let cap = 300;
        let pp = PronyParams::new(cap, 8);
        let mut msgs: Vec<Vec<i64>> = (0..cap)
            .map(|i| {
                (0..8)
                    .map(|s| ((i * 31 + s * 7) % 100_000) as i64)
                    .collect()
            })
            .collect();
        let got = sketch_of(&pp, &mut rng, &msgs).decode().unwrap();
        msgs.sort();
        assert_eq!(got, msgs);
    }

    #[test]
    fn single_element_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        let pp = PronyParams::new(4, 1);
        let sk = sketch_of(&pp, &mut rng, &[vec![42]]);
        assert_eq!(sk.decode().unwrap(), vec![vec![42]]);
    }

    #[test]
    fn multi_element_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        let pp = PronyParams::new(8, 1);
        let msgs = singletons(&[400, 7, 13, 19, 100, 200, 300, 1]);
        let got = sketch_of(&pp, &mut rng, &msgs).decode().unwrap();
        let mut want = msgs;
        want.sort();
        assert_eq!(got, want);
    }

    /// Capacity is a hard bound, unlike peeling's probabilistic one: exactly
    /// `capacity` contributors must decode.
    #[test]
    fn full_capacity_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([9u8; 32]);
        let cap = 32;
        let pp = PronyParams::new(cap, 3);
        let mut msgs: Vec<Vec<i64>> = (0..cap)
            .map(|i| vec![i as i64 * 7 + 1, i as i64 * 11, 1 << 33])
            .collect();
        let got = sketch_of(&pp, &mut rng, &msgs).decode().unwrap();
        msgs.sort();
        assert_eq!(got, msgs);
    }

    /// The sum of encodings, as KAHE computes it, unpacked.
    fn sum_encodings(pp: &PronyParams, encodings: &[Vec<KahePoly>]) -> PronySketch {
        let summed: Vec<KahePoly> = (0..PronySketch::n_polys(pp))
            .map(|i| {
                encodings
                    .iter()
                    .map(|e| e[i])
                    .fold(KahePoly::default(), |a, b| a + b)
            })
            .collect();
        PronySketch::unpack(pp, &summed)
    }

    /// The summed sketch of one client per message.
    fn sketch_of<R: Rng>(pp: &PronyParams, rng: &mut R, msgs: &[Vec<i64>]) -> PronySketch {
        let encodings: Vec<Vec<KahePoly>> = msgs
            .iter()
            .map(|m| PronySketch::encode(pp, rng, m))
            .collect();
        sum_encodings(pp, &encodings)
    }

    fn singletons(xs: &[i64]) -> Vec<Vec<i64>> {
        xs.iter().map(|&x| vec![x]).collect()
    }

    #[test]
    fn cover_adds_nothing() {
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        let pp = PronyParams::new(8, 2);
        let a = PronySketch::encode(&pp, &mut rng, &[5, 6]);
        let cover = PronySketch::cover(&pp);
        assert_eq!(cover.len(), PronySketch::n_polys(&pp));
        let sum = sum_encodings(&pp, &[a, cover.clone()]);
        assert_eq!(sum.decode().unwrap(), vec![vec![5, 6]]);
        assert_eq!(
            sum_encodings(&pp, &[cover]).decode().unwrap(),
            Vec::<Vec<i64>>::new()
        );
    }

    /// Encodings are `n_polys` long and their cells centered in
    /// `[−p/2, p/2)`, which the KAHE noise budget's `ρ·t/2` term assumes.
    #[test]
    fn encoded_cells_are_centered() {
        let mut rng = ChaCha20Rng::from_seed([8u8; 32]);
        let pp = PronyParams::new(16, 3);
        let half = (pp.p / 2) as i64;
        for i in 0..16 {
            let enc = PronySketch::encode(&pp, &mut rng, &[i, i * 3, -i]);
            assert_eq!(enc.len(), PronySketch::n_polys(&pp));
            for poly in enc {
                assert!(poly.coeffs().iter().all(|&c| c >= -half && c < half));
            }
        }
    }

    /// The encoding of a known key holds `r^j` and `x_ℓ·r^j` in the layout
    /// `unpack` reads, including the zero padding of an odd payload.
    #[test]
    fn encoding_layout() {
        let pp = PronyParams::new(5, 3);
        let ext = Ext::new(pp.p);
        let r = F2 { a: 12345, b: 678 };
        let payload = [7i64, -2, 9];
        let sk = sum_encodings(&pp, &[PronySketch::encode_with_key(&pp, &payload, r)]);
        let c = pp.capacity;
        let pow = |j: usize| (0..j).fold(F2::ONE, |acc, _| ext.mul(acc, r));
        let xs = [F2 { a: 7, b: pp.p - 2 }, F2 { a: 9, b: 0 }];
        for j in 0..c {
            assert_eq!(sk.k[j], pow(j + 1), "k[{j}]");
            for (l, &x) in xs.iter().enumerate() {
                assert_eq!(sk.cell(l * c + j), ext.mul(x, pow(j)), "v[{l}][{j}]");
            }
        }
    }

    /// More contributors than the capacity: `Λ` is then essentially a random
    /// polynomial of degree `C`, which splits into distinct linear factors
    /// with probability about `1/C!`.
    #[test]
    fn overflowing_capacity_is_detected() {
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        let pp = PronyParams::new(16, 1);
        let msgs = singletons(&(0..20).collect::<Vec<_>>());
        let sk = sketch_of(&pp, &mut rng, &msgs);
        assert_eq!(sk.decode(), Err(PronyError::NotSplit));
    }

    /// The number of contributors comes from the power sums alone, for every
    /// count from an empty sketch to full capacity.
    #[test]
    fn contributor_count_from_power_sums() {
        let mut rng = ChaCha20Rng::from_seed([12u8; 32]);
        let cap = 12;
        let pp = PronyParams::new(cap, 2);
        for k in [0usize, 1, 2, cap - 1, cap] {
            let mut want: Vec<Vec<i64>> = (0..k as i64).map(|i| vec![i, 1000 - i]).collect();
            let got = sketch_of(&pp, &mut rng, &want).decode().unwrap();
            want.sort();
            assert_eq!(got, want, "k = {k}");
        }
    }

    /// Without contributors, every payload cell must be zero.
    #[test]
    fn payload_without_contributors_is_detected() {
        let mut rng = ChaCha20Rng::from_seed([14u8; 32]);
        let pp = PronyParams::new(8, 2);
        let mut sk = sketch_of(&pp, &mut rng, &[]);
        sk.v[PRONY_OMEGA * 3] = 5;
        assert_eq!(sk.decode(), Err(PronyError::CheckFailed));
    }

    /// Two contributors with the same key: `Λ` is not squarefree. Detected,
    /// not returned as garbage.
    #[test]
    fn duplicate_key_is_detected() {
        let pp = PronyParams::new(8, 1);
        let r = F2 { a: 12345, b: 678 };
        let encodings = [
            PronySketch::encode_with_key(&pp, &[11], r),
            PronySketch::encode_with_key(&pp, &[22], r),
            PronySketch::encode_with_key(&pp, &[33], F2 { a: 999, b: 1 }),
        ];
        assert_eq!(
            sum_encodings(&pp, &encodings).decode(),
            Err(PronyError::NotSplit)
        );
    }

    /// Below capacity, a contribution that is not a Vandermonde row disagrees
    /// with the payload columns the solve does not use.
    #[test]
    fn malformed_contribution_is_detected() {
        let mut rng = ChaCha20Rng::from_seed([13u8; 32]);
        let pp = PronyParams::new(8, 1);
        let mut sk = sketch_of(&pp, &mut rng, &singletons(&[3, 4, 5]));
        sk.v[PRONY_OMEGA * 2] += 7;
        assert_eq!(sk.decode(), Err(PronyError::CheckFailed));
    }

    #[test]
    #[should_panic(expected = "payload arity must match")]
    fn payload_arity_mismatch_panics() {
        let mut rng = ChaCha20Rng::from_seed([29u8; 32]);
        PronySketch::encode(&PronyParams::new(8, 3), &mut rng, &[1, 2]);
    }
}
