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

    /// `x mod p` for `x < 2^{2k}`, which covers sums of up to three products
    /// below `p²` (HAC 14.42 — the estimate leaves a residue below `3p`, hence
    /// at most two corrections).
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
        let bb = f.reduce(x.b as u128 * y.b as u128);
        F2 {
            a: f.add(
                f.reduce(x.a as u128 * y.a as u128),
                f.reduce(NONRESIDUE as u128 * bb as u128),
            ),
            b: f.reduce(x.a as u128 * y.b as u128 + x.b as u128 * y.a as u128),
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

/// Schoolbook, accumulating the three coordinate products of each output
/// coefficient unreduced and reducing once at the end. Each term is below
/// `2p² < 2^73`, so `u128` has room for degrees far past any capacity here.
fn poly_mul(a: &[F2], b: &[F2], ext: Ext) -> Vec<F2> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let n = a.len() + b.len() - 1;
    let (mut aa, mut bb, mut ab) = (vec![0u128; n], vec![0u128; n], vec![0u128; n]);
    for (i, &x) in a.iter().enumerate() {
        if x.is_zero() {
            continue;
        }
        let (xa, xb) = (x.a as u128, x.b as u128);
        for (j, &y) in b.iter().enumerate() {
            aa[i + j] += xa * y.a as u128;
            bb[i + j] += xb * y.b as u128;
            ab[i + j] += xa * y.b as u128 + xb * y.a as u128;
        }
    }
    let f = ext.f;
    trim(
        (0..n)
            .map(|k| F2 {
                a: f.add(
                    f.reduce_wide(aa[k]),
                    f.reduce(NONRESIDUE as u128 * f.reduce_wide(bb[k]) as u128),
                ),
                b: f.reduce_wide(ab[k]),
            })
            .collect(),
    )
}

/// `(quotient, remainder)` of `a / b`. `b` must be nonzero.
///
/// Long division with deferred reduction: each coefficient of `a` keeps the
/// three coordinate products subtracted from it unreduced in `u128`, and is
/// only reduced when it becomes the leading coefficient or part of the
/// remainder. Subtracting `c·b` is adding `(−c)·b`, so the accumulators only
/// grow; each gets at most `deg a` terms below `2p² < 2^73`.
fn poly_divmod(a: &[F2], b: &[F2], ext: Ext) -> (Vec<F2>, Vec<F2>) {
    debug_assert!(!b.is_empty());
    if a.len() < b.len() {
        return (Vec::new(), a.to_vec());
    }
    let f = ext.f;
    let reduce = |aa: u128, bb: u128, ab: u128| F2 {
        a: f.add(
            f.reduce_wide(aa),
            f.reduce(NONRESIDUE as u128 * f.reduce_wide(bb) as u128),
        ),
        b: f.reduce_wide(ab),
    };
    let mut aa: Vec<u128> = a.iter().map(|x| x.a as u128).collect();
    let mut bb = vec![0u128; a.len()];
    let mut ab: Vec<u128> = a.iter().map(|x| x.b as u128).collect();
    let inv_lead = ext.inv(*b.last().unwrap());
    let top = b.len() - 1;
    let mut q = vec![F2::ZERO; a.len() - top];
    for shift in (0..q.len()).rev() {
        let lead = reduce(aa[shift + top], bb[shift + top], ab[shift + top]);
        let c = ext.mul(lead, inv_lead);
        if c.is_zero() {
            continue;
        }
        q[shift] = c;
        let nc = ext.neg(c);
        let (ca, cb) = (nc.a as u128, nc.b as u128);
        for (i, y) in b[..top].iter().enumerate() {
            let k = shift + i;
            aa[k] += y.a as u128 * ca;
            bb[k] += y.b as u128 * cb;
            ab[k] += y.a as u128 * cb + y.b as u128 * ca;
        }
    }
    let rem = (0..top).map(|k| reduce(aa[k], bb[k], ab[k])).collect();
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

#[derive(Clone, Debug, PartialEq)]
pub struct PronySketch {
    params: PronyParams,
    /// `k[j − 1] = Σ_i r_i^j`, `j ∈ [1, capacity]`.
    k: Vec<F2>,
    /// `v[ℓ][j] = Σ_i x_{i,ℓ} · r_i^j`, `j ∈ [0, capacity)`.
    v: Vec<Vec<F2>>,
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
    pub fn new(params: PronyParams) -> Self {
        let c = params.capacity;
        let v = (0..params.payload_elements())
            .map(|_| vec![F2::ZERO; c])
            .collect();
        Self {
            k: vec![F2::ZERO; c],
            v,
            params,
        }
    }

    /// Insert one element under a fresh random key.
    /// May be called repeatedly; one item per client is enforced by client code,
    /// not by this multiset encoding.
    pub fn insert<R: Rng>(&mut self, rng: &mut R, payload: &[i64]) {
        let ext = Ext::new(self.params.p);
        let r = loop {
            let r = ext.random(rng);
            if !r.is_zero() {
                break r;
            }
        };
        self.insert_with_key(payload, r);
    }

    /// Insert under a given key `r`, a nonzero element of `F`.
    fn insert_with_key(&mut self, payload: &[i64], r: F2) {
        assert_eq!(
            payload.len(),
            self.params.payload_symbols,
            "payload arity must match params.payload_symbols",
        );
        let p = self.params.p;
        assert!(
            !r.is_zero() && r.a < p && r.b < p,
            "r must be a nonzero element of F"
        );
        let ext = Ext::new(p);
        let c = self.params.capacity;

        // r^0, …, r^C: the power sums take r^1..r^C, the payload r^0..r^{C−1}.
        let mut pows = Vec::with_capacity(c + 1);
        let mut cur = F2::ONE;
        for _ in 0..=c {
            pows.push(cur);
            cur = ext.mul(cur, r);
        }

        for (slot, &pw) in self.k.iter_mut().zip(pows[1..].iter()) {
            *slot = ext.add(*slot, pw);
        }
        // The payload's symbols in pairs, the last one padded with zero.
        let xs: Vec<F2> = payload
            .chunks(PRONY_OMEGA)
            .map(|s| F2 {
                a: canon(s[0], p),
                b: s.get(1).map_or(0, |&y| canon(y, p)),
            })
            .collect();
        // x·r^j with x fixed per row: both coordinates are sums of two
        // products below p², one reduction each.
        let f = ext.f;
        self.v
            .par_iter_mut()
            .zip(xs.par_iter())
            .for_each(|(row, &x)| {
                let (xa, xb) = (x.a as u128, x.b as u128);
                let xb3 = f.mul(NONRESIDUE, x.b) as u128;
                for (slot, &pw) in row.iter_mut().zip(pows.iter()) {
                    let (pa, pb) = (pw.a as u128, pw.b as u128);
                    let prod = F2 {
                        a: f.reduce(xa * pa + xb3 * pb),
                        b: f.reduce(xa * pb + xb * pa),
                    };
                    *slot = ext.add(*slot, prod);
                }
            });
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
        for m in 1..=c {
            let mut acc = F2::ZERO;
            for i in 1..=m {
                let term = ext.mul(e[m - i], self.k[i - 1]);
                acc = if i % 2 == 1 {
                    ext.add(acc, term)
                } else {
                    ext.sub(acc, term)
                };
            }
            e[m] = ext.scale(acc, f.inv(m as u64 % p));
        }
        // The number of contributors. Since `e_m = 0` for every `m > ρ̂`, the
        // power sums are exactly those of the roots of `Λ` once it splits, so
        // they need no separate check.
        let rho_hat = (0..=c).rev().find(|&m| !e[m].is_zero()).unwrap_or(0);
        if rho_hat == 0 {
            return if self.v.iter().all(|row| row.iter().all(|x| x.is_zero())) {
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
        let xs: Vec<Vec<F2>> = self
            .v
            .par_iter()
            .map(|row| {
                let w = &row[..rho_hat];
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
        let bad = self.v.par_iter().zip(xs.par_iter()).any(|(row, xrow)| {
            rpow.iter().enumerate().any(|(idx, rj)| {
                let (mut aa, mut bb, mut ab) = (0u128, 0u128, 0u128);
                for (x, pw) in xrow.iter().zip(rj.iter()) {
                    aa += x.a as u128 * pw.a as u128;
                    bb += x.b as u128 * pw.b as u128;
                    ab += x.a as u128 * pw.b as u128 + x.b as u128 * pw.a as u128;
                }
                let got = F2 {
                    a: f.add(
                        f.reduce_wide(aa),
                        f.reduce(NONRESIDUE as u128 * f.reduce_wide(bb) as u128),
                    ),
                    b: f.reduce_wide(ab),
                };
                got != row[rho_hat + idx]
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

    /// Number of `KahePoly`s required to pack this sketch.
    pub fn n_polys(params: &PronyParams) -> usize {
        params.total_scalars().div_ceil(N)
    }

    /// Cover traffic: zero polys, same count as `pack()`. Contributes nothing
    /// to the sum, so it does not consume capacity.
    pub fn cover(params: &PronyParams) -> Vec<KahePoly> {
        Self::new(params.clone()).pack()
    }

    /// Flatten `K` then `V[0] … V[ξ_F−1]` into `KahePoly` coefficient slots,
    /// each element of `F` as its two coordinates.
    pub fn pack(&self) -> Vec<KahePoly> {
        let p = self.params.p;
        let total = self.params.total_scalars();
        let n_polys = total.div_ceil(N);
        let mut polys = Vec::with_capacity(n_polys);
        let mut buf = [0i64; N];
        let mut written = 0usize;
        let mut feed = |val: u64, polys: &mut Vec<KahePoly>| {
            buf[written % N] = center(val, p);
            written += 1;
            if written.is_multiple_of(N) {
                polys.push(KahePoly::from_coeffs(buf));
                buf = [0i64; N];
            }
        };
        for x in self.k.iter().chain(self.v.iter().flatten()) {
            feed(x.a, &mut polys);
            feed(x.b, &mut polys);
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
        let p = params.p;
        let mut flat: Vec<u64> = Vec::with_capacity(polys.len() * N);
        for poly in polys {
            let mut q = *poly;
            q.normalize();
            flat.extend(q.coeffs().iter().map(|&x| canon(x, p)));
        }
        let el = |i: usize| F2 {
            a: flat[2 * i],
            b: flat[2 * i + 1],
        };
        let c = params.capacity;
        let k = (0..c).map(el).collect();
        let v = (0..params.payload_elements())
            .map(|l| (0..c).map(|j| el(c + l * c + j)).collect())
            .collect();
        Self {
            params: params.clone(),
            k,
            v,
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
                let y = rng.gen::<u128>() % (3 * pp);
                assert_eq!(f.reduce(y), (y % p as u128) as u64, "reduce {y} mod {p}");
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
            .map(|i| vec![i as i64 * 7 + 1, i as i64 * 11, 1 << 33])
            .collect();
        for m in &msgs {
            sk.insert(&mut rng, m);
        }
        let mut got = sk.decode().unwrap();
        msgs.sort();
        got.sort();
        assert_eq!(got, msgs);
    }

    /// The sum of packed sketches, as KAHE computes it.
    fn sum_packed(sketches: &[PronySketch]) -> PronySketch {
        let packed: Vec<Vec<KahePoly>> = sketches.iter().map(PronySketch::pack).collect();
        let summed: Vec<KahePoly> = (0..packed[0].len())
            .map(|i| {
                packed
                    .iter()
                    .map(|p| p[i])
                    .fold(KahePoly::default(), |a, b| a + b)
            })
            .collect();
        PronySketch::unpack(&sketches[0].params, &summed)
    }

    #[test]
    fn union_via_packed_sum() {
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
        let sum = sum_packed(&[a, b]);
        let mut got: Vec<i64> = sum.decode().unwrap().into_iter().map(|t| t[0]).collect();
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
        let sum = sum_packed(&[a, cover]);
        assert_eq!(sum.decode().unwrap(), vec![vec![5, 6]]);
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

    /// More contributors than the capacity: `Λ` is then essentially a random
    /// polynomial of degree `C`, which splits into distinct linear factors
    /// with probability about `1/C!`.
    #[test]
    fn overflowing_capacity_is_detected() {
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        let pp = PronyParams::new(16, 1);
        let mut sk = PronySketch::new(pp);
        for i in 0..20 {
            sk.insert(&mut rng, &[i]);
        }
        assert_eq!(sk.decode(), Err(PronyError::NotSplit));
    }

    /// The number of contributors comes from the power sums alone, for every
    /// count from an empty sketch to full capacity.
    #[test]
    fn contributor_count_from_power_sums() {
        let mut rng = ChaCha20Rng::from_seed([12u8; 32]);
        let cap = 12;
        for k in [0usize, 1, 2, cap - 1, cap] {
            let mut sk = PronySketch::new(PronyParams::new(cap, 2));
            let mut want: Vec<Vec<i64>> = (0..k as i64).map(|i| vec![i, 1000 - i]).collect();
            for m in &want {
                sk.insert(&mut rng, m);
            }
            let mut got = sk.decode().unwrap();
            want.sort();
            got.sort();
            assert_eq!(got, want, "k = {k}");
        }
    }

    /// Without contributors, every payload cell must be zero.
    #[test]
    fn payload_without_contributors_is_detected() {
        let mut sk = PronySketch::new(PronyParams::new(8, 2));
        sk.v[0][3] = F2 { a: 5, b: 0 };
        assert_eq!(sk.decode(), Err(PronyError::CheckFailed));
    }

    /// Two contributors with the same key: `Λ` is not squarefree. Detected,
    /// not returned as garbage.
    #[test]
    fn duplicate_key_is_detected() {
        let pp = PronyParams::new(8, 1);
        let mut sk = PronySketch::new(pp);
        let r = F2 { a: 12345, b: 678 };
        sk.insert_with_key(&[11], r);
        sk.insert_with_key(&[22], r);
        sk.insert_with_key(&[33], F2 { a: 999, b: 1 });
        assert_eq!(sk.decode(), Err(PronyError::NotSplit));
    }

    /// Below capacity, a contribution that is not a Vandermonde row disagrees
    /// with the payload columns the solve does not use.
    #[test]
    fn malformed_contribution_is_detected() {
        let mut rng = ChaCha20Rng::from_seed([13u8; 32]);
        let pp = PronyParams::new(8, 1);
        let mut sk = PronySketch::new(pp);
        for x in [3i64, 4, 5] {
            sk.insert(&mut rng, &[x]);
        }
        sk.v[0][2] = Ext::new(PRONY_PRIME).add(sk.v[0][2], F2 { a: 7, b: 0 });
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
