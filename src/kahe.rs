//! Key-additive homomorphic encryption (Willow-style RLWE-based KAHE).
//!
//! Lives on its own ring `R_{q_kahe}` (chipmunk's `KahePoly`, q ≈ 2^28),
//! decoupled from the chipmunk Ring-SIS CS ring.
//! The KAHE q is chosen for headroom in the noise budget — chipmunk's q is
//! tied to its multi-signature size optimization and would be wasteful here.
//!
//! Per-poly form (Willow `EncryptPolynomial`):
//!
//! ```text
//! Enc(m, sk):  sample e ← D_{σ_e};   c = m + A·sk + t·e   mod q_kahe
//! Dec(c, sk):  ((c − A·sk) mod q_kahe) reduced mod t   (centered representatives)
//! ```
//!
//! Parameters:
//! - `A ∈ R_{q_kahe}^{μ × κ}` — public matrix.
//! - `sk ← D_{σ_s}^κ` — discrete Gaussian secret key (σ_s = 15.72).
//! - `e ← D_{σ_e}^μ` — discrete Gaussian fresh error (single `D_σ`, σ_e = 15.72).
//! - `t_modulus` — plaintext modulus. Plaintext lives in `R_t^μ` with centered
//!   representatives in `[-t/2, t/2)`. Aggregate decryption returns `Σm mod t`.
//!
//! Correctness budget at ρ aggregations: need `t·8σ_e·√ρ + ρ·t/2 < q_kahe/2`.
//! At σ_e = 15.72, q_kahe = 271_163_393, t = 2^16: holds for ρ ≲ 240,
//! covering the S=8, N=100 operating point.
//!
//! `KaheKey` (fresh, low-norm) feeds `Enc`; `KaheAggKey` (in `R_{q_kahe}^κ`,
//! recovered by Shamir interpolation over `R_{q_cs}` and lifted via
//! [`lift_cs_to_kahe`] in `protocol::verify`) feeds `Dec`. Hiding-given-
//! aggregate-key reduces to (Hint-)RLWE rather than to LHL.
//!
//! Bridge layer. KAHE secret-key components are *small* (Gaussian σ_s = 15.72,
//! `‖sk‖_∞ ≤ 8σ_s ≈ 126` w.o.p.). Shamir-recovered sums-of-keys are also small
//! (`‖Σ sk‖_∞ ≤ N_clients · 8σ_s`). Both fit losslessly in both rings under
//! centered representation, so we cross between `KahePoly` and `CsPoly` by
//! coefficient-wise re-interpretation: see [`kahe_to_cs_centered`] (client
//! side, KAHE→CS for Shamir input) and [`lift_cs_to_kahe`] (verifier side,
//! CS→KAHE for decryption).

use chipmunk_code::{
    pointwise_dot_kahe, CsPoly, KaheNTTPoly, KahePoly, Polynomial, CS_MODULUS_OVER_TWO,
    KAHE_MODULUS_OVER_TWO, N,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

/// KAHE scheme contract.
pub trait KaheScheme {
    type Params;
    type Key: Clone;
    type AggKey: Clone;
    type Message: Clone;
    type Ciphertext: Clone;

    fn setup<R: Rng>(rng: &mut R) -> Self::Params;
    fn gen<R: Rng>(rng: &mut R, pp: &Self::Params) -> Self::Key;
    fn enc<R: Rng>(
        rng: &mut R,
        pp: &Self::Params,
        k: &Self::Key,
        m: &Self::Message,
    ) -> Self::Ciphertext;
    fn dec(pp: &Self::Params, c: &Self::Ciphertext, k: &Self::AggKey) -> Self::Message;
    fn agg_ctxt(cs: &[Self::Ciphertext]) -> Self::Ciphertext;
    fn agg_key(ks: &[Self::Key]) -> Self::AggKey;
}

/// Public parameters. `a_matrices_ntt[i]` is a `μ × κ` (NTT-resident) public
/// matrix for chunk `i ∈ [0, l)`. One Shamir+CS round amortizes its commit
/// overhead across `l` ciphertexts that share the same `sk` but each use a
/// different `A_i` — security reduces to standard multi-sample Module-LWE
/// with the same secret. `t_modulus` is the plaintext modulus; both `sigma_s`
/// (key) and `sigma_e` (error) are Gaussian standard deviations.
pub struct KaheParams {
    pub a_matrices_ntt: Vec<Vec<Vec<KaheNTTPoly>>>,
    pub mu_kahe: usize,
    pub kappa_kahe: usize,
    /// Number of ciphertext chunks emitted per `enc` call (and consumed per
    /// `dec`). Messages and ciphertexts are flat `Vec<KahePoly>` of length
    /// `mu_kahe · l`.
    pub l: usize,
    pub sigma_s: f64,
    pub sigma_e: f64,
    pub t_modulus: u32,
}

/// A *fresh* KAHE key — discrete Gaussian over `R^κ`. Only valid input to
/// [`KaheScheme::enc`].
#[derive(Clone)]
pub struct KaheKey(Vec<KahePoly>);

impl KaheKey {
    pub(crate) fn inner(&self) -> &[KahePoly] {
        &self.0
    }
    pub(crate) fn component(&self, k: usize) -> &KahePoly {
        &self.0[k]
    }
}

/// Aggregate KAHE key — element of `R_{q_kahe}^κ`, output of
/// [`KaheScheme::agg_key`] or constructed by the verifier via
/// [`KaheAggKey::from_components`] after Shamir interpolation (and the
/// HVC→KAHE bridge). Only valid input to [`KaheScheme::dec`].
#[derive(Clone)]
pub struct KaheAggKey(Vec<KahePoly>);

impl KaheAggKey {
    pub(crate) fn inner(&self) -> &[KahePoly] {
        &self.0
    }
    pub fn from_components(components: Vec<KahePoly>) -> Self {
        Self(components)
    }
}

/// Sample a single discrete-Gaussian-rounded polynomial coefficient.
/// Round-to-nearest-integer applied to a continuous Gaussian; clamped at the
/// 8σ tail (Willow's convention).
fn sample_dg<R: Rng>(rng: &mut R, sigma: f64) -> i32 {
    let normal = Normal::new(0.0, sigma).expect("σ > 0");
    let tail = (8.0 * sigma).ceil() as i32;
    loop {
        let x = normal.sample(rng).round() as i32;
        if x.abs() <= tail {
            return x;
        }
    }
}

fn sample_dg_poly<R: Rng>(rng: &mut R, sigma: f64) -> KahePoly {
    let mut coeffs = [0i32; N];
    for c in coeffs.iter_mut() {
        *c = sample_dg(rng, sigma);
    }
    KahePoly::from_coeffs(coeffs)
}

/// `(A_chunk · sk)[i]` where `sk` is already in NTT representation. Used by
/// `enc`/`dec` to amortize the κ-NTT of the secret across all `l·μ` rows.
fn pad_poly(
    a_matrices_ntt: &[Vec<Vec<KaheNTTPoly>>],
    sk_ntt: &[KaheNTTPoly],
    chunk: usize,
    i: usize,
) -> KahePoly {
    KahePoly::from(&pointwise_dot_kahe(&a_matrices_ntt[chunk][i], sk_ntt))
}

fn vec_add(a: &[KahePoly], b: &[KahePoly]) -> Vec<KahePoly> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect()
}

fn vec_zero(len: usize) -> Vec<KahePoly> {
    vec![KahePoly::default(); len]
}

/// Accumulate `Σ_client cs[client][pos]` into an `i64[N]` accumulator. Summing
/// ρ ciphertext coefficients (each `< q/2 ≈ 2^27`) reaches ~`ρ·2^27`, which
/// overflows i32 for ρ ≳ 15 — so we widen to i64 and reduce once at the end,
/// instead of the per-step mod-q the old fold-of-adds required.
fn accumulate_pos(acc: &mut [i64; N], cs: &[Vec<KahePoly>], pos: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { accumulate_pos_avx2(acc, cs, pos) };
            return;
        }
    }
    for client in cs {
        let src = client[pos].coeffs();
        for i in 0..N {
            acc[i] += src[i] as i64;
        }
    }
}

/// AVX2: widen 4 i32 coefficients to i64 (`cvtepi32_epi64`) and accumulate into
/// 4-wide i64 lanes. `N` is a multiple of 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_pos_avx2(acc: &mut [i64; N], cs: &[Vec<KahePoly>], pos: usize) {
    use std::arch::x86_64::*;
    debug_assert_eq!(N % 4, 0);
    for client in cs {
        let src = client[pos].coeffs().as_ptr();
        let mut i = 0usize;
        while i < N {
            let v = _mm_loadu_si128(src.add(i) as *const __m128i);
            let v64 = _mm256_cvtepi32_epi64(v);
            let a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
            let s = _mm256_add_epi64(a, v64);
            _mm256_storeu_si256(acc.as_mut_ptr().add(i) as *mut __m256i, s);
            i += 4;
        }
    }
}

/// `poly[i] · scale` mod q, widened to i64 (q_kahe ≈ 2^28).
fn scale_poly(poly: &KahePoly, scale: i32) -> KahePoly {
    let mut coeffs = [0i32; N];
    let q = chipmunk_code::KAHE_MODULUS as i64;
    for (out, &c) in coeffs.iter_mut().zip(poly.coeffs().iter()) {
        let prod = (c as i64) * (scale as i64) % q;
        *out = prod as i32;
    }
    let mut p = KahePoly::from_coeffs(coeffs);
    p.normalize();
    p
}

/// Reduce one (centered) integer coefficient mod `t` to centered range
/// `[-t/2, t/2)`.
fn reduce_centered(x: i32, t: i32) -> i32 {
    let r = x.rem_euclid(t);
    let half = t / 2;
    if r >= half {
        r - t
    } else {
        r
    }
}

/// Reduce a polynomial coefficient-wise mod `t` (centered).
///
/// Crucially uses `normalize()` (centered `[-q/2, q/2]`) and not `lift()`
/// (which puts coefficients in `[0, q)` and would skew the mod-t residue
/// whenever `q ≢ 0 (mod t)`).
fn poly_mod_t(poly: &KahePoly, t: u32) -> KahePoly {
    let mut p = *poly;
    p.normalize();
    let t_i = t as i32;
    let mut coeffs = [0i32; N];
    for (out, &c) in coeffs.iter_mut().zip(p.coeffs().iter()) {
        *out = reduce_centered(c, t_i);
    }
    KahePoly::from_coeffs(coeffs)
}

// ---------------------------------------------------------------
// Bridge: HVC ↔ KAHE via centered-representative re-interpretation
// ---------------------------------------------------------------

/// Lift a `CsPoly` (centered repr in `[-q_cs/2, q_cs/2]`) into a `KahePoly`
/// at q_kahe by reinterpreting coefficients as signed integers. Lossless iff
/// `‖input‖_∞ ≤ q_cs/2`, which holds for Shamir-recovered sums of small
/// Gaussian keys (`‖Σ sk‖_∞ ≪ q_cs/2` at any realistic n_clients).
pub fn lift_cs_to_kahe(p: &CsPoly) -> KahePoly {
    let mut q = *p;
    q.normalize();
    debug_assert!(
        q.coeffs().iter().all(|&c| c.abs() <= CS_MODULUS_OVER_TWO),
        "lift_cs_to_kahe: input not centered after normalize()"
    );
    KahePoly::from_signed_coeffs(q.coeffs())
}

/// Reduce a small `KahePoly` (a fresh KAHE secret key — Gaussian σ_s ≈ 15.72,
/// `‖·‖_∞ ≤ 8σ_s ≈ 126`) into a `CsPoly` by centered-rep re-interpretation.
/// Lossless iff `‖input‖_∞ ≤ q_cs/2`. Used at the client side to feed Shamir
/// sharing, which operates over `R_{q_cs}`.
pub fn kahe_to_cs_centered(p: &KahePoly) -> CsPoly {
    let mut q = *p;
    q.normalize();
    debug_assert!(
        q.coeffs().iter().all(|&c| c.abs() <= CS_MODULUS_OVER_TWO),
        "kahe_to_cs_centered: coefficient magnitude exceeds q_cs/2 = {} \
         (callers must only bridge small Gaussian-bounded values)",
        CS_MODULUS_OVER_TWO
    );
    CsPoly::from_coeffs(*q.coeffs())
}

pub struct Kahe;

impl Kahe {
    /// Setup with explicit dimensions and Gaussian widths. `sigma_s` = key std,
    /// `sigma_e` = error std. Defaults `SIGMA_S_DEFAULT = SIGMA_E_DEFAULT =
    /// 15.72` (single `D_σ`) target ~128-bit RLWE at N=2048.
    pub fn setup_with_dims<R: Rng>(
        rng: &mut R,
        mu_kahe: usize,
        kappa_kahe: usize,
        l: usize,
        sigma_s: f64,
        sigma_e: f64,
        t_modulus: u32,
    ) -> KaheParams {
        assert!(mu_kahe >= 1, "μ_kahe must be ≥ 1");
        assert!(kappa_kahe >= 1, "κ_kahe must be ≥ 1");
        assert!(l >= 1, "l must be ≥ 1");
        assert!(t_modulus >= 2, "t_modulus must be ≥ 2");
        // Sample directly NTT-resident — uniform-in-NTT slot is statistically
        // equivalent to NTT(uniform coeff poly) and saves `l·μ·κ` forward NTTs.
        let a_matrices_ntt: Vec<Vec<Vec<KaheNTTPoly>>> = (0..l)
            .map(|_| {
                (0..mu_kahe)
                    .map(|_| {
                        (0..kappa_kahe)
                            .map(|_| KaheNTTPoly::rand_ntt_poly(rng))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        KaheParams {
            a_matrices_ntt,
            mu_kahe,
            kappa_kahe,
            l,
            sigma_s,
            sigma_e,
            t_modulus,
        }
    }
}

/// q_kahe = 271_163_393 (≈2^28) at N=2048. Single `D_σ` for key and error
/// (Nils's spec), σ_s = σ_e = 15.72 — sized for ~128-bit RLWE at N=2048.
pub const SIGMA_S_DEFAULT: f64 = 15.72;
pub const SIGMA_E_DEFAULT: f64 = 15.72;
/// 2^16. Per-coefficient noise budget `t·8σ_e·√ρ + ρ·t/2 < q_kahe/2` holds for
/// ρ ≲ 240 at (t, σ, q) = (2^16, 15.72, 271_163_393); comfortably covers the
/// S=8, N=100 operating point.
pub const T_MODULUS_DEFAULT: u32 = 65536;

impl KaheScheme for Kahe {
    type Params = KaheParams;
    type Key = KaheKey;
    type AggKey = KaheAggKey;
    type Message = Vec<KahePoly>;
    type Ciphertext = Vec<KahePoly>;

    /// `(μ, κ, l) = (1, 1, 1)`, σ_s=σ_e=15.72, `t = 2^16` at q_kahe = 271_163_393.
    /// κ_kahe=1 → plain RLWE secret (single ring element); one Shamir-share
    /// component per CS opening.
    fn setup<R: Rng>(rng: &mut R) -> KaheParams {
        Self::setup_with_dims(
            rng,
            1,
            1,
            1,
            SIGMA_S_DEFAULT,
            SIGMA_E_DEFAULT,
            T_MODULUS_DEFAULT,
        )
    }

    fn gen<R: Rng>(rng: &mut R, pp: &KaheParams) -> KaheKey {
        let polys = (0..pp.kappa_kahe)
            .map(|_| sample_dg_poly(rng, pp.sigma_s))
            .collect();
        KaheKey(polys)
    }

    /// Batch-encrypt `l` chunks of `μ` plaintext polys under one key `sk`.
    /// For chunk `i ∈ [0, l)`: `c_i = m_i + A_i·sk + t·e_i`, with a fresh
    /// `e_i ← D_{σ_e}^μ`. The κ-NTT of `sk` is computed once and reused
    /// across all `l` chunks.
    fn enc<R: Rng>(rng: &mut R, pp: &KaheParams, k: &KaheKey, m: &Vec<KahePoly>) -> Vec<KahePoly> {
        debug_assert_eq!(k.inner().len(), pp.kappa_kahe);
        debug_assert_eq!(m.len(), pp.mu_kahe * pp.l);
        let t = pp.t_modulus as i32;
        let sk_ntt: Vec<KaheNTTPoly> = k.inner().iter().map(KaheNTTPoly::from).collect();
        let total = pp.mu_kahe * pp.l;
        let seeds = crate::fork_seeds(rng, total);
        (0..total)
            .into_par_iter()
            .map(|idx| {
                let (chunk, i) = (idx / pp.mu_kahe, idx % pp.mu_kahe);
                let pad = pad_poly(&pp.a_matrices_ntt, &sk_ntt, chunk, i);
                let mut item_rng = ChaCha20Rng::from_seed(seeds[idx]);
                let e = sample_dg_poly(&mut item_rng, pp.sigma_e);
                let te = scale_poly(&e, t);
                m[idx] + pad + te
            })
            .collect()
    }

    /// `((c_i − A_i·sk_agg) mod q_kahe) reduced mod t`, per chunk. `sk_agg`'s
    /// κ-NTT is computed once and shared across all `l` chunks.
    fn dec(pp: &KaheParams, c: &Vec<KahePoly>, k: &KaheAggKey) -> Vec<KahePoly> {
        debug_assert_eq!(k.inner().len(), pp.kappa_kahe);
        debug_assert_eq!(c.len(), pp.mu_kahe * pp.l);
        let sk_ntt: Vec<KaheNTTPoly> = k.inner().iter().map(KaheNTTPoly::from).collect();
        let total = pp.mu_kahe * pp.l;
        (0..total)
            .into_par_iter()
            .map(|idx| {
                let (chunk, i) = (idx / pp.mu_kahe, idx % pp.mu_kahe);
                let pad = pad_poly(&pp.a_matrices_ntt, &sk_ntt, chunk, i);
                let raw = c[idx] - pad;
                poly_mod_t(&raw, pp.t_modulus)
            })
            .collect()
    }

    fn agg_ctxt(cs: &[Vec<KahePoly>]) -> Vec<KahePoly> {
        if cs.is_empty() {
            return Vec::new();
        }
        let len = cs[0].len();
        let q = chipmunk_code::KAHE_MODULUS as i64;
        let half = q / 2;
        // Per ciphertext-poly position, sum across all clients in i64 (no
        // overflow), then reduce mod q to a centered representative once. The
        // result is ≡ Σ cᵢ (mod q), which is all `dec` (which subtracts the
        // aggregate pad then reduces mod t) requires.
        (0..len)
            .into_par_iter()
            .map(|pos| {
                let mut acc = [0i64; N];
                accumulate_pos(&mut acc, cs, pos);
                let mut coeffs = [0i32; N];
                for (out, &a) in coeffs.iter_mut().zip(acc.iter()) {
                    let mut r = a.rem_euclid(q);
                    if r > half {
                        r -= q;
                    }
                    *out = r as i32;
                }
                KahePoly::from_coeffs(coeffs)
            })
            .collect()
    }

    fn agg_key(ks: &[KaheKey]) -> KaheAggKey {
        if ks.is_empty() {
            return KaheAggKey(Vec::new());
        }
        let len = ks[0].inner().len();
        let summed = ks
            .iter()
            .fold(vec_zero(len), |acc, x| vec_add(&acc, x.inner()));
        KaheAggKey(summed)
    }
}

// Suppress unused warning for re-exported KAHE_MODULUS_OVER_TWO consumers
// (used by bridge debug assertions transitively).
const _: i32 = KAHE_MODULUS_OVER_TWO;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    /// Random message with coefficients centered in `[-t/2, t/2)`.
    fn rand_message_poly<R: Rng>(rng: &mut R, t: u32) -> KahePoly {
        let half = t / 2;
        let mut coeffs = [0i32; N];
        for c in coeffs.iter_mut() {
            *c = (rng.gen_range(0..t) as i32) - half as i32;
        }
        KahePoly::from_coeffs(coeffs)
    }

    #[test]
    fn round_trip() {
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        let pp = Kahe::setup(&mut rng);
        let k = Kahe::gen(&mut rng, &pp);
        let m: Vec<KahePoly> = (0..pp.mu_kahe)
            .map(|_| rand_message_poly(&mut rng, pp.t_modulus))
            .collect();
        let c = Kahe::enc(&mut rng, &pp, &k, &m);
        let agg = Kahe::agg_key(std::slice::from_ref(&k));
        assert_eq!(Kahe::dec(&pp, &c, &agg), m);
    }

    #[test]
    fn additive_homomorphism() {
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        let pp = Kahe::setup(&mut rng);
        let n = 5;
        let keys: Vec<KaheKey> = (0..n).map(|_| Kahe::gen(&mut rng, &pp)).collect();
        let msgs: Vec<Vec<KahePoly>> = (0..n)
            .map(|_| {
                (0..pp.mu_kahe)
                    .map(|_| rand_message_poly(&mut rng, pp.t_modulus))
                    .collect()
            })
            .collect();
        let ctxts: Vec<Vec<KahePoly>> = keys
            .iter()
            .zip(&msgs)
            .map(|(k, m)| Kahe::enc(&mut rng, &pp, k, m))
            .collect();

        let agg_c = Kahe::agg_ctxt(&ctxts);
        let agg_k = Kahe::agg_key(&keys);
        let agg_m_expected: Vec<KahePoly> = (0..pp.mu_kahe)
            .map(|i| {
                let summed = msgs.iter().fold(KahePoly::default(), |a, x| a + x[i]);
                poly_mod_t(&summed, pp.t_modulus)
            })
            .collect();

        assert_eq!(Kahe::dec(&pp, &agg_c, &agg_k), agg_m_expected);
    }

    #[test]
    fn shamir_recovered_agg_key_round_trip() {
        use crate::sss::{ShamirParams, ShamirSharing};

        let mut rng = ChaCha20Rng::from_seed([23u8; 32]);
        let pp = Kahe::setup(&mut rng);
        let n_servers = 4;
        let t = 3;
        let shamir = ShamirParams::new(t, n_servers);
        let k = Kahe::gen(&mut rng, &pp);

        // Shamir runs over CsPoly (R_{q_cs}). Bridge KAHE → CS for shares,
        // recover at q_cs, bridge CS → KAHE for decryption.
        let recovered_components: Vec<KahePoly> = (0..pp.kappa_kahe)
            .map(|c| {
                let secret_cs = kahe_to_cs_centered(k.component(c));
                let shares = ShamirSharing::share(&mut rng, &shamir, &secret_cs);
                let samples: Vec<(usize, CsPoly)> = (0..t).map(|i| (i, shares[i])).collect();
                let recovered_cs = ShamirSharing::recover(&shamir, &samples).unwrap();
                lift_cs_to_kahe(&recovered_cs)
            })
            .collect();
        let agg = KaheAggKey::from_components(recovered_components);

        let m: Vec<KahePoly> = (0..pp.mu_kahe)
            .map(|_| rand_message_poly(&mut rng, pp.t_modulus))
            .collect();
        let c = Kahe::enc(&mut rng, &pp, &k, &m);
        assert_eq!(Kahe::dec(&pp, &c, &agg), m);
    }

    /// Aggregating ρ=200 ciphertexts decrypts to `Σm mod t` at the default
    /// parameters (sanity check for the noise budget at the bench operating point).
    #[test]
    fn aggregate_200_decrypts() {
        let mut rng = ChaCha20Rng::from_seed([29u8; 32]);
        let pp = Kahe::setup(&mut rng);
        let n = 200;
        let keys: Vec<KaheKey> = (0..n).map(|_| Kahe::gen(&mut rng, &pp)).collect();
        let msgs: Vec<Vec<KahePoly>> = (0..n)
            .map(|_| {
                (0..pp.mu_kahe)
                    .map(|_| rand_message_poly(&mut rng, pp.t_modulus))
                    .collect()
            })
            .collect();
        let ctxts: Vec<Vec<KahePoly>> = keys
            .iter()
            .zip(&msgs)
            .map(|(k, m)| Kahe::enc(&mut rng, &pp, k, m))
            .collect();

        let agg_c = Kahe::agg_ctxt(&ctxts);
        let agg_k = Kahe::agg_key(&keys);
        let agg_m_expected: Vec<KahePoly> = (0..pp.mu_kahe)
            .map(|i| {
                let summed = msgs.iter().fold(KahePoly::default(), |a, x| a + x[i]);
                poly_mod_t(&summed, pp.t_modulus)
            })
            .collect();
        assert_eq!(Kahe::dec(&pp, &agg_c, &agg_k), agg_m_expected);
    }

    #[test]
    fn round_trip_l4() {
        let mut rng = ChaCha20Rng::from_seed([13u8; 32]);
        let pp = Kahe::setup_with_dims(
            &mut rng,
            5,
            5,
            4,
            SIGMA_S_DEFAULT,
            SIGMA_E_DEFAULT,
            T_MODULUS_DEFAULT,
        );
        let k = Kahe::gen(&mut rng, &pp);
        let m: Vec<KahePoly> = (0..pp.mu_kahe * pp.l)
            .map(|_| rand_message_poly(&mut rng, pp.t_modulus))
            .collect();
        let c = Kahe::enc(&mut rng, &pp, &k, &m);
        assert_eq!(c.len(), pp.mu_kahe * pp.l);
        let agg = Kahe::agg_key(std::slice::from_ref(&k));
        assert_eq!(Kahe::dec(&pp, &c, &agg), m);
    }

    #[test]
    fn additive_homomorphism_l4() {
        let mut rng = ChaCha20Rng::from_seed([17u8; 32]);
        let pp = Kahe::setup_with_dims(
            &mut rng,
            3,
            5,
            4,
            SIGMA_S_DEFAULT,
            SIGMA_E_DEFAULT,
            T_MODULUS_DEFAULT,
        );
        let n = 5;
        let keys: Vec<KaheKey> = (0..n).map(|_| Kahe::gen(&mut rng, &pp)).collect();
        let msgs: Vec<Vec<KahePoly>> = (0..n)
            .map(|_| {
                (0..pp.mu_kahe * pp.l)
                    .map(|_| rand_message_poly(&mut rng, pp.t_modulus))
                    .collect()
            })
            .collect();
        let ctxts: Vec<Vec<KahePoly>> = keys
            .iter()
            .zip(&msgs)
            .map(|(k, m)| Kahe::enc(&mut rng, &pp, k, m))
            .collect();
        let agg_c = Kahe::agg_ctxt(&ctxts);
        let agg_k = Kahe::agg_key(&keys);
        let agg_m_expected: Vec<KahePoly> = (0..pp.mu_kahe * pp.l)
            .map(|i| {
                let summed = msgs.iter().fold(KahePoly::default(), |a, x| a + x[i]);
                poly_mod_t(&summed, pp.t_modulus)
            })
            .collect();
        assert_eq!(Kahe::dec(&pp, &agg_c, &agg_k), agg_m_expected);
    }

    #[test]
    fn bridge_round_trip_small_values() {
        // Realistic Shamir-recovered range: |coef| ≤ N_clients · 8σ_s ≈ 3600.
        let mut rng = ChaCha20Rng::from_seed([41u8; 32]);
        for _ in 0..50 {
            let mut coeffs = [0i32; N];
            for c in coeffs.iter_mut() {
                *c = rng.gen_range(-3600i32..=3600);
            }
            let cs = CsPoly::from_coeffs(coeffs);
            let kahe = lift_cs_to_kahe(&cs);
            // The lift preserves centered representatives for small inputs.
            for (a, b) in cs.coeffs().iter().zip(kahe.coeffs().iter()) {
                assert_eq!(*a, *b);
            }
            // Round trip via kahe_to_cs_centered.
            let back = kahe_to_cs_centered(&kahe);
            for (a, b) in cs.coeffs().iter().zip(back.coeffs().iter()) {
                assert_eq!(*a, *b);
            }
        }
    }
}
