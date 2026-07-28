//! Key-additive homomorphic encryption (Willow-style RLWE-based KAHE).
//!
//! Lives on its own ring `R_{q_kahe}` (chipmunk's `KahePoly`, a 64-bit ring
//! with q ≈ 2^48.3), decoupled from the chipmunk Ring-SIS CS ring.
//! The KAHE q is chosen for headroom in the noise budget — chipmunk's q is
//! tied to its multi-signature size optimization and would be wasteful here.
//!
//! Per-poly form:
//!
//! ```text
//! Enc(m, sk):  sample e ← D_{σ_e}^μ;  c = m + a·sk + t·e   mod q_kahe
//! Dec(c, sk):  ((c − a·sk) mod q_kahe) reduced mod t   (centered representatives)
//! ```
//!
//! Parameters:
//! - `a ∈ R_{q_kahe}^μ` — public vector.
//! - `sk ← D_{σ_s}` — a single discrete-Gaussian ring element (σ_s = 15.72).
//! - `e ← D_{σ_e}^μ` — discrete Gaussian fresh error (single `D_σ`, σ_e = 15.72).
//! - `t_modulus` — plaintext modulus. Plaintext lives in `R_t^μ` with centered
//!   representatives in `[-t/2, t/2)`. Aggregate decryption returns `Σm mod t`.
//!
//! Correctness budget at ρ aggregations: need `t·8σ_e·√ρ + ρ·t/2 < q_kahe/2`.
//! At σ_e = 15.72, q_kahe = 347_280_875_347_969, t = 2^36: holds for ρ ≲ 349,
//! covering the S=8, N=100 operating point.
//!
//! `KaheKey` (fresh, low-norm) feeds `Enc`; `KaheAggKey` (in `R_{q_kahe}`,
//! recovered by Shamir interpolation over `R_{q_cs}` and lifted via
//! [`lift_cs_to_kahe`] in `protocol::verify`) feeds `Dec`. Hiding-given-
//! aggregate-key reduces to (Hint-)RLWE rather than to LHL.
//!
//! Bridge layer. KAHE secret keys are *small* (Gaussian σ_s = 15.72,
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

/// Public parameters. `a_ntt[i]` is the length-`μ` (NTT-resident) public vector
/// `a` for chunk `i ∈ [0, l)`. One Shamir+CS round amortizes its commit
/// overhead across `l` ciphertexts that share the same `sk` but each use a
/// different `a_i` — security reduces to standard multi-sample RLWE with the
/// same secret. `t_modulus` is the plaintext modulus; both `sigma_s` (key) and
/// `sigma_e` (error) are Gaussian standard deviations.
pub struct KaheParams {
    pub a_ntt: Vec<Vec<KaheNTTPoly>>,
    pub mu_kahe: usize,
    /// Number of ciphertext chunks emitted per `enc` call (and consumed per
    /// `dec`). Messages and ciphertexts are flat `Vec<KahePoly>` of length
    /// `mu_kahe · l`.
    pub l: usize,
    pub sigma_s: f64,
    pub sigma_e: f64,
    pub t_modulus: u64,
}

/// A *fresh* KAHE key — one discrete-Gaussian ring element. Only valid input to
/// [`KaheScheme::enc`].
#[derive(Clone)]
pub struct KaheKey(KahePoly);

impl KaheKey {
    pub(crate) fn inner(&self) -> &KahePoly {
        &self.0
    }
}

/// Aggregate KAHE key — element of `R_{q_kahe}`, output of
/// [`KaheScheme::agg_key`] or constructed by the verifier via
/// [`KaheAggKey::from_component`] after Shamir interpolation (and the
/// CS→KAHE bridge). Only valid input to [`KaheScheme::dec`].
#[derive(Clone)]
pub struct KaheAggKey(KahePoly);

impl KaheAggKey {
    pub(crate) fn inner(&self) -> &KahePoly {
        &self.0
    }
    pub fn from_component(component: KahePoly) -> Self {
        Self(component)
    }
}

/// Sample a single discrete-Gaussian-rounded polynomial coefficient.
/// Round-to-nearest-integer applied to a continuous Gaussian; clamped at the
/// 8σ tail (Willow's convention).
fn sample_dg<R: Rng>(rng: &mut R, sigma: f64) -> i64 {
    let normal = Normal::new(0.0, sigma).expect("σ > 0");
    let tail = (8.0 * sigma).ceil() as i64;
    loop {
        let x = normal.sample(rng).round() as i64;
        if x.abs() <= tail {
            return x;
        }
    }
}

fn sample_dg_poly<R: Rng>(rng: &mut R, sigma: f64) -> KahePoly {
    let mut coeffs = [0i64; N];
    for c in coeffs.iter_mut() {
        *c = sample_dg(rng, sigma);
    }
    KahePoly::from_coeffs(coeffs)
}

/// `(a_chunk[i] · sk)` where `sk` is already in NTT representation. Used by
/// `enc`/`dec` to amortize the secret's single NTT across all `l·μ` rows.
fn pad_poly(
    a_ntt: &[Vec<KaheNTTPoly>],
    sk_ntt: &KaheNTTPoly,
    chunk: usize,
    i: usize,
) -> KahePoly {
    KahePoly::from(&pointwise_dot_kahe(
        std::slice::from_ref(&a_ntt[chunk][i]),
        std::slice::from_ref(sk_ntt),
    ))
}

/// Accumulate `Σ_client cs[client][pos]` into an `i64[N]` accumulator. Summing
/// ρ centered coefficients (each `≤ q/2 < 2^49`) reaches `ρ·2^49 < 2^63` for
/// any realistic ρ, so plain i64 accumulation with one final reduction is safe.
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
            acc[i] += src[i];
        }
    }
}

/// AVX2: accumulate 4-wide i64 lanes. `N` is a multiple of 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_pos_avx2(acc: &mut [i64; N], cs: &[Vec<KahePoly>], pos: usize) {
    use std::arch::x86_64::*;
    debug_assert_eq!(N % 4, 0);
    for client in cs {
        let src = client[pos].coeffs().as_ptr();
        let mut i = 0usize;
        while i < N {
            let v = _mm256_loadu_si256(src.add(i) as *const __m256i);
            let a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
            let s = _mm256_add_epi64(a, v);
            _mm256_storeu_si256(acc.as_mut_ptr().add(i) as *mut __m256i, s);
            i += 4;
        }
    }
}

/// `poly[i] · scale` mod q. Products reach `q/2 · t ≈ 2^86`, so i128.
fn scale_poly(poly: &KahePoly, scale: i64) -> KahePoly {
    let mut coeffs = [0i64; N];
    let q = chipmunk_code::KAHE_MODULUS as i128;
    for (out, &c) in coeffs.iter_mut().zip(poly.coeffs().iter()) {
        *out = ((c as i128) * (scale as i128) % q) as i64;
    }
    let mut p = KahePoly::from_coeffs(coeffs);
    p.normalize();
    p
}

/// Reduce one (centered) integer coefficient mod `t` to centered range
/// `[-t/2, t/2)`.
fn reduce_centered(x: i64, t: i64) -> i64 {
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
fn poly_mod_t(poly: &KahePoly, t: u64) -> KahePoly {
    let mut p = *poly;
    p.normalize();
    let t_i = t as i64;
    let mut coeffs = [0i64; N];
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
    let coeffs: [i64; N] = core::array::from_fn(|i| q.coeffs()[i] as i64);
    KahePoly::from_signed_coeffs(&coeffs)
}

/// Reduce a small `KahePoly` (a fresh KAHE secret key — Gaussian σ_s ≈ 15.72,
/// `‖·‖_∞ ≤ 8σ_s ≈ 126`) into a `CsPoly` by centered-rep re-interpretation.
/// Lossless iff `‖input‖_∞ ≤ q_cs/2`. Used at the client side to feed Shamir
/// sharing, which operates over `R_{q_cs}`.
pub fn kahe_to_cs_centered(p: &KahePoly) -> CsPoly {
    let mut q = *p;
    q.normalize();
    debug_assert!(
        q.coeffs()
            .iter()
            .all(|&c| c.abs() <= CS_MODULUS_OVER_TWO as i64),
        "kahe_to_cs_centered: coefficient magnitude exceeds q_cs/2 = {} \
         (callers must only bridge small Gaussian-bounded values)",
        CS_MODULUS_OVER_TWO
    );
    let coeffs: [i32; N] = core::array::from_fn(|i| q.coeffs()[i] as i32);
    CsPoly::from_coeffs(coeffs)
}

pub struct Kahe;

impl Kahe {
    /// Setup with explicit dimensions and Gaussian widths. `sigma_s` = key std,
    /// `sigma_e` = error std. Defaults `SIGMA_S_DEFAULT = SIGMA_E_DEFAULT =
    /// 15.72` (single `D_σ`) target ~128-bit RLWE at N=2048.
    pub fn setup_with_dims<R: Rng>(
        rng: &mut R,
        mu_kahe: usize,
        l: usize,
        sigma_s: f64,
        sigma_e: f64,
        t_modulus: u64,
    ) -> KaheParams {
        assert!(mu_kahe >= 1, "μ_kahe must be ≥ 1");
        assert!(l >= 1, "l must be ≥ 1");
        assert!(t_modulus >= 2, "t_modulus must be ≥ 2");
        // Sample directly NTT-resident — uniform-in-NTT slot is statistically
        // equivalent to NTT(uniform coeff poly) and saves `l·μ` forward NTTs.
        let a_ntt: Vec<Vec<KaheNTTPoly>> = (0..l)
            .map(|_| (0..mu_kahe).map(|_| KaheNTTPoly::rand_ntt_poly(rng)).collect())
            .collect();
        KaheParams {
            a_ntt,
            mu_kahe,
            l,
            sigma_s,
            sigma_e,
            t_modulus,
        }
    }
}

/// q_kahe = 347_280_875_347_969 (≈2^48.3) at N=2048. Single `D_σ` for key and
/// error (Nils's spec), σ_s = σ_e = 15.72 — sized for ~128-bit RLWE at N=2048.
pub const SIGMA_S_DEFAULT: f64 = 15.72;
pub const SIGMA_E_DEFAULT: f64 = 15.72;
/// 2^36. Per-coefficient noise budget `t·8σ_e·√ρ + ρ·t/2 < q_kahe/2` holds for
/// ρ ≲ 349 at (t, σ, q) = (2^36, 15.72, 347_280_875_347_969); comfortably
/// covers the S=8, N=100 operating point.
pub const T_MODULUS_DEFAULT: u64 = 1 << 36;

impl KaheScheme for Kahe {
    type Params = KaheParams;
    type Key = KaheKey;
    type AggKey = KaheAggKey;
    type Message = Vec<KahePoly>;
    type Ciphertext = Vec<KahePoly>;

    /// `(μ, l) = (1, 1)`, σ_s=σ_e=15.72, `t = 2^36` at q_kahe ≈ 2^48.3.
    fn setup<R: Rng>(rng: &mut R) -> KaheParams {
        Self::setup_with_dims(
            rng,
            1,
            1,
            SIGMA_S_DEFAULT,
            SIGMA_E_DEFAULT,
            T_MODULUS_DEFAULT,
        )
    }

    fn gen<R: Rng>(rng: &mut R, pp: &KaheParams) -> KaheKey {
        KaheKey(sample_dg_poly(rng, pp.sigma_s))
    }

    /// Batch-encrypt up to `l` chunks of `μ` plaintext polys under one key
    /// `sk`. For chunk `i ∈ [0, l)`: `c_i = m_i + a_i·sk + t·e_i`, with a
    /// fresh `e_i ← D_{σ_e}^μ`. The NTT of `sk` is computed once and reused
    /// across all chunks. The final chunk may be partial (`m.len() ≤ μ·l`):
    /// unused entries of `a` are simply not used — fewer RLWE samples under
    /// the same secret — so ciphertexts are not padded to chunk boundaries.
    fn enc<R: Rng>(rng: &mut R, pp: &KaheParams, k: &KaheKey, m: &Vec<KahePoly>) -> Vec<KahePoly> {
        debug_assert!(m.len() <= pp.mu_kahe * pp.l);
        let t = pp.t_modulus as i64;
        let sk_ntt = KaheNTTPoly::from(k.inner());
        let total = m.len();
        let seeds = crate::fork_seeds(rng, total);
        (0..total)
            .into_par_iter()
            .map(|idx| {
                let (chunk, i) = (idx / pp.mu_kahe, idx % pp.mu_kahe);
                let pad = pad_poly(&pp.a_ntt, &sk_ntt, chunk, i);
                let mut item_rng = ChaCha20Rng::from_seed(seeds[idx]);
                let e = sample_dg_poly(&mut item_rng, pp.sigma_e);
                let te = scale_poly(&e, t);
                m[idx] + pad + te
            })
            .collect()
    }

    /// `((c_i − a_i·sk_agg) mod q_kahe) reduced mod t`, per chunk. `sk_agg`'s
    /// NTT is computed once and shared across all chunks. Accepts a partial
    /// final chunk, mirroring `enc`.
    fn dec(pp: &KaheParams, c: &Vec<KahePoly>, k: &KaheAggKey) -> Vec<KahePoly> {
        debug_assert!(c.len() <= pp.mu_kahe * pp.l);
        let sk_ntt = KaheNTTPoly::from(k.inner());
        let total = c.len();
        (0..total)
            .into_par_iter()
            .map(|idx| {
                let (chunk, i) = (idx / pp.mu_kahe, idx % pp.mu_kahe);
                let pad = pad_poly(&pp.a_ntt, &sk_ntt, chunk, i);
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
                let mut coeffs = [0i64; N];
                for (out, &a) in coeffs.iter_mut().zip(acc.iter()) {
                    let mut r = a.rem_euclid(q);
                    if r > half {
                        r -= q;
                    }
                    *out = r;
                }
                KahePoly::from_coeffs(coeffs)
            })
            .collect()
    }

    fn agg_key(ks: &[KaheKey]) -> KaheAggKey {
        KaheAggKey(
            ks.iter()
                .fold(KahePoly::default(), |acc, k| acc + *k.inner()),
        )
    }
}

// Suppress unused warning for re-exported KAHE_MODULUS_OVER_TWO consumers
// (used by bridge debug assertions transitively).
const _: i64 = KAHE_MODULUS_OVER_TWO;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    /// Random message with coefficients centered in `[-t/2, t/2)`.
    fn rand_message_poly<R: Rng>(rng: &mut R, t: u64) -> KahePoly {
        let half = (t / 2) as i64;
        let mut coeffs = [0i64; N];
        for c in coeffs.iter_mut() {
            *c = rng.gen_range(0..t) as i64 - half;
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
        let secret_cs = kahe_to_cs_centered(k.inner());
        let shares = ShamirSharing::share(&mut rng, &shamir, &secret_cs);
        let samples: Vec<(usize, CsPoly)> = (0..t).map(|i| (i, shares[i])).collect();
        let recovered_cs = ShamirSharing::recover(&shamir, &samples).unwrap();
        let agg = KaheAggKey::from_component(lift_cs_to_kahe(&recovered_cs));

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
                assert_eq!(*a as i64, *b);
            }
            // Round trip via kahe_to_cs_centered.
            let back = kahe_to_cs_centered(&kahe);
            for (a, b) in cs.coeffs().iter().zip(back.coeffs().iter()) {
                assert_eq!(*a, *b);
            }
        }
    }
}
