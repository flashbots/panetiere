//! Key-additive homomorphic encryption
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

pub struct KaheParams {
    pub a_ntt: Vec<KaheNTTPoly>,
    pub mu_kahe: usize,
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

/// Aggregate KAHE key — element of `R_{q_kahe}`
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

fn pad_poly(a_ntt: &[KaheNTTPoly], sk_ntt: &KaheNTTPoly, i: usize) -> KahePoly {
    KahePoly::from(&pointwise_dot_kahe(
        std::slice::from_ref(&a_ntt[i]),
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
    pub fn setup_with_dims<R: Rng>(
        rng: &mut R,
        mu_kahe: usize,
        sigma_s: f64,
        sigma_e: f64,
        t_modulus: u64,
    ) -> KaheParams {
        assert!(mu_kahe >= 1, "μ_kahe must be ≥ 1");
        assert!(t_modulus >= 2, "t_modulus must be ≥ 2");
        // Sample directly NTT-resident — uniform-in-NTT slot is statistically
        // equivalent to NTT(uniform coeff poly) and saves `μ` forward NTTs.
        let a_ntt: Vec<KaheNTTPoly> = (0..mu_kahe)
            .map(|_| KaheNTTPoly::rand_ntt_poly(rng))
            .collect();
        KaheParams {
            a_ntt,
            mu_kahe,
            sigma_s,
            sigma_e,
            t_modulus,
        }
    }
}

pub const SIGMA_S_DEFAULT: f64 = 15.72;
pub const SIGMA_E_DEFAULT: f64 = 15.72;
pub const T_MODULUS_DEFAULT: u64 = 1 << 36;

impl KaheScheme for Kahe {
    type Params = KaheParams;
    type Key = KaheKey;
    type AggKey = KaheAggKey;
    type Message = Vec<KahePoly>;
    type Ciphertext = Vec<KahePoly>;

    /// `μ = 1`, σ_s=σ_e=15.72, `t = 2^36` at q_kahe ≈ 2^48.3.
    fn setup<R: Rng>(rng: &mut R) -> KaheParams {
        Self::setup_with_dims(rng, 1, SIGMA_S_DEFAULT, SIGMA_E_DEFAULT, T_MODULUS_DEFAULT)
    }

    fn gen<R: Rng>(rng: &mut R, pp: &KaheParams) -> KaheKey {
        KaheKey(sample_dg_poly(rng, pp.sigma_s))
    }

    /// Encrypt up to `μ` plaintext polys under one key `sk`. For position
    /// `i ∈ [0, μ)`: `c_i = m_i + a_i·sk + t·e_i`, with a fresh `e_i ← D_{σ_e}`.
    fn enc<R: Rng>(rng: &mut R, pp: &KaheParams, k: &KaheKey, m: &Vec<KahePoly>) -> Vec<KahePoly> {
        debug_assert!(m.len() <= pp.mu_kahe);
        let t = pp.t_modulus as i64;
        let sk_ntt = KaheNTTPoly::from(k.inner());
        let total = m.len();
        let seeds = crate::fork_seeds(rng, total);
        (0..total)
            .into_par_iter()
            .map(|i| {
                let pad = pad_poly(&pp.a_ntt, &sk_ntt, i);
                let mut item_rng = ChaCha20Rng::from_seed(seeds[i]);
                let e = sample_dg_poly(&mut item_rng, pp.sigma_e);
                let te = scale_poly(&e, t);
                m[i] + pad + te
            })
            .collect()
    }

    /// `((c_i − a_i·sk_agg) mod q_kahe) reduced mod t`, per position.
    fn dec(pp: &KaheParams, c: &Vec<KahePoly>, k: &KaheAggKey) -> Vec<KahePoly> {
        debug_assert!(c.len() <= pp.mu_kahe);
        let sk_ntt = KaheNTTPoly::from(k.inner());
        let total = c.len();
        (0..total)
            .into_par_iter()
            .map(|i| {
                let pad = pad_poly(&pp.a_ntt, &sk_ntt, i);
                let raw = c[i] - pad;
                poly_mod_t(&raw, pp.t_modulus)
            })
            .collect()
    }

    fn agg_ctxt(cs: &[Vec<KahePoly>]) -> Vec<KahePoly> {
        if cs.is_empty() {
            return Vec::new();
        }
        let len = cs[0].len();
        let q = chipmunk_code::KAHE_MODULUS;
        let half = q / 2;
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
    fn round_trip_wide_mu() {
        let mut rng = ChaCha20Rng::from_seed([13u8; 32]);
        let pp = Kahe::setup_with_dims(
            &mut rng,
            20,
            SIGMA_S_DEFAULT,
            SIGMA_E_DEFAULT,
            T_MODULUS_DEFAULT,
        );
        let k = Kahe::gen(&mut rng, &pp);
        let m: Vec<KahePoly> = (0..pp.mu_kahe)
            .map(|_| rand_message_poly(&mut rng, pp.t_modulus))
            .collect();
        let c = Kahe::enc(&mut rng, &pp, &k, &m);
        assert_eq!(c.len(), pp.mu_kahe);
        let agg = Kahe::agg_key(std::slice::from_ref(&k));
        assert_eq!(Kahe::dec(&pp, &c, &agg), m);
    }

    /// A message shorter than `μ` is legal — the trailing positions are unused.
    #[test]
    fn round_trip_partial_width() {
        let mut rng = ChaCha20Rng::from_seed([19u8; 32]);
        let pp = Kahe::setup_with_dims(
            &mut rng,
            20,
            SIGMA_S_DEFAULT,
            SIGMA_E_DEFAULT,
            T_MODULUS_DEFAULT,
        );
        let k = Kahe::gen(&mut rng, &pp);
        let m: Vec<KahePoly> = (0..13)
            .map(|_| rand_message_poly(&mut rng, pp.t_modulus))
            .collect();
        let c = Kahe::enc(&mut rng, &pp, &k, &m);
        assert_eq!(c.len(), 13);
        let agg = Kahe::agg_key(std::slice::from_ref(&k));
        assert_eq!(Kahe::dec(&pp, &c, &agg), m);
    }

    #[test]
    fn additive_homomorphism_wide_mu() {
        let mut rng = ChaCha20Rng::from_seed([17u8; 32]);
        let pp = Kahe::setup_with_dims(
            &mut rng,
            12,
            SIGMA_S_DEFAULT,
            SIGMA_E_DEFAULT,
            T_MODULUS_DEFAULT,
        );
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
