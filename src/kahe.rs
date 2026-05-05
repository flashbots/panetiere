//! Key-additive homomorphic encryption.
//!
//! Two layers:
//!
//! 1. [`RingOtp`] — pad-expansion + OTP-style enc/dec primitive. `expand(sk) =
//!    A·sk ∈ R_q^μ` for a public matrix `A ∈ R_q^{μ × κ}`. `enc(sk, m) = m +
//!    expand(sk)`, `dec(sk, c) = c − expand(sk)`. `expand` is `R`-linear, so
//!    the OTP works on both fresh (short) and aggregate (large-norm) seeds —
//!    callers supply the seed as a raw `&[HVCPoly]`.
//!
//! 2. [`Kahe`] — the KAHE scheme (impls [`KaheScheme`]). `Gen` samples a
//!    *short* seed (Lemma 8 hiding regime). `Enc`/`Dec` wrap [`RingOtp`] but
//!    constrain the key types: [`KaheKey`] (fresh, short) feeds `Enc`,
//!    [`KaheAggKey`] (aggregate, in `R_q^κ`) feeds `Dec`. The scheme also
//!    provides `agg_ctxt` and `agg_key`. The Shamir bridge (`Σ sk_j`
//!    interpolated from per-server share sums) lives in `protocol::verify`,
//!    where it constructs a `KaheAggKey` via [`KaheAggKey::from_components`].
//!
//! Hiding (BDLOP §4 LHL, Lemma 8) is proven for fresh keys living in the short
//! ball `B_β^κ`. Aggregate keys live in all of `R_q^κ` and are only valid as
//! `Dec` input on aggregated ciphertexts; the [`KaheKey`] / [`KaheAggKey`]
//! newtypes enforce this distinction at the type level.

use chipmunk_code::{pointwise_dot, HVCNTTPoly, HVCPoly, Polynomial};
use rand::Rng;

/// KAHE scheme contract.
pub trait KaheScheme {
    type Params;
    type Key: Clone;
    type AggKey: Clone;
    type Message: Clone;
    type Ciphertext: Clone;

    fn setup<R: Rng>(rng: &mut R) -> Self::Params;
    fn gen<R: Rng>(rng: &mut R, pp: &Self::Params) -> Self::Key;
    fn enc(pp: &Self::Params, k: &Self::Key, m: &Self::Message) -> Self::Ciphertext;
    fn dec(pp: &Self::Params, c: &Self::Ciphertext, k: &Self::AggKey) -> Self::Message;
    fn agg_ctxt(cs: &[Self::Ciphertext]) -> Self::Ciphertext;
    fn agg_key(ks: &[Self::Key]) -> Self::AggKey;
}

/// Public matrix-form parameters. `a_matrix_ntt` is `μ × κ`. `sk_bound` is
/// the KAHE secret-key infinity-norm bound (Lemma 8 `β`).
pub struct KaheParams {
    pub a_matrix_ntt: Vec<Vec<HVCNTTPoly>>,
    pub mu_kahe: usize,
    pub kappa_kahe: usize,
    pub sk_bound: u32,
}

/// Pad-expansion + OTP-style enc/dec primitive. Key-agnostic: callers supply
/// the seed as a raw `&[HVCPoly]`. Reads only the pad-shape fields of
/// [`KaheParams`] (`a_matrix_ntt`, dims); ignores `sk_bound`.
pub struct RingOtp;

impl RingOtp {
    pub fn expand(pp: &KaheParams, sk: &[HVCPoly]) -> Vec<HVCPoly> {
        debug_assert_eq!(sk.len(), pp.kappa_kahe);
        matvec(&pp.a_matrix_ntt, sk)
    }

    /// `c = m + expand(sk)`.
    pub fn enc(pp: &KaheParams, sk: &[HVCPoly], m: &[HVCPoly]) -> Vec<HVCPoly> {
        debug_assert_eq!(m.len(), pp.mu_kahe);
        let pad = Self::expand(pp, sk);
        vec_add(m, &pad)
    }

    /// `m = c − expand(sk)`.
    pub fn dec(pp: &KaheParams, sk: &[HVCPoly], c: &[HVCPoly]) -> Vec<HVCPoly> {
        debug_assert_eq!(c.len(), pp.mu_kahe);
        let pad = Self::expand(pp, sk);
        vec_sub(c, &pad)
    }
}

/// A *fresh* KAHE key — short (`‖·‖∞ ≤ β`), sampled by [`KaheScheme::gen`].
/// Only valid input to [`KaheScheme::enc`].
#[derive(Clone)]
pub struct KaheKey(Vec<HVCPoly>);

impl KaheKey {
    pub(crate) fn inner(&self) -> &[HVCPoly] {
        &self.0
    }
    /// Component-wise access for the SSS bridge in the protocol layer.
    pub(crate) fn component(&self, k: usize) -> &HVCPoly {
        &self.0[k]
    }
}

/// An aggregate KAHE key — element of `R_q^κ`, output of
/// [`KaheScheme::agg_key`] or constructed by the verifier via
/// [`KaheAggKey::from_components`] after Shamir interpolation. Only valid input
/// to [`KaheScheme::dec`] on an aggregated ciphertext.
#[derive(Clone)]
pub struct KaheAggKey(Vec<HVCPoly>);

impl KaheAggKey {
    pub(crate) fn inner(&self) -> &[HVCPoly] {
        &self.0
    }
    /// Construct from per-component polys (one per `κ_kahe` slot). Used by
    /// `protocol::verify` to wrap the Shamir-interpolated aggregate key.
    pub fn from_components(components: Vec<HVCPoly>) -> Self {
        Self(components)
    }
}

fn matvec(a_matrix_ntt: &[Vec<HVCNTTPoly>], sk: &[HVCPoly]) -> Vec<HVCPoly> {
    let sk_ntt: Vec<HVCNTTPoly> = sk.iter().map(HVCNTTPoly::from).collect();
    a_matrix_ntt
        .iter()
        .map(|row| HVCPoly::from(&pointwise_dot(row, &sk_ntt)))
        .collect()
}

fn vec_add(a: &[HVCPoly], b: &[HVCPoly]) -> Vec<HVCPoly> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect()
}

fn vec_sub(a: &[HVCPoly], b: &[HVCPoly]) -> Vec<HVCPoly> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect()
}

fn vec_zero(len: usize) -> Vec<HVCPoly> {
    vec![HVCPoly::default(); len]
}

/// The KAHE scheme used by flashnet. Built on [`RingOtp`] with short-key
/// `Gen`, type-separated fresh / aggregate keys, and a Shamir `recover_key`.
pub struct Kahe;

impl Kahe {
    /// Setup with explicit dimensions. `(μ, κ, β)` must jointly satisfy
    /// Lemma 8 for the target ρ and λ; this constructor does not check.
    pub fn setup_with_dims<R: Rng>(
        rng: &mut R,
        mu_kahe: usize,
        kappa_kahe: usize,
        sk_bound: u32,
    ) -> KaheParams {
        assert!(mu_kahe >= 1, "μ_kahe must be ≥ 1");
        assert!(kappa_kahe >= 1, "κ_kahe must be ≥ 1");
        let a_matrix_ntt: Vec<Vec<HVCNTTPoly>> = (0..mu_kahe)
            .map(|_| {
                (0..kappa_kahe)
                    .map(|_| HVCNTTPoly::from(&HVCPoly::rand_poly(rng)))
                    .collect()
            })
            .collect();
        KaheParams {
            a_matrix_ntt,
            mu_kahe,
            kappa_kahe,
            sk_bound,
        }
    }
}

impl KaheScheme for Kahe {
    type Params = KaheParams;
    type Key = KaheKey;
    type AggKey = KaheAggKey;
    type Message = Vec<HVCPoly>;
    type Ciphertext = Vec<HVCPoly>;

    /// Default-dimension setup: `(μ, κ, β) = (1, 6, 64)` — satisfies Lemma 8
    /// at λ=128 for ρ ≤ 2²⁰ with the chipmunk HVC modulus.
    fn setup<R: Rng>(rng: &mut R) -> KaheParams {
        Self::setup_with_dims(rng, 1, 6, 64)
    }

    fn gen<R: Rng>(rng: &mut R, pp: &KaheParams) -> KaheKey {
        // Short low-norm secret: each component has coefficients in [-sk_bound, sk_bound].
        let polys = (0..pp.kappa_kahe)
            .map(|_| HVCPoly::rand_mod_p(rng, pp.sk_bound))
            .collect();
        KaheKey(polys)
    }

    fn enc(pp: &KaheParams, k: &KaheKey, m: &Vec<HVCPoly>) -> Vec<HVCPoly> {
        debug_assert_eq!(k.inner().len(), pp.kappa_kahe);
        RingOtp::enc(pp, k.inner(), m)
    }

    fn dec(pp: &KaheParams, c: &Vec<HVCPoly>, k: &KaheAggKey) -> Vec<HVCPoly> {
        debug_assert_eq!(k.inner().len(), pp.kappa_kahe);
        RingOtp::dec(pp, k.inner(), c)
    }

    fn agg_ctxt(cs: &[Vec<HVCPoly>]) -> Vec<HVCPoly> {
        if cs.is_empty() {
            return Vec::new();
        }
        let len = cs[0].len();
        cs.iter().fold(vec_zero(len), |acc, x| vec_add(&acc, x))
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn round_trip() {
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        let pp = Kahe::setup(&mut rng);
        let k = Kahe::gen(&mut rng, &pp);
        let m: Vec<HVCPoly> = (0..pp.mu_kahe).map(|_| HVCPoly::rand_poly(&mut rng)).collect();
        let c = Kahe::enc(&pp, &k, &m);
        // Dec takes an AggKey; a single fresh key lifts via agg_key([k]).
        let agg = Kahe::agg_key(std::slice::from_ref(&k));
        assert_eq!(Kahe::dec(&pp, &c, &agg), m);
    }

    #[test]
    fn additive_homomorphism() {
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        let pp = Kahe::setup(&mut rng);
        let n = 5;
        let keys: Vec<KaheKey> = (0..n).map(|_| Kahe::gen(&mut rng, &pp)).collect();
        let msgs: Vec<Vec<HVCPoly>> = (0..n)
            .map(|_| (0..pp.mu_kahe).map(|_| HVCPoly::rand_poly(&mut rng)).collect())
            .collect();
        let ctxts: Vec<Vec<HVCPoly>> = keys
            .iter()
            .zip(&msgs)
            .map(|(k, m)| Kahe::enc(&pp, k, m))
            .collect();

        let agg_c = Kahe::agg_ctxt(&ctxts);
        let agg_k = Kahe::agg_key(&keys);
        let agg_m_expected: Vec<HVCPoly> = (0..pp.mu_kahe)
            .map(|i| msgs.iter().fold(HVCPoly::default(), |a, x| a + x[i]))
            .collect();

        assert_eq!(Kahe::dec(&pp, &agg_c, &agg_k), agg_m_expected);
    }

    /// Shamir-interpolated aggregate key (the `verify.rs` path) decrypts the
    /// single-client ciphertext correctly.
    #[test]
    fn shamir_recovered_agg_key_round_trip() {
        use crate::sss::{ShamirParams, ShamirSharing};

        let mut rng = ChaCha20Rng::from_seed([23u8; 32]);
        let pp = Kahe::setup(&mut rng);
        let n_servers = 4;
        let t = 3;
        let shamir = ShamirParams::new(t, n_servers);
        let k = Kahe::gen(&mut rng, &pp);

        // Shamir-share each component, then interpolate from the first t
        // servers — exactly what verify.rs does.
        let recovered_components: Vec<HVCPoly> = (0..pp.kappa_kahe)
            .map(|c| {
                let shares = ShamirSharing::share(&mut rng, &shamir, k.component(c));
                let samples: Vec<(usize, HVCPoly)> =
                    (0..t).map(|i| (i, shares[i])).collect();
                ShamirSharing::recover(&shamir, &samples)
            })
            .collect();
        let agg = KaheAggKey::from_components(recovered_components);

        let m: Vec<HVCPoly> = (0..pp.mu_kahe).map(|_| HVCPoly::rand_poly(&mut rng)).collect();
        let c = Kahe::enc(&pp, &k, &m);
        assert_eq!(Kahe::dec(&pp, &c, &agg), m);
    }

    /// `RingOtp::enc` / `dec` directly (no Key newtypes) round-trips on a raw seed.
    #[test]
    fn ring_otp_raw_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
        let pp = Kahe::setup_with_dims(&mut rng, 1, 6, 64);
        let sk: Vec<HVCPoly> = (0..pp.kappa_kahe)
            .map(|_| HVCPoly::rand_mod_p(&mut rng, pp.sk_bound))
            .collect();
        let m: Vec<HVCPoly> = (0..pp.mu_kahe).map(|_| HVCPoly::rand_poly(&mut rng)).collect();
        let c = RingOtp::enc(&pp, &sk, &m);
        assert_eq!(RingOtp::dec(&pp, &sk, &c), m);
    }
}
