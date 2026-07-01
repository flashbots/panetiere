pub mod aggregator;
pub mod client;
pub mod server;
pub mod verify;

use rand::Rng;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct ClientId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct ServerId(pub u32);

use chipmunk_code::KahePoly;

use crate::cs::{Cs, HidingMerkleCommitment};
use crate::kahe::{Kahe, KaheParams, KaheScheme, SIGMA_E_DEFAULT, SIGMA_S_DEFAULT, T_MODULUS_DEFAULT};
use crate::sss::ShamirParams;

/// `KahePoly` count of one KAHE plaintext message (`μ_kahe · l`); every client
/// contributes exactly this many, real or cover.
pub fn message_polys(pp: &ProtocolParams) -> usize {
    pp.kahe.mu_kahe * pp.kahe.l
}

/// All-zero KAHE message (cover): contributes nothing to the aggregated sum.
pub fn zero_message(pp: &ProtocolParams) -> Vec<KahePoly> {
    vec![KahePoly::default(); message_polys(pp)]
}

/// Bundle of public parameters carried through one protocol session.
///
/// Couples `μ_cs = κ_kahe` so each per-server CS leaf-commit packs the full
/// `κ_kahe`-component share vector — a single CS pipeline replaces the
/// `κ_kahe` parallel CS instances of the earlier shape.
pub struct ProtocolParams {
    pub kahe: KaheParams,
    pub cs: <HidingMerkleCommitment as Cs>::Params,
    pub shamir: ShamirParams,
}

impl ProtocolParams {
    /// Build params with `n_servers` servers and threshold
    /// `t = max(⌊n/2⌋ + 1, n − 2)`.
    pub fn setup<R: Rng>(rng: &mut R, n_servers: usize) -> Self {
        let t = (n_servers / 2 + 1).max(n_servers.saturating_sub(2));
        Self::setup_with_threshold(rng, n_servers, t)
    }

    pub fn setup_with_threshold<R: Rng>(rng: &mut R, n_servers: usize, t: usize) -> Self {
        let kahe = Kahe::setup(rng);
        // μ_cs = κ_kahe (coupling); κ_cs = 5 (BDLOP hiding bound, spec).
        let cs = HidingMerkleCommitment::setup_with_dims(rng, n_servers, kahe.kappa_kahe, 5);
        let shamir = ShamirParams::new(t, n_servers);
        Self { kahe, cs, shamir }
    }

    /// Explicit KAHE dimensions `(μ, κ)`; `l = 1`, `σ_s = σ_e = 15.72`,
    /// `t_modulus = 2^16` (the `*_DEFAULT` constants).
    pub fn setup_with_kahe_dims<R: Rng>(
        rng: &mut R,
        n_servers: usize,
        mu_kahe: usize,
        kappa_kahe: usize,
    ) -> Self {
        Self::setup_with_kahe_dims_full(
            rng,
            n_servers,
            mu_kahe,
            kappa_kahe,
            1,
            SIGMA_S_DEFAULT,
            SIGMA_E_DEFAULT,
            T_MODULUS_DEFAULT,
        )
    }

    /// Explicit KAHE dimensions and Gaussian / plaintext-modulus parameters.
    /// `l` is the per-round chunk multiplier: one Shamir+CS pass covers `l`
    /// ciphertext chunks (each `μ`-wide) under independent matrices `A_i`.
    /// Threshold `t = max(⌊n/2⌋ + 1, n − 2)`. CS μ_cs is coupled to κ_kahe.
    pub fn setup_with_kahe_dims_full<R: Rng>(
        rng: &mut R,
        n_servers: usize,
        mu_kahe: usize,
        kappa_kahe: usize,
        l: usize,
        sigma_s: f64,
        sigma_e: f64,
        t_modulus: u32,
    ) -> Self {
        let t = (n_servers / 2 + 1).max(n_servers.saturating_sub(2));
        let kahe =
            Kahe::setup_with_dims(rng, mu_kahe, kappa_kahe, l, sigma_s, sigma_e, t_modulus);
        // μ_cs = κ_kahe (coupling); κ_cs = 5 (BDLOP hiding bound, spec).
        let cs = HidingMerkleCommitment::setup_with_dims(rng, n_servers, kahe.kappa_kahe, 5);
        let shamir = ShamirParams::new(t, n_servers);
        Self { kahe, cs, shamir }
    }
}
