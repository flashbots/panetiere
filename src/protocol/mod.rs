pub mod client;
pub mod server;
pub mod verify;

use rand::Rng;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct ClientId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct ServerId(pub u32);

use crate::cs::{Cs, HidingMerkleCommitment};
use crate::kahe::{Kahe, KaheParams, KaheScheme, SIGMA_E_DEFAULT, SIGMA_S_DEFAULT, T_MODULUS_DEFAULT};
use crate::sss::ShamirParams;

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
        let cs = HidingMerkleCommitment::setup_with_dims(rng, n_servers, kahe.kappa_kahe, 8);
        let shamir = ShamirParams::new(t, n_servers);
        Self { kahe, cs, shamir }
    }

    /// Explicit KAHE dimensions `(μ, κ)` with Willow defaults
    /// (`σ_s = 4.5`, `σ_e = √2·σ_s`, `t_modulus = 64`).
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
        let cs = HidingMerkleCommitment::setup_with_dims(rng, n_servers, kahe.kappa_kahe, 8);
        let shamir = ShamirParams::new(t, n_servers);
        Self { kahe, cs, shamir }
    }
}
