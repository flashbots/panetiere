pub mod aggregator;
pub mod client;
pub mod server;
pub mod verify;

use rand::Rng;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct ClientId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct ServerId(pub u32);

/// Identifier of one protocol execution.
///
/// Must be unique per execution: sealed openings are bound to it, so reusing one
/// across rounds lets an envelope be replayed from an earlier round into a later
/// one.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SessionId(pub [u8; 32]);

/// AEAD associated data for one sealed opening. Binding the context here makes a
/// mismatch an authentication failure rather than a check the caller can skip.
pub fn opening_aad(sid: &SessionId, client: ClientId, server: ServerId) -> Vec<u8> {
    const DOMAIN: &[u8] = b"panetiere/opening/v1";
    let mut aad = Vec::with_capacity(DOMAIN.len() + 32 + 8);
    aad.extend_from_slice(DOMAIN);
    aad.extend_from_slice(&sid.0);
    aad.extend_from_slice(&client.0.to_le_bytes());
    aad.extend_from_slice(&server.0.to_le_bytes());
    aad
}

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
/// The KAHE key is a single ring element, so each per-server CS leaf commits
/// exactly one Shamir share: `μ_cs = 1`.
pub struct ProtocolParams {
    pub kahe: KaheParams,
    pub cs: <HidingMerkleCommitment as Cs>::Params,
    pub shamir: ShamirParams,
    /// Anonymity floor: never decrypt over fewer distinct clients. Default 1.
    pub min_clients: usize,
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
        let cs = HidingMerkleCommitment::setup_with_dims(rng, n_servers, MU_CS, KAPPA_CS);
        let shamir = ShamirParams::new(t, n_servers);
        Self { kahe, cs, shamir, min_clients: 1 }
    }

    /// Explicit KAHE width `μ`; `l = 1`, `σ_s = σ_e = 15.72`,
    /// `t_modulus = 2^36` (the `*_DEFAULT` constants).
    pub fn setup_with_kahe_dims<R: Rng>(rng: &mut R, n_servers: usize, mu_kahe: usize) -> Self {
        Self::setup_with_kahe_dims_full(
            rng,
            n_servers,
            mu_kahe,
            1,
            SIGMA_S_DEFAULT,
            SIGMA_E_DEFAULT,
            T_MODULUS_DEFAULT,
        )
    }

    /// Explicit KAHE dimensions and Gaussian / plaintext-modulus parameters.
    /// `l` is the per-round chunk multiplier: one Shamir+CS pass covers `l`
    /// ciphertext chunks (each `μ`-wide) under independent vectors `a_i`.
    /// Threshold `t = max(⌊n/2⌋ + 1, n − 2)`.
    pub fn setup_with_kahe_dims_full<R: Rng>(
        rng: &mut R,
        n_servers: usize,
        mu_kahe: usize,
        l: usize,
        sigma_s: f64,
        sigma_e: f64,
        t_modulus: u64,
    ) -> Self {
        let t = (n_servers / 2 + 1).max(n_servers.saturating_sub(2));
        let kahe = Kahe::setup_with_dims(rng, mu_kahe, l, sigma_s, sigma_e, t_modulus);
        let cs = HidingMerkleCommitment::setup_with_dims(rng, n_servers, MU_CS, KAPPA_CS);
        let shamir = ShamirParams::new(t, n_servers);
        Self { kahe, cs, shamir, min_clients: 1 }
    }
}

/// One committed value per server: the Shamir share of the single KAHE key.
const MU_CS: usize = 1;

/// BDLOP randomness width. 5 is the exact minimum the statistical
/// addition-hiding bound admits at `MU_CS = 1`: 4 fails it outright, and
/// `MU_CS = 2` would need 8.
const KAPPA_CS: usize = 5;
