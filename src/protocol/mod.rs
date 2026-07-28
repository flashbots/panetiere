pub mod aggregator;
pub mod client;
pub mod dispute;
pub mod recipient;
pub mod server;
pub mod verify;

use rand::Rng;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct ClientId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct ServerId(pub u32);

/// An RS lane. The first `n_servers` nodes are the threshold servers (they also
/// hold an opening); any beyond that carry a coded share and nothing else,
/// which is what lets `n` exceed `S` and `k` grow without paying more openings.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct NodeId(pub u32);

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
use crate::digest::DigestParams;
use crate::kahe::{Kahe, KaheParams, KaheScheme, SIGMA_E_DEFAULT, SIGMA_S_DEFAULT, T_MODULUS_DEFAULT};
use crate::rs::RsParams;
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
    /// Set only in the RS-sharded ingress mode; `None` is the broadcast flow.
    pub rs: Option<RsParams>,
    pub digest: Option<DigestParams>,
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
        Self { kahe, cs, shamir, min_clients: 1, rs: None, digest: None }
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
        Self { kahe, cs, shamir, min_clients: 1, rs: None, digest: None }
    }
}

impl ProtocolParams {
    /// RS-sharded ingress mode. `payload_polys` is the plaintext width, and
    /// also `μ_kahe` — the digest is an Ajtai hash of the *ciphertext*, so it
    /// needs no plaintext slot and no encryption.
    ///
    /// `n_nodes` may exceed `n_servers`: the extra nodes are lanes that hold a
    /// coded share and no key material. `rho_max` bounds the aggregation the
    /// digest ring can represent exactly, and is what the verifier range-checks.
    pub fn setup_rs_mode<R: Rng>(
        rng: &mut R,
        n_servers: usize,
        payload_polys: usize,
        k: usize,
        n_nodes: usize,
        t_modulus: u64,
        rho_max: usize,
        digest_seed: [u8; 32],
    ) -> Self {
        assert!(payload_polys >= 1, "payload must be ≥ 1 poly");
        assert!(
            n_nodes >= n_servers,
            "every threshold server is also a lane; n_nodes ≥ n_servers"
        );
        let t = (n_servers / 2 + 1).max(n_servers.saturating_sub(2));
        let kahe = Kahe::setup_with_dims(
            rng,
            payload_polys,
            1,
            SIGMA_S_DEFAULT,
            SIGMA_E_DEFAULT,
            t_modulus,
        );
        Self {
            kahe,
            cs: HidingMerkleCommitment::setup_with_dims(rng, n_servers, MU_CS, KAPPA_CS),
            shamir: ShamirParams::new(t, n_servers),
            min_clients: 1,
            rs: Some(RsParams::new(k, n_nodes)),
            digest: Some(DigestParams::from_seed(digest_seed, payload_polys, rho_max)),
        }
    }
}

/// Exact crypto bytes each role puts on the wire in one round. Framing and
/// serialisation overhead are the transport's business and are not counted.
pub struct RoundWireSizes {
    /// One client's bulletin post: commitment plus its ciphertext.
    pub client_post: usize,
    /// One server's aggregated opening plus its share.
    pub server_entry: usize,
}

/// Sizes without sampling a CRS — the lengths depend only on the server count,
/// the ciphertext width and `ρ`.
pub fn round_wire_sizes(n_servers: usize, ctxt_polys: usize, rho: u32) -> RoundWireSizes {
    RoundWireSizes {
        client_post: crate::bulletin::ClientBulletinEntry::packed_len(ctxt_polys),
        server_entry: crate::cs::aggregated_server_crypto_len_for(
            n_servers, MU_CS, KAPPA_CS, rho,
        ),
    }
}

/// One committed value per server: the Shamir share of the single KAHE key.
const MU_CS: usize = 1;

/// BDLOP randomness width. 5 is the exact minimum the statistical
/// addition-hiding bound admits at `MU_CS = 1`: 4 fails it outright, and
/// `MU_CS = 2` would need 8.
const KAPPA_CS: usize = 5;
