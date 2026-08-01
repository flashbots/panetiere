pub mod aggregator;
pub mod client;
pub mod recipient;
pub mod server;
pub mod verify;

use chipmunk_code::KahePoly;
use rand::Rng;

use crate::cs::{Cs, HidingMerkleCommitment};
use crate::kahe::{
    Kahe, KaheParams, KaheScheme, SIGMA_E_DEFAULT, SIGMA_S_DEFAULT, T_MODULUS_DEFAULT,
};
use crate::rs::RsParams;
use crate::share_commitment::ShareCommitmentParams;
use crate::sss::ShamirParams;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct ClientId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct ServerId(pub u32);

/// Erasure coding lane node id
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct NodeId(pub u32);

/// Client's unique per-round session id to prevent replays
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SessionId(pub [u8; 32]);

/// AEAD associated data for one sealed opening.
pub fn opening_aad(sid: &SessionId, client: ClientId, server: ServerId) -> Vec<u8> {
    const DOMAIN: &[u8] = b"panetiere/opening/v1";
    let mut aad = Vec::with_capacity(DOMAIN.len() + 32 + 8);
    aad.extend_from_slice(DOMAIN);
    aad.extend_from_slice(&sid.0);
    aad.extend_from_slice(&client.0.to_le_bytes());
    aad.extend_from_slice(&server.0.to_le_bytes());
    aad
}

/// `KahePoly` count of one KAHE plaintext message
pub fn message_polys(pp: &ProtocolParams) -> usize {
    pp.kahe.mu_kahe
}

/// All-zero KAHE message (cover)
pub fn zero_message(pp: &ProtocolParams) -> Vec<KahePoly> {
    vec![KahePoly::default(); message_polys(pp)]
}

/// Bundle of public parameters carried through one protocol session.
pub struct ProtocolParams {
    pub kahe: KaheParams,
    pub cs: <HidingMerkleCommitment as Cs>::Params,
    pub shamir: ShamirParams,
    pub min_clients: usize,
    /// Set only in the RS-sharded ingress mode; `None` is the broadcast flow.
    pub rs: Option<RsParams>,
    pub share_comm: Option<ShareCommitmentParams>,
}

impl ProtocolParams {
    /// Build params with `n_servers` servers and threshold `t = max(⌊n/2⌋ + 1, n − 2)`.
    pub fn setup<R: Rng>(rng: &mut R, n_servers: usize) -> Self {
        let t = (n_servers / 2 + 1).max(n_servers.saturating_sub(2));
        Self::setup_with_threshold(rng, n_servers, t)
    }

    pub fn setup_with_threshold<R: Rng>(rng: &mut R, n_servers: usize, t: usize) -> Self {
        let kahe = Kahe::setup(rng);
        let cs = HidingMerkleCommitment::setup_with_dims(rng, n_servers, MU_CS, KAPPA_CS);
        let shamir = ShamirParams::new(t, n_servers);
        Self {
            kahe,
            cs,
            shamir,
            min_clients: 1,
            rs: None,
            share_comm: None,
        }
    }

    pub fn setup_with_kahe_dims<R: Rng>(rng: &mut R, n_servers: usize, mu_kahe: usize) -> Self {
        Self::setup_with_kahe_dims_full(
            rng,
            n_servers,
            mu_kahe,
            SIGMA_S_DEFAULT,
            SIGMA_E_DEFAULT,
            T_MODULUS_DEFAULT,
        )
    }

    pub fn setup_with_kahe_dims_full<R: Rng>(
        rng: &mut R,
        n_servers: usize,
        mu_kahe: usize,
        sigma_s: f64,
        sigma_e: f64,
        t_modulus: u64,
    ) -> Self {
        let t = (n_servers / 2 + 1).max(n_servers.saturating_sub(2));
        let kahe = Kahe::setup_with_dims(rng, mu_kahe, sigma_s, sigma_e, t_modulus);
        let cs = HidingMerkleCommitment::setup_with_dims(rng, n_servers, MU_CS, KAPPA_CS);
        let shamir = ShamirParams::new(t, n_servers);
        Self {
            kahe,
            cs,
            shamir,
            min_clients: 1,
            rs: None,
            share_comm: None,
        }
    }
}

impl ProtocolParams {
    pub fn setup_rs_mode<R: Rng>(
        rng: &mut R,
        n_servers: usize,
        payload_polys: usize,
        k: usize,
        n_nodes: usize,
        t_modulus: u64,
        rho_max: usize,
        crs_seed: [u8; 32],
    ) -> Self {
        assert!(payload_polys >= 1, "payload must be ≥ 1 poly");
        assert!(
            n_nodes >= n_servers,
            "every threshold server is also a lane; n_nodes ≥ n_servers"
        );
        let t = (n_servers / 2 + 1).max(n_servers.saturating_sub(2));
        let kahe =
            Kahe::setup_with_dims(rng, payload_polys, SIGMA_S_DEFAULT, SIGMA_E_DEFAULT, t_modulus);
        let rs = RsParams::new(k, n_nodes);
        let block_len = rs.block_len(payload_polys);
        Self {
            kahe,
            cs: HidingMerkleCommitment::setup_with_dims(rng, n_servers, MU_CS, KAPPA_CS),
            shamir: ShamirParams::new(t, n_servers),
            min_clients: 1,
            rs: Some(rs),
            share_comm: Some(ShareCommitmentParams::from_seed(
                crs_seed, block_len, n_nodes, rho_max,
            )),
        }
    }
}

/// Exact crypto bytes each role puts on the wire in one round.
pub struct RoundWireSizes {
    /// One client's bulletin post: commitment plus its ciphertext.
    pub client_post: usize,
    /// One server's aggregated opening plus its share.
    pub server_entry: usize,
}

/// Sizes without sampling a CRS
pub fn round_wire_sizes(n_servers: usize, ctxt_polys: usize, rho: u32) -> RoundWireSizes {
    RoundWireSizes {
        client_post: crate::bulletin::ClientBulletinEntry::packed_len(ctxt_polys),
        server_entry: crate::cs::aggregated_server_crypto_len_for(n_servers, MU_CS, KAPPA_CS, rho),
    }
}

/// One committed value per server: the Shamir share of the single KAHE key
const MU_CS: usize = 1;

/// BDLOP randomness width
const KAPPA_CS: usize = 5;
