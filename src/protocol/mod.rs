pub mod client;
pub mod recipient;
pub mod server;
pub mod verify;

use crate::KahePoly;
use rand::Rng;

use crate::cs::{Cs, HidingMerkleCommitment, KAPPA_CS, MU_CS};
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

/// The Shamir threshold `⌊γ/2⌋ + 1` for `γ = n_servers`, under which both
/// anonymity and guaranteed output delivery hold as long as the honest servers
/// are a majority.
///
/// A threshold `t` gives anonymity against up to `t − 1` corrupt servers and
/// guaranteed output delivery against up to `min(t − 1, γ − t)`. For
/// `t = ⌊γ/2⌋ + 1` these are `⌊γ/2⌋` and `⌈γ/2⌉ − 1`, i.e. every corrupt
/// minority. This is the threshold `t_s` of the paper. A larger `t` trades output
/// delivery for anonymity; see [`ProtocolParams::with_threshold`].
pub fn honest_majority_threshold(n_servers: usize) -> usize {
    n_servers / 2 + 1
}

/// Bundle of public parameters carried through one protocol session.
pub struct ProtocolParams {
    pub kahe: KaheParams,
    pub cs: <HidingMerkleCommitment as Cs>::Params,
    pub shamir: ShamirParams,
    pub min_clients: usize,
    /// Aggregate capacity derived at setup from commitment and KAHE bounds.
    pub max_clients: usize,
    /// Set only in the RS-sharded ingress mode; `None` is the broadcast flow.
    pub rs: Option<RsParams>,
    pub share_comm: Option<ShareCommitmentParams>,
}

impl ProtocolParams {
    /// Build params with `n_servers` servers and the honest-majority threshold
    /// ([`honest_majority_threshold`]).
    pub fn setup<R: Rng>(rng: &mut R, n_servers: usize) -> Self {
        Self::setup_with_threshold(rng, n_servers, honest_majority_threshold(n_servers))
    }

    pub fn setup_with_threshold<R: Rng>(rng: &mut R, n_servers: usize, t: usize) -> Self {
        let kahe = Kahe::setup(rng);
        let cs = HidingMerkleCommitment::setup_with_dims(rng, n_servers, MU_CS, KAPPA_CS);
        let shamir = ShamirParams::new(t, n_servers);
        Self {
            max_clients: aggregate_capacity(&kahe, &cs),
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
        let t = honest_majority_threshold(n_servers);
        let kahe = Kahe::setup_with_dims(rng, mu_kahe, sigma_s, sigma_e, t_modulus);
        let cs = HidingMerkleCommitment::setup_with_dims(rng, n_servers, MU_CS, KAPPA_CS);
        let shamir = ShamirParams::new(t, n_servers);
        Self {
            max_clients: aggregate_capacity(&kahe, &cs),
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
    /// Replace the Shamir threshold, keeping everything else. Every `setup*`
    /// uses [`honest_majority_threshold`]; this sets any other `1 ≤ t ≤ n_servers`.
    pub fn with_threshold(mut self, t: usize) -> Self {
        self.shamir = ShamirParams::new(t, self.shamir.n);
        self
    }

    #[allow(clippy::too_many_arguments)]
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
        let t = honest_majority_threshold(n_servers);
        let kahe = Kahe::setup_with_dims(
            rng,
            payload_polys,
            SIGMA_S_DEFAULT,
            SIGMA_E_DEFAULT,
            t_modulus,
        );
        let rs = RsParams::new(k, n_nodes);
        let block_len = rs.block_len(payload_polys);
        let cs = HidingMerkleCommitment::setup_with_dims(rng, n_servers, MU_CS, KAPPA_CS);
        Self {
            max_clients: aggregate_capacity(&kahe, &cs).min(rho_max),
            kahe,
            cs,
            shamir: ShamirParams::new(t, n_servers),
            min_clients: 1,
            rs: Some(rs),
            share_comm: Some(ShareCommitmentParams::from_seed(
                crs_seed, block_len, n_nodes, rho_max,
            )),
        }
    }
}

fn aggregate_capacity(kahe: &KaheParams, cs: &crate::cs::CsParams) -> usize {
    let commitment_cap =
        (cs.r_bound / cs.beta_cs).min(cs.beta_agg_hvc / chipmunk_code::ZETA) as usize;
    let t = kahe.t_modulus as f64;
    // Eight-sigma aggregate noise plus the worst-case centered message sum.
    (1..=commitment_cap)
        .take_while(|&rho| {
            let rho = rho as f64;
            t * (8.0 * kahe.sigma_e * rho.sqrt() + rho / 2.0)
                < crate::rings::KAHE_MODULUS as f64 / 2.0
        })
        .last()
        .unwrap_or(0)
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn every_setup_uses_the_honest_majority_threshold() {
        let mut rng = ChaCha20Rng::from_seed([93; 32]);
        for (gamma, t) in [
            (1, 1),
            (2, 2),
            (3, 2),
            (4, 3),
            (7, 4),
            (8, 5),
            (16, 9),
            (64, 33),
        ] {
            assert_eq!(honest_majority_threshold(gamma), t);
        }
        for gamma in [7, 8] {
            let t = honest_majority_threshold(gamma);
            assert_eq!(ProtocolParams::setup(&mut rng, gamma).shamir.t, t);
            let pp = ProtocolParams::setup_with_kahe_dims_full(
                &mut rng,
                gamma,
                1,
                SIGMA_S_DEFAULT,
                SIGMA_E_DEFAULT,
                T_MODULUS_DEFAULT,
            );
            assert_eq!(pp.shamir.t, t);
            let pp = ProtocolParams::setup_rs_mode(
                &mut rng,
                gamma,
                1,
                gamma - 2,
                gamma,
                T_MODULUS_DEFAULT,
                300,
                [94; 32],
            );
            assert_eq!(pp.shamir.t, t);
            let pp = pp.with_threshold(gamma - 1);
            assert_eq!((pp.shamir.t, pp.shamir.n), (gamma - 1, gamma));
        }
    }

    #[test]
    fn aggregate_capacity_tracks_parameter_bounds() {
        let mut rng = ChaCha20Rng::from_seed([91; 32]);
        let pp = ProtocolParams::setup(&mut rng, 4);
        assert_eq!(pp.max_clients, 300);
        let at_cap: Vec<_> = (0..300).map(ClientId).collect();
        assert_eq!(verify::check_client_count(&pp, &at_cap), Ok(()));
        let above_cap: Vec<_> = (0..301).map(ClientId).collect();
        assert_eq!(
            verify::check_client_count(&pp, &above_cap),
            Err(verify::VerifyError::TooManyClients { got: 301, max: 300 })
        );

        let pp = ProtocolParams::setup_with_kahe_dims_full(
            &mut rng,
            4,
            1,
            SIGMA_S_DEFAULT,
            SIGMA_E_DEFAULT,
            T_MODULUS_DEFAULT * 2,
        );
        assert!(pp.max_clients > 0 && pp.max_clients < 300);
        let budget = |rho: usize| {
            pp.kahe.t_modulus as f64
                * (8.0 * pp.kahe.sigma_e * (rho as f64).sqrt() + rho as f64 / 2.0)
        };
        let half_q = crate::rings::KAHE_MODULUS as f64 / 2.0;
        assert!(budget(pp.max_clients) < half_q);
        assert!(budget(pp.max_clients + 1) >= half_q);

        for rho_max in [3, 400] {
            let pp = ProtocolParams::setup_rs_mode(
                &mut rng,
                4,
                1,
                3,
                4,
                T_MODULUS_DEFAULT,
                rho_max,
                [92; 32],
            );
            assert_eq!(pp.max_clients, rho_max.min(300));
        }
    }
}
