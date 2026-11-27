pub mod client;
pub mod recipient;
pub mod server;
pub mod verify;

pub use crate::cs::{Commitment, Opening};

use crate::KahePoly;
use rand::Rng;

use crate::cs::{Cs, HidingMerkleCommitment, KAPPA_CS, MU_CS};
use crate::kahe::{Kahe, KaheParams, KaheScheme, SIGMA};
use crate::rings::{PLAINTEXT_MODULUS, RHO_MAX};
use crate::sss::ShamirParams;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct ClientId(pub(crate) u32);

impl ClientId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn get(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct ServerId(pub(crate) u32);

impl ServerId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn get(self) -> u32 {
        self.0
    }
}

/// Client's unique per-round session id to prevent replays
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SessionId(pub(crate) [u8; 32]);

impl SessionId {
    pub fn new(id: [u8; 32]) -> Self {
        Self(id)
    }
}

/// AEAD associated data for one sealed opening.
pub(crate) fn opening_aad(sid: &SessionId, client: ClientId, server: ServerId) -> Vec<u8> {
    const DOMAIN: &[u8] = b"panetiere/opening/v1";
    let mut aad = Vec::with_capacity(DOMAIN.len() + 32 + 8);
    aad.extend_from_slice(DOMAIN);
    aad.extend_from_slice(&sid.0);
    aad.extend_from_slice(&client.0.to_le_bytes());
    aad.extend_from_slice(&server.0.to_le_bytes());
    aad
}

/// All-zero KAHE message (cover)
pub(crate) fn zero_message(pp: &ProtocolParams) -> Vec<KahePoly> {
    vec![KahePoly::default(); pp.message_polys()]
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
    pub(crate) kahe: KaheParams,
    pub(crate) cs: <HidingMerkleCommitment as Cs>::Params,
    pub(crate) shamir: ShamirParams,
    pub(crate) min_clients: usize,
    /// Aggregate capacity derived at setup from commitment and KAHE bounds.
    pub(crate) max_clients: usize,
}

impl ProtocolParams {
    /// Build params with `n_servers` servers and the honest-majority threshold
    /// ([`honest_majority_threshold`]).
    pub fn setup<R: Rng>(rng: &mut R, n_servers: usize) -> Self {
        Self::setup_with_threshold(rng, n_servers, honest_majority_threshold(n_servers))
    }

    fn setup_with_threshold<R: Rng>(rng: &mut R, n_servers: usize, t: usize) -> Self {
        let kahe = Kahe::setup(rng);
        let cs = HidingMerkleCommitment::setup_with_dims(rng, n_servers, MU_CS, KAPPA_CS);
        let shamir = ShamirParams::new(t, n_servers);
        Self {
            max_clients: aggregate_capacity(&kahe, &cs),
            kahe,
            cs,
            shamir,
            min_clients: 1,
        }
    }

    pub fn setup_with_kahe_dims<R: Rng>(rng: &mut R, n_servers: usize, mu_kahe: usize) -> Self {
        Self::setup_with_kahe_dims_full(rng, n_servers, mu_kahe, PLAINTEXT_MODULUS)
    }

    pub(crate) fn setup_with_kahe_dims_full<R: Rng>(
        rng: &mut R,
        n_servers: usize,
        mu_kahe: usize,
        t_modulus: u64,
    ) -> Self {
        let t = honest_majority_threshold(n_servers);
        let kahe = Kahe::setup_with_dims(rng, mu_kahe, t_modulus);
        let cs = HidingMerkleCommitment::setup_with_dims(rng, n_servers, MU_CS, KAPPA_CS);
        let shamir = ShamirParams::new(t, n_servers);
        Self {
            max_clients: aggregate_capacity(&kahe, &cs),
            kahe,
            cs,
            shamir,
            min_clients: 1,
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

    /// Replace the smallest client set a round decodes over (the anonymity
    /// floor). Every `setup*` uses 1.
    pub fn with_min_clients(mut self, min_clients: usize) -> Self {
        self.min_clients = min_clients;
        self
    }

    pub fn n_servers(&self) -> usize {
        self.shamir.n
    }

    /// The Shamir threshold `t`.
    pub fn threshold(&self) -> usize {
        self.shamir.t
    }

    /// Largest client set a round can decode, derived at setup from the
    /// commitment and KAHE noise bounds.
    pub fn max_clients(&self) -> usize {
        self.max_clients
    }

    /// `KahePoly` count of one KAHE plaintext message.
    pub fn message_polys(&self) -> usize {
        self.kahe.mu_kahe
    }

    /// The KAHE plaintext modulus `t`.
    pub fn plaintext_modulus(&self) -> u64 {
        self.kahe.t_modulus
    }

    /// The KAHE ciphertext modulus `q`.
    pub fn ciphertext_modulus(&self) -> u64 {
        crate::rings::KAHE_MODULUS as u64
    }

    /// The parameter `σ` of the discrete Gaussian distribution of the
    /// coefficients of KAHE keys and errors, with probabilities proportional
    /// to `exp(−x²/(2σ²))`.
    pub fn sigma(&self) -> f64 {
        SIGMA
    }
}

fn aggregate_capacity(kahe: &KaheParams, cs: &crate::cs::CsParams) -> usize {
    // The reference parameters are generated for at most `RHO_MAX` clients, and
    // the bridge of the summed keys back into KAHE's ring relies on it.
    let commitment_cap = ((cs.r_bound / cs.beta_cs).min(cs.beta_agg_hvc / chipmunk_code::ZETA)
        as usize)
        .min(RHO_MAX as usize);
    let t = kahe.t_modulus as f64;
    // Eight-sigma aggregate noise plus the worst-case centered message sum.
    (1..=commitment_cap)
        .take_while(|&rho| {
            let rho = rho as f64;
            t * (8.0 * SIGMA * rho.sqrt() + rho / 2.0) < crate::rings::KAHE_MODULUS as f64 / 2.0
        })
        .last()
        .unwrap_or(0)
}

/// Exact crypto bytes each role puts on the wire in one round.
pub struct RoundWireSizes {
    /// One client's bulletin post: commitment plus its ciphertext.
    client_post: usize,
    /// One server's aggregated opening plus its share.
    server_entry: usize,
}

impl RoundWireSizes {
    /// One client's bulletin post: commitment plus its ciphertext.
    pub fn client_post(&self) -> usize {
        self.client_post
    }

    /// One server's aggregated opening plus its share.
    pub fn server_entry(&self) -> usize {
        self.server_entry
    }
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
            let pp =
                ProtocolParams::setup_with_kahe_dims_full(&mut rng, gamma, 1, PLAINTEXT_MODULUS);
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

        let pp = ProtocolParams::setup_with_kahe_dims_full(&mut rng, 4, 1, PLAINTEXT_MODULUS * 2);
        assert!(pp.max_clients > 0 && pp.max_clients < 300);
        let budget = |rho: usize| {
            pp.kahe.t_modulus as f64 * (8.0 * SIGMA * (rho as f64).sqrt() + rho as f64 / 2.0)
        };
        let half_q = crate::rings::KAHE_MODULUS as f64 / 2.0;
        assert!(budget(pp.max_clients) < half_q);
        assert!(budget(pp.max_clients + 1) >= half_q);
    }
}
