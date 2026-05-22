use rand::Rng;

use chipmunk_code::HVCPoly;

use crate::bulletin::ClientPublic;
use crate::cs::{Commitment, Cs, HidingMerkleCommitment, Opening};
use crate::kahe::{kahe_to_hvc_centered, Kahe, KaheKey, KaheScheme};
use crate::sss::ShamirSharing;

use super::message::{ClientId, ServerId};
use super::ProtocolParams;

/// Output of a single client round (README steps 1–6).
///
/// One CS commit per client (with `μ_cs = κ_kahe`), so each per-server
/// private payload is a single `Opening` — its `s()` is the `κ_kahe`-vector
/// of Shamir shares for that server.
pub struct ClientRound {
    pub client_id: ClientId,
    pub public: ClientPublic,
    pub private: Vec<(ServerId, Opening)>,
}

/// Phase 1 — KAHE keygen + encrypt of the client's message.
/// Returns the ciphertext (public) and the secret key (consumed by phase 2).
pub fn kahe_encrypt<R: Rng>(
    rng: &mut R,
    pp: &ProtocolParams,
    message: &<Kahe as KaheScheme>::Message,
) -> (<Kahe as KaheScheme>::Ciphertext, KaheKey) {
    let key = Kahe::gen(rng, &pp.kahe);
    let ctxt = Kahe::enc(rng, &pp.kahe, &key, message);
    (ctxt, key)
}

/// Phase 2 — bridge each KAHE-key component into R_{q_cs} via centered-rep
/// re-interpretation, Shamir-share each component across `n_servers`, then
/// transpose into per-server share vectors of length `κ_kahe`.
///
/// Asserts the structural couplings (`n_servers == pp.cs.n_servers == pp.shamir.n`
/// and `μ_cs == κ_kahe`) — these are invariants of `ProtocolParams` setup but
/// re-checked here so a misuse of this phase fails loudly.
pub fn shamir_share<R: Rng>(
    rng: &mut R,
    pp: &ProtocolParams,
    key: &KaheKey,
    n_servers: usize,
) -> Vec<Vec<HVCPoly>> {
    assert_eq!(n_servers, pp.cs.n_servers, "server count must match CS params");
    assert_eq!(n_servers, pp.shamir.n, "server count must match Shamir params");
    let kappa_kahe = pp.kahe.kappa_kahe;
    assert_eq!(
        kappa_kahe, pp.cs.mu_cs,
        "ProtocolParams must couple μ_cs = κ_kahe"
    );

    // shares_per_component[k][i] = f_k(point_{i+1}).
    let shares_per_component: Vec<Vec<HVCPoly>> = (0..kappa_kahe)
        .map(|k| {
            let secret_hvc = kahe_to_hvc_centered(key.component(k));
            ShamirSharing::share(rng, &pp.shamir, &secret_hvc)
        })
        .collect();
    (0..n_servers)
        .map(|i| (0..kappa_kahe).map(|k| shares_per_component[k][i]).collect())
        .collect()
}

/// Phase 3 — CS commit to the per-server share matrix. Returns the public
/// commitment and the per-server openings.
pub fn cs_commit<R: Rng>(
    rng: &mut R,
    pp: &ProtocolParams,
    shares_per_server: &[Vec<HVCPoly>],
) -> (Commitment, Vec<Opening>) {
    HidingMerkleCommitment::commit(rng, &pp.cs, shares_per_server)
}

pub fn run_client_round<R: Rng>(
    rng: &mut R,
    pp: &ProtocolParams,
    client_id: ClientId,
    message: <Kahe as KaheScheme>::Message,
    servers: &[ServerId],
) -> ClientRound {
    let (ctxt, key) = kahe_encrypt(rng, pp, &message);
    let shares_per_server = shamir_share(rng, pp, &key, servers.len());
    let (comm, openings) = cs_commit(rng, pp, &shares_per_server);

    let private: Vec<(ServerId, Opening)> =
        servers.iter().copied().zip(openings.into_iter()).collect();

    ClientRound {
        client_id,
        public: ClientPublic { ctxt, comm },
        private,
    }
}
