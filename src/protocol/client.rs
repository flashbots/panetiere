use rand::Rng;

use crate::bulletin::ClientPublic;
use crate::cs::{Cs, HidingMerkleCommitment, Opening};
use crate::kahe::{kahe_to_hvc_centered, Kahe, KaheScheme};
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

pub fn run_client_round<R: Rng>(
    rng: &mut R,
    pp: &ProtocolParams,
    client_id: ClientId,
    message: <Kahe as KaheScheme>::Message,
    servers: &[ServerId],
) -> ClientRound {
    let key = Kahe::gen(rng, &pp.kahe);
    let ctxt = Kahe::enc(rng, &pp.kahe, &key, &message);

    let n = servers.len();
    assert_eq!(n, pp.cs.n_servers, "server count must match CS params");
    assert_eq!(n, pp.shamir.n, "server count must match Shamir params");
    let kappa_kahe = pp.kahe.kappa_kahe;
    assert_eq!(
        kappa_kahe, pp.cs.mu_cs,
        "ProtocolParams must couple μ_cs = κ_kahe"
    );

    // Per-component Shamir shares: shares_per_component[k][i] = f_k(point_{i+1}).
    // Shamir lives in R_{q_cs}; KAHE keys live in R_{q_kahe}. Bridge each
    // component (small Gaussian) into HVC via centered-rep re-interpretation.
    let shares_per_component: Vec<Vec<chipmunk_code::HVCPoly>> = (0..kappa_kahe)
        .map(|k| {
            let secret_hvc = kahe_to_hvc_centered(key.component(k));
            ShamirSharing::share(rng, &pp.shamir, &secret_hvc)
        })
        .collect();
    // Transpose into per-server share vectors (length κ_kahe each).
    let shares_per_server: Vec<Vec<chipmunk_code::HVCPoly>> = (0..n)
        .map(|i| {
            (0..kappa_kahe)
                .map(|k| shares_per_component[k][i])
                .collect()
        })
        .collect();

    let (comm, openings) = HidingMerkleCommitment::commit(rng, &pp.cs, &shares_per_server);

    let private: Vec<(ServerId, Opening)> =
        servers.iter().copied().zip(openings.into_iter()).collect();

    ClientRound {
        client_id,
        public: ClientPublic { ctxt, comm },
        private,
    }
}
