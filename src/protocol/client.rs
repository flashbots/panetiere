use rand::{CryptoRng, Rng};

use chipmunk_code::CsPoly;

use crate::bulletin::ClientBulletinEntry;
use crate::cs::{fresh_opening_pack_bounds, Commitment, Cs, HidingMerkleCommitment, Opening};
use crate::kahe::{kahe_to_cs_centered, Kahe, KaheKey, KaheScheme};
use crate::pke;
use crate::sss::ShamirSharing;

use super::{ClientId, ServerId};
use super::ProtocolParams;

/// Output of a single client round (README steps 1–6).
///
/// One CS commit per client (with `μ_cs = κ_kahe`), so each server's
/// sealed opening wraps a single `Opening` — its `s()` is the
/// `κ_kahe`-vector of Shamir shares for that server.
pub struct ClientRound {
    pub client_id: ClientId,
    pub encrypted_message: ClientBulletinEntry,
    /// Per-server ECIES envelope over the bit-packed `Opening`.
    pub sealed_openings: Vec<(ServerId, Vec<u8>)>,
}

/// Pack (fresh bounds) + ECIES-seal one per-server opening.
pub fn seal_opening<R: CryptoRng + Rng>(
    rng: &mut R,
    pp: &ProtocolParams,
    opening: &Opening,
    recipient: &pke::PublicKey,
) -> Vec<u8> {
    let (r_b, s_b, t_b) = fresh_opening_pack_bounds(&pp.cs);
    pke::encrypt(rng, recipient, &opening.pack(r_b, s_b, t_b).to_bytes())
}

/// Sample a fresh KAHE secret key.
pub fn kahe_keygen<R: Rng>(rng: &mut R, pp: &ProtocolParams) -> KaheKey {
    Kahe::gen(rng, &pp.kahe)
}

/// Encrypt `message` under `key`.
pub fn kahe_encrypt<R: Rng>(
    rng: &mut R,
    pp: &ProtocolParams,
    key: &KaheKey,
    message: &<Kahe as KaheScheme>::Message,
) -> <Kahe as KaheScheme>::Ciphertext {
    Kahe::enc(rng, &pp.kahe, key, message)
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
) -> Vec<Vec<CsPoly>> {
    assert_eq!(n_servers, pp.cs.n_servers, "server count must match CS params");
    assert_eq!(n_servers, pp.shamir.n, "server count must match Shamir params");
    let kappa_kahe = pp.kahe.kappa_kahe;
    assert_eq!(
        kappa_kahe, pp.cs.mu_cs,
        "ProtocolParams must couple μ_cs = κ_kahe"
    );

    // shares_per_component[k][i] = f_k(point_{i+1}).
    let shares_per_component: Vec<Vec<CsPoly>> = (0..kappa_kahe)
        .map(|k| {
            let secret_cs = kahe_to_cs_centered(key.component(k));
            ShamirSharing::share(rng, &pp.shamir, &secret_cs)
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
    shares_per_server: &[Vec<CsPoly>],
) -> (Commitment, Vec<Opening>) {
    HidingMerkleCommitment::commit(rng, &pp.cs, shares_per_server)
}

pub fn run_client_round<R: CryptoRng + Rng>(
    rng: &mut R,
    pp: &ProtocolParams,
    client_id: ClientId,
    message: <Kahe as KaheScheme>::Message,
    servers: &[(ServerId, pke::PublicKey)],
) -> ClientRound {
    let key = kahe_keygen(rng, pp);
    let ctxt = kahe_encrypt(rng, pp, &key, &message);
    let shares_per_server = shamir_share(rng, pp, &key, servers.len());
    let (comm, openings) = cs_commit(rng, pp, &shares_per_server);

    let sealed_openings = servers
        .iter()
        .zip(openings.iter())
        .map(|((sid, xpub), opening)| (*sid, seal_opening(rng, pp, opening, xpub)))
        .collect();

    ClientRound {
        client_id,
        encrypted_message: ClientBulletinEntry { ctxt, comm },
        sealed_openings,
    }
}
