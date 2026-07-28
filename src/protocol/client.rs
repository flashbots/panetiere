use rand::{CryptoRng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use chipmunk_code::CsPoly;

use crate::bulletin::ClientBulletinEntry;
use crate::cs::{fresh_opening_pack_bounds, Commitment, Cs, HidingMerkleCommitment, Opening};
use crate::kahe::{kahe_to_cs_centered, Kahe, KaheKey, KaheScheme};
use crate::pke;
use crate::sss::ShamirSharing;

use super::ProtocolParams;
use super::{opening_aad, ClientId, ServerId, SessionId};

/// Output of a single client round (README steps 1–6).
///
/// One CS commit per client, so each server's sealed opening wraps a single
/// `Opening` — its `s()` is that server's Shamir share of the KAHE key.
pub struct ClientRound {
    pub client_id: ClientId,
    pub encrypted_message: ClientBulletinEntry,
    /// Per-server ECIES envelope over the bit-packed `Opening`.
    pub sealed_openings: Vec<(ServerId, Vec<u8>)>,
}

/// Pack (fresh bounds) + ECIES-seal one per-server opening, bound to
/// `(sid, client_id, server_id)`.
pub fn seal_opening<R: CryptoRng + Rng>(
    rng: &mut R,
    pp: &ProtocolParams,
    sid: &SessionId,
    client_id: ClientId,
    server_id: ServerId,
    opening: &Opening,
    recipient: &pke::PublicKey,
) -> Vec<u8> {
    let (r_b, s_b, t_b) = fresh_opening_pack_bounds(&pp.cs);
    pke::encrypt(
        rng,
        recipient,
        &opening.pack(r_b, s_b, t_b).to_bytes(),
        &opening_aad(sid, client_id, server_id),
    )
}

/// Batch form of [`seal_opening`] over one client's per-server openings.
/// Independent per server; forked seeds keep the result deterministic
/// regardless of thread schedule.
pub fn seal_openings<R: CryptoRng + Rng>(
    rng: &mut R,
    pp: &ProtocolParams,
    sid: &SessionId,
    client_id: ClientId,
    openings: &[Opening],
    servers: &[(ServerId, pke::PublicKey)],
) -> Vec<(ServerId, Vec<u8>)> {
    let seeds = crate::fork_seeds(rng, servers.len());
    servers
        .par_iter()
        .zip(openings.par_iter())
        .zip(seeds.par_iter())
        .map(|(((sid_server, xpub), opening), seed)| {
            let mut item_rng = ChaCha20Rng::from_seed(*seed);
            (
                *sid_server,
                seal_opening(&mut item_rng, pp, sid, client_id, *sid_server, opening, xpub),
            )
        })
        .collect()
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

/// Phase 2 — bridge the KAHE key into R_{q_cs} via centered-rep
/// re-interpretation and Shamir-share it across `n_servers`. `out[i]` is
/// server `i`'s share.
///
/// Asserts the structural couplings (`n_servers == pp.cs.n_servers == pp.shamir.n`
/// and `μ_cs == 1`) — these are invariants of `ProtocolParams` setup but
/// re-checked here so a misuse of this phase fails loudly.
pub fn shamir_share<R: Rng>(
    rng: &mut R,
    pp: &ProtocolParams,
    key: &KaheKey,
    n_servers: usize,
) -> Vec<CsPoly> {
    assert_eq!(n_servers, pp.cs.n_servers, "server count must match CS params");
    assert_eq!(n_servers, pp.shamir.n, "server count must match Shamir params");
    assert_eq!(pp.cs.mu_cs, 1, "one committed share per server");

    ShamirSharing::share(rng, &pp.shamir, &kahe_to_cs_centered(key.inner()))
}

/// Phase 3 — CS commit to the per-server shares. The commitment scheme takes a
/// `μ_cs`-vector per position, which the protocol instantiates at `μ_cs = 1`.
/// Returns the public commitment and the per-server openings.
pub fn cs_commit<R: Rng>(
    rng: &mut R,
    pp: &ProtocolParams,
    shares_per_server: &[CsPoly],
) -> (Commitment, Vec<Opening>) {
    let as_vectors: Vec<Vec<CsPoly>> = shares_per_server.iter().map(|s| vec![*s]).collect();
    HidingMerkleCommitment::commit(rng, &pp.cs, &as_vectors)
}

pub fn run_client_round<R: CryptoRng + Rng>(
    rng: &mut R,
    pp: &ProtocolParams,
    sid: &SessionId,
    client_id: ClientId,
    message: <Kahe as KaheScheme>::Message,
    servers: &[(ServerId, pke::PublicKey)],
) -> ClientRound {
    let key = kahe_keygen(rng, pp);
    let ctxt = kahe_encrypt(rng, pp, &key, &message);
    let shares_per_server = shamir_share(rng, pp, &key, servers.len());
    let (comm, openings) = cs_commit(rng, pp, &shares_per_server);

    let sealed_openings = seal_openings(rng, pp, sid, client_id, &openings, servers);

    ClientRound {
        client_id,
        encrypted_message: ClientBulletinEntry { ctxt, comm },
        sealed_openings,
    }
}
