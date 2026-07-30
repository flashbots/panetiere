use rand::{CryptoRng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use chipmunk_code::CsPoly;

use crate::bulletin::{ClientBulletinEntry, RsClientBulletinEntry};
use crate::cs::{fresh_opening_pack_bounds, Commitment, Cs, HidingMerkleCommitment, Opening};
use crate::digest::{digest, embed};
use crate::kahe::{kahe_to_cs_centered, Kahe, KaheKey, KaheScheme};
use crate::pke;
use crate::rs::{Rs, Share};
use crate::sig::SigningKey;
use crate::sss::ShamirSharing;

use super::ProtocolParams;
use super::{opening_aad, ClientId, ServerId, SessionId};

/// Pack one per-server opening, bound to `(sid, client_id, server_id)`.
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

/// Batch form of [`seal_opening`]
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
                seal_opening(
                    &mut item_rng,
                    pp,
                    sid,
                    client_id,
                    *sid_server,
                    opening,
                    xpub,
                ),
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

/// Bridge the KAHE key into R_{q_cs} via centered-rep and Shamir-share it. One
/// share per server, `pp.shamir.n` of them.
pub fn shamir_share<R: Rng>(rng: &mut R, pp: &ProtocolParams, key: &KaheKey) -> Vec<CsPoly> {
    ShamirSharing::share(rng, &pp.shamir, &kahe_to_cs_centered(key.inner()))
}

/// Commit to the per-server shares.
/// Returns the public commitment and the per-server openings.
pub fn cs_commit<R: Rng>(
    rng: &mut R,
    pp: &ProtocolParams,
    shares_per_server: &[CsPoly],
) -> (Commitment, Vec<Opening>) {
    let as_vectors: Vec<Vec<CsPoly>> = shares_per_server.iter().map(|s| vec![*s]).collect();
    HidingMerkleCommitment::commit(rng, &pp.cs, &as_vectors)
}

/// Output of a single client round
pub struct ClientRound {
    pub client_id: ClientId,
    pub encrypted_message: ClientBulletinEntry,
    pub sealed_openings: Vec<(ServerId, Vec<u8>)>,
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
    let shares_per_server = shamir_share(rng, pp, &key);
    let (comm, openings) = cs_commit(rng, pp, &shares_per_server);

    let sealed_openings = seal_openings(rng, pp, sid, client_id, &openings, servers);

    ClientRound {
        client_id,
        encrypted_message: ClientBulletinEntry { ctxt, comm },
        sealed_openings,
    }
}

/// Output of one RS-mode client round.
pub struct RsClientRound {
    pub client_id: ClientId,
    pub bulletin: RsClientBulletinEntry,
    pub sealed_openings: Vec<(ServerId, Vec<u8>)>,
    pub rs_shares: Vec<Share>,
}

/// RS-sharded ingress round.
pub fn run_client_round_rs<R: CryptoRng + Rng>(
    rng: &mut R,
    pp: &ProtocolParams,
    sid: &SessionId,
    client_id: ClientId,
    message: <Kahe as KaheScheme>::Message,
    servers: &[(ServerId, pke::PublicKey)],
    signing_key: &SigningKey,
) -> RsClientRound {
    let rs = pp.rs.as_ref().expect("RS mode params");
    let dp = pp.digest.as_ref().expect("digest params");

    let key = kahe_keygen(rng, pp);
    let ctxt = kahe_encrypt(rng, pp, &key, &message);
    let embedded = embed(&ctxt);
    let h = digest(dp, &embedded);
    let rs_shares = Rs::encode(rs, &embedded);

    let shares_per_server = shamir_share(rng, pp, &key);
    let (comm, openings) = cs_commit(rng, pp, &shares_per_server);
    let sealed_openings = seal_openings(rng, pp, sid, client_id, &openings, servers);

    let sig = signing_key.sign(&RsClientBulletinEntry::signing_bytes(
        sid, client_id, &comm, &h,
    ));

    RsClientRound {
        client_id,
        bulletin: RsClientBulletinEntry {
            comm,
            digest: h,
            pubkey: signing_key.verifying_key().to_sec1_bytes(),
            sig,
        },
        sealed_openings,
        rs_shares,
    }
}
