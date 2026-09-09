use rand::{CryptoRng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use crate::{CsPoly, KaheNTTPoly, KahePoly};

use super::ProtocolParams;
use super::{opening_aad, ClientId, ServerId, SessionId};
use crate::bulletin::{ClientBulletinEntry, RsClientBulletinEntry};
use crate::cs::{fresh_opening_pack_bounds, Commitment, Cs, HidingMerkleCommitment, Opening};
use crate::kahe::{kahe_to_cs_centered, Kahe, KaheKey, KaheScheme};
use crate::pke;
use crate::rs::{Rs, Share};
use crate::share_commitment::{commit_shares, SharePath};
use crate::sig::SigningKey;
use crate::sss::ShamirSharing;

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
/// The protocol assumes clients execute this honestly: the share commitment
/// does not prove degree consistency.
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

pub struct ClientRound {
    client_id: ClientId,
    encrypted_message: ClientBulletinEntry,
    sealed_openings: Vec<(ServerId, Vec<u8>)>,
}

impl ClientRound {
    pub fn new<R: CryptoRng + Rng>(
        rng: &mut R,
        pp: &ProtocolParams,
        sid: &SessionId,
        client_id: ClientId,
        servers: &[(ServerId, pke::PublicKey)],
    ) -> Self {
        let key = kahe_keygen(rng, pp);
        let ctxt = kahe_encrypt(rng, pp, &key, &super::zero_message(pp));
        let shares_per_server = shamir_share(rng, pp, &key);
        let (comm, openings) = cs_commit(rng, pp, &shares_per_server);
        let sealed_openings = seal_openings(rng, pp, sid, client_id, &openings, servers);
        Self {
            client_id,
            encrypted_message: ClientBulletinEntry { ctxt, comm },
            sealed_openings,
        }
    }

    pub fn finalize(mut self, message: &[KahePoly]) -> ClientRoundOutput {
        assert_eq!(
            self.encrypted_message.ctxt.len(),
            message.len(),
            "zero round/message width mismatch",
        );
        self.encrypted_message
            .ctxt
            .par_iter_mut()
            .zip(message.par_iter())
            .for_each(|(ciphertext, message)| *ciphertext += *message);
        ClientRoundOutput {
            client_id: self.client_id,
            encrypted_message: self.encrypted_message,
            sealed_openings: self.sealed_openings,
        }
    }
}

/// Output of a single client round.
pub struct ClientRoundOutput {
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
) -> ClientRoundOutput {
    ClientRound::new(rng, pp, sid, client_id, servers).finalize(&message)
}

pub struct RsClientRound {
    client_id: ClientId,
    comm: Commitment,
    sealed_openings: Vec<(ServerId, Vec<u8>)>,
    rs_shares: Vec<Share>,
}

impl RsClientRound {
    pub fn new<R: CryptoRng + Rng>(
        rng: &mut R,
        pp: &ProtocolParams,
        sid: &SessionId,
        client_id: ClientId,
        servers: &[(ServerId, pke::PublicKey)],
    ) -> Self {
        let rs = pp.rs.as_ref().expect("RS mode params");
        let key = kahe_keygen(rng, pp);
        let embedded = Kahe::enc_ntt(rng, &pp.kahe, &key, &super::zero_message(pp));
        let rs_shares = Rs::encode(rs, &embedded);
        let shares_per_server = shamir_share(rng, pp, &key);
        let (comm, openings) = cs_commit(rng, pp, &shares_per_server);
        let sealed_openings = seal_openings(rng, pp, sid, client_id, &openings, servers);
        Self {
            client_id,
            comm,
            sealed_openings,
            rs_shares,
        }
    }

    pub fn finalize(
        mut self,
        pp: &ProtocolParams,
        sid: &SessionId,
        message: &[KahePoly],
        signing_key: &SigningKey,
    ) -> RsClientRoundOutput {
        let rs = pp.rs.as_ref().expect("RS mode params");
        let scp = pp.share_comm.as_ref().expect("share-commitment params");
        assert_eq!(
            pp.kahe.mu_kahe,
            message.len(),
            "zero round/message width mismatch",
        );
        let message_ntt: Vec<KaheNTTPoly> = message.par_iter().map(KaheNTTPoly::from).collect();
        let message_shares = Rs::encode(rs, &message_ntt);
        assert_eq!(self.rs_shares.len(), message_shares.len());
        self.rs_shares
            .par_iter_mut()
            .zip(message_shares.par_iter())
            .for_each(|(share, mask)| {
                assert_eq!(share.len(), mask.len());
                share
                    .iter_mut()
                    .zip(mask.iter())
                    .for_each(|(value, mask)| *value += *mask);
            });
        let (share_root, share_paths) = commit_shares(scp, &self.rs_shares);
        let sig = signing_key.sign(&RsClientBulletinEntry::signing_bytes(
            sid,
            self.client_id,
            &self.comm,
            &share_root,
        ));
        RsClientRoundOutput {
            client_id: self.client_id,
            bulletin: RsClientBulletinEntry {
                comm: self.comm,
                share_root,
                pubkey: signing_key.verifying_key().to_sec1_bytes(),
                sig,
            },
            sealed_openings: self.sealed_openings,
            rs_shares: self.rs_shares,
            share_paths,
        }
    }
}

/// Output of one RS-mode client round.
pub struct RsClientRoundOutput {
    pub client_id: ClientId,
    pub bulletin: RsClientBulletinEntry,
    pub sealed_openings: Vec<(ServerId, Vec<u8>)>,
    pub rs_shares: Vec<Share>,
    /// Index-aligned with `rs_shares`: lane `j` receives `(rs_shares[j], share_paths[j])`.
    pub share_paths: Vec<SharePath>,
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
) -> RsClientRoundOutput {
    RsClientRound::new(rng, pp, sid, client_id, servers).finalize(pp, sid, &message, signing_key)
}
