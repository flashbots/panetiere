use rand::{CryptoRng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use crate::{CsPoly, KahePoly};

use super::ProtocolParams;
use super::{opening_aad, ClientId, ServerId, SessionId};
use crate::bulletin::ClientBulletinEntry;
use crate::cs::{fresh_opening_pack_bounds, Commitment, Cs, HidingMerkleCommitment, Opening};
use crate::kahe::{kahe_to_cs_centered, Kahe, KaheKey, KaheScheme};
use crate::pke;
use crate::sss::ShamirSharing;

/// Pack one per-server opening, bound to `(sid, client_id, server_id)`.
pub(crate) fn seal_opening<R: CryptoRng + Rng>(
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
pub(crate) fn seal_openings<R: CryptoRng + Rng>(
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
pub(crate) fn kahe_keygen<R: Rng>(rng: &mut R, pp: &ProtocolParams) -> KaheKey {
    Kahe::gen(rng, &pp.kahe)
}

/// Encrypt `message` under `key`.
pub(crate) fn kahe_encrypt<R: Rng>(
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
pub(crate) fn shamir_share<R: Rng>(rng: &mut R, pp: &ProtocolParams, key: &KaheKey) -> Vec<CsPoly> {
    ShamirSharing::share(rng, &pp.shamir, &kahe_to_cs_centered(key.inner()))
}

/// Commit to the per-server shares.
/// Returns the public commitment and the per-server openings.
pub(crate) fn cs_commit<R: Rng>(
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
    pub(crate) client_id: ClientId,
    pub(crate) encrypted_message: ClientBulletinEntry,
    pub(crate) sealed_openings: Vec<(ServerId, Vec<u8>)>,
}

impl ClientRoundOutput {
    pub fn client_id(&self) -> ClientId {
        self.client_id
    }

    /// The client's bulletin post: its ciphertext and commitment.
    pub fn post(&self) -> &ClientBulletinEntry {
        &self.encrypted_message
    }

    /// One sealed opening per server, each for that server only.
    pub fn sealed_openings(&self) -> &[(ServerId, Vec<u8>)] {
        &self.sealed_openings
    }

    /// The post and the sealed openings, without copying either.
    pub fn into_parts(self) -> (ClientBulletinEntry, Vec<(ServerId, Vec<u8>)>) {
        (self.encrypted_message, self.sealed_openings)
    }
}

pub fn run_client_round<R: CryptoRng + Rng>(
    rng: &mut R,
    pp: &ProtocolParams,
    sid: &SessionId,
    client_id: ClientId,
    message: Vec<KahePoly>,
    servers: &[(ServerId, pke::PublicKey)],
) -> ClientRoundOutput {
    ClientRound::new(rng, pp, sid, client_id, servers).finalize(&message)
}
