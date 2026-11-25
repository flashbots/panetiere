use std::collections::HashMap;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use crate::bulletin::ServerBulletinEntry;
use crate::cs::{
    batch_capacity, batch_weights, Commitment, Cs, HidingMerkleCommitment, Opening, PackedOpening,
};
use crate::pke;

use super::{opening_aad, ClientId, ProtocolParams, ServerId, SessionId};

pub fn unseal_opening(
    key: &pke::PrivateKey,
    sid: &SessionId,
    client_id: ClientId,
    server_id: ServerId,
    sealed: &[u8],
) -> Option<Opening> {
    let plain = pke::decrypt(key, sealed, &opening_aad(sid, client_id, server_id)).ok()?;
    let packed = PackedOpening::from_bytes(&plain)?;
    Opening::from_packed(&packed).ok()
}

pub fn unseal_openings(
    key: &pke::PrivateKey,
    sid: &SessionId,
    server_id: ServerId,
    sealed: &[(ClientId, Vec<u8>)],
) -> Vec<Option<Opening>> {
    sealed
        .par_iter()
        .map(|(cid, s)| unseal_opening(key, sid, *cid, server_id, s))
        .collect()
}

pub struct ServerInbox {
    pub(crate) server_id: ServerId,
    pub(crate) items: Vec<(ClientId, Opening)>,
}

impl ServerInbox {
    /// The openings `server_id` unsealed this round, one per client.
    pub fn new(server_id: ServerId, items: Vec<(ClientId, Opening)>) -> Self {
        Self { server_id, items }
    }
}

#[derive(Debug, PartialEq)]
pub enum ServerRoundError {
    MissingClient(ClientId),
    /// This client's opening does not open its own commitment at this server's
    /// position. The server posts nothing for the round, as in the paper, and
    /// the client is named so the round can be re-run without it.
    BadOpening(ClientId),
}

/// Random combinations checked per server round: a bad opening slips past
/// each with probability at most `2^-15`, so nine leave `2^-135`, the
/// smallest count past the 128-bit target.
const BATCH_COMBOS: usize = 9;

/// The paper's server step: every canonical client's opening must open that
/// client's own commitment at this server's position, within the fresh norm
/// bounds, or the server posts nothing. Only then are the openings summed, so
/// a bad client can never make an honest server's aggregate fail at the
/// recipient.
pub fn run_server_round(
    pp: &ProtocolParams,
    inbox: &ServerInbox,
    canonical: &[ClientId],
    commitments: &[(ClientId, Commitment)],
) -> Result<ServerBulletinEntry, ServerRoundError> {
    let cs = &pp.cs;
    let index: HashMap<ClientId, usize> = inbox
        .items
        .iter()
        .enumerate()
        .map(|(i, (cid, _))| (*cid, i))
        .collect();
    let comm_index: HashMap<ClientId, usize> = commitments
        .iter()
        .enumerate()
        .map(|(i, (cid, _))| (*cid, i))
        .collect();

    let mut opening_refs: Vec<&Opening> = Vec::with_capacity(canonical.len());
    let mut comm_refs: Vec<&Commitment> = Vec::with_capacity(canonical.len());
    for cid in canonical {
        let i = *index
            .get(cid)
            .ok_or(ServerRoundError::MissingClient(*cid))?;
        let c = *comm_index
            .get(cid)
            .ok_or(ServerRoundError::MissingClient(*cid))?;
        opening_refs.push(&inbox.items[i].1);
        comm_refs.push(&commitments[c].1);
    }

    let position = inbox.server_id.0 as usize;
    // position, shape and norms per client, then the ring arithmetic once
    // over random combinations of the whole set; only a failing batch pays
    // for the per-client check that names the culprit
    let culprit = canonical
        .par_iter()
        .zip(opening_refs.par_iter())
        .find_map_first(|(cid, op)| {
            (op.server_index != position
                || op.path_index != position
                || !HidingMerkleCommitment::check_fresh_shape(cs, op))
            .then_some(*cid)
        });
    if let Some(cid) = culprit {
        return Err(ServerRoundError::BadOpening(cid));
    }
    // one batch per `batch_capacity` openings, so a round larger than the i32
    // accumulators allow is still checked in batches rather than per client
    let capacity = batch_capacity(cs);
    let mut rng = ChaCha20Rng::from_entropy();
    let batched = capacity > 0
        && opening_refs
            .chunks(capacity)
            .zip(comm_refs.chunks(capacity))
            .all(|(openings, comms)| {
                let weights = batch_weights(&mut rng, BATCH_COMBOS, openings.len());
                HidingMerkleCommitment::verify_fresh_batch_shaped(cs, comms, openings, &weights)
            });
    if !batched {
        let culprit = canonical
            .par_iter()
            .zip(opening_refs.par_iter())
            .zip(comm_refs.par_iter())
            .find_map_first(|((cid, op), comm)| {
                (!HidingMerkleCommitment::verify_fresh(cs, comm, op)).then_some(*cid)
            });
        if let Some(cid) = culprit {
            return Err(ServerRoundError::BadOpening(cid));
        }
    }

    // Every opening passed the shape and position checks above, so the sum
    // cannot fail. An empty set gives the zero opening, as in the paper, where
    // each server starts from d_i = 0 and posts it if I is empty.
    let agg_open = HidingMerkleCommitment::sum_openings(cs, position, &opening_refs)
        .expect("every opening passed the shape and position checks");

    Ok(ServerBulletinEntry {
        server_id: inbox.server_id,
        clients: canonical.to_vec(),
        agg_open,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cs::CsParams;
    use crate::CsPoly;

    const SERVERS: usize = 4;
    const POSITION: u32 = 1;

    /// One client's commitment and its opening at `POSITION`.
    fn commit_one(rng: &mut ChaCha20Rng, cs: &CsParams) -> (Commitment, Opening) {
        let shares: Vec<Vec<CsPoly>> = (0..SERVERS)
            .map(|_| (0..cs.mu_cs).map(|_| CsPoly::rand_poly(rng)).collect())
            .collect();
        let (comm, mut openings) = HidingMerkleCommitment::commit(rng, cs, &shares);
        (comm, openings.swap_remove(POSITION as usize))
    }

    /// A round of `clients` honest clients, as the server at `POSITION` sees it.
    fn round(
        rng: &mut ChaCha20Rng,
        cs: &CsParams,
        clients: u32,
    ) -> (Vec<ClientId>, Vec<(ClientId, Commitment)>, ServerInbox) {
        let canonical: Vec<ClientId> = (0..clients).map(ClientId).collect();
        let (commitments, items) = canonical
            .iter()
            .map(|&cid| {
                let (comm, opening) = commit_one(rng, cs);
                ((cid, comm), (cid, opening))
            })
            .unzip();
        let inbox = ServerInbox::new(ServerId(POSITION), items);
        (canonical, commitments, inbox)
    }

    /// A server's entry survives its wire form: it parses back to the same
    /// bytes, still verifies against the summed commitments, and has the
    /// predicted length. Truncated or extended bytes do not parse.
    #[test]
    fn server_entry_round_trips_through_its_wire_form() {
        let mut rng = ChaCha20Rng::from_seed([80u8; 32]);
        let pp = ProtocolParams::setup(&mut rng, SERVERS);
        for clients in [1u32, 5] {
            let (canonical, commitments, inbox) = round(&mut rng, &pp.cs, clients);
            let entry = run_server_round(&pp, &inbox, &canonical, &commitments).unwrap();
            let bytes = entry.to_bytes(&pp);
            assert_eq!(
                bytes.len(),
                8 + 4 * clients as usize + pp.cs.aggregated_server_crypto_len(clients)
            );

            let parsed = ServerBulletinEntry::try_from_bytes(&bytes).expect("entry parses");
            assert_eq!(parsed.server_id, entry.server_id);
            assert_eq!(parsed.clients, entry.clients);
            assert_eq!(parsed.to_bytes(&pp), bytes);
            let refs: Vec<&Commitment> = commitments.iter().map(|(_, c)| c).collect();
            let summed = HidingMerkleCommitment::sum_commitment_refs(&refs);
            let share = HidingMerkleCommitment::verify(&pp.cs, &summed, &entry.agg_open);
            assert!(share.is_some());
            assert_eq!(
                HidingMerkleCommitment::verify(&pp.cs, &summed, &parsed.agg_open),
                share
            );

            assert!(ServerBulletinEntry::try_from_bytes(&bytes[..bytes.len() - 1]).is_none());
            let mut longer = bytes.clone();
            longer.push(0);
            assert!(ServerBulletinEntry::try_from_bytes(&longer).is_none());
            assert!(ServerBulletinEntry::try_from_bytes(&bytes[..7]).is_none());
        }
    }

    /// A batch that fails falls back to the per-client check, which names
    /// the client whose opening is bad.
    #[test]
    fn failed_batch_names_the_bad_opening() {
        let mut rng = ChaCha20Rng::from_seed([81u8; 32]);
        let pp = ProtocolParams::setup(&mut rng, SERVERS);
        let (canonical, commitments, mut inbox) = round(&mut rng, &pp.cs, 5);
        assert!(run_server_round(&pp, &inbox, &canonical, &commitments).is_ok());

        // still within the fresh bounds, but not an opening of its leaf
        inbox.items[3].1.r_mut().swap(0, 1);
        assert_eq!(
            run_server_round(&pp, &inbox, &canonical, &commitments).err(),
            Some(ServerRoundError::BadOpening(ClientId(3)))
        );
    }

    /// An opening with `r` at the aggregate bound is algebraically valid, so
    /// the batch accepts it; only the per-client shape gate turns it away.
    #[test]
    fn oversized_randomness_is_turned_away_before_the_batch() {
        let mut rng = ChaCha20Rng::from_seed([82u8; 32]);
        let mut pp = ProtocolParams::setup(&mut rng, SERVERS);
        let (canonical, mut commitments, mut inbox) = round(&mut rng, &pp.cs, 5);

        let fresh_bound = pp.cs.beta_cs;
        pp.cs.beta_cs = pp.cs.r_bound;
        let (comm, opening) = commit_one(&mut rng, &pp.cs);
        pp.cs.beta_cs = fresh_bound;
        commitments[2].1 = comm;
        inbox.items[2].1 = opening;
        assert_eq!(
            run_server_round(&pp, &inbox, &canonical, &commitments).err(),
            Some(ServerRoundError::BadOpening(ClientId(2)))
        );
    }

    /// An empty round posts the zero opening, as the paper's server does for
    /// an empty I (d_i = 0), and that opening verifies against the empty sum
    /// of commitments with the zero share.
    #[test]
    fn empty_round_posts_the_zero_opening() {
        let mut rng = ChaCha20Rng::from_seed([84u8; 32]);
        let pp = ProtocolParams::setup(&mut rng, SERVERS);
        let inbox = ServerInbox::new(ServerId(POSITION), Vec::new());
        let entry = run_server_round(&pp, &inbox, &[], &[]).expect("an empty round is valid");
        assert!(entry.clients.is_empty());
        assert_eq!(entry.agg_open.server_index, POSITION as usize);
        let empty_sum = HidingMerkleCommitment::sum_commitment_refs(&[]);
        assert_eq!(
            HidingMerkleCommitment::verify(&pp.cs, &empty_sum, &entry.agg_open),
            Some(vec![CsPoly::default()])
        );
        // but not against an actual commitment
        let (comm, _) = commit_one(&mut rng, &pp.cs);
        assert!(HidingMerkleCommitment::verify(&pp.cs, &comm, &entry.agg_open).is_none());
    }

    /// More clients than one batch holds still verify, and a bad one is
    /// still named.
    #[test]
    fn round_larger_than_a_batch() {
        let mut rng = ChaCha20Rng::from_seed([83u8; 32]);
        let mut pp = ProtocolParams::setup(&mut rng, SERVERS);
        let (canonical, commitments, mut inbox) = round(&mut rng, &pp.cs, 5);
        // a looser fresh bound still accepts these openings but leaves room
        // for only two per batch
        pp.cs.beta_cs = i32::MAX as u32 / (2 * (crate::cs::BATCH_WEIGHT_BOUND as u32 - 1));
        assert_eq!(batch_capacity(&pp.cs), 2);
        assert!(run_server_round(&pp, &inbox, &canonical, &commitments).is_ok());

        // still within the fresh bounds, but not an opening of its leaf
        inbox.items[4].1.r_mut().swap(0, 1);
        assert_eq!(
            run_server_round(&pp, &inbox, &canonical, &commitments).err(),
            Some(ServerRoundError::BadOpening(ClientId(4)))
        );
    }

    // an opening filed under another server, or at another tree position,
    // is turned away by name, whichever of the two is off
    #[test]
    fn opening_for_another_position_names_the_client() {
        let mut rng = ChaCha20Rng::from_seed([84u8; 32]);
        let pp = ProtocolParams::setup(&mut rng, SERVERS);
        let (canonical, commitments, inbox) = round(&mut rng, &pp.cs, 5);

        let mut wrong_server = ServerInbox {
            server_id: inbox.server_id,
            items: inbox.items.clone(),
        };
        wrong_server.items[1].1.server_index += 1;
        assert_eq!(
            run_server_round(&pp, &wrong_server, &canonical, &commitments).err(),
            Some(ServerRoundError::BadOpening(ClientId(1)))
        );

        let mut wrong_position = ServerInbox {
            server_id: inbox.server_id,
            items: inbox.items,
        };
        wrong_position.items[3].1.path_index += 1;
        assert_eq!(
            run_server_round(&pp, &wrong_position, &canonical, &commitments).err(),
            Some(ServerRoundError::BadOpening(ClientId(3)))
        );
    }

    // a client missing from the inbox or from the commitments is named
    #[test]
    fn missing_client_is_named() {
        let mut rng = ChaCha20Rng::from_seed([85u8; 32]);
        let pp = ProtocolParams::setup(&mut rng, SERVERS);
        let (canonical, mut commitments, mut inbox) = round(&mut rng, &pp.cs, 4);
        let (cid, comm) = commitments.remove(2);
        assert_eq!(
            run_server_round(&pp, &inbox, &canonical, &commitments).err(),
            Some(ServerRoundError::MissingClient(cid))
        );
        commitments.push((cid, comm));
        let (absent, _) = inbox.items.remove(0);
        assert_eq!(
            run_server_round(&pp, &inbox, &canonical, &commitments).err(),
            Some(ServerRoundError::MissingClient(absent))
        );
    }
}
