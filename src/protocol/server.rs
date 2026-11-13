use std::collections::HashMap;

use crate::{CsPoly, RsNTTPoly};
use chipmunk_code::HVCPoly;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use crate::bulletin::{RsNodeBulletinEntry, ServerBulletinEntry};
use crate::cs::{
    batch_capacity, batch_weights, Commitment, Cs, CsParams, HidingMerkleCommitment, Opening,
    PackedOpening,
};
use crate::hvc_sum::sum_hvc_polys;
use crate::pke;
use crate::rs::Share;
use crate::share_commitment::{
    ingest_share, open_share, verify_aggregated, ShareCommitmentParams, ShareOpeningAcc, SharePath,
};

use super::{opening_aad, ClientId, NodeId, ServerId, SessionId};

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
    pub server_id: ServerId,
    pub items: Vec<(ClientId, Opening)>,
}

#[derive(Debug, PartialEq)]
pub enum ServerRoundError {
    MissingClient(ClientId),
    /// Ingest rejection: `(share, path)` does not open the client's signed root.
    BadShare(ClientId),
    /// The lane's aggregate does not open although every client's share does —
    /// the roster exceeds `ρ_max`, so the digit sums left the bound.
    AggregateOverCapacity,
    /// Some client's opening disagrees with the others on shape or position, so
    /// the set cannot be summed.
    MalformedOpening,
    /// This client's opening does not open its own commitment at this server's
    /// position. The server posts nothing for the round, as in the paper, and
    /// the client is named so the round can be re-run without it.
    BadOpening(ClientId),
}

pub struct RsNodeInbox {
    pub node_id: NodeId,
    /// As received from each client; the signed root comes off the bulletin.
    pub items: Vec<(ClientId, Share, SharePath)>,
}

/// Fold the canonical set's shares digit-wise, then prove the sum opens
/// `Σ roots` at this lane's position — one hash of the aggregate, not ρ of them,
/// which linearity makes equivalent. The post carries its own proof either way,
/// so a lane is trusted for nothing.
///
/// Only if the aggregate fails to open does the lane re-check per client, which
/// is what names the culprit. That case requires a client that shipped a share
/// disagreeing with its own signed root — excluded while clients run in TEEs,
/// and paid for only when it happens.
pub fn run_rs_node_round(
    scp: &ShareCommitmentParams,
    inbox: &RsNodeInbox,
    canonical: &[ClientId],
    roots: &[(ClientId, HVCPoly)],
) -> Result<RsNodeBulletinEntry, ServerRoundError> {
    let index: HashMap<ClientId, usize> = inbox
        .items
        .iter()
        .enumerate()
        .map(|(i, (cid, _, _))| (*cid, i))
        .collect();
    let root_index: HashMap<ClientId, usize> = roots
        .iter()
        .enumerate()
        .map(|(i, (cid, _))| (*cid, i))
        .collect();

    let lane = inbox.node_id.0 as usize;
    let mut selected: Vec<usize> = Vec::with_capacity(canonical.len());
    for cid in canonical {
        let i = *index
            .get(cid)
            .ok_or(ServerRoundError::MissingClient(*cid))?;
        let r = *root_index
            .get(cid)
            .ok_or(ServerRoundError::MissingClient(*cid))?;
        selected.push(i);
        debug_assert_eq!(roots[r].0, *cid);
    }

    let (agg, share_sum) = canonical
        .par_iter()
        .zip(selected.par_iter())
        .try_fold(
            || {
                (
                    ShareOpeningAcc::zero(scp, lane),
                    vec![RsNTTPoly::default(); scp.block_len],
                )
            },
            |(mut acc, mut sum), (cid, &i)| {
                let (_, share, path) = &inbox.items[i];
                let o =
                    open_share(scp, lane, share, path).ok_or(ServerRoundError::BadShare(*cid))?;
                acc.add(&o);
                for (s, p) in sum.iter_mut().zip(share.iter()) {
                    *s += *p;
                }
                Ok((acc, sum))
            },
        )
        .try_reduce(
            || {
                (
                    ShareOpeningAcc::zero(scp, lane),
                    vec![RsNTTPoly::default(); scp.block_len],
                )
            },
            |(mut a, mut sa), (b, sb)| {
                a.merge(&b);
                for (s, p) in sa.iter_mut().zip(sb.iter()) {
                    *s += *p;
                }
                Ok((a, sa))
            },
        )?;
    let agg = agg.finish();

    let root_refs: Vec<&HVCPoly> = canonical
        .iter()
        .map(|cid| &roots[root_index[cid]].1)
        .collect();
    let summed_root = sum_hvc_polys(&root_refs);
    if !verify_aggregated(scp, &summed_root, &share_sum, &agg) {
        // The aggregate does not open, so some client's share disagrees with
        // its own signed root. Now — and only now — pay per client to name it.
        let culprit = canonical
            .par_iter()
            .zip(selected.par_iter())
            .find_map_first(|(cid, &i)| {
                let root = &roots[root_index[cid]].1;
                let (_, share, path) = &inbox.items[i];
                ingest_share(scp, root, lane, share, path)
                    .is_none()
                    .then_some(*cid)
            });
        return Err(culprit.map_or(
            ServerRoundError::AggregateOverCapacity,
            ServerRoundError::BadShare,
        ));
    }

    Ok(RsNodeBulletinEntry {
        node_id: inbox.node_id,
        clients: canonical.to_vec(),
        share_sum,
        agg_open: agg,
    })
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
    cs: &CsParams,
    inbox: &ServerInbox,
    canonical: &[ClientId],
    commitments: &[(ClientId, Commitment)],
) -> Result<ServerBulletinEntry, ServerRoundError> {
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

    let agg_open = HidingMerkleCommitment::sum_openings(&opening_refs)
        .ok_or(ServerRoundError::MalformedOpening)?;
    let agg_share: CsPoly = agg_open.s()[0];

    Ok(ServerBulletinEntry {
        server_id: inbox.server_id,
        clients: canonical.to_vec(),
        agg_open,
        agg_share,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let inbox = ServerInbox {
            server_id: ServerId(POSITION),
            items,
        };
        (canonical, commitments, inbox)
    }

    /// A batch that fails falls back to the per-client check, which names
    /// the client whose opening is bad.
    #[test]
    fn failed_batch_names_the_bad_opening() {
        let mut rng = ChaCha20Rng::from_seed([81u8; 32]);
        let cs = HidingMerkleCommitment::setup(&mut rng, SERVERS);
        let (canonical, commitments, mut inbox) = round(&mut rng, &cs, 5);
        assert!(run_server_round(&cs, &inbox, &canonical, &commitments).is_ok());

        inbox.items[3].1.s_mut()[0] += CsPoly::from_coeffs([1; crate::N]);
        assert_eq!(
            run_server_round(&cs, &inbox, &canonical, &commitments).err(),
            Some(ServerRoundError::BadOpening(ClientId(3)))
        );
    }

    /// An opening with `r` at the aggregate bound is algebraically valid, so
    /// the batch accepts it; only the per-client shape gate turns it away.
    #[test]
    fn oversized_randomness_is_turned_away_before_the_batch() {
        let mut rng = ChaCha20Rng::from_seed([82u8; 32]);
        let mut cs = HidingMerkleCommitment::setup(&mut rng, SERVERS);
        let (canonical, mut commitments, mut inbox) = round(&mut rng, &cs, 5);

        let fresh_bound = cs.beta_cs;
        cs.beta_cs = cs.r_bound;
        let (comm, opening) = commit_one(&mut rng, &cs);
        cs.beta_cs = fresh_bound;
        commitments[2].1 = comm;
        inbox.items[2].1 = opening;
        assert_eq!(
            run_server_round(&cs, &inbox, &canonical, &commitments).err(),
            Some(ServerRoundError::BadOpening(ClientId(2)))
        );
    }

    /// More clients than one batch holds still verify, and a bad one is
    /// still named.
    #[test]
    fn round_larger_than_a_batch() {
        let mut rng = ChaCha20Rng::from_seed([83u8; 32]);
        let mut cs = HidingMerkleCommitment::setup(&mut rng, SERVERS);
        let (canonical, commitments, mut inbox) = round(&mut rng, &cs, 5);
        // a looser fresh bound still accepts these openings but leaves room
        // for only two per batch
        cs.beta_cs = i32::MAX as u32 / (2 * (crate::cs::BATCH_WEIGHT_BOUND as u32 - 1));
        assert_eq!(batch_capacity(&cs), 2);
        assert!(run_server_round(&cs, &inbox, &canonical, &commitments).is_ok());

        inbox.items[4].1.s_mut()[0] += CsPoly::from_coeffs([1; crate::N]);
        assert_eq!(
            run_server_round(&cs, &inbox, &canonical, &commitments).err(),
            Some(ServerRoundError::BadOpening(ClientId(4)))
        );
    }
}
