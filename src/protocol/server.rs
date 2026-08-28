use std::collections::HashMap;

use crate::{CsPoly, DgtNTTPoly};
use chipmunk_code::HVCPoly;
use rayon::prelude::*;

use crate::bulletin::{RsNodeBulletinEntry, ServerBulletinEntry};
use crate::cs::{Cs, HidingMerkleCommitment, Opening, PackedOpening};
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
                    vec![DgtNTTPoly::default(); scp.block_len],
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
                    vec![DgtNTTPoly::default(); scp.block_len],
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

pub fn run_server_round(
    inbox: &ServerInbox,
    canonical: &[ClientId],
) -> Result<ServerBulletinEntry, ServerRoundError> {
    let index: HashMap<ClientId, usize> = inbox
        .items
        .iter()
        .enumerate()
        .map(|(i, (cid, _))| (*cid, i))
        .collect();

    let mut opening_refs: Vec<&Opening> = Vec::with_capacity(canonical.len());
    for cid in canonical {
        let i = *index
            .get(cid)
            .ok_or(ServerRoundError::MissingClient(*cid))?;
        let (_, op) = &inbox.items[i];
        opening_refs.push(op);
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
