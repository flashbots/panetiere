use std::collections::HashMap;

use chipmunk_code::CsPoly;
use rayon::prelude::*;

use crate::bulletin::{RsNodeBulletinEntry, ServerBulletinEntry};
use crate::cs::{Cs, HidingMerkleCommitment, Opening, PackedOpening};
use crate::pke;
use crate::rs::{Rs, Share};

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
}

pub struct RsNodeInbox {
    pub node_id: NodeId,
    pub items: Vec<(ClientId, Share)>,
}

pub fn run_rs_node_round(
    inbox: &RsNodeInbox,
    canonical: &[ClientId],
) -> Result<RsNodeBulletinEntry, ServerRoundError> {
    let index: HashMap<ClientId, usize> = inbox
        .items
        .iter()
        .enumerate()
        .map(|(i, (cid, _))| (*cid, i))
        .collect();

    let mut selected: Vec<&[chipmunk_code::DgtNTTPoly]> = Vec::with_capacity(canonical.len());
    for cid in canonical {
        let i = *index
            .get(cid)
            .ok_or(ServerRoundError::MissingClient(*cid))?;
        selected.push(inbox.items[i].1.as_slice());
    }

    Ok(RsNodeBulletinEntry {
        node_id: inbox.node_id,
        clients: canonical.to_vec(),
        share_sum: Rs::sum_shares(&selected),
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
    let agg_open = HidingMerkleCommitment::sum_openings(&opening_refs);
    let agg_share: CsPoly = agg_open.s()[0];

    Ok(ServerBulletinEntry {
        server_id: inbox.server_id,
        clients: canonical.to_vec(),
        agg_open,
        agg_share,
    })
}
