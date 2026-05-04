use std::collections::HashMap;

use crate::bulletin::ServerPublic;
use crate::cs::{Cs, HidingMerkleCommitment, Opening};
use crate::sss::{AdditiveSharing, Sss};

use super::message::{ClientId, ServerId};

/// Per-server inbox: openings received privately from each client. The key
/// share is `Opening::s()`; we don't carry a separate copy.
pub struct ServerInbox {
    pub server_id: ServerId,
    pub items: Vec<(ClientId, Opening)>,
}

#[derive(Debug, PartialEq)]
pub enum ServerRoundError {
    MissingClient(ClientId),
}

/// Aggregate inbox entries for a canonical client set (README steps 7–8).
/// Returns `Err(MissingClient)` if any canonical client is absent.
///
/// Indexes the inbox by `ClientId` first so the per-canonical-client lookup is
/// O(1). Building the index is O(|inbox|), giving an overall O(|inbox| + N)
/// instead of the O(N · |inbox|) linear scan we'd otherwise pay.
pub fn run_server_round(
    inbox: &ServerInbox,
    canonical: &[ClientId],
) -> Result<ServerPublic, ServerRoundError> {
    let index: HashMap<ClientId, usize> = inbox
        .items
        .iter()
        .enumerate()
        .map(|(i, (cid, _))| (*cid, i))
        .collect();

    let mut opening_refs: Vec<&Opening> = Vec::with_capacity(canonical.len());
    let mut shares = Vec::with_capacity(canonical.len());
    for cid in canonical {
        let i = *index
            .get(cid)
            .ok_or(ServerRoundError::MissingClient(*cid))?;
        let (_, op) = &inbox.items[i];
        opening_refs.push(op);
        shares.push(*op.s());
    }
    let agg_open = HidingMerkleCommitment::sum_openings(&opening_refs);
    let agg_share = AdditiveSharing::recover(&shares); // for additive sharing, recover = sum
    Ok(ServerPublic {
        server_id: inbox.server_id,
        clients: canonical.to_vec(),
        agg_open,
        agg_share,
    })
}
