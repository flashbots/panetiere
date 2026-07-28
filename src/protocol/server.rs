use std::collections::HashMap;

use chipmunk_code::CsPoly;
use rayon::prelude::*;

use crate::bulletin::ServerBulletinEntry;
use crate::cs::{Cs, HidingMerkleCommitment, Opening, PackedOpening};
use crate::pke;

use super::{opening_aad, ClientId, ServerId, SessionId};

/// Open one client's ECIES envelope into its `Opening`; `None` on a bad seal, a
/// malformed packed opening, or a `(sid, client_id, server_id)` other than the
/// one it was sealed under.
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

/// Batch form of [`unseal_opening`] over a server's whole inbox, independent
/// per item. Each entry carries its `ClientId`: it is part of the bound context.
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

/// Per-server inbox: one `Opening` per client (its `s()[0]` is that client's
/// Shamir share for this server).
pub struct ServerInbox {
    pub server_id: ServerId,
    pub items: Vec<(ClientId, Opening)>,
}

#[derive(Debug, PartialEq)]
pub enum ServerRoundError {
    MissingClient(ClientId),
}

/// Sum the canonical clients' openings into a single aggregated `Opening` and
/// extract the summed share.
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
    // For Shamir t-of-n with linear interpolation, summing per-server shares
    // across canonical clients gives the share of `Σ sk_j` at this server's
    // point. `agg_share` mirrors `agg_open.s()`.
    let agg_share: CsPoly = agg_open.s()[0];

    Ok(ServerBulletinEntry {
        server_id: inbox.server_id,
        clients: canonical.to_vec(),
        agg_open,
        agg_share,
    })
}
