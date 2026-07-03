use std::collections::HashMap;

use chipmunk_code::CsPoly;
use rayon::prelude::*;

use crate::bulletin::ServerBulletinEntry;
use crate::cs::{Cs, HidingMerkleCommitment, Opening, PackedOpening};
use crate::pke;

use super::{ClientId, ServerId};

/// Open one client's ECIES envelope into its `Opening`; `None` on a bad seal
/// or a malformed packed opening.
pub fn unseal_opening(key: &pke::PrivateKey, sealed: &[u8]) -> Option<Opening> {
    let plain = pke::decrypt(key, sealed).ok()?;
    let packed = PackedOpening::from_bytes(&plain)?;
    Opening::from_packed(&packed).ok()
}

/// Batch form of [`unseal_opening`] over a server's whole inbox, independent
/// per item.
pub fn unseal_openings(key: &pke::PrivateKey, sealed: &[Vec<u8>]) -> Vec<Option<Opening>> {
    sealed.par_iter().map(|s| unseal_opening(key, s)).collect()
}

/// Per-server inbox: one `Opening` per client (its `s()` is the κ_kahe-vector
/// of that client's Shamir shares for this server).
pub struct ServerInbox {
    pub server_id: ServerId,
    pub items: Vec<(ClientId, Opening)>,
}

#[derive(Debug, PartialEq)]
pub enum ServerRoundError {
    MissingClient(ClientId),
}

/// Sum the canonical clients' openings into a single aggregated `Opening` and
/// extract the summed κ_kahe-component share vector.
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
    // point. `agg_share` mirrors `agg_open.s()` componentwise.
    let agg_share: Vec<CsPoly> = agg_open.s().to_vec();

    Ok(ServerBulletinEntry {
        server_id: inbox.server_id,
        clients: canonical.to_vec(),
        agg_open,
        agg_share,
    })
}
