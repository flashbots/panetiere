use crate::bulletin::ServerPublic;
use crate::cs::{Cs, HidingMerkleCommitment, Opening};
use crate::kahe::{Kahe, RingOtp};
use crate::sss::{AdditiveSharing, Sss};

use super::message::{ClientId, ServerId};

/// Per-server inbox: openings + key shares received privately from each client.
pub struct ServerInbox {
    pub server_id: ServerId,
    pub items: Vec<(ClientId, Opening, <RingOtp as Kahe>::Key)>,
}

/// Aggregate inbox entries for a canonical client set (README steps 7–8).
/// Returns `None` if any canonical client is missing.
pub fn run_server_round(
    inbox: &ServerInbox,
    canonical: &[ClientId],
) -> Option<ServerPublic> {
    let mut openings = Vec::with_capacity(canonical.len());
    let mut shares = Vec::with_capacity(canonical.len());
    for cid in canonical {
        let (_, op, sh) = inbox.items.iter().find(|(c, _, _)| c == cid)?;
        openings.push(op.clone());
        shares.push(*sh);
    }
    let agg_open = HidingMerkleCommitment::sum_openings(&openings);
    let agg_share = AdditiveSharing::recover(&shares); // for additive sharing, recover = sum
    Some(ServerPublic {
        server_id: inbox.server_id,
        clients: canonical.to_vec(),
        agg_open,
        agg_share,
    })
}
