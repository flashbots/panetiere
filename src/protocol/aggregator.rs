//! Public aggregator round (aggregated flow only — the direct flow is unchanged).
//!
//! An aggregator serves a group of clients. It sums the **public** parts of
//! their bulletin entries — the KAHE ciphertext and the CS commitment — into a
//! single per-group aggregate. Openings are not touched: they stay sealed to
//! the servers, so the threshold/privacy model is identical to the direct flow.
//!
//! `agg_ctxt` and `sum_commitments` are coefficient-wise sums, hence
//! associative: the verifier re-sums per-group aggregates across groups with
//! the same two operations and recovers the same value a direct
//! `aggregate_and_decrypt` would over every individual entry.

use std::collections::HashSet;

use crate::KahePoly;

use crate::bulletin::ClientBulletinEntry;
use crate::cs::{Commitment, HidingMerkleCommitment};
use crate::kahe::Kahe;

use super::ClientId;

/// One group's aggregated public contribution.
pub struct AggregatedPublic {
    /// The group's client set, sorted and deduplicated.
    pub clients: Vec<ClientId>,
    /// Σ over the group of each client's ciphertext.
    pub summed_ctxt: Vec<KahePoly>,
    /// Σ over the group of each client's commitment.
    pub summed_comm: Commitment,
}

/// Sum the public ciphertexts and commitments of a group's clients. At most one
/// entry per `ClientId` contributes (first occurrence wins); the returned
/// `clients` is the sorted, deduplicated set actually summed.
pub fn run_aggregator_round(entries: &[(ClientId, ClientBulletinEntry)]) -> AggregatedPublic {
    let mut seen: HashSet<ClientId> = HashSet::with_capacity(entries.len());
    let mut clients: Vec<ClientId> = Vec::with_capacity(entries.len());
    let mut ctxts: Vec<&[KahePoly]> = Vec::with_capacity(entries.len());
    let mut comms: Vec<&Commitment> = Vec::with_capacity(entries.len());
    for (cid, entry) in entries {
        if !seen.insert(*cid) {
            continue;
        }
        clients.push(*cid);
        ctxts.push(&entry.ctxt);
        comms.push(&entry.comm);
    }
    clients.sort_unstable();

    AggregatedPublic {
        clients,
        summed_ctxt: Kahe::agg_ctxt_refs(&ctxts),
        summed_comm: HidingMerkleCommitment::sum_commitment_refs(&comms),
    }
}
