//! Narrowing a digest failure to a group of clients.
//!
//! The digest check is aggregate: `Σ_i (H(m_i) − h_i) ≠ 0` says *someone* is
//! inconsistent, not who. Per-client attribution is not available
//! non-interactively — verifying one client's digest needs its individual
//! `m_i`, which needs `sk_i`, which exists only as Shamir shares and whose
//! reconstruction would deanonymise that client.
//!
//! What is available is bisection, because [`run_server_round`] and
//! [`run_node_round`] both take an *arbitrary* client subset as their canonical
//! set. Re-running them over each half localises the inconsistency: the digest
//! is linear, so a nonzero total forces a nonzero half.
//!
//! **This burns the round's anonymity down to `min_clients` along the disputed
//! branch.** Decrypting over a subset is exactly what the floor permits, so
//! this is not a new exposure, but it is a real cost — the search stops as soon
//! as a further split would breach the floor, and the answer is a *group*, not
//! a client. Call it deliberately, not on every failed round.
//!
//! [`run_server_round`]: crate::protocol::server::run_server_round
//! [`run_node_round`]: crate::protocol::server::run_node_round

use crate::bulletin::{RsClientBulletinEntry, RsNodeBulletinEntry, ServerBulletinEntry};
use crate::protocol::verify::{aggregate_and_decrypt_rs, VerifyError};
use crate::protocol::{ClientId, ProtocolParams, SessionId};

/// Everything a dispute round needs from the servers and lanes for one subset.
pub type SubsetOutputs = (Vec<ServerBulletinEntry>, Vec<RsNodeBulletinEntry>);

#[derive(Debug, PartialEq)]
pub enum DisputeOutcome {
    /// Smallest testable group still exhibiting the failure.
    Culprits(Vec<ClientId>),
    /// Both halves verified — the inconsistency is not attributable to a subset
    /// (a lying lane rather than a client, most likely).
    NotAttributable(Vec<ClientId>),
    /// A subset re-run could not be produced or failed for an unrelated reason.
    Inconclusive(Vec<ClientId>),
}

/// Bisect `failing` until a further split would drop a half below `floor`.
/// `rerun` re-executes the servers and lanes over the given subset; returning
/// `None` aborts the search.
///
/// `floor` is the anonymity the caller is willing to spend — the returned group
/// is no smaller, and every client in it has been decrypted over a set that
/// size. `pp.min_clients` is the natural choice, but it is *not* read from `pp`
/// on purpose: it defaults to 1, and a silent bisection down to a single client
/// is precisely what this must not do by accident.
pub fn bisect_digest_failure<F>(
    pp: &ProtocolParams,
    sid: &SessionId,
    client_entries: &[(ClientId, RsClientBulletinEntry)],
    failing: &[ClientId],
    floor: usize,
    mut rerun: F,
) -> DisputeOutcome
where
    F: FnMut(&[ClientId]) -> Option<SubsetOutputs>,
{
    assert!(floor >= 1, "anonymity floor must be ≥ 1");
    let mut cur = failing.to_vec();
    cur.sort();
    cur.dedup();

    loop {
        if cur.len() < 2 * floor {
            return DisputeOutcome::Culprits(cur);
        }
        let mid = cur.len() / 2;
        let halves = [cur[..mid].to_vec(), cur[mid..].to_vec()];

        let mut next: Option<Vec<ClientId>> = None;
        for half in halves {
            let Some((servers, nodes)) = rerun(&half) else {
                return DisputeOutcome::Inconclusive(cur);
            };
            match aggregate_and_decrypt_rs(pp, sid, &half, client_entries, &servers, &nodes) {
                Err(VerifyError::DigestMismatch) => {
                    next = Some(half);
                    break;
                }
                Ok(_) => continue,
                Err(_) => return DisputeOutcome::Inconclusive(cur),
            }
        }
        match next {
            Some(h) => cur = h,
            // Linearity says a nonzero total forces a nonzero half, so both
            // halves clean means the corruption was not in the client posts.
            None => return DisputeOutcome::NotAttributable(cur),
        }
    }
}
