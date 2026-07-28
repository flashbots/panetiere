//! The recipient: decide the canonical client set, then decode over it.
//!
//! [`verify`](crate::protocol::verify) takes the canonical set as an argument, so
//! on its own it cannot answer "which set?" — and every caller that had to answer
//! it grew its own copy of the rule. This module owns that decision, in two
//! layers:
//!
//! - [`recover_once`] is a single attempt over a set the caller names. A faulty
//!   share fails the round and names the server; nothing is retried.
//! - [`recover_direct`] and [`recover_aggregated`] choose the set per
//!   [`SetPolicy`], then drop servers whose shares fail their opening until `t`
//!   honest ones remain. Threshold decryption tolerates that; the excluded
//!   servers come back in [`Recovered::culprits`] for the caller to attribute.
//!
//! No logging here — rejections come back as [`RejectReason`] so the caller
//! renders them in its own format.

use std::collections::BTreeMap;

use chipmunk_code::KahePoly;

use crate::bulletin::{ClientBulletinEntry, ServerBulletinEntry};
use crate::cs::{Cs, HidingMerkleCommitment};
use crate::kahe::{Kahe, KaheScheme};

use super::verify::{
    aggregate_and_decrypt, check_anonymity_floor, decrypt_aggregate, VerifyError,
};
use super::{ClientId, ProtocolParams, ServerId};

/// Where the canonical set comes from.
pub enum SetSource<'a> {
    /// Decode strictly over this set — a leader announced it.
    Anchored(&'a [ClientId]),
    /// No announcer: take the largest set that at least `t` servers agree on.
    Majority,
}

/// The canonical-set rule. `max_clients` is a single number the caller computes
/// once; passing it here is what keeps the announced set, the accepted set and
/// the served set from drifting apart.
///
/// `min_clients` is the caller's anonymity floor, taken alongside — not instead
/// of — `pp.min_clients`; the stricter of the two applies. A deployment's
/// configured floor is usually above the parameter default, and reading only
/// `pp` would quietly discard it.
pub struct SetPolicy<'a> {
    pub source: SetSource<'a>,
    pub max_clients: usize,
    pub min_clients: usize,
}

impl<'a> SetPolicy<'a> {
    pub fn anchored(set: &'a [ClientId], min_clients: usize, max_clients: usize) -> Self {
        Self {
            source: SetSource::Anchored(set),
            max_clients,
            min_clients,
        }
    }

    pub fn majority(min_clients: usize, max_clients: usize) -> Self {
        Self {
            source: SetSource::Majority,
            max_clients,
            min_clients,
        }
    }
}

pub struct Recovered {
    pub canonical: Vec<ClientId>,
    pub plaintext: Vec<KahePoly>,
    /// Servers dropped to reach the threshold, in exclusion order.
    pub culprits: Vec<ServerId>,
}

/// Summarised rather than derived: `plaintext` is megabytes at realistic widths.
impl std::fmt::Debug for Recovered {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recovered")
            .field("canonical", &self.canonical.len())
            .field("plaintext_polys", &self.plaintext.len())
            .field("culprits", &self.culprits)
            .finish()
    }
}

#[derive(Debug, PartialEq)]
pub enum RecipientError {
    /// Too few shares to attempt anything. Expected for a round whose peers have
    /// not reported yet, so callers usually log this at trace level.
    BelowShareThreshold { have: usize, need: usize },
    /// Every candidate was rejected, in the order tried.
    NoCandidate(Vec<CandidateRejection>),
}

#[derive(Debug, PartialEq)]
pub struct CandidateRejection {
    pub canonical_len: usize,
    pub reason: RejectReason,
}

#[derive(Debug, PartialEq)]
pub enum RejectReason {
    BelowAnonymityFloor { got: usize, min: usize },
    AboveMaxClients { got: usize, max: usize },
    TooFewAgreeingServers { agreeing: usize, need: usize },
    MissingClientPublics { have: usize, need: usize },
    /// A canonical client's group aggregate never arrived.
    GroupAggregateMissing { group_of: ClientId },
    /// A group aggregate covers a different membership than the canonical set.
    GroupMembershipMismatch { got: usize, expected: usize },
    /// Dropped so many faulty shares that the threshold is out of reach.
    ExhaustedShares {
        remaining: usize,
        need: usize,
        excluded: usize,
    },
    /// A verify failure that names no server, so nothing can be excluded.
    Unattributable(VerifyError),
}

fn sorted_dedup(set: &[ClientId]) -> Vec<ClientId> {
    let mut v = set.to_vec();
    v.sort();
    v.dedup();
    v
}

/// Candidate sets in the order they should be tried, each with the servers that
/// shared over exactly it.
fn candidates(
    policy: &SetPolicy,
    server_outputs: &[ServerBulletinEntry],
) -> Vec<(Vec<ClientId>, Vec<ServerBulletinEntry>)> {
    match policy.source {
        SetSource::Anchored(set) => {
            let canonical = sorted_dedup(set);
            let agreeing = server_outputs
                .iter()
                .filter(|sp| sorted_dedup(&sp.clients) == canonical)
                .cloned()
                .collect();
            vec![(canonical, agreeing)]
        }
        SetSource::Majority => {
            // BTreeMap, and a tie broken by the set itself, so the choice is a
            // function of the inputs and not of map iteration order.
            let mut groups: BTreeMap<Vec<ClientId>, Vec<ServerBulletinEntry>> = BTreeMap::new();
            for sp in server_outputs {
                groups
                    .entry(sorted_dedup(&sp.clients))
                    .or_default()
                    .push(sp.clone());
            }
            let mut out: Vec<_> = groups.into_iter().collect();
            out.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
            out
        }
    }
}

/// Shape checks every candidate must clear before any crypto runs.
fn admit(
    pp: &ProtocolParams,
    policy: &SetPolicy,
    canonical: &[ClientId],
    agreeing: usize,
) -> Result<(), RejectReason> {
    let floor = policy.min_clients.max(pp.min_clients);
    if canonical.len() < floor {
        return Err(RejectReason::BelowAnonymityFloor {
            got: canonical.len(),
            min: floor,
        });
    }
    debug_assert!(check_anonymity_floor(pp, canonical).is_ok());
    if canonical.len() > policy.max_clients {
        return Err(RejectReason::AboveMaxClients {
            got: canonical.len(),
            max: policy.max_clients,
        });
    }
    if agreeing < pp.shamir.t {
        return Err(RejectReason::TooFewAgreeingServers {
            agreeing,
            need: pp.shamir.t,
        });
    }
    Ok(())
}

/// One decode attempt over a caller-named set. A bad share aborts and is named
/// by index into `server_outputs`; nothing is excluded or retried.
pub fn recover_once(
    pp: &ProtocolParams,
    canonical: &[ClientId],
    publics: &[(ClientId, ClientBulletinEntry)],
    server_outputs: &[ServerBulletinEntry],
) -> Result<Vec<KahePoly>, VerifyError> {
    aggregate_and_decrypt(pp, canonical, publics, server_outputs)
}

/// Drop servers whose shares fail their opening until `t` remain, then decode.
/// `decode` runs one attempt over the surviving outputs.
fn exclude_and_decode<F>(
    t: usize,
    mut outputs: Vec<ServerBulletinEntry>,
    mut decode: F,
) -> Result<(Vec<KahePoly>, Vec<ServerId>), RejectReason>
where
    F: FnMut(&[ServerBulletinEntry]) -> Result<Vec<KahePoly>, VerifyError>,
{
    let mut culprits = Vec::new();
    loop {
        if outputs.len() < t {
            return Err(RejectReason::ExhaustedShares {
                remaining: outputs.len(),
                need: t,
                excluded: culprits.len(),
            });
        }
        match decode(&outputs) {
            Ok(plain) => return Ok((plain, culprits)),
            Err(VerifyError::InvalidServerOpening(i))
            | Err(VerifyError::ShareOpeningMismatch(i))
                if i < outputs.len() =>
            {
                culprits.push(outputs[i].server_id);
                outputs.remove(i);
            }
            Err(e) => return Err(RejectReason::Unattributable(e)),
        }
    }
}

/// Direct flow: clients posted their own ciphertext and commitment.
///
/// `publics` is looked up per candidate set; a client in the set with no post
/// rejects that candidate rather than silently shrinking it.
pub fn recover_direct<L>(
    pp: &ProtocolParams,
    policy: &SetPolicy,
    publics: L,
    server_outputs: &[ServerBulletinEntry],
) -> Result<Recovered, RecipientError>
where
    L: Fn(ClientId) -> Option<ClientBulletinEntry>,
{
    if server_outputs.len() < pp.shamir.t {
        return Err(RecipientError::BelowShareThreshold {
            have: server_outputs.len(),
            need: pp.shamir.t,
        });
    }
    let mut rejections = Vec::new();
    for (canonical, agreeing) in candidates(policy, server_outputs) {
        let reject = |reason| CandidateRejection {
            canonical_len: canonical.len(),
            reason,
        };
        if let Err(reason) = admit(pp, policy, &canonical, agreeing.len()) {
            rejections.push(reject(reason));
            continue;
        }
        let entries: Vec<(ClientId, ClientBulletinEntry)> = canonical
            .iter()
            .filter_map(|cid| publics(*cid).map(|p| (*cid, p)))
            .collect();
        if entries.len() != canonical.len() {
            rejections.push(reject(RejectReason::MissingClientPublics {
                have: entries.len(),
                need: canonical.len(),
            }));
            continue;
        }
        match exclude_and_decode(pp.shamir.t, agreeing, |outs| {
            aggregate_and_decrypt(pp, &canonical, &entries, outs)
        }) {
            Ok((plaintext, culprits)) => {
                return Ok(Recovered {
                    canonical,
                    plaintext,
                    culprits,
                })
            }
            Err(reason) => rejections.push(reject(reason)),
        }
    }
    Err(RecipientError::NoCandidate(rejections))
}

/// Aggregated flow: aggregators already summed each group's public parts, so the
/// recipient re-sums the groups instead of the individual posts.
///
/// `groups` is `(that group's client set, its summed entry)`. How clients map to
/// groups is the caller's topology and is not inspected — only that the groups
/// jointly cover exactly the canonical set.
pub fn recover_aggregated(
    pp: &ProtocolParams,
    policy: &SetPolicy,
    groups: &[(Vec<ClientId>, ClientBulletinEntry)],
    server_outputs: &[ServerBulletinEntry],
) -> Result<Recovered, RecipientError> {
    if server_outputs.len() < pp.shamir.t {
        return Err(RecipientError::BelowShareThreshold {
            have: server_outputs.len(),
            need: pp.shamir.t,
        });
    }
    let mut rejections = Vec::new();
    for (canonical, agreeing) in candidates(policy, server_outputs) {
        let reject = |reason| CandidateRejection {
            canonical_len: canonical.len(),
            reason,
        };
        if let Err(reason) = admit(pp, policy, &canonical, agreeing.len()) {
            rejections.push(reject(reason));
            continue;
        }

        // Every canonical client must appear in exactly one supplied group, and
        // the groups must contribute nothing outside the set.
        let covered: Vec<ClientId> = sorted_dedup(
            &groups
                .iter()
                .flat_map(|(cs, _)| cs.iter().copied())
                .collect::<Vec<_>>(),
        );
        if covered != canonical {
            let missing = canonical.iter().find(|c| !covered.contains(c));
            rejections.push(reject(match missing {
                Some(c) => RejectReason::GroupAggregateMissing { group_of: *c },
                None => RejectReason::GroupMembershipMismatch {
                    got: covered.len(),
                    expected: canonical.len(),
                },
            }));
            continue;
        }

        let total_ctxt = Kahe::agg_ctxt(
            &groups.iter().map(|(_, e)| e.ctxt.clone()).collect::<Vec<_>>(),
        );
        let total_comm = HidingMerkleCommitment::sum_commitments(
            &groups.iter().map(|(_, e)| e.comm.clone()).collect::<Vec<_>>(),
        );
        match exclude_and_decode(pp.shamir.t, agreeing, |outs| {
            decrypt_aggregate(pp, &total_ctxt, &total_comm, outs)
        }) {
            Ok((plaintext, culprits)) => {
                return Ok(Recovered {
                    canonical,
                    plaintext,
                    culprits,
                })
            }
            Err(reason) => rejections.push(reject(reason)),
        }
    }
    Err(RecipientError::NoCandidate(rejections))
}
