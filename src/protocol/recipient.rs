//! The recipient: decide the canonical client set, then decode over it.
//!
//! - [`recover_once`] is a single attempt over a set the caller names. A faulty
//!   share fails the round and names the server; nothing is retried.
//! - [`recover_direct`] and [`recover_aggregated`] admit the set per
//!   [`SetPolicy`], then drop servers whose shares fail their opening until `t`
//!   honest ones remain. Threshold decryption tolerates that; the excluded
//!   servers come back in [`Recovered::culprits`] for the caller to attribute.

use chipmunk_code::KahePoly;

use crate::bulletin::{ClientBulletinEntry, ServerBulletinEntry};
use crate::cs::{Cs, HidingMerkleCommitment};
use crate::kahe::{Kahe, KaheScheme};

use super::verify::{
    aggregate_and_decrypt, check_anonymity_floor, decrypt_aggregate, VerifyError,
};
use super::{ClientId, ProtocolParams, ServerId};

/// The canonical-set rule: decode strictly over the anchor the caller names.
pub struct SetPolicy<'a> {
    pub anchor: &'a [ClientId],
    pub max_clients: usize,
    pub min_clients: usize,
}

impl<'a> SetPolicy<'a> {
    pub fn anchored(set: &'a [ClientId], min_clients: usize, max_clients: usize) -> Self {
        Self {
            anchor: set,
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
    BelowShareThreshold { have: usize, need: usize },
    Rejected(RejectReason),
}

#[derive(Debug, PartialEq)]
pub enum RejectReason {
    BelowAnonymityFloor { got: usize, min: usize },
    AboveMaxClients { got: usize, max: usize },
    TooFewAgreeingServers { agreeing: usize, need: usize },
    MissingClientPublics { have: usize, need: usize },
    GroupAggregateMissing { group_of: ClientId },
    GroupMembershipMismatch { got: usize, expected: usize },
    ExhaustedShares {
        remaining: usize,
        need: usize,
        excluded: usize,
    },
    Unattributable(VerifyError),
}

fn sorted_dedup(set: &[ClientId]) -> Vec<ClientId> {
    let mut v = set.to_vec();
    v.sort();
    v.dedup();
    v
}

/// The anchor, and the servers that shared over exactly it.
fn anchored_set(
    policy: &SetPolicy,
    server_outputs: &[ServerBulletinEntry],
) -> (Vec<ClientId>, Vec<ServerBulletinEntry>) {
    let canonical = sorted_dedup(policy.anchor);
    let agreeing = server_outputs
        .iter()
        .filter(|sp| sorted_dedup(&sp.clients) == canonical)
        .cloned()
        .collect();
    (canonical, agreeing)
}

/// Shape checks the set must clear before any crypto runs.
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

/// One decode attempt over a caller-named set.
pub fn recover_once(
    pp: &ProtocolParams,
    canonical: &[ClientId],
    publics: &[(ClientId, ClientBulletinEntry)],
    server_outputs: &[ServerBulletinEntry],
) -> Result<Vec<KahePoly>, VerifyError> {
    aggregate_and_decrypt(pp, canonical, publics, server_outputs)
}

/// Drop servers whose shares fail their opening until `t` remain, then decode.
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
    let (canonical, agreeing) = anchored_set(policy, server_outputs);
    admit(pp, policy, &canonical, agreeing.len()).map_err(RecipientError::Rejected)?;

    let entries: Vec<(ClientId, ClientBulletinEntry)> = canonical
        .iter()
        .filter_map(|cid| publics(*cid).map(|p| (*cid, p)))
        .collect();
    if entries.len() != canonical.len() {
        return Err(RecipientError::Rejected(
            RejectReason::MissingClientPublics {
                have: entries.len(),
                need: canonical.len(),
            },
        ));
    }
    let (plaintext, culprits) = exclude_and_decode(pp.shamir.t, agreeing, |outs| {
        aggregate_and_decrypt(pp, &canonical, &entries, outs)
    })
    .map_err(RecipientError::Rejected)?;
    Ok(Recovered {
        canonical,
        plaintext,
        culprits,
    })
}

/// Aggregated flow: aggregators already summed each group's public parts, so the
/// recipient re-sums the groups instead of the individual posts.
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
    let (canonical, agreeing) = anchored_set(policy, server_outputs);
    admit(pp, policy, &canonical, agreeing.len()).map_err(RecipientError::Rejected)?;

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
        return Err(RecipientError::Rejected(match missing {
            Some(c) => RejectReason::GroupAggregateMissing { group_of: *c },
            None => RejectReason::GroupMembershipMismatch {
                got: covered.len(),
                expected: canonical.len(),
            },
        }));
    }

    let total_ctxt =
        Kahe::agg_ctxt(&groups.iter().map(|(_, e)| e.ctxt.clone()).collect::<Vec<_>>());
    let total_comm = HidingMerkleCommitment::sum_commitments(
        &groups.iter().map(|(_, e)| e.comm.clone()).collect::<Vec<_>>(),
    );
    let (plaintext, culprits) = exclude_and_decode(pp.shamir.t, agreeing, |outs| {
        decrypt_aggregate(pp, &total_ctxt, &total_comm, outs)
    })
    .map_err(RecipientError::Rejected)?;
    Ok(Recovered {
        canonical,
        plaintext,
        culprits,
    })
}
