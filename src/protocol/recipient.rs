//! The recipient: decide the canonical client set, then decode over it.
//!
//! - [`recover_unverified_once`] is a single attempt over a set the caller names. A faulty
//!   share fails the round and names the server; nothing is retried.
//! - [`recover_unverified_direct`] admits the set per
//!   [`SetPolicy`], then drop servers whose shares fail their opening until `t`
//!   honest ones remain. Threshold decryption tolerates that; the excluded
//!   servers come back in [`Recovered::culprits`] for the caller to attribute.

use crate::KahePoly;

use crate::bulletin::{ClientBulletinEntry, ServerBulletinEntry};

use super::verify::{aggregate_and_decrypt_unverified, VerifyError};
use super::{ClientId, ProtocolParams, ServerId};

/// The canonical-set rule: decode strictly over the anchor the caller names.
pub struct SetPolicy<'a> {
    anchor: &'a [ClientId],
    max_clients: usize,
    min_clients: usize,
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
    pub(crate) canonical: Vec<ClientId>,
    pub(crate) plaintext: Vec<KahePoly>,
    /// Servers dropped to reach the threshold, in exclusion order.
    pub(crate) culprits: Vec<ServerId>,
}

impl Recovered {
    /// The client set the plaintext was decoded over.
    pub fn canonical(&self) -> &[ClientId] {
        &self.canonical
    }

    /// `Σ m_i mod t` over the canonical set, for the payload layer to decode.
    pub fn plaintext(&self) -> &[KahePoly] {
        &self.plaintext
    }

    /// Servers dropped to reach the threshold, in exclusion order.
    pub fn culprits(&self) -> &[ServerId] {
        &self.culprits
    }
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
    BelowAnonymityFloor {
        got: usize,
        min: usize,
    },
    AboveMaxClients {
        got: usize,
        max: usize,
    },
    TooFewAgreeingServers {
        agreeing: usize,
        need: usize,
    },
    MissingClientPublics {
        have: usize,
        need: usize,
    },
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
        .filter_map(|sp| {
            let clients = sorted_dedup(&sp.clients);
            if clients.len() != sp.clients.len() || clients != canonical {
                return None;
            }
            let mut sp = sp.clone();
            sp.clients = clients;
            Some(sp)
        })
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
    let max = policy.max_clients.min(pp.max_clients);
    if canonical.len() > max {
        return Err(RejectReason::AboveMaxClients {
            got: canonical.len(),
            max,
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
pub fn recover_unverified_once(
    pp: &ProtocolParams,
    canonical: &[ClientId],
    publics: &[(ClientId, ClientBulletinEntry)],
    server_outputs: &[ServerBulletinEntry],
) -> Result<Vec<KahePoly>, VerifyError> {
    aggregate_and_decrypt_unverified(pp, canonical, publics, server_outputs)
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
/// Requires authentic, session-bound posts consistent with the servers' view;
/// recovery does not authenticate the supplied ciphertexts.
pub fn recover_unverified_direct<L>(
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
        aggregate_and_decrypt_unverified(pp, &canonical, &entries, outs)
    })
    .map_err(RecipientError::Rejected)?;
    Ok(Recovered {
        canonical,
        plaintext,
        culprits,
    })
}
