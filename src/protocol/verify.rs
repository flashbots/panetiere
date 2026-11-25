use std::collections::{HashMap, HashSet};

use crate::{CsPoly, KahePoly};
use rayon::prelude::*;

use crate::bulletin::{ClientBulletinEntry, ServerBulletinEntry};
use crate::cs::{Commitment, Cs, HidingMerkleCommitment};
use crate::kahe::{lift_cs_to_kahe, Kahe, KaheAggKey, KaheScheme};
use crate::sss::ShamirSharing;
pub use crate::sss::SssError;

use super::ProtocolParams;
use super::{ClientId, ServerId};

#[derive(Debug, PartialEq)]
pub enum VerifyError {
    MissingClient(ClientId),
    DuplicateClient(ClientId),
    InvalidServerOpening(usize),
    NoServers,
    BadServerCoverage,
    InconsistentCanonical(ServerId),
    /// This client's ciphertext does not have exactly `μ` ring elements.
    WrongCiphertextLength(ClientId),
    /// The summed ciphertext does not have exactly `μ` ring elements.
    WrongAggregateLength,
    AnonymitySetTooSmall {
        got: usize,
        min: usize,
    },
    TooManyClients {
        got: usize,
        max: usize,
    },
    ShareRecovery(SssError),
}

/// Verify server `i`'s aggregated opening against the summed commitment, at
/// its own position, and return the share it opens to: the server's Shamir
/// share of the summed key.
fn verify_one_server(
    pp: &ProtocolParams,
    summed_comm: &Commitment,
    i: usize,
    sp: &ServerBulletinEntry,
) -> Result<CsPoly, VerifyError> {
    if sp.agg_open.path_index != sp.server_id.0 as usize {
        return Err(VerifyError::InvalidServerOpening(i));
    }
    HidingMerkleCommitment::verify(&pp.cs, summed_comm, &sp.agg_open)
        .map(|share| share[0])
        .ok_or(VerifyError::InvalidServerOpening(i))
}

/// [`verify_one_server`] for every server, in parallel; the first failing
/// server in `server_outputs`' order is reported.
fn verify_servers(
    pp: &ProtocolParams,
    summed_comm: &Commitment,
    server_outputs: &[ServerBulletinEntry],
) -> Result<Vec<CsPoly>, VerifyError> {
    let results: Vec<Result<CsPoly, VerifyError>> = server_outputs
        .par_iter()
        .enumerate()
        .map(|(i, sp)| verify_one_server(pp, summed_comm, i, sp))
        .collect();
    results.into_iter().collect()
}

// Valid openings do not prove degree consistency. With inconsistent client
// shares, the first t reports can recover a different key than another subset.
fn recover_agg_key(
    pp: &ProtocolParams,
    server_outputs: &[ServerBulletinEntry],
    shares: &[CsPoly],
) -> Result<KahePoly, VerifyError> {
    let samples: Vec<(usize, CsPoly)> = server_outputs
        .iter()
        .zip(shares)
        .take(pp.shamir.t)
        .map(|(sp, &share)| (sp.server_id.0 as usize, share))
        .collect();
    let recovered_cs =
        ShamirSharing::recover(&pp.shamir, &samples).map_err(VerifyError::ShareRecovery)?;
    Ok(lift_cs_to_kahe(&recovered_cs))
}

fn check_unique_clients(clients: &[ClientId]) -> Result<(), VerifyError> {
    let mut seen = HashSet::with_capacity(clients.len());
    for &cid in clients {
        if !seen.insert(cid) {
            return Err(VerifyError::DuplicateClient(cid));
        }
    }
    Ok(())
}

pub(super) fn check_client_count(
    pp: &ProtocolParams,
    clients: &[ClientId],
) -> Result<(), VerifyError> {
    let got = clients.iter().collect::<HashSet<_>>().len();
    if got > pp.max_clients {
        return Err(VerifyError::TooManyClients {
            got,
            max: pp.max_clients,
        });
    }
    if got < pp.min_clients {
        return Err(VerifyError::AnonymitySetTooSmall {
            got,
            min: pp.min_clients,
        });
    }
    Ok(())
}

pub(crate) fn aggregate_and_decrypt_unverified(
    pp: &ProtocolParams,
    canonical: &[ClientId],
    client_entries: &[(ClientId, ClientBulletinEntry)],
    server_outputs: &[ServerBulletinEntry],
) -> Result<Vec<KahePoly>, VerifyError> {
    if server_outputs.is_empty() {
        return Err(VerifyError::NoServers);
    }
    check_unique_clients(canonical)?;
    check_client_count(pp, canonical)?;
    // Every ciphertext has exactly μ ring elements, so the sum is positional.
    let mu = pp.kahe.mu_kahe;
    let t = pp.shamir.t;

    if server_outputs.len() < t {
        return Err(VerifyError::BadServerCoverage);
    }
    let mut seen: HashSet<u32> = HashSet::with_capacity(server_outputs.len());
    for sp in server_outputs {
        if (sp.server_id.0 as usize) >= pp.cs.n_servers || !seen.insert(sp.server_id.0) {
            return Err(VerifyError::BadServerCoverage);
        }
    }

    for sp in server_outputs {
        if sp.clients != canonical {
            return Err(VerifyError::InconsistentCanonical(sp.server_id));
        }
    }

    let pub_index: HashMap<ClientId, usize> = client_entries
        .iter()
        .enumerate()
        .map(|(i, (cid, _))| (*cid, i))
        .collect();

    let mut ctxts: Vec<&[KahePoly]> = Vec::with_capacity(canonical.len());
    for cid in canonical {
        let i = *pub_index.get(cid).ok_or(VerifyError::MissingClient(*cid))?;
        let (_, p) = &client_entries[i];
        if p.ctxt.len() != mu {
            return Err(VerifyError::WrongCiphertextLength(*cid));
        }
        ctxts.push(&p.ctxt);
    }
    // An empty set sums to zero.
    let summed_ctxt = if ctxts.is_empty() {
        vec![KahePoly::default(); mu]
    } else {
        Kahe::agg_ctxt_refs(&ctxts)
    };

    let comms: Vec<&Commitment> = canonical
        .iter()
        .map(|cid| {
            let i = pub_index[cid];
            &client_entries[i].1.comm
        })
        .collect();
    let summed_comm = HidingMerkleCommitment::sum_commitment_refs(&comms);

    // Single CS verification per server (μ_cs = κ_kahe packs all components).
    let shares = verify_servers(pp, &summed_comm, server_outputs)?;

    // Lagrange-interpolate the KAHE key from the first `t` servers' summed
    // Shamir shares (R_{q_cs}), then bridge into R_{q_kahe} via centered-rep
    // lift before wrapping as `KaheAggKey`.
    let agg_key = KaheAggKey::from_component(recover_agg_key(pp, server_outputs, &shares)?);

    Ok(Kahe::dec(&pp.kahe, &summed_ctxt, &agg_key))
}

/// Decrypt from client posts that are already summed over the whole client
/// set, so there is no per-client `canonical` to cross-check.
pub fn decrypt_unverified_sum(
    pp: &ProtocolParams,
    summed_ctxt: &[KahePoly],
    summed_comm: &Commitment,
    server_outputs: &[ServerBulletinEntry],
) -> Result<Vec<KahePoly>, VerifyError> {
    if server_outputs.is_empty() {
        return Err(VerifyError::NoServers);
    }
    let t = pp.shamir.t;

    if server_outputs.len() < t {
        return Err(VerifyError::BadServerCoverage);
    }
    if summed_ctxt.len() != pp.kahe.mu_kahe {
        return Err(VerifyError::WrongAggregateLength);
    }
    let mut seen: HashSet<u32> = HashSet::with_capacity(server_outputs.len());
    for sp in server_outputs {
        if (sp.server_id.0 as usize) >= pp.cs.n_servers || !seen.insert(sp.server_id.0) {
            return Err(VerifyError::BadServerCoverage);
        }
    }

    for sp in &server_outputs[1..] {
        if sp.clients != server_outputs[0].clients {
            return Err(VerifyError::InconsistentCanonical(sp.server_id));
        }
    }
    check_unique_clients(&server_outputs[0].clients)?;
    check_client_count(pp, &server_outputs[0].clients)?;

    let shares = verify_servers(pp, summed_comm, server_outputs)?;

    let agg_key = KaheAggKey::from_component(recover_agg_key(pp, server_outputs, &shares)?);

    Ok(Kahe::dec(&pp.kahe, &summed_ctxt.to_vec(), &agg_key))
}
