use std::collections::{HashMap, HashSet};

use chipmunk_code::{HVCPoly, KahePoly};

use crate::bulletin::{ClientPublic, ServerPublic};
use crate::cs::{Cs, HidingMerkleCommitment};
use crate::kahe::{lift_hvc_to_kahe, Kahe, KaheAggKey, KaheScheme};
use crate::sss::ShamirSharing;

use super::message::{ClientId, ServerId};
use super::ProtocolParams;

#[derive(Debug, PartialEq)]
pub enum VerifyError {
    MissingClient(ClientId),
    InvalidServerOpening(usize),
    ShareOpeningMismatch(usize),
    NoServers,
    BadServerCoverage,
    InconsistentCanonical(ServerId),
    InconsistentKappa(ServerId),
}

/// Public verifier (README step 9):
/// 1. sum ciphertexts and commitments over the canonical client set
/// 2. for each server, check `agg_open` opens the summed commitment
/// 3. recover the aggregate KAHE key from any `t` per-server `agg_share` vectors
///    via Shamir interpolation (componentwise)
/// 4. decrypt the summed ciphertext with `Σ sk_j`
pub fn aggregate_and_decrypt(
    pp: &ProtocolParams,
    canonical: &[ClientId],
    publics: &[(ClientId, ClientPublic)],
    server_outputs: &[ServerPublic],
) -> Result<Vec<KahePoly>, VerifyError> {
    if server_outputs.is_empty() {
        return Err(VerifyError::NoServers);
    }
    let kappa_kahe = pp.kahe.kappa_kahe;
    let mu_kahe = pp.kahe.mu_kahe;
    let t = pp.shamir.t;

    if server_outputs.len() < t {
        return Err(VerifyError::BadServerCoverage);
    }
    let mut seen: HashSet<u32> = HashSet::with_capacity(server_outputs.len());
    for sp in server_outputs {
        if (sp.server_id.0 as usize) >= pp.cs.n_servers || !seen.insert(sp.server_id.0) {
            return Err(VerifyError::BadServerCoverage);
        }
        if sp.agg_share.len() != kappa_kahe {
            return Err(VerifyError::InconsistentKappa(sp.server_id));
        }
    }

    for sp in server_outputs {
        if sp.clients != canonical {
            return Err(VerifyError::InconsistentCanonical(sp.server_id));
        }
    }

    let pub_index: HashMap<ClientId, usize> = publics
        .iter()
        .enumerate()
        .map(|(i, (cid, _))| (*cid, i))
        .collect();

    let mut ctxts: Vec<Vec<KahePoly>> = Vec::with_capacity(canonical.len());
    let mut comms = Vec::with_capacity(canonical.len());
    for cid in canonical {
        let i = *pub_index
            .get(cid)
            .ok_or(VerifyError::MissingClient(*cid))?;
        let (_, p) = &publics[i];
        if p.ctxt.len() != mu_kahe {
            return Err(VerifyError::InconsistentKappa(server_outputs[0].server_id));
        }
        ctxts.push(p.ctxt.clone());
        comms.push(p.comm.clone());
    }

    let summed_ctxt = Kahe::agg_ctxt(&ctxts);
    let summed_comm = HidingMerkleCommitment::sum_commitments(&comms);

    // Single CS verification per server (μ_cs = κ_kahe packs all components).
    for (i, sp) in server_outputs.iter().enumerate() {
        if !HidingMerkleCommitment::verify(&pp.cs, &summed_comm, &sp.agg_open) {
            return Err(VerifyError::InvalidServerOpening(i));
        }
        if sp.agg_share.as_slice() != sp.agg_open.s() {
            return Err(VerifyError::ShareOpeningMismatch(i));
        }
    }

    // Lagrange-interpolate each KAHE key component from the first `t`
    // servers' summed Shamir shares (R_{q_cs}), then bridge each into
    // R_{q_kahe} via centered-rep lift before wrapping as `KaheAggKey`.
    let recovered_components: Vec<KahePoly> = (0..kappa_kahe)
        .map(|k| {
            let samples: Vec<(usize, HVCPoly)> = server_outputs
                .iter()
                .take(t)
                .map(|sp| (sp.server_id.0 as usize, sp.agg_share[k]))
                .collect();
            let recovered_hvc = ShamirSharing::recover(&pp.shamir, &samples);
            lift_hvc_to_kahe(&recovered_hvc)
        })
        .collect();
    let agg_key = KaheAggKey::from_components(recovered_components);

    Ok(Kahe::dec(&pp.kahe, &summed_ctxt, &agg_key))
}
