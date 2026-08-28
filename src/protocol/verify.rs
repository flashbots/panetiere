use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::{CsPoly, KahePoly};
use rayon::prelude::*;

use crate::bulletin::{
    ClientBulletinEntry, RsClientBulletinEntry, RsNodeBulletinEntry, ServerBulletinEntry,
};
use crate::cs::{Commitment, Cs, HidingMerkleCommitment};
use crate::hvc_sum::sum_hvc_polys;
use crate::kahe::{lift_cs_to_kahe, Kahe, KaheAggKey, KaheScheme};
use crate::rs::{Rs, RsError};
use crate::share_commitment::verify_aggregated;
use crate::sig;
use crate::sss::{ShamirSharing, SssError};
use crate::DgtNTTPoly;
use chipmunk_code::HVCPoly;

use super::ProtocolParams;
use super::{ClientId, NodeId, ServerId, SessionId};

#[derive(Debug, PartialEq)]
pub enum VerifyError {
    MissingClient(ClientId),
    InvalidServerOpening(usize),
    ShareOpeningMismatch(usize),
    NoServers,
    BadServerCoverage,
    InconsistentCanonical(ServerId),
    InconsistentCiphertextLen(ServerId),
    AnonymitySetTooSmall {
        got: usize,
        min: usize,
    },
    ShareRecovery(SssError),
    /// RS mode only.
    NotRsMode,
    NotEnoughNodes,
    BadNodeCoverage,
    InconsistentNodeCanonical(NodeId),
    BadSignature(ClientId),
    Reconstruct(RsError),
    /// These lanes' posts do not open the summed signed roots.
    LaneOpeningFailed(Vec<NodeId>),
    /// Reconstruction exceeded `ρ_max·q_kahe/2`, so it is not a sum of honest
    /// ciphertexts.
    CiphertextOutOfRange,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct VerifyTimings {
    pub agg_ctxt_us: f64,
    pub sum_comm_us: f64,
    pub opening_verify_us: f64,
    pub interpolation_us: f64,
    pub kahe_dec_us: f64,
}

fn verify_one_server(
    pp: &ProtocolParams,
    summed_comm: &Commitment,
    i: usize,
    sp: &ServerBulletinEntry,
) -> Result<(), VerifyError> {
    if !HidingMerkleCommitment::verify(&pp.cs, summed_comm, &sp.agg_open) {
        return Err(VerifyError::InvalidServerOpening(i));
    }
    if sp.agg_open.path_index != sp.server_id.0 as usize {
        return Err(VerifyError::InvalidServerOpening(i));
    }
    if sp.agg_share != sp.agg_open.s()[0] {
        return Err(VerifyError::ShareOpeningMismatch(i));
    }
    Ok(())
}

fn recover_agg_key(
    pp: &ProtocolParams,
    server_outputs: &[ServerBulletinEntry],
) -> Result<KahePoly, VerifyError> {
    let samples: Vec<(usize, CsPoly)> = server_outputs
        .iter()
        .take(pp.shamir.t)
        .map(|sp| (sp.server_id.0 as usize, sp.agg_share))
        .collect();
    let recovered_cs =
        ShamirSharing::recover(&pp.shamir, &samples).map_err(VerifyError::ShareRecovery)?;
    Ok(lift_cs_to_kahe(&recovered_cs))
}

pub(super) fn check_anonymity_floor(
    pp: &ProtocolParams,
    clients: &[ClientId],
) -> Result<(), VerifyError> {
    let got = clients.iter().collect::<HashSet<_>>().len();
    if got < pp.min_clients {
        return Err(VerifyError::AnonymitySetTooSmall {
            got,
            min: pp.min_clients,
        });
    }
    Ok(())
}

pub fn aggregate_and_decrypt(
    pp: &ProtocolParams,
    canonical: &[ClientId],
    client_entries: &[(ClientId, ClientBulletinEntry)],
    server_outputs: &[ServerBulletinEntry],
) -> Result<Vec<KahePoly>, VerifyError> {
    aggregate_and_decrypt_timed(pp, canonical, client_entries, server_outputs).map(|(m, _)| m)
}

pub fn aggregate_and_decrypt_timed(
    pp: &ProtocolParams,
    canonical: &[ClientId],
    client_entries: &[(ClientId, ClientBulletinEntry)],
    server_outputs: &[ServerBulletinEntry],
) -> Result<(Vec<KahePoly>, VerifyTimings), VerifyError> {
    if server_outputs.is_empty() {
        return Err(VerifyError::NoServers);
    }
    check_anonymity_floor(pp, canonical)?;
    // A ciphertext may be shorter than μ; all clients must agree on the exact
    // length, since agg_ctxt sums positionally.
    let max_ctxt_len = pp.kahe.mu_kahe;
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

    let mut tt = VerifyTimings::default();
    let now = Instant::now();
    let mut ctxts: Vec<&[KahePoly]> = Vec::with_capacity(canonical.len());
    let mut ctxt_len: Option<usize> = None;
    for cid in canonical {
        let i = *pub_index.get(cid).ok_or(VerifyError::MissingClient(*cid))?;
        let (_, p) = &client_entries[i];
        let expected = *ctxt_len.get_or_insert(p.ctxt.len());
        if p.ctxt.len() != expected || p.ctxt.len() > max_ctxt_len {
            return Err(VerifyError::InconsistentCiphertextLen(
                server_outputs[0].server_id,
            ));
        }
        ctxts.push(&p.ctxt);
    }
    let summed_ctxt = Kahe::agg_ctxt_refs(&ctxts);
    tt.agg_ctxt_us = now.elapsed().as_secs_f64() * 1e6;

    let now = Instant::now();
    let comms: Vec<&Commitment> = canonical
        .iter()
        .map(|cid| {
            let i = pub_index[cid];
            &client_entries[i].1.comm
        })
        .collect();
    let summed_comm = HidingMerkleCommitment::sum_commitment_refs(&comms);
    tt.sum_comm_us = now.elapsed().as_secs_f64() * 1e6;

    // Single CS verification per server (μ_cs = κ_kahe packs all components).
    let now = Instant::now();
    let results: Vec<Result<(), VerifyError>> = server_outputs
        .par_iter()
        .enumerate()
        .map(|(i, sp)| verify_one_server(pp, &summed_comm, i, sp))
        .collect();
    for r in results {
        r?;
    }
    tt.opening_verify_us = now.elapsed().as_secs_f64() * 1e6;

    // Lagrange-interpolate the KAHE key from the first `t` servers' summed
    // Shamir shares (R_{q_cs}), then bridge into R_{q_kahe} via centered-rep
    // lift before wrapping as `KaheAggKey`.
    let now = Instant::now();
    let agg_key = KaheAggKey::from_component(recover_agg_key(pp, server_outputs)?);
    tt.interpolation_us = now.elapsed().as_secs_f64() * 1e6;

    let now = Instant::now();
    let m = Kahe::dec(&pp.kahe, &summed_ctxt, &agg_key);
    tt.kahe_dec_us = now.elapsed().as_secs_f64() * 1e6;

    Ok((m, tt))
}

#[derive(Clone, Copy, Default, Debug)]
pub struct RsVerifyTimings {
    pub sig_verify_us: f64,
    pub sum_comm_us: f64,
    pub opening_verify_us: f64,
    pub interpolation_us: f64,
    /// Per-lane aggregated-opening verification against the summed roots.
    pub lane_open_us: f64,
    pub reconstruct_us: f64,
    /// Re-encode the reconstruction and compare every reporting lane's post.
    pub crosscheck_us: f64,
    pub kahe_dec_us: f64,
}

pub fn aggregate_and_decrypt_rs(
    pp: &ProtocolParams,
    sid: &SessionId,
    canonical: &[ClientId],
    client_entries: &[(ClientId, RsClientBulletinEntry)],
    server_outputs: &[ServerBulletinEntry],
    node_outputs: &[RsNodeBulletinEntry],
) -> Result<(Vec<KahePoly>, RsVerifyTimings), VerifyError> {
    let rs = pp.rs.as_ref().ok_or(VerifyError::NotRsMode)?;
    let scp = pp.share_comm.as_ref().ok_or(VerifyError::NotRsMode)?;
    if server_outputs.is_empty() {
        return Err(VerifyError::NoServers);
    }
    check_anonymity_floor(pp, canonical)?;
    if server_outputs.len() < pp.shamir.t {
        return Err(VerifyError::BadServerCoverage);
    }
    if node_outputs.len() < rs.k {
        return Err(VerifyError::NotEnoughNodes);
    }

    let mut seen: HashSet<u32> = HashSet::with_capacity(server_outputs.len());
    for sp in server_outputs {
        if (sp.server_id.0 as usize) >= pp.cs.n_servers || !seen.insert(sp.server_id.0) {
            return Err(VerifyError::BadServerCoverage);
        }
        if sp.clients != canonical {
            return Err(VerifyError::InconsistentCanonical(sp.server_id));
        }
    }
    // `Σ sk` and `Σ ct` must cover exactly the same set or decryption yields
    // noise, so the lanes are held to the same roster as the servers.
    let mut seen_nodes: HashSet<u32> = HashSet::with_capacity(node_outputs.len());
    for np in node_outputs {
        if (np.node_id.0 as usize) >= rs.n || !seen_nodes.insert(np.node_id.0) {
            return Err(VerifyError::BadNodeCoverage);
        }
        if np.clients != canonical {
            return Err(VerifyError::InconsistentNodeCanonical(np.node_id));
        }
    }

    let pub_index: HashMap<ClientId, usize> = client_entries
        .iter()
        .enumerate()
        .map(|(i, (cid, _))| (*cid, i))
        .collect();
    let entries: Vec<&RsClientBulletinEntry> = canonical
        .iter()
        .map(|cid| {
            pub_index
                .get(cid)
                .map(|i| &client_entries[*i].1)
                .ok_or(VerifyError::MissingClient(*cid))
        })
        .collect::<Result<_, _>>()?;

    let mut tt = RsVerifyTimings::default();

    let now = Instant::now();
    let sig_results: Vec<Result<(), VerifyError>> = canonical
        .par_iter()
        .zip(entries.par_iter())
        .map(|(cid, e)| {
            let vk = sig::VerifyingKey::from_sec1_bytes(&e.pubkey)
                .map_err(|_| VerifyError::BadSignature(*cid))?;
            vk.verify(
                &RsClientBulletinEntry::signing_bytes(sid, *cid, &e.comm, &e.share_root),
                &e.sig,
            )
            .map_err(|_| VerifyError::BadSignature(*cid))
        })
        .collect();
    for r in sig_results {
        r?;
    }
    tt.sig_verify_us = now.elapsed().as_secs_f64() * 1e6;

    let now = Instant::now();
    let comms: Vec<Commitment> = entries.iter().map(|e| e.comm.clone()).collect();
    let summed_comm = HidingMerkleCommitment::sum_commitments(&comms);
    let root_refs: Vec<&HVCPoly> = entries.iter().map(|e| &e.share_root).collect();
    let summed_root = sum_hvc_polys(&root_refs);
    tt.sum_comm_us = now.elapsed().as_secs_f64() * 1e6;

    let now = Instant::now();
    let results: Vec<Result<(), VerifyError>> = server_outputs
        .par_iter()
        .enumerate()
        .map(|(i, sp)| verify_one_server(pp, &summed_comm, i, sp))
        .collect();
    for r in results {
        r?;
    }
    tt.opening_verify_us = now.elapsed().as_secs_f64() * 1e6;

    let now = Instant::now();
    let agg_key = KaheAggKey::from_component(recover_agg_key(pp, server_outputs)?);
    tt.interpolation_us = now.elapsed().as_secs_f64() * 1e6;

    // Each lane's post is its own proof: `A·share_sum` must open the summed
    // signed roots at that lane's position. Systematic sums are additionally
    // norm-gated — that shortness is what makes the Ajtai layer binding for
    // them. Parity sums are full-range and get no gate; their binding comes
    // from the reconstruction bound (an in-bound forgery is a short kernel
    // vector of a rotated `A`, i.e. SIS) plus the re-encode check below.
    let now = Instant::now();
    let bad: Vec<NodeId> = node_outputs
        .par_iter()
        .filter(|np| {
            let systematic_ok = (np.node_id.0 as usize) >= rs.k
                || crate::digest::centered_within_bound(scp.rho_max, &np.share_sum).is_some();
            np.agg_open.lane_index != np.node_id.0 as usize
                || !systematic_ok
                || !verify_aggregated(scp, &summed_root, &np.share_sum, &np.agg_open)
        })
        .map(|np| np.node_id)
        .collect();
    if !bad.is_empty() {
        return Err(VerifyError::LaneOpeningFailed(bad));
    }
    tt.lane_open_us = now.elapsed().as_secs_f64() * 1e6;

    // Reconstruct systematic-first: those sums are individually pinned, and a
    // full systematic set makes reconstruction a concatenation.
    let ell = pp.kahe.mu_kahe;
    let now = Instant::now();
    let mut ordered: Vec<&RsNodeBulletinEntry> = node_outputs.iter().collect();
    ordered.sort_by_key(|np| ((np.node_id.0 as usize) >= rs.k, np.node_id.0));
    let samples: Vec<(usize, &[DgtNTTPoly])> = ordered
        .iter()
        .map(|np| (np.node_id.0 as usize, np.share_sum.as_slice()))
        .collect();
    let summed_ntt = Rs::reconstruct(rs, ell, &samples).map_err(VerifyError::Reconstruct)?;
    tt.reconstruct_us = now.elapsed().as_secs_f64() * 1e6;

    let now = Instant::now();
    let centered = crate::digest::centered_within_bound(scp.rho_max, &summed_ntt)
        .ok_or(VerifyError::CiphertextOutOfRange)?;
    // The reconstruction is in-bound, so it is the committed codeword; any
    // reporting lane whose post disagrees with its re-encoding lied.
    let expected = Rs::encode(rs, &summed_ntt);
    let liars: Vec<NodeId> = node_outputs
        .iter()
        .filter(|np| expected[np.node_id.0 as usize] != np.share_sum)
        .map(|np| np.node_id)
        .collect();
    if !liars.is_empty() {
        return Err(VerifyError::LaneOpeningFailed(liars));
    }
    tt.crosscheck_us = now.elapsed().as_secs_f64() * 1e6;

    let now = Instant::now();
    let summed_ctxt = crate::digest::to_kahe(&centered);
    let plain = Kahe::dec(&pp.kahe, &summed_ctxt, &agg_key);
    tt.kahe_dec_us = now.elapsed().as_secs_f64() * 1e6;

    Ok((plain, tt))
}

/// Aggregated-flow verifier: ciphertext and commitment are already summed over
/// the whole client set, so there is no per-client `canonical` to cross-check.
pub fn decrypt_aggregate(
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
    // Shorter than μ is allowed; μ bounds the length.
    if summed_ctxt.is_empty() || summed_ctxt.len() > pp.kahe.mu_kahe {
        return Err(VerifyError::InconsistentCiphertextLen(
            server_outputs[0].server_id,
        ));
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
    check_anonymity_floor(pp, &server_outputs[0].clients)?;

    let results: Vec<Result<(), VerifyError>> = server_outputs
        .par_iter()
        .enumerate()
        .map(|(i, sp)| verify_one_server(pp, summed_comm, i, sp))
        .collect();
    for r in results {
        r?;
    }

    let agg_key = KaheAggKey::from_component(recover_agg_key(pp, server_outputs)?);

    Ok(Kahe::dec(&pp.kahe, &summed_ctxt.to_vec(), &agg_key))
}
