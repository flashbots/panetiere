use std::collections::HashMap;

use crate::bulletin::{ClientPublic, ServerPublic};
use crate::cs::{Commitment, Cs, HidingMerkleCommitment};
use crate::kahe::{Kahe, RingOtp};
use chipmunk_code::HVCPoly;

use super::message::ClientId;

#[derive(Debug, PartialEq)]
pub enum VerifyError {
    MissingClient(ClientId),
    InvalidServerOpening(usize),
    ShareOpeningMismatch(usize),
    NoServers,
}

/// Public verifier (README step 9):
/// 1. sum ciphertexts and commitments over the canonical client set
/// 2. for each server, check its `agg_open` opens the summed commitment
/// 3. recover the aggregate KAHE key from the per-server `agg_share`s
/// 4. decrypt the summed ciphertext
pub fn aggregate_and_decrypt(
    pp: &<HidingMerkleCommitment as Cs>::Params,
    canonical: &[ClientId],
    publics: &[(ClientId, ClientPublic)],
    server_outputs: &[ServerPublic],
) -> Result<HVCPoly, VerifyError> {
    if server_outputs.is_empty() {
        return Err(VerifyError::NoServers);
    }

    // Index the public-bulletin entries by ClientId so the per-canonical lookup
    // is O(1); avoids O(N · |publics|) linear scans on large client sets.
    let pub_index: HashMap<ClientId, usize> = publics
        .iter()
        .enumerate()
        .map(|(i, (cid, _))| (*cid, i))
        .collect();

    let mut ctxts = Vec::with_capacity(canonical.len());
    let mut comms: Vec<Commitment> = Vec::with_capacity(canonical.len());
    for cid in canonical {
        let i = *pub_index
            .get(cid)
            .ok_or(VerifyError::MissingClient(*cid))?;
        let (_, p) = &publics[i];
        ctxts.push(p.ctxt);
        comms.push(p.comm.clone());
    }

    let summed_ctxt = RingOtp::agg_ctxt(&ctxts);
    let summed_comm = HidingMerkleCommitment::sum_commitments(&comms);

    for (i, sp) in server_outputs.iter().enumerate() {
        if !HidingMerkleCommitment::verify(pp, &summed_comm, &sp.agg_open) {
            return Err(VerifyError::InvalidServerOpening(i));
        }
        // The aggregated share and the `s` inside the aggregated opening are
        // both pointwise sums of the same per-client `share_i = opening_i.s`.
        // They must match — otherwise a server has tampered with one of them.
        if sp.agg_share != sp.agg_open.s {
            return Err(VerifyError::ShareOpeningMismatch(i));
        }
    }

    let agg_key = RingOtp::agg_key(
        &server_outputs
            .iter()
            .map(|s| s.agg_share)
            .collect::<Vec<_>>(),
    );
    Ok(RingOtp::dec(&summed_ctxt, &agg_key))
}
