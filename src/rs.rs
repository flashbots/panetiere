//! Systematic Reed–Solomon for sharding ingress ciphertexts. It codes the two
//! KAHE prime channels independently and keeps every polynomial in NTT form.

use crate::rings::KAHE_RNS_MODULI;
use crate::{RsNTTPoly, N};
use rayon::prelude::*;

use crate::bulletin::rs_poly_packed_len;

pub type Share = Vec<RsNTTPoly>;

/// Append a share's NTT-domain coefficients, little-endian, [`rs_poly_packed_len`]
/// per poly.
pub fn pack_share(share: &[RsNTTPoly], out: &mut Vec<u8>) {
    out.reserve(share.len() * rs_poly_packed_len());
    for limb in 0..2 {
        for p in share {
            for &c in &p.residues()[limb] {
                debug_assert!(c < KAHE_RNS_MODULI[limb]);
                let bytes = c.to_le_bytes();
                out.extend_from_slice(&bytes[..3]);
            }
        }
    }
}

/// Inverse of [`pack_share`], requiring the exact length for `n_polys`.
pub fn unpack_share(bytes: &[u8], n_polys: usize) -> Option<Share> {
    if bytes.len() != n_polys.checked_mul(rs_poly_packed_len())? {
        return None;
    }
    let mut polys = vec![[[0u32; N]; 2]; n_polys];
    let mut chunks = bytes.chunks_exact(3);
    for limb in 0..2 {
        for poly in &mut polys {
            for c in &mut poly[limb] {
                let b = chunks.next()?;
                *c = u32::from_le_bytes([b[0], b[1], b[2], 0]);
                if *c >= KAHE_RNS_MODULI[limb] {
                    return None;
                }
            }
        }
    }
    if !chunks.remainder().is_empty() {
        return None;
    }
    polys.into_iter().map(RsNTTPoly::from_residues).collect()
}

#[derive(Debug, PartialEq, Eq)]
pub enum RsError {
    NotEnoughShares,
    DuplicateIndex,
    IndexOutOfRange,
    ShareLenMismatch,
}

#[derive(Clone, Debug)]
pub struct RsParams {
    /// Blocks needed to reconstruct.
    pub k: usize,
    /// Coded shares emitted, one per node.
    pub n: usize,
}

impl RsParams {
    pub fn new(k: usize, n: usize) -> Self {
        assert!(k >= 1, "k must be ≥ 1");
        assert!(n >= k, "n must be ≥ k");
        assert!(
            n < KAHE_RNS_MODULI.into_iter().min().unwrap() as usize,
            "points 1..=n must be distinct in every RNS channel"
        );
        Self { k, n }
    }

    /// Polys per block; the final block is zero-padded up to it.
    pub fn block_len(&self, ctxt_len: usize) -> usize {
        ctxt_len.div_ceil(self.k)
    }
}

fn pow_channel(base: u32, mut exp: u32, q: u32) -> u32 {
    let mut acc = 1u64;
    let mut b = base as u64;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = acc * b % q as u64;
        }
        b = b * b % q as u64;
        exp >>= 1;
    }
    acc as u32
}

fn lagrange_at(xs: &[u32], x: u32) -> Vec<[u32; 2]> {
    xs.iter()
        .enumerate()
        .map(|(i, &xi)| {
            core::array::from_fn(|limb| {
                let q = KAHE_RNS_MODULI[limb];
                let mut num = 1u64;
                let mut den = 1u64;
                for (j, &xj) in xs.iter().enumerate() {
                    if j == i {
                        continue;
                    }
                    num = num * ((x + q - xj) % q) as u64 % q as u64;
                    den = den * ((xi + q - xj) % q) as u64 % q as u64;
                }
                let inv = pow_channel(den as u32, q - 2, q);
                (num * inv as u64 % q as u64) as u32
            })
        })
        .collect()
}

fn combine(blocks: &[&[RsNTTPoly]], coefs: &[[u32; 2]], block_len: usize) -> Vec<RsNTTPoly> {
    (0..block_len)
        .into_par_iter()
        .map(|pos| {
            let mut out = [[0u32; N]; 2];
            for limb in 0..2 {
                let q = KAHE_RNS_MODULI[limb] as u64;
                for (block, coef) in blocks.iter().zip(coefs) {
                    let w = coef[limb] as u64;
                    let src = &block[pos].residues()[limb];
                    for i in 0..N {
                        out[limb][i] = ((out[limb][i] as u64 + src[i] as u64 * w) % q) as u32;
                    }
                }
            }
            RsNTTPoly::from_residues(out).unwrap()
        })
        .collect()
}

fn data_points(k: usize) -> Vec<u32> {
    (1..=k as u32).collect()
}

pub struct Rs;

impl Rs {
    /// Split `ctxt` into `k` zero-padded blocks.
    pub fn split_blocks(params: &RsParams, ctxt: &[RsNTTPoly]) -> Vec<Vec<RsNTTPoly>> {
        let bl = params.block_len(ctxt.len());
        (0..params.k)
            .map(|b| {
                let start = (b * bl).min(ctxt.len());
                let end = ((b + 1) * bl).min(ctxt.len());
                let mut v = ctxt[start..end].to_vec();
                v.resize(bl, RsNTTPoly::default());
                v
            })
            .collect()
    }

    /// One share per node. Shares `0..k` alias the blocks, so only the `n−k`
    /// parity evaluations cost arithmetic — `(n−k)·ctxt_len·N` modmuls,
    /// independent of `k`.
    pub fn encode(params: &RsParams, ctxt: &[RsNTTPoly]) -> Vec<Share> {
        let mut out = Self::split_blocks(params, ctxt);
        let bl = params.block_len(ctxt.len());
        let refs: Vec<&[RsNTTPoly]> = out.iter().map(Vec::as_slice).collect();
        let parity: Vec<Share> = (params.k..params.n)
            .map(|j| {
                combine(
                    &refs,
                    &lagrange_at(&data_points(params.k), (j + 1) as _),
                    bl,
                )
            })
            .collect();
        drop(refs);
        out.extend(parity);
        out
    }

    /// Interpolate the `k` blocks back from any `k` `(node_index, share)`
    /// samples and flatten to `ctxt_len` polys. Extra samples are ignored.
    pub fn reconstruct(
        params: &RsParams,
        ctxt_len: usize,
        samples: &[(usize, &[RsNTTPoly])],
    ) -> Result<Vec<RsNTTPoly>, RsError> {
        let k = params.k;
        if samples.len() < k {
            return Err(RsError::NotEnoughShares);
        }
        let used = &samples[..k];
        let bl = params.block_len(ctxt_len);
        for (i, (idx, share)) in used.iter().enumerate() {
            if *idx >= params.n {
                return Err(RsError::IndexOutOfRange);
            }
            if share.len() != bl {
                return Err(RsError::ShareLenMismatch);
            }
            if used[..i].iter().any(|(o, _)| o == idx) {
                return Err(RsError::DuplicateIndex);
            }
        }

        let mut flat: Vec<RsNTTPoly> = if used.iter().enumerate().all(|(b, (idx, _))| *idx == b) {
            used.iter().flat_map(|(_, s)| s.iter().copied()).collect()
        } else {
            let xs: Vec<u32> = used.iter().map(|(idx, _)| (*idx + 1) as u32).collect();
            let shares: Vec<&[RsNTTPoly]> = used.iter().map(|(_, s)| *s).collect();
            (0..k)
                .flat_map(|b| combine(&shares, &lagrange_at(&xs, (b + 1) as _), bl))
                .collect()
        };
        flat.truncate(ctxt_len);
        Ok(flat)
    }

    /// Positional sum of one node's shares across clients — the lane round.
    pub fn sum_shares(shares: &[&[RsNTTPoly]]) -> Share {
        if shares.is_empty() {
            return Vec::new();
        }
        let len = shares[0].len();
        (0..len)
            .into_par_iter()
            .map(|pos| {
                let mut acc = RsNTTPoly::default();
                for s in shares {
                    acc += s[pos];
                }
                acc
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KahePoly;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn rand_ctxt(rng: &mut ChaCha20Rng, len: usize) -> Vec<RsNTTPoly> {
        (0..len)
            .map(|_| RsNTTPoly::from_kahe(&KahePoly::rand_poly(rng)))
            .collect()
    }

    #[test]
    fn systematic_shares_alias_the_blocks() {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        let params = RsParams::new(4, 6);
        let ct = rand_ctxt(&mut rng, 9);
        let shares = Rs::encode(&params, &ct);
        let blocks = Rs::split_blocks(&params, &ct);
        assert_eq!(shares.len(), 6);
        for b in 0..4 {
            assert_eq!(shares[b], blocks[b]);
        }
    }

    #[test]
    fn any_k_of_n_reconstruct() {
        let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
        for (k, n, len) in [(1usize, 3usize, 5usize), (4, 6, 9), (3, 8, 12), (5, 5, 5)] {
            let params = RsParams::new(k, n);
            let ct = rand_ctxt(&mut rng, len);
            let shares = Rs::encode(&params, &ct);
            for start in 0..=(n - k) {
                let samples: Vec<(usize, &[RsNTTPoly])> = (start..start + k)
                    .map(|j| (j, shares[j].as_slice()))
                    .collect();
                let back = Rs::reconstruct(&params, len, &samples).unwrap();
                assert_eq!(back, ct, "k={k} n={n} len={len} start={start}");
            }
        }
    }

    #[test]
    fn reconstruct_rejects_bad_sample_sets() {
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        let params = RsParams::new(3, 5);
        let ct = rand_ctxt(&mut rng, 7);
        let shares = Rs::encode(&params, &ct);
        let one: Vec<(usize, &[RsNTTPoly])> = vec![(0, shares[0].as_slice())];
        assert_eq!(
            Rs::reconstruct(&params, 7, &one),
            Err(RsError::NotEnoughShares)
        );
        let dup: Vec<(usize, &[RsNTTPoly])> = vec![
            (1, shares[1].as_slice()),
            (1, shares[1].as_slice()),
            (2, shares[2].as_slice()),
        ];
        assert_eq!(
            Rs::reconstruct(&params, 7, &dup),
            Err(RsError::DuplicateIndex)
        );
        let stub = [RsNTTPoly::default()];
        let short: Vec<(usize, &[RsNTTPoly])> = vec![
            (0, &stub[..]),
            (1, shares[1].as_slice()),
            (2, shares[2].as_slice()),
        ];
        assert_eq!(
            Rs::reconstruct(&params, 7, &short),
            Err(RsError::ShareLenMismatch)
        );
    }

    #[test]
    fn share_wire_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([5u8; 32]);
        let share = rand_ctxt(&mut rng, 4);
        let mut bytes = Vec::new();
        pack_share(&share, &mut bytes);
        assert_eq!(bytes.len(), 4 * rs_poly_packed_len());
        assert_eq!(unpack_share(&bytes, 4).unwrap(), share);
        assert!(unpack_share(&bytes, 3).is_none());
        assert!(unpack_share(&bytes[..bytes.len() - 1], 4).is_none());
    }

    #[test]
    fn share_wire_rejects_noncanonical_residue() {
        let mut rng = ChaCha20Rng::from_seed([15u8; 32]);
        let share = rand_ctxt(&mut rng, 1);
        let mut bytes = Vec::new();
        pack_share(&share, &mut bytes);
        bytes[..3].copy_from_slice(&KAHE_RNS_MODULI[0].to_le_bytes()[..3]);
        assert!(unpack_share(&bytes, 1).is_none());
    }

    /// The property the whole mode rests on: coding commutes with aggregation,
    /// so summing lane `j` across clients gives lane `j` of the summed
    /// ciphertext modulo `q_kahe`.
    #[test]
    fn summing_shares_reconstructs_the_integer_sum() {
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let params = RsParams::new(4, 7);
        let len = 11;
        let cts: Vec<Vec<KahePoly>> = (0..6)
            .map(|_| (0..len).map(|_| KahePoly::rand_poly(&mut rng)).collect())
            .collect();
        let embedded: Vec<Vec<RsNTTPoly>> = cts
            .iter()
            .map(|c| c.iter().map(RsNTTPoly::from_kahe).collect())
            .collect();
        let per_client: Vec<Vec<Share>> = embedded.iter().map(|c| Rs::encode(&params, c)).collect();

        let lane_sums: Vec<Share> = (0..params.n)
            .map(|j| {
                let col: Vec<&[RsNTTPoly]> = per_client.iter().map(|s| s[j].as_slice()).collect();
                Rs::sum_shares(&col)
            })
            .collect();

        let samples: Vec<(usize, &[RsNTTPoly])> = (3..3 + params.k)
            .map(|j| (j, lane_sums[j].as_slice()))
            .collect();
        let got = Rs::reconstruct(&params, len, &samples).unwrap();

        for pos in 0..len {
            let recovered = got[pos].to_centered_coeffs();
            for i in 0..N {
                let want: i64 = cts
                    .iter()
                    .map(|c| {
                        let mut p = c[pos];
                        p.normalize();
                        p.coeffs()[i]
                    })
                    .sum();
                let want = crate::rings::center_i64(want, crate::KAHE_MODULUS);
                assert_eq!(recovered[i], want, "pos {pos} coeff {i}");
            }
        }
    }
}
