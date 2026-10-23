//! Systematic Reed–Solomon for sharding ingress ciphertexts, over the KAHE
//! ring's prime modulus. Every polynomial stays in NTT form.

use crate::{RsNTTPoly, KAHE_MODULUS, N};
use rayon::prelude::*;

use crate::bulletin::rs_poly_packed_len;

const Q: u64 = KAHE_MODULUS as u64;

/// Bytes per NTT-domain coefficient on the wire.
pub(crate) const RS_COEFF_BYTES: usize = 6;
const _: () = assert!(Q < 1 << (8 * RS_COEFF_BYTES));

pub type Share = Vec<RsNTTPoly>;

/// Append a share's NTT-domain coefficients, little-endian, [`rs_poly_packed_len`]
/// per poly.
pub fn pack_share(share: &[RsNTTPoly], out: &mut Vec<u8>) {
    out.reserve(share.len() * rs_poly_packed_len());
    for p in share {
        for &c in p.coeffs() {
            debug_assert!(c < Q);
            out.extend_from_slice(&c.to_le_bytes()[..RS_COEFF_BYTES]);
        }
    }
}

/// Inverse of [`pack_share`], requiring the exact length for `n_polys`.
pub fn unpack_share(bytes: &[u8], n_polys: usize) -> Option<Share> {
    if bytes.len() != n_polys.checked_mul(rs_poly_packed_len())? {
        return None;
    }
    let mut out = Vec::with_capacity(n_polys);
    for chunk in bytes.chunks_exact(rs_poly_packed_len()) {
        let mut coeffs = [0u64; N];
        for (c, b) in coeffs.iter_mut().zip(chunk.chunks_exact(RS_COEFF_BYTES)) {
            let mut word = [0u8; 8];
            word[..RS_COEFF_BYTES].copy_from_slice(b);
            *c = u64::from_le_bytes(word);
        }
        out.push(RsNTTPoly::from_canonical(coeffs)?);
    }
    Some(out)
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
        assert!((n as u64) < Q, "points 1..=n must be distinct mod q");
        Self { k, n }
    }

    /// Polys per block; the final block is zero-padded up to it.
    pub fn block_len(&self, ctxt_len: usize) -> usize {
        ctxt_len.div_ceil(self.k)
    }
}

#[inline]
fn mul_q(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % Q as u128) as u64
}

#[inline]
fn sub_q(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + Q - b
    }
}

fn pow_q(base: u64, mut exp: u64) -> u64 {
    let mut acc = 1u64;
    let mut b = base % Q;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul_q(acc, b);
        }
        b = mul_q(b, b);
        exp >>= 1;
    }
    acc
}

/// `q` is prime, so Fermat gives the inverse.
#[inline]
fn inv_q(x: u64) -> u64 {
    debug_assert!(x % Q != 0, "inverse of zero");
    pow_q(x, Q - 2)
}

/// Shoup precomputation `⌊w·2^64/q⌋`, valid because `q < 2^62`.
#[inline]
fn shoup_precomp(w: u64) -> u64 {
    (((w as u128) << 64) / Q as u128) as u64
}

/// `x·w mod q` given `w_shoup = ⌊w·2^64/q⌋`, for `x, w < q`.
#[inline(always)]
fn shoup_mul(x: u64, w: u64, w_shoup: u64) -> u64 {
    let q_hat = ((x as u128 * w_shoup as u128) >> 64) as u64;
    let r = x.wrapping_mul(w).wrapping_sub(q_hat.wrapping_mul(Q));
    if r >= Q {
        r - Q
    } else {
        r
    }
}

/// Lagrange basis for the points `xs`, evaluated at `x`.
fn lagrange_at(xs: &[u64], x: u64) -> Vec<u64> {
    (0..xs.len())
        .map(|i| {
            let mut num = 1u64;
            let mut den = 1u64;
            for (j, &xj) in xs.iter().enumerate() {
                if j == i {
                    continue;
                }
                num = mul_q(num, sub_q(x, xj));
                den = mul_q(den, sub_q(xs[i], xj));
            }
            mul_q(num, inv_q(den))
        })
        .collect()
}

/// `Σ_b coefs[b] · blocks[b]`, positionally over `block_len` polys. Inputs are
/// canonical in `[0, q)`, so this is a bare Shoup multiply-accumulate and
/// `acc + p < 2q < 2^63` never overflows.
fn combine(blocks: &[&[RsNTTPoly]], coefs: &[u64], block_len: usize) -> Vec<RsNTTPoly> {
    let shoup: Vec<u64> = coefs.iter().map(|&w| shoup_precomp(w)).collect();
    (0..block_len)
        .into_par_iter()
        .map(|pos| {
            let mut acc = [0u64; N];
            for (b, blk) in blocks.iter().enumerate() {
                let w = coefs[b];
                if w == 0 {
                    continue;
                }
                let ws = shoup[b];
                let src = blk[pos].coeffs();
                for i in 0..N {
                    let s = acc[i] + shoup_mul(src[i], w, ws);
                    acc[i] = if s >= Q { s - Q } else { s };
                }
            }
            // Each step reduces back under q, so the sum stays canonical.
            RsNTTPoly::from_canonical(acc).expect("combine keeps coefficients canonical")
        })
        .collect()
}

fn data_points(k: usize) -> Vec<u64> {
    (1..=k as u64).collect()
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
                    &lagrange_at(&data_points(params.k), (j + 1) as u64),
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
    /// Performs erasure recovery, not error correction or codeword validation.
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
            let xs: Vec<u64> = used.iter().map(|(idx, _)| (*idx + 1) as u64).collect();
            let shares: Vec<&[RsNTTPoly]> = used.iter().map(|(_, s)| *s).collect();
            (0..k)
                .flat_map(|b| combine(&shares, &lagrange_at(&xs, (b + 1) as u64), bl))
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
        bytes[..RS_COEFF_BYTES].copy_from_slice(&Q.to_le_bytes()[..RS_COEFF_BYTES]);
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
