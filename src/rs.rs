//! Systematic Reed–Solomon over the digest ring, for sharding the ingress
//! ciphertext across nodes.

use chipmunk_code::{DgtNTTPoly, DGT_MODULUS, N};
use rayon::prelude::*;

const Q: u64 = DGT_MODULUS;

pub type Share = Vec<DgtNTTPoly>;

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

/// `q_dgt` is prime, so Fermat gives the inverse.
#[inline]
fn inv_q(x: u64) -> u64 {
    debug_assert!(!x.is_multiple_of(Q), "inverse of zero");
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

/// `Σ_b coefs[b] · blocks[b]`, positionally over `block_len` polys.
///
/// Inputs are canonical in `[0, q)` already — the digest ring's wire invariant —
/// so this is a bare Shoup multiply-accumulate with no normalisation, and
/// `acc + p < 2^62` never overflows.
fn combine(blocks: &[&[DgtNTTPoly]], coefs: &[u64], block_len: usize) -> Vec<DgtNTTPoly> {
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
            DgtNTTPoly::from_raw(&acc)
        })
        .collect()
}

fn data_points(k: usize) -> Vec<u64> {
    (1..=k as u64).collect()
}

pub struct Rs;

impl Rs {
    /// Split `ctxt` into `k` zero-padded blocks.
    pub fn split_blocks(params: &RsParams, ctxt: &[DgtNTTPoly]) -> Vec<Vec<DgtNTTPoly>> {
        let bl = params.block_len(ctxt.len());
        (0..params.k)
            .map(|b| {
                let start = (b * bl).min(ctxt.len());
                let end = ((b + 1) * bl).min(ctxt.len());
                let mut v = ctxt[start..end].to_vec();
                v.resize(bl, DgtNTTPoly::default());
                v
            })
            .collect()
    }

    /// One share per node. Shares `0..k` alias the blocks, so only the `n−k`
    /// parity evaluations cost arithmetic — `(n−k)·ctxt_len·N` modmuls,
    /// independent of `k`.
    pub fn encode(params: &RsParams, ctxt: &[DgtNTTPoly]) -> Vec<Share> {
        let mut out = Self::split_blocks(params, ctxt);
        let bl = params.block_len(ctxt.len());
        let refs: Vec<&[DgtNTTPoly]> = out.iter().map(Vec::as_slice).collect();
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
    pub fn reconstruct(
        params: &RsParams,
        ctxt_len: usize,
        samples: &[(usize, &[DgtNTTPoly])],
    ) -> Result<Vec<DgtNTTPoly>, RsError> {
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

        let mut flat: Vec<DgtNTTPoly> = if used.iter().enumerate().all(|(b, (idx, _))| *idx == b) {
            used.iter().flat_map(|(_, s)| s.iter().copied()).collect()
        } else {
            let xs: Vec<u64> = used.iter().map(|(idx, _)| (*idx + 1) as u64).collect();
            let shares: Vec<&[DgtNTTPoly]> = used.iter().map(|(_, s)| *s).collect();
            (0..k)
                .flat_map(|b| combine(&shares, &lagrange_at(&xs, (b + 1) as u64), bl))
                .collect()
        };
        flat.truncate(ctxt_len);
        Ok(flat)
    }

    /// Positional sum of one node's shares across clients — the lane round.
    pub fn sum_shares(shares: &[&[DgtNTTPoly]]) -> Share {
        if shares.is_empty() {
            return Vec::new();
        }
        let len = shares[0].len();
        (0..len)
            .into_par_iter()
            .map(|pos| {
                let mut acc = DgtNTTPoly::default();
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
    use chipmunk_code::KahePoly;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn rand_ctxt(rng: &mut ChaCha20Rng, len: usize) -> Vec<DgtNTTPoly> {
        (0..len)
            .map(|_| DgtNTTPoly::from_kahe(&KahePoly::rand_poly(rng)))
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
                let samples: Vec<(usize, &[DgtNTTPoly])> = (start..start + k)
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
        let one: Vec<(usize, &[DgtNTTPoly])> = vec![(0, shares[0].as_slice())];
        assert_eq!(
            Rs::reconstruct(&params, 7, &one),
            Err(RsError::NotEnoughShares)
        );
        let dup: Vec<(usize, &[DgtNTTPoly])> = vec![
            (1, shares[1].as_slice()),
            (1, shares[1].as_slice()),
            (2, shares[2].as_slice()),
        ];
        assert_eq!(
            Rs::reconstruct(&params, 7, &dup),
            Err(RsError::DuplicateIndex)
        );
        let stub = [DgtNTTPoly::default()];
        let short: Vec<(usize, &[DgtNTTPoly])> = vec![
            (0, &stub[..]),
            (1, shares[1].as_slice()),
            (2, shares[2].as_slice()),
        ];
        assert_eq!(
            Rs::reconstruct(&params, 7, &short),
            Err(RsError::ShareLenMismatch)
        );
    }

    /// The property the whole mode rests on: coding commutes with aggregation,
    /// so summing lane `j` across clients gives lane `j` of the summed
    /// ciphertext — and the reconstruction is the exact *integer* sum,
    /// recoverable as such rather than reduced mod `q_kahe`.
    #[test]
    fn summing_shares_reconstructs_the_integer_sum() {
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let params = RsParams::new(4, 7);
        let len = 11;
        let cts: Vec<Vec<KahePoly>> = (0..6)
            .map(|_| (0..len).map(|_| KahePoly::rand_poly(&mut rng)).collect())
            .collect();
        let embedded: Vec<Vec<DgtNTTPoly>> = cts
            .iter()
            .map(|c| c.iter().map(DgtNTTPoly::from_kahe).collect())
            .collect();
        let per_client: Vec<Vec<Share>> = embedded.iter().map(|c| Rs::encode(&params, c)).collect();

        let lane_sums: Vec<Share> = (0..params.n)
            .map(|j| {
                let col: Vec<&[DgtNTTPoly]> = per_client.iter().map(|s| s[j].as_slice()).collect();
                Rs::sum_shares(&col)
            })
            .collect();

        let samples: Vec<(usize, &[DgtNTTPoly])> = (3..3 + params.k)
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
                assert_eq!(recovered[i], want, "pos {pos} coeff {i}");
            }
        }
    }
}
