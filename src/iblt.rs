//! Invertible Bloom Lookup Table over `Z_q^L`.
//!
//! Adapted from `adcnet/blind-auction/ibf.go`, but native to flashnet's
//! polynomial ring. Buckets are length-`L` `Vec<i32>` with pointwise add/sub
//! mod `q = HVC_MODULUS`, instead of the 385-bit prime field used in adcnet.
//! Each 48-byte chunk is split into `L = ⌈384 / base_bits⌉` base-`2^base_bits`
//! limbs that occupy a bucket.
//!
//! Correctness condition for aggregation across `N` clients:
//! `N * (2^base_bits − 1) < q`. Misconfigured `base_bits` corrupts the IBLT
//! silently (counters stay consistent but bucket limbs wrap mod q).

use chipmunk_code::{HVCPoly, HVC_MODULUS, N};
// `Polynomial` trait is brought into scope only inside fn bodies that need it.
use chipmunk_code::Polynomial as _;
use sha2::{Digest, Sha256};

pub const IBLT_N_LEVELS: usize = 4;
pub const IBLT_SHRINK: f64 = 0.75;
pub const IBLT_CHUNK_BYTES: usize = 48;
pub const IBLT_CHUNK_BITS: usize = IBLT_CHUNK_BYTES * 8;

const Q: i32 = HVC_MODULUS;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IbltParams {
    pub message_slots: u32,
    pub base_bits: u32,
}

impl IbltParams {
    pub fn limbs_per_chunk(&self) -> usize {
        (IBLT_CHUNK_BITS + self.base_bits as usize - 1) / self.base_bits as usize
    }

    pub fn level_size(&self, level: usize) -> usize {
        let mut size = self.message_slots as f64;
        for _ in 0..level {
            size *= IBLT_SHRINK;
        }
        size as usize
    }

    pub fn total_buckets(&self) -> usize {
        (0..IBLT_N_LEVELS).map(|l| self.level_size(l)).sum()
    }

    /// Largest `N_clients` for which per-coefficient sums don't wrap mod q.
    pub fn max_clients(&self) -> u32 {
        let b = 1u32 << self.base_bits;
        (HVC_MODULUS as u32) / (b - 1).max(1)
    }
}

#[derive(Clone, Debug)]
pub struct IbltVector {
    pub params: IbltParams,
    pub chunks: [Vec<Vec<i32>>; IBLT_N_LEVELS],
    pub counters: [Vec<i32>; IBLT_N_LEVELS],
}

#[derive(Debug, PartialEq)]
pub enum IbltError {
    UnexpectedZeroCounter,
    ParamsMismatch,
}

#[inline]
fn add_mod(a: i32, b: i32) -> i32 {
    let s = a + b;
    if s >= Q {
        s - Q
    } else {
        s
    }
}

#[inline]
fn sub_mod(a: i32, b: i32) -> i32 {
    let s = a - b;
    if s < 0 {
        s + Q
    } else {
        s
    }
}

fn add_limbs(dst: &mut [i32], src: &[i32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = add_mod(*d, s);
    }
}

fn sub_limbs(dst: &mut [i32], src: &[i32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = sub_mod(*d, s);
    }
}

/// Decompose `chunk` into base-`2^base_bits` limbs, little-endian (limbs[0] = LSB).
/// `base_bits` must be in `[1, 16]`.
pub fn chunk_to_limbs(chunk: &[u8; IBLT_CHUNK_BYTES], base_bits: u32) -> Vec<i32> {
    assert!(base_bits >= 1 && base_bits <= 16);
    let b = base_bits as usize;
    let l = (IBLT_CHUNK_BITS + b - 1) / b;
    let mut limbs = vec![0i32; l];
    let mask = (1u64 << b) - 1;

    let mut acc: u64 = 0;
    let mut acc_bits: usize = 0;
    let mut limb_idx = 0;
    // Iterate bytes LSB-first (last byte to first).
    for byte_idx in (0..IBLT_CHUNK_BYTES).rev() {
        acc |= (chunk[byte_idx] as u64) << acc_bits;
        acc_bits += 8;
        while acc_bits >= b && limb_idx < l {
            limbs[limb_idx] = (acc & mask) as i32;
            acc >>= b;
            acc_bits -= b;
            limb_idx += 1;
        }
    }
    if limb_idx < l {
        limbs[limb_idx] = acc as i32;
    }
    limbs
}

/// Inverse of `chunk_to_limbs`. Limbs above 384 bits are dropped.
pub fn limbs_to_chunk(limbs: &[i32], base_bits: u32) -> [u8; IBLT_CHUNK_BYTES] {
    assert!(base_bits >= 1 && base_bits <= 16);
    let b = base_bits as usize;
    let mut out = [0u8; IBLT_CHUNK_BYTES];

    let mut acc: u64 = 0;
    let mut acc_bits: usize = 0;
    let mut byte_idx = IBLT_CHUNK_BYTES;
    for &limb in limbs {
        acc |= (limb as u64) << acc_bits;
        acc_bits += b;
        while acc_bits >= 8 && byte_idx > 0 {
            byte_idx -= 1;
            out[byte_idx] = (acc & 0xff) as u8;
            acc >>= 8;
            acc_bits -= 8;
        }
    }
    out
}

/// Bucket index for a chunk at a given level.
/// SHA-256(level_decimal_ascii || chunk)[..8] big-endian, mod `items_in_level`.
pub fn chunk_index(
    chunk: &[u8; IBLT_CHUNK_BYTES],
    level: usize,
    items_in_level: usize,
) -> u64 {
    let level_str = format!("{}", level);
    let mut hasher = Sha256::new();
    hasher.update(level_str.as_bytes());
    hasher.update(chunk);
    let hash = hasher.finalize();
    let mut prefix = [0u8; 8];
    prefix.copy_from_slice(&hash[0..8]);
    u64::from_be_bytes(prefix) % items_in_level as u64
}

impl IbltVector {
    pub fn new(params: IbltParams) -> Self {
        let l = params.limbs_per_chunk();
        let chunks = [
            vec![vec![0i32; l]; params.level_size(0)],
            vec![vec![0i32; l]; params.level_size(1)],
            vec![vec![0i32; l]; params.level_size(2)],
            vec![vec![0i32; l]; params.level_size(3)],
        ];
        let counters = [
            vec![0i32; params.level_size(0)],
            vec![0i32; params.level_size(1)],
            vec![0i32; params.level_size(2)],
            vec![0i32; params.level_size(3)],
        ];
        Self {
            params,
            chunks,
            counters,
        }
    }

    pub fn insert_chunk(&mut self, chunk: [u8; IBLT_CHUNK_BYTES]) {
        let limbs = chunk_to_limbs(&chunk, self.params.base_bits);
        for level in 0..IBLT_N_LEVELS {
            let level_size = self.chunks[level].len();
            let idx = chunk_index(&chunk, level, level_size) as usize;
            add_limbs(&mut self.chunks[level][idx], &limbs);
            self.counters[level][idx] = add_mod(self.counters[level][idx], 1);
        }
    }

    /// Pointwise add another IBLT (must share params). Bucket limbs and counters
    /// reduce mod q. The summed IBLT decodes to the union of inserted chunks.
    pub fn add_assign(&mut self, other: &Self) -> Result<(), IbltError> {
        if self.params != other.params {
            return Err(IbltError::ParamsMismatch);
        }
        for level in 0..IBLT_N_LEVELS {
            for i in 0..self.chunks[level].len() {
                add_limbs(&mut self.chunks[level][i], &other.chunks[level][i]);
                self.counters[level][i] =
                    add_mod(self.counters[level][i], other.counters[level][i]);
            }
        }
        Ok(())
    }

    /// Number of HVCPoly slots needed to pack an IBLT with these params.
    pub fn n_polys(params: &IbltParams) -> usize {
        let total = params.total_buckets() * (params.limbs_per_chunk() + 1);
        total.div_ceil(N).max(1)
    }

    /// Flatten chunk-limbs and counters into a coefficient stream and pack into
    /// HVCPolys. Layout: all chunk-limbs in level/slot order, then all counters
    /// in level/slot order. Final poly is zero-padded.
    pub fn pack(&self) -> Vec<HVCPoly> {
        let l = self.params.limbs_per_chunk();
        let total = self.params.total_buckets() * (l + 1);
        let mut coeffs: Vec<i32> = Vec::with_capacity(total);
        for level in 0..IBLT_N_LEVELS {
            for bucket in &self.chunks[level] {
                coeffs.extend_from_slice(bucket);
            }
        }
        for level in 0..IBLT_N_LEVELS {
            coeffs.extend_from_slice(&self.counters[level]);
        }
        debug_assert_eq!(coeffs.len(), total);

        let n_polys = Self::n_polys(&self.params);
        let mut polys = Vec::with_capacity(n_polys);
        for chunk in coeffs.chunks(N) {
            let mut arr = [0i32; N];
            arr[..chunk.len()].copy_from_slice(chunk);
            polys.push(HVCPoly::from_coeffs(arr));
        }
        while polys.len() < n_polys {
            polys.push(HVCPoly::from_coeffs([0i32; N]));
        }
        polys
    }

    /// Inverse of `pack`. Reads coefficients from polys (lifted to `[0, q)`).
    /// Caller must supply the same `IbltParams` used at pack time.
    pub fn unpack(params: &IbltParams, polys: &[HVCPoly]) -> Self {
        let l = params.limbs_per_chunk();

        let mut coeffs: Vec<i32> = Vec::with_capacity(polys.len() * N);
        for poly in polys {
            let mut p = *poly;
            p.lift();
            coeffs.extend_from_slice(p.coeffs());
        }

        let mut chunks = [
            vec![vec![0i32; l]; params.level_size(0)],
            vec![vec![0i32; l]; params.level_size(1)],
            vec![vec![0i32; l]; params.level_size(2)],
            vec![vec![0i32; l]; params.level_size(3)],
        ];
        let mut counters = [
            vec![0i32; params.level_size(0)],
            vec![0i32; params.level_size(1)],
            vec![0i32; params.level_size(2)],
            vec![0i32; params.level_size(3)],
        ];

        let mut idx = 0;
        for level in 0..IBLT_N_LEVELS {
            for bucket in chunks[level].iter_mut() {
                bucket.copy_from_slice(&coeffs[idx..idx + l]);
                idx += l;
            }
        }
        for level in 0..IBLT_N_LEVELS {
            for c in counters[level].iter_mut() {
                *c = coeffs[idx];
                idx += 1;
            }
        }

        Self {
            params: params.clone(),
            chunks,
            counters,
        }
    }

    /// Queue-based peeling: mirrors the structure of `ibf.go::Recover`. Returns
    /// the multiset of chunks present in the IBLT.
    pub fn recover(&self) -> Result<Vec<[u8; IBLT_CHUNK_BYTES]>, IbltError> {
        let mut working = self.clone();
        let mut recovered: Vec<[u8; IBLT_CHUNK_BYTES]> = Vec::new();

        let mut queue: Vec<(usize, usize)> = Vec::new();
        for level in 0..IBLT_N_LEVELS {
            for i in 0..working.counters[level].len() {
                if working.counters[level][i] == 1 {
                    queue.push((level, i));
                }
            }
        }

        let mut head = 0;
        while head < queue.len() {
            let (level, idx) = queue[head];
            head += 1;
            if working.counters[level][idx] != 1 {
                continue;
            }

            let chunk_limbs = working.chunks[level][idx].clone();
            let chunk = limbs_to_chunk(&chunk_limbs, working.params.base_bits);
            recovered.push(chunk);

            for inner_level in 0..IBLT_N_LEVELS {
                let level_size = working.chunks[inner_level].len();
                let inner_idx = chunk_index(&chunk, inner_level, level_size) as usize;
                if working.counters[inner_level][inner_idx] == 0 {
                    return Err(IbltError::UnexpectedZeroCounter);
                }
                sub_limbs(&mut working.chunks[inner_level][inner_idx], &chunk_limbs);
                working.counters[inner_level][inner_idx] =
                    sub_mod(working.counters[inner_level][inner_idx], 1);
                if working.counters[inner_level][inner_idx] == 1 {
                    queue.push((inner_level, inner_idx));
                }
            }
        }

        Ok(recovered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    fn rand_chunk<R: Rng>(rng: &mut R) -> [u8; IBLT_CHUNK_BYTES] {
        let mut c = [0u8; IBLT_CHUNK_BYTES];
        rng.fill(&mut c[..]);
        c
    }

    fn sorted<T: Ord + Clone>(v: &[T]) -> Vec<T> {
        let mut v = v.to_vec();
        v.sort();
        v
    }

    #[test]
    fn limb_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        for &b in &[7u32, 8, 10, 12, 16] {
            for _ in 0..16 {
                let c = rand_chunk(&mut rng);
                let limbs = chunk_to_limbs(&c, b);
                let c2 = limbs_to_chunk(&limbs, b);
                assert_eq!(c, c2, "base_bits={}", b);
                let max = 1i32 << b;
                assert!(
                    limbs.iter().all(|&l| l >= 0 && l < max),
                    "limb out of range at b={}",
                    b
                );
                assert_eq!(limbs.len(), (384 + b as usize - 1) / b as usize);
            }
        }
    }

    #[test]
    fn limbs_zero_round_trip() {
        let zero = [0u8; IBLT_CHUNK_BYTES];
        for &b in &[7u32, 8, 12, 16] {
            assert_eq!(limbs_to_chunk(&chunk_to_limbs(&zero, b), b), zero);
        }
    }

    #[test]
    fn insert_then_recover_single() {
        let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
        let params = IbltParams {
            message_slots: 100,
            base_bits: 12,
        };
        let mut iblt = IbltVector::new(params);
        let chunk = rand_chunk(&mut rng);
        iblt.insert_chunk(chunk);
        let recovered = iblt.recover().unwrap();
        assert_eq!(recovered, vec![chunk]);
    }

    #[test]
    fn recover_twenty_chunks() {
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        let params = IbltParams {
            message_slots: 100,
            base_bits: 12,
        };
        let mut iblt = IbltVector::new(params);
        let chunks: Vec<_> = (0..20).map(|_| rand_chunk(&mut rng)).collect();
        for c in &chunks {
            iblt.insert_chunk(*c);
        }
        let recovered = iblt.recover().unwrap();
        assert_eq!(sorted(&recovered), sorted(&chunks));
    }

    #[test]
    fn union_via_add_assign() {
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let params = IbltParams {
            message_slots: 14,
            base_bits: 12,
        };
        let mut a = IbltVector::new(params.clone());
        let mut b = IbltVector::new(params);
        let chunks_a: Vec<_> = (0..IBLT_N_LEVELS).map(|_| rand_chunk(&mut rng)).collect();
        let chunks_b: Vec<_> = (0..IBLT_N_LEVELS).map(|_| rand_chunk(&mut rng)).collect();
        for c in &chunks_a {
            a.insert_chunk(*c);
        }
        for c in &chunks_b {
            b.insert_chunk(*c);
        }
        a.add_assign(&b).unwrap();
        let recovered = a.recover().unwrap();
        let expected: Vec<_> = chunks_a.iter().chain(&chunks_b).copied().collect();
        assert_eq!(sorted(&recovered), sorted(&expected));
    }

    #[test]
    fn add_assign_param_mismatch() {
        let p1 = IbltParams {
            message_slots: 10,
            base_bits: 12,
        };
        let p2 = IbltParams {
            message_slots: 11,
            base_bits: 12,
        };
        let mut a = IbltVector::new(p1);
        let b = IbltVector::new(p2);
        assert_eq!(a.add_assign(&b), Err(IbltError::ParamsMismatch));
    }

    #[test]
    fn pack_unpack_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([10u8; 32]);
        for &b in &[7u32, 10, 12] {
            let params = IbltParams {
                message_slots: 32,
                base_bits: b,
            };
            let mut iblt = IbltVector::new(params.clone());
            for _ in 0..7 {
                iblt.insert_chunk(rand_chunk(&mut rng));
            }
            let polys = iblt.pack();
            assert_eq!(polys.len(), IbltVector::n_polys(&params));
            let recovered = IbltVector::unpack(&params, &polys);
            assert_eq!(recovered.params, iblt.params);
            for level in 0..IBLT_N_LEVELS {
                assert_eq!(recovered.chunks[level], iblt.chunks[level]);
                assert_eq!(recovered.counters[level], iblt.counters[level]);
            }
            // Round-trip preserves recoverability of inserted chunks.
            assert!(recovered.recover().is_ok());
        }
    }

    #[test]
    fn pack_is_linear_under_hvcpoly_add() {
        // Two clients, each builds an IBLT and packs. Pointwise HVCPoly add of
        // their packed polys must equal the pack of `a.add_assign(b)`.
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        let params = IbltParams {
            message_slots: 16,
            base_bits: 12,
        };
        let mut a = IbltVector::new(params.clone());
        let mut b = IbltVector::new(params.clone());
        for _ in 0..3 {
            a.insert_chunk(rand_chunk(&mut rng));
            b.insert_chunk(rand_chunk(&mut rng));
        }

        let pa = a.pack();
        let pb = b.pack();
        let summed: Vec<HVCPoly> = pa.iter().zip(&pb).map(|(x, y)| *x + *y).collect();

        let mut union = a.clone();
        union.add_assign(&b).unwrap();
        let pu = union.pack();

        // Lift both sides into [0, q) for byte-equal comparison.
        for (s, u) in summed.iter().zip(&pu) {
            let mut s = *s;
            let mut u = *u;
            s.lift();
            u.lift();
            assert_eq!(s.coeffs(), u.coeffs());
        }

        // And the union recovers the right chunks.
        let recovered = IbltVector::unpack(&params, &summed).recover().unwrap();
        let expected: Vec<_> = a
            .recover()
            .unwrap()
            .into_iter()
            .chain(b.recover().unwrap())
            .collect();
        assert_eq!(sorted(&recovered), sorted(&expected));
    }

    #[test]
    fn max_clients_matches_constraint() {
        for &b in &[7u32, 8, 10, 12] {
            let p = IbltParams {
                message_slots: 1,
                base_bits: b,
            };
            let n = p.max_clients() as i64;
            let base = 1i64 << b;
            assert!(
                n * (base - 1) < HVC_MODULUS as i64,
                "constraint violated: N={}, B={}, q={}",
                n,
                base,
                HVC_MODULUS
            );
        }
    }
}
