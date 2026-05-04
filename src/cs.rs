//! Commitment scheme: hiding leaf commitment built on a homomorphic Merkle tree.
//!
//! Per share `i`, the leaf pair at positions `(2i, 2i+1)` of a `Tree<HVCHash>` is
//! `(a·R_i, b·R_i + s_i)`, where `R_i` is a low-norm vector of length `r_len`.
//! Tree height = log2(n_leaves), n_leaves = (2·n_servers).next_power_of_two().
//!
//! Openings are stored in *decomposed* form along the path. Reason: chipmunk's
//! `decom_then_hash(l, r) = hash_separate_inputs(l.decompose_r(), r.decompose_r())`
//! is linear over its decomposed inputs but non-linear over raw `(l, r)`, so
//! summing two openings pointwise must happen on the decomposed representation.
//! Verification walks bottom→top using `hash_separate_inputs` and `projection_r`
//! (the inverse of `decompose_r`, which is linear).

use chipmunk_code::{
    pointwise_dot, pointwise_sum, pointwise_sum_polys, HVCHash, HVCNTTPoly, HVCPoly,
    Polynomial, Tree, HVC_WIDTH, N as POLY_N,
};
// Note: HVCPoly::decompose_r returns [HVCPoly; HVC_WIDTH]; we keep openings in
// Vec<HVCPoly> form to share code with sum_openings.
use rand::Rng;

pub trait Cs {
    type Params;
    type Secret;
    type Commitment: Clone;
    type Opening: Clone;

    fn setup<R: Rng>(rng: &mut R, num_servers: usize) -> Self::Params;
    fn commit<R: Rng>(
        rng: &mut R,
        pp: &Self::Params,
        shares: &[Self::Secret],
    ) -> (Self::Commitment, Vec<Self::Opening>);
    fn verify(pp: &Self::Params, c: &Self::Commitment, o: &Self::Opening) -> bool;
    fn sum_commitments(cs: &[Self::Commitment]) -> Self::Commitment;
    /// Take borrowed slices to avoid cloning openings (each can be ~100s of KB).
    /// Callers with owned `Vec<Opening>` should `.iter().collect()` into a
    /// `Vec<&Opening>` first.
    fn sum_openings(os: &[&Self::Opening]) -> Self::Opening;
}

pub struct CsParams {
    /// Public commitment matrix `a`, kept NTT-resident so each `dot(a, R)`
    /// converts only `R` to NTT and skips per-pair NTT round-trips.
    pub a_ntt: Vec<HVCNTTPoly>,
    pub b_ntt: Vec<HVCNTTPoly>,
    pub r_len: usize,
    pub r_bound: u32,
    pub r_half_weight: usize,
    pub hasher: HVCHash,
    pub n_servers: usize,
    pub n_leaves: usize,
}

#[derive(Clone)]
pub struct Commitment {
    pub root: HVCPoly,
}

/// Single-allocation opening. All polynomials live in one contiguous
/// `Box<[HVCPoly]>` so the per-opening data streams in one shot during the
/// opening-major sum scan. Layout:
///
/// ```text
///   data[0 .. r_len]                            = r
///   data[r_len]                                 = s
///   data[r_len + 1 + 2k·W .. r_len + 1 + (2k+1)·W]   = path level k, left  (decomposed, HVC_WIDTH polys)
///   data[r_len + 1 + (2k+1)·W .. r_len + 1 + (2k+2)·W] = path level k, right
/// ```
/// where `W = HVC_WIDTH` and `k ∈ [0, path_len)`. Path index 0 is the topmost
/// pair (just below the root); index `path_len-1` is the leaf level.
#[derive(Clone)]
pub struct Opening {
    pub server_index: usize,
    pub path_index: usize,
    r_len: usize,
    path_len: usize,
    data: Box<[HVCPoly]>,
}

fn opening_total_polys(r_len: usize, path_len: usize) -> usize {
    r_len + 1 + path_len * 2 * HVC_WIDTH
}

impl Opening {
    pub fn r_len(&self) -> usize {
        self.r_len
    }
    pub fn path_len(&self) -> usize {
        self.path_len
    }

    pub fn r(&self) -> &[HVCPoly] {
        &self.data[..self.r_len]
    }
    pub fn r_mut(&mut self) -> &mut [HVCPoly] {
        &mut self.data[..self.r_len]
    }

    pub fn s(&self) -> &HVCPoly {
        &self.data[self.r_len]
    }
    pub fn s_mut(&mut self) -> &mut HVCPoly {
        &mut self.data[self.r_len]
    }

    /// `(left, right)` decomposed pair at `level` of the Merkle path.
    pub fn path_node(&self, level: usize) -> (&[HVCPoly], &[HVCPoly]) {
        let base = self.r_len + 1 + level * 2 * HVC_WIDTH;
        (
            &self.data[base..base + HVC_WIDTH],
            &self.data[base + HVC_WIDTH..base + 2 * HVC_WIDTH],
        )
    }
}

pub struct HidingMerkleCommitment;

/// Compute the leaf pair `(a·R, b·R + s)` for one share. Converts `R` to NTT
/// once and reuses it for both halves; runs the SIMD-vectorized `pointwise_dot`
/// twice (one MAC each) and INTTs once per side.
///
/// Per share: `r_len` forward NTTs + 2 INTTs (vs. the old shape's
/// `2 · r_len` forward NTTs + `2 · r_len` INTTs through `HVCPoly::Mul`).
fn leaf_pair(
    a_ntt: &[HVCNTTPoly],
    b_ntt: &[HVCNTTPoly],
    r: &[HVCPoly],
    s: &HVCPoly,
) -> (HVCPoly, HVCPoly) {
    let r_ntt: Vec<HVCNTTPoly> = r.iter().map(HVCNTTPoly::from).collect();
    let leaf_l = HVCPoly::from(&pointwise_dot(a_ntt, &r_ntt));
    let leaf_r = HVCPoly::from(&pointwise_dot(b_ntt, &r_ntt));
    (leaf_l, leaf_r + *s)
}

fn position_list(index: usize, depth: usize) -> Vec<bool> {
    // Length depth+1; position_list[i] = bit (depth - i) of index.
    // position_list[i] for i in 1..depth tells whether the path-node at level i
    // is the right (true) or left (false) child of its parent at level i-1.
    (0..=depth).map(|i| ((index >> i) & 1) != 0).rev().collect()
}


impl Cs for HidingMerkleCommitment {
    type Params = CsParams;
    type Secret = HVCPoly;
    type Commitment = Commitment;
    type Opening = Opening;

    fn setup<R: Rng>(rng: &mut R, num_servers: usize) -> CsParams {
        let r_len = 8;
        let r_bound = 64;
        let r_half_weight = 1;
        // Sample a, b once and immediately convert to NTT domain. The
        // commitment scheme only uses these inside `dot(a, R)` / `dot(b, R)`,
        // so storing them in NTT form lets every dot skip r_len NTT
        // conversions.
        let a_ntt = (0..r_len)
            .map(|_| HVCNTTPoly::from(&HVCPoly::rand_poly(rng)))
            .collect();
        let b_ntt = (0..r_len)
            .map(|_| HVCNTTPoly::from(&HVCPoly::rand_poly(rng)))
            .collect();
        let hasher = HVCHash::init(rng);
        let n_leaves = (2 * num_servers).next_power_of_two().max(2);
        CsParams {
            a_ntt,
            b_ntt,
            r_len,
            r_bound,
            r_half_weight,
            hasher,
            n_servers: num_servers,
            n_leaves,
        }
    }

    fn commit<R: Rng>(
        rng: &mut R,
        pp: &CsParams,
        shares: &[HVCPoly],
    ) -> (Commitment, Vec<Opening>) {
        assert_eq!(shares.len(), pp.n_servers);
        // Path length depends only on tree shape, so we can pre-size each
        // opening's data buffer and write R, share, and path nodes into it
        // directly — no separate Vec<Vec> for the per-client R values.
        let path_len = pp.n_leaves.next_power_of_two().trailing_zeros() as usize;
        let total = opening_total_polys(pp.r_len, path_len);

        let mut leaves = vec![HVCPoly::default(); pp.n_leaves];
        let mut datas: Vec<Vec<HVCPoly>> = Vec::with_capacity(pp.n_servers);
        for (i, s) in shares.iter().enumerate() {
            let mut data: Vec<HVCPoly> = Vec::with_capacity(total);
            for _ in 0..pp.r_len {
                data.push(HVCPoly::rand_balanced_ternary(rng, pp.r_half_weight));
            }
            let (leaf_l, leaf_r) = leaf_pair(&pp.a_ntt, &pp.b_ntt, &data[..pp.r_len], s);
            leaves[2 * i] = leaf_l;
            leaves[2 * i + 1] = leaf_r;
            data.push(*s);
            datas.push(data);
        }
        let tree = Tree::<HVCHash>::new_with_leaf_nodes(&leaves, &pp.hasher);
        let root = tree.root();

        let openings = datas
            .into_iter()
            .enumerate()
            .map(|(i, mut data)| {
                let raw_path = tree.gen_proof(2 * i);
                debug_assert_eq!(raw_path.nodes.len(), path_len);
                for (l, r) in raw_path.nodes.iter() {
                    data.extend_from_slice(&l.decompose_r());
                    data.extend_from_slice(&r.decompose_r());
                }
                debug_assert_eq!(data.len(), total);
                Opening {
                    server_index: i,
                    path_index: raw_path.index,
                    r_len: pp.r_len,
                    path_len,
                    data: data.into_boxed_slice(),
                }
            })
            .collect();

        (Commitment { root }, openings)
    }

    fn verify(pp: &CsParams, c: &Commitment, o: &Opening) -> bool {
        if o.r_len() != pp.r_len {
            return false;
        }
        if !o.r().iter().all(|p| p.infinity_norm() <= pp.r_bound) {
            return false;
        }
        if o.path_len() == 0 {
            return false;
        }
        // The flat layout enforces HVC_WIDTH-sized sides at construction time;
        // no per-level length check needed.

        let (expected_leaf_l, expected_leaf_r) =
            leaf_pair(&pp.a_ntt, &pp.b_ntt, o.r(), o.s());

        let last_idx = o.path_len() - 1;
        let (last_l, last_r) = o.path_node(last_idx);
        if HVCPoly::projection_r(last_l) != expected_leaf_l {
            return false;
        }
        if HVCPoly::projection_r(last_r) != expected_leaf_r {
            return false;
        }

        // Top of path hashes to the root.
        let (top_l, top_r) = o.path_node(0);
        if pp.hasher.hash_separate_inputs(top_l, top_r) != c.root {
            return false;
        }

        // Internal: hash of each level matches the projected sibling at the level above.
        let pos = position_list(o.path_index, o.path_len());
        for i in 1..o.path_len() {
            let (cur_l, cur_r) = o.path_node(i);
            let parent = pp.hasher.hash_separate_inputs(cur_l, cur_r);
            let (prev_l, prev_r) = o.path_node(i - 1);
            let expected = if pos[i] {
                HVCPoly::projection_r(prev_r)
            } else {
                HVCPoly::projection_r(prev_l)
            };
            if parent != expected {
                return false;
            }
        }
        true
    }

    fn sum_commitments(cs: &[Commitment]) -> Commitment {
        let refs: Vec<&HVCPoly> = cs.iter().map(|c| &c.root).collect();
        Commitment {
            root: pointwise_sum_polys(&refs),
        }
    }

    fn sum_openings(os: &[&Opening]) -> Opening {
        assert!(!os.is_empty());
        let idx = os[0].server_index;
        let r_len = os[0].r_len();
        let path_len = os[0].path_len();
        let path_index = os[0].path_index;
        for o in os {
            assert_eq!(o.server_index, idx, "sum_openings: differing server_index");
            assert_eq!(o.r_len(), r_len, "sum_openings: differing r length");
            assert_eq!(
                o.path_len(),
                path_len,
                "sum_openings: differing path length"
            );
            assert_eq!(o.path_index, path_index, "sum_openings: differing path index");
        }

        // Opening-major accumulation: visit each opening once and dispatch its
        // 39-ish polys to their target accumulators. The flat-buffer Opening
        // means the per-opening polys are contiguous in memory and stream in
        // sequentially.
        //
        // The accumulators below are allocated fresh on every call (~78 KB at
        // S=12). For long-running servers that run many sum_openings rounds,
        // hoisting these into a caller-owned `SumScratch { acc_r, acc_s,
        // acc_path_l, acc_path_r }` and passing it into a `sum_openings_in`
        // method is a viable optimization — saves the malloc + first-touch
        // page-faulting per round. We removed that API for v1 because the
        // single-call bench shape didn't measure the difference and the
        // simpler signature is easier to plumb through `Cs`. Re-introduce if
        // a multi-round profile shows the alloc cost mattering.
        let mut acc_r: Vec<[i32; POLY_N]> = vec![[0i32; POLY_N]; r_len];
        let mut acc_s = [0i32; POLY_N];
        let mut acc_path_l: Vec<Vec<[i32; POLY_N]>> =
            (0..path_len).map(|_| vec![[0i32; POLY_N]; HVC_WIDTH]).collect();
        let mut acc_path_r: Vec<Vec<[i32; POLY_N]>> =
            (0..path_len).map(|_| vec![[0i32; POLY_N]; HVC_WIDTH]).collect();

        for o in os.iter() {
            let r_slice = o.r();
            for j in 0..r_len {
                pointwise_sum(&mut acc_r[j], &[r_slice[j].coeffs()]);
            }
            pointwise_sum(&mut acc_s, &[o.s().coeffs()]);
            for k in 0..path_len {
                let (path_l, path_r) = o.path_node(k);
                for j in 0..HVC_WIDTH {
                    pointwise_sum(&mut acc_path_l[k][j], &[path_l[j].coeffs()]);
                    pointwise_sum(&mut acc_path_r[k][j], &[path_r[j].coeffs()]);
                }
            }
        }

        // Build a flat-layout output Opening directly.
        let total = opening_total_polys(r_len, path_len);
        let mut data: Vec<HVCPoly> = Vec::with_capacity(total);
        for a in acc_r.iter() {
            data.push(HVCPoly::from_coeffs(*a));
        }
        data.push(HVCPoly::from_coeffs(acc_s));
        for k in 0..path_len {
            for a in acc_path_l[k].iter() {
                data.push(HVCPoly::from_coeffs(*a));
            }
            for a in acc_path_r[k].iter() {
                data.push(HVCPoly::from_coeffs(*a));
            }
        }
        debug_assert_eq!(data.len(), total);

        Opening {
            server_index: idx,
            path_index,
            r_len,
            path_len,
            data: data.into_boxed_slice(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn rand_shares<R: Rng>(rng: &mut R, n: usize) -> Vec<HVCPoly> {
        (0..n).map(|_| HVCPoly::rand_poly(rng)).collect()
    }

    #[test]
    fn commit_verify() {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        for n_servers in [1usize, 2, 3, 4, 7] {
            let pp = HidingMerkleCommitment::setup(&mut rng, n_servers);
            let shares = rand_shares(&mut rng, n_servers);
            let (comm, openings) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares);
            assert_eq!(openings.len(), n_servers);
            for o in &openings {
                assert!(HidingMerkleCommitment::verify(&pp, &comm, o));
            }
        }
    }

    #[test]
    fn high_norm_r_rejected() {
        let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
        let pp = HidingMerkleCommitment::setup(&mut rng, 4);
        let shares = rand_shares(&mut rng, 4);
        let (comm, mut openings) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares);
        openings[1].r_mut()[0] = HVCPoly::rand_poly(&mut rng);
        assert!(!HidingMerkleCommitment::verify(&pp, &comm, &openings[1]));
    }

    #[test]
    fn wrong_s_rejected() {
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let pp = HidingMerkleCommitment::setup(&mut rng, 4);
        let shares = rand_shares(&mut rng, 4);
        let (comm, mut openings) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares);
        *openings[0].s_mut() = *openings[0].s() + HVCPoly::rand_poly(&mut rng);
        assert!(!HidingMerkleCommitment::verify(&pp, &comm, &openings[0]));
    }

    #[test]
    fn sum_homomorphism() {
        let mut rng = ChaCha20Rng::from_seed([5u8; 32]);
        let n_servers = 4;
        let pp = HidingMerkleCommitment::setup(&mut rng, n_servers);

        let shares_a = rand_shares(&mut rng, n_servers);
        let shares_b = rand_shares(&mut rng, n_servers);
        let (comm_a, opens_a) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares_a);
        let (comm_b, opens_b) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares_b);

        let comm_sum = HidingMerkleCommitment::sum_commitments(&[comm_a, comm_b]);

        for i in 0..n_servers {
            let summed = HidingMerkleCommitment::sum_openings(&[&opens_a[i], &opens_b[i]]);
            assert_eq!(*summed.s(), shares_a[i] + shares_b[i]);
            assert!(HidingMerkleCommitment::verify(&pp, &comm_sum, &summed));
        }
    }
}
