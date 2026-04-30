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

use chipmunk_code::path::Path;
use chipmunk_code::{HVCHash, HVCPoly, Polynomial, Tree, HVC_WIDTH};
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
    pub a: Vec<HVCPoly>,
    pub b: Vec<HVCPoly>,
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

/// Decomposed path: each level stores `(decompose_r(left), decompose_r(right))`,
/// each side a `Vec<HVCPoly>` of length `HVC_WIDTH`. Index 0 is the topmost
/// pair (just below the root); the last entry is at the leaf level.
#[derive(Clone)]
pub struct Opening {
    pub server_index: usize,
    pub r: Vec<HVCPoly>,
    pub s: HVCPoly,
    pub path_nodes: Vec<(Vec<HVCPoly>, Vec<HVCPoly>)>,
    pub path_index: usize,
}

pub struct HidingMerkleCommitment;

fn dot(a: &[HVCPoly], r: &[HVCPoly]) -> HVCPoly {
    a.iter()
        .zip(r)
        .map(|(x, y)| *x * *y)
        .fold(HVCPoly::default(), |acc, x| acc + x)
}

fn sum_polys<I: IntoIterator<Item = HVCPoly>>(iter: I) -> HVCPoly {
    iter.into_iter().fold(HVCPoly::default(), |a, x| a + x)
}

fn position_list(index: usize, depth: usize) -> Vec<bool> {
    // Length depth+1; position_list[i] = bit (depth - i) of index.
    // position_list[i] for i in 1..depth tells whether the path-node at level i
    // is the right (true) or left (false) child of its parent at level i-1.
    (0..=depth).map(|i| ((index >> i) & 1) != 0).rev().collect()
}

fn decompose_path(path: &Path<HVCHash>) -> Vec<(Vec<HVCPoly>, Vec<HVCPoly>)> {
    path.nodes
        .iter()
        .map(|(l, r)| (l.decompose_r().to_vec(), r.decompose_r().to_vec()))
        .collect()
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
        let a = (0..r_len).map(|_| HVCPoly::rand_poly(rng)).collect();
        let b = (0..r_len).map(|_| HVCPoly::rand_poly(rng)).collect();
        let hasher = HVCHash::init(rng);
        let n_leaves = (2 * num_servers).next_power_of_two().max(2);
        CsParams {
            a,
            b,
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
        let mut leaves = vec![HVCPoly::default(); pp.n_leaves];
        let mut rs: Vec<Vec<HVCPoly>> = Vec::with_capacity(pp.n_servers);
        for (i, s) in shares.iter().enumerate() {
            let r: Vec<HVCPoly> = (0..pp.r_len)
                .map(|_| HVCPoly::rand_balanced_ternary(rng, pp.r_half_weight))
                .collect();
            leaves[2 * i] = dot(&pp.a, &r);
            leaves[2 * i + 1] = dot(&pp.b, &r) + *s;
            rs.push(r);
        }
        let tree = Tree::<HVCHash>::new_with_leaf_nodes(&leaves, &pp.hasher);
        let root = tree.root();

        let openings = (0..pp.n_servers)
            .map(|i| {
                let raw_path = tree.gen_proof(2 * i);
                Opening {
                    server_index: i,
                    r: rs[i].clone(),
                    s: shares[i],
                    path_nodes: decompose_path(&raw_path),
                    path_index: raw_path.index,
                }
            })
            .collect();

        (Commitment { root }, openings)
    }

    fn verify(pp: &CsParams, c: &Commitment, o: &Opening) -> bool {
        if o.r.len() != pp.r_len {
            return false;
        }
        if !o.r.iter().all(|p| p.infinity_norm() <= pp.r_bound) {
            return false;
        }
        if o.path_nodes.is_empty() {
            return false;
        }
        // Each side must hold exactly HVC_WIDTH decomposed polynomials.
        if !o
            .path_nodes
            .iter()
            .all(|(l, r)| l.len() == HVC_WIDTH && r.len() == HVC_WIDTH)
        {
            return false;
        }

        let expected_leaf_l = dot(&pp.a, &o.r);
        let expected_leaf_r = dot(&pp.b, &o.r) + o.s;

        let last = o.path_nodes.last().unwrap();
        if HVCPoly::projection_r(&last.0) != expected_leaf_l {
            return false;
        }
        if HVCPoly::projection_r(&last.1) != expected_leaf_r {
            return false;
        }

        // Top of path hashes to the root.
        if pp
            .hasher
            .hash_separate_inputs(&o.path_nodes[0].0, &o.path_nodes[0].1)
            != c.root
        {
            return false;
        }

        // Internal: hash of each level matches the projected sibling at the level above.
        let pos = position_list(o.path_index, o.path_nodes.len());
        for i in 1..o.path_nodes.len() {
            let parent = pp
                .hasher
                .hash_separate_inputs(&o.path_nodes[i].0, &o.path_nodes[i].1);
            let expected = if pos[i] {
                HVCPoly::projection_r(&o.path_nodes[i - 1].1)
            } else {
                HVCPoly::projection_r(&o.path_nodes[i - 1].0)
            };
            if parent != expected {
                return false;
            }
        }
        true
    }

    fn sum_commitments(cs: &[Commitment]) -> Commitment {
        Commitment {
            root: sum_polys(cs.iter().map(|c| c.root)),
        }
    }

    fn sum_openings(os: &[&Opening]) -> Opening {
        assert!(!os.is_empty());
        let idx = os[0].server_index;
        let r_len = os[0].r.len();
        let path_len = os[0].path_nodes.len();
        let path_index = os[0].path_index;
        for o in os {
            assert_eq!(o.server_index, idx, "sum_openings: differing server_index");
            assert_eq!(o.r.len(), r_len, "sum_openings: differing r length");
            assert_eq!(
                o.path_nodes.len(),
                path_len,
                "sum_openings: differing path length"
            );
            assert_eq!(o.path_index, path_index, "sum_openings: differing path index");
        }
        let r: Vec<HVCPoly> = (0..r_len)
            .map(|j| sum_polys(os.iter().map(|o| o.r[j])))
            .collect();
        let s = sum_polys(os.iter().map(|o| o.s));
        let path_nodes: Vec<(Vec<HVCPoly>, Vec<HVCPoly>)> = (0..path_len)
            .map(|k| {
                let l: Vec<HVCPoly> = (0..HVC_WIDTH)
                    .map(|j| sum_polys(os.iter().map(|o| o.path_nodes[k].0[j])))
                    .collect();
                let r: Vec<HVCPoly> = (0..HVC_WIDTH)
                    .map(|j| sum_polys(os.iter().map(|o| o.path_nodes[k].1[j])))
                    .collect();
                (l, r)
            })
            .collect();
        Opening {
            server_index: idx,
            r,
            s,
            path_nodes,
            path_index,
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
        openings[1].r[0] = HVCPoly::rand_poly(&mut rng);
        assert!(!HidingMerkleCommitment::verify(&pp, &comm, &openings[1]));
    }

    #[test]
    fn wrong_s_rejected() {
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let pp = HidingMerkleCommitment::setup(&mut rng, 4);
        let shares = rand_shares(&mut rng, 4);
        let (comm, mut openings) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares);
        openings[0].s = openings[0].s + HVCPoly::rand_poly(&mut rng);
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
            assert_eq!(summed.s, shares_a[i] + shares_b[i]);
            assert!(HidingMerkleCommitment::verify(&pp, &comm_sum, &summed));
        }
    }
}
