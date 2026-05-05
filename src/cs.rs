//! Commitment scheme: §5.3 hiding-vector-commitment composition.
//!
//! `BDLOP_Leaf(s; r) = (c¹, c²) = (a^T r,  B r + s)` is a hiding commitment
//! to a `μ_cs`-component share vector `s ∈ R_q^{μ_cs}` under randomness
//! `r ∈ B^{κ_cs}_{β,q}`. `a` is a length-`κ_cs` row vector, `B` is a
//! `μ_cs × κ_cs` matrix. The chipmunk Merkle tree is the (non-hiding) vector
//! commitment over the `n_servers` leaf-commits; their composition is
//! statistically `ρ`-addition hiding + position binding (paper Theorem 12).
//!
//! For server `i`, the `1 + μ_cs` ring elements `(c¹_i, c²_{i,0}, ...,
//! c²_{i,μ_cs-1})` (zero-padded to `block_size = (1 + μ_cs).next_power_of_two()`)
//! occupy consecutive chipmunk tree positions
//! `[block_size·i .. block_size·(i+1))`. Each opening stores:
//!
//! 1. `r` (length `κ_cs`) and the share vector `s` (length `μ_cs`).
//! 2. The decomposed leaf-block subtree — every node from the leaves up to
//!    (but excluding) the leaf-block-root, in `decompose_r` form. That's
//!    `2·block_size − 2` decomposed polys.
//! 3. The chipmunk Merkle path *above* the leaf-block — `stored_path_len =
//!    log₂(n_leaves) − log₂(block_size)` `(left, right)` pairs in decomposed
//!    form.
//!
//! Why store the entire block subtree decomposed?  `chipmunk::decompose_r` is
//! *non-linear* in raw values, so we cannot recompute decompositions of
//! summed `(r, s)` from scratch. But `hash_separate_inputs` is *linear over
//! decomposed inputs*, so summing the stored decompositions pointwise keeps
//! `sum_openings` correct.
//!
//! `Verify` reconstructs the leaf raw values from `(r, s, A, B)`, checks each
//! stored leaf decomp projects back to its expected raw value (`projection_r`
//! is linear), then walks the block subtree pairwise via
//! `hash_separate_inputs`, checking each internal-node decomp against the
//! freshly computed parent. Finally it walks the stored chipmunk path above
//! the block.

use std::cell::RefCell;

use chipmunk_code::{
    pointwise_dot, pointwise_sum_polys, HVCHash, HVCNTTPoly, HVCPoly, Polynomial, Tree,
    HVC_MODULUS, HVC_WIDTH, N as POLY_N,
};
use rand::Rng;

thread_local! {
    /// Reusable accumulator for [`HidingMerkleCommitment::sum_openings`]. Avoids
    /// re-allocating ~600 KB per call (the dominant memmove source before).
    static SUM_OPENING_ACC: RefCell<Vec<[i32; POLY_N]>> = const { RefCell::new(Vec::new()) };
}

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
    fn sum_openings(os: &[&Self::Opening]) -> Self::Opening;
}

/// BDLOP matrix-form public parameters.
pub struct CsParams {
    pub a_ntt: Vec<HVCNTTPoly>,
    pub b_matrix_ntt: Vec<Vec<HVCNTTPoly>>,
    pub mu_cs: usize,
    pub kappa_cs: usize,
    pub r_bound: u32,
    pub r_half_weight: usize,
    pub hasher: HVCHash,
    pub n_servers: usize,
    pub n_leaves: usize,
}

impl CsParams {
    pub fn block_size(&self) -> usize {
        (1 + self.mu_cs).next_power_of_two()
    }

    pub fn block_height(&self) -> usize {
        self.block_size().trailing_zeros() as usize
    }

    pub fn total_path_len(&self) -> usize {
        self.n_leaves.trailing_zeros() as usize
    }

    pub fn stored_path_len(&self) -> usize {
        self.total_path_len() - self.block_height()
    }
}

#[derive(Clone)]
pub struct Commitment {
    pub root: HVCPoly,
}

#[derive(Clone)]
pub struct Opening {
    pub server_index: usize,
    pub path_index: usize,
    kappa_cs: usize,
    mu_cs: usize,
    block_size: usize,
    stored_path_len: usize,
    data: Box<[HVCPoly]>,
}

fn block_subtree_node_count(block_size: usize) -> usize {
    2 * block_size - 2
}

fn block_level_size(block_size: usize, level: usize) -> usize {
    block_size >> level
}

fn block_level_offset(block_size: usize, level: usize) -> usize {
    let mut acc = 0usize;
    let mut count = block_size;
    for _ in 0..level {
        acc += count;
        count /= 2;
    }
    acc
}

fn opening_total_polys(
    kappa_cs: usize,
    mu_cs: usize,
    block_size: usize,
    stored_path_len: usize,
) -> usize {
    kappa_cs + mu_cs + block_subtree_node_count(block_size) * HVC_WIDTH + stored_path_len * 2 * HVC_WIDTH
}

impl Opening {
    pub fn kappa_cs(&self) -> usize {
        self.kappa_cs
    }
    pub fn mu_cs(&self) -> usize {
        self.mu_cs
    }
    pub fn block_size(&self) -> usize {
        self.block_size
    }
    pub fn stored_path_len(&self) -> usize {
        self.stored_path_len
    }

    pub fn r(&self) -> &[HVCPoly] {
        &self.data[..self.kappa_cs]
    }
    pub fn r_mut(&mut self) -> &mut [HVCPoly] {
        &mut self.data[..self.kappa_cs]
    }

    /// The committed share vector, length `μ_cs`.
    pub fn s(&self) -> &[HVCPoly] {
        &self.data[self.kappa_cs..self.kappa_cs + self.mu_cs]
    }
    pub fn s_mut(&mut self) -> &mut [HVCPoly] {
        &mut self.data[self.kappa_cs..self.kappa_cs + self.mu_cs]
    }

    /// Decomposed node at block-subtree `level` (0 = leaves), position `idx`.
    pub fn block_node(&self, level: usize, idx: usize) -> &[HVCPoly] {
        let base_polys = self.kappa_cs + self.mu_cs;
        let level_offset = block_level_offset(self.block_size, level);
        let start = base_polys + (level_offset + idx) * HVC_WIDTH;
        &self.data[start..start + HVC_WIDTH]
    }

    pub fn path_node(&self, level: usize) -> (&[HVCPoly], &[HVCPoly]) {
        let base = self.kappa_cs
            + self.mu_cs
            + block_subtree_node_count(self.block_size) * HVC_WIDTH
            + level * 2 * HVC_WIDTH;
        (
            &self.data[base..base + HVC_WIDTH],
            &self.data[base + HVC_WIDTH..base + 2 * HVC_WIDTH],
        )
    }
}

pub struct HidingMerkleCommitment;

/// Compute the `(1 + μ_cs)` raw BDLOP leaf-block elements:
/// `[c¹, c²_0, c²_1, ..., c²_{μ-1}]` where `c¹ = a^T r`, `c²_k = B[k]·r + s[k]`.
fn leaf_block(
    a_ntt: &[HVCNTTPoly],
    b_matrix_ntt: &[Vec<HVCNTTPoly>],
    r: &[HVCPoly],
    s: &[HVCPoly],
) -> Vec<HVCPoly> {
    debug_assert_eq!(s.len(), b_matrix_ntt.len());
    let r_ntt: Vec<HVCNTTPoly> = r.iter().map(HVCNTTPoly::from).collect();
    let mu = b_matrix_ntt.len();
    let mut leaves = Vec::with_capacity(1 + mu);
    leaves.push(HVCPoly::from(&pointwise_dot(a_ntt, &r_ntt)));
    for k in 0..mu {
        let leaf = HVCPoly::from(&pointwise_dot(&b_matrix_ntt[k], &r_ntt)) + s[k];
        leaves.push(leaf);
    }
    leaves
}

fn position_list(index: usize, depth: usize) -> Vec<bool> {
    (0..=depth).map(|i| ((index >> i) & 1) != 0).rev().collect()
}

impl HidingMerkleCommitment {
    /// Inherent variant accepting `μ_cs` and `κ_cs` directly.
    pub fn setup_with_dims<R: Rng>(
        rng: &mut R,
        num_servers: usize,
        mu_cs: usize,
        kappa_cs: usize,
    ) -> CsParams {
        assert!(mu_cs >= 1, "μ_cs must be ≥ 1");
        assert!(kappa_cs >= 1, "κ_cs must be ≥ 1");
        let r_bound = 64;
        let r_half_weight = 1;
        let a_ntt: Vec<HVCNTTPoly> = (0..kappa_cs)
            .map(|_| HVCNTTPoly::from(&HVCPoly::rand_poly(rng)))
            .collect();
        let b_matrix_ntt: Vec<Vec<HVCNTTPoly>> = (0..mu_cs)
            .map(|_| {
                (0..kappa_cs)
                    .map(|_| HVCNTTPoly::from(&HVCPoly::rand_poly(rng)))
                    .collect()
            })
            .collect();
        let hasher = HVCHash::init(rng);
        let block_size = (1 + mu_cs).next_power_of_two();
        let n_leaves = (block_size * num_servers).next_power_of_two().max(2);
        CsParams {
            a_ntt,
            b_matrix_ntt,
            mu_cs,
            kappa_cs,
            r_bound,
            r_half_weight,
            hasher,
            n_servers: num_servers,
            n_leaves,
        }
    }
}

impl Cs for HidingMerkleCommitment {
    type Params = CsParams;
    /// A "share" is a `μ_cs`-component vector — the per-server slice of the
    /// committed message.
    type Secret = Vec<HVCPoly>;
    type Commitment = Commitment;
    type Opening = Opening;

    fn setup<R: Rng>(rng: &mut R, num_servers: usize) -> CsParams {
        Self::setup_with_dims(rng, num_servers, 1, 8)
    }

    fn commit<R: Rng>(
        rng: &mut R,
        pp: &CsParams,
        shares: &[Vec<HVCPoly>],
    ) -> (Commitment, Vec<Opening>) {
        assert_eq!(shares.len(), pp.n_servers);
        for s in shares {
            assert_eq!(s.len(), pp.mu_cs, "each share must have μ_cs components");
        }
        let block_size = pp.block_size();
        let block_height = pp.block_height();
        let total_path_len = pp.total_path_len();
        let stored_path_len = pp.stored_path_len();
        let total = opening_total_polys(pp.kappa_cs, pp.mu_cs, block_size, stored_path_len);

        let mut leaves_full = vec![HVCPoly::default(); pp.n_leaves];
        let mut server_raw_blocks: Vec<Vec<Vec<HVCPoly>>> = Vec::with_capacity(pp.n_servers);
        let mut server_rs: Vec<Vec<HVCPoly>> = Vec::with_capacity(pp.n_servers);
        for (i, s_vec) in shares.iter().enumerate() {
            let r: Vec<HVCPoly> = (0..pp.kappa_cs)
                .map(|_| HVCPoly::rand_balanced_ternary(rng, pp.r_half_weight))
                .collect();
            let raw_leaves = leaf_block(&pp.a_ntt, &pp.b_matrix_ntt, &r, s_vec);

            let mut levels: Vec<Vec<HVCPoly>> = Vec::with_capacity(block_height + 1);
            let mut level_polys: Vec<HVCPoly> = raw_leaves
                .iter()
                .copied()
                .chain(
                    std::iter::repeat(HVCPoly::default())
                        .take(block_size - raw_leaves.len()),
                )
                .collect();
            levels.push(level_polys.clone());
            while level_polys.len() > 1 {
                let next: Vec<HVCPoly> = level_polys
                    .chunks(2)
                    .map(|p| pp.hasher.decom_then_hash(&p[0], &p[1]))
                    .collect();
                levels.push(next.clone());
                level_polys = next;
            }
            for (k, leaf) in levels[0].iter().enumerate() {
                leaves_full[block_size * i + k] = *leaf;
            }
            server_raw_blocks.push(levels);
            server_rs.push(r);
        }

        let tree = Tree::<HVCHash>::new_with_leaf_nodes(&leaves_full, &pp.hasher);
        let root = tree.root();

        let openings: Vec<Opening> = (0..pp.n_servers)
            .map(|i| {
                let mut data: Vec<HVCPoly> = Vec::with_capacity(total);
                data.extend_from_slice(&server_rs[i]);
                data.extend_from_slice(&shares[i]);
                for h in 0..block_height {
                    for node in &server_raw_blocks[i][h] {
                        data.extend_from_slice(&node.decompose_r());
                    }
                }
                let raw_path = tree.gen_proof(block_size * i);
                debug_assert_eq!(raw_path.nodes.len(), total_path_len);
                for (l, r) in raw_path.nodes.iter().take(stored_path_len) {
                    data.extend_from_slice(&l.decompose_r());
                    data.extend_from_slice(&r.decompose_r());
                }
                debug_assert_eq!(data.len(), total);
                Opening {
                    server_index: i,
                    path_index: raw_path.index,
                    kappa_cs: pp.kappa_cs,
                    mu_cs: pp.mu_cs,
                    block_size,
                    stored_path_len,
                    data: data.into_boxed_slice(),
                }
            })
            .collect();

        (Commitment { root }, openings)
    }

    fn verify(pp: &CsParams, c: &Commitment, o: &Opening) -> bool {
        if o.kappa_cs() != pp.kappa_cs {
            return false;
        }
        if o.mu_cs() != pp.mu_cs {
            return false;
        }
        if o.block_size() != pp.block_size() {
            return false;
        }
        if o.stored_path_len() != pp.stored_path_len() {
            return false;
        }
        if !o.r().iter().all(|p| p.infinity_norm() <= pp.r_bound) {
            return false;
        }

        let block_size = pp.block_size();
        let block_height = pp.block_height();

        let raw = leaf_block(&pp.a_ntt, &pp.b_matrix_ntt, o.r(), o.s());
        let mut expected_level: Vec<HVCPoly> = raw
            .iter()
            .copied()
            .chain(std::iter::repeat(HVCPoly::default()).take(block_size - raw.len()))
            .collect();

        for h in 0..block_height {
            let level_size = block_level_size(block_size, h);
            for k in 0..level_size {
                let stored = o.block_node(h, k);
                if HVCPoly::projection_r(stored) != expected_level[k] {
                    return false;
                }
            }
            let next_size = level_size / 2;
            let mut next: Vec<HVCPoly> = Vec::with_capacity(next_size);
            for m in 0..next_size {
                let l = o.block_node(h, 2 * m);
                let r = o.block_node(h, 2 * m + 1);
                next.push(pp.hasher.hash_separate_inputs(l, r));
            }
            expected_level = next;
        }
        debug_assert_eq!(expected_level.len(), 1);
        let leaf_block_root = expected_level[0];

        let stored_path_len = o.stored_path_len();
        if stored_path_len == 0 {
            return c.root == leaf_block_root;
        }

        let (top_l, top_r) = o.path_node(0);
        if pp.hasher.hash_separate_inputs(top_l, top_r) != c.root {
            return false;
        }

        let block_height = pp.block_height();
        let above_block_index = o.path_index >> block_height;
        let pos = position_list(above_block_index, stored_path_len);
        for i in 1..stored_path_len {
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

        let last = stored_path_len - 1;
        let (last_l, last_r) = o.path_node(last);
        let stored_running = if pos[stored_path_len] {
            HVCPoly::projection_r(last_r)
        } else {
            HVCPoly::projection_r(last_l)
        };
        if stored_running != leaf_block_root {
            return false;
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
        let kappa_cs = os[0].kappa_cs();
        let mu_cs = os[0].mu_cs();
        let block_size = os[0].block_size();
        let stored_path_len = os[0].stored_path_len();
        let path_index = os[0].path_index;
        let total = os[0].data.len();
        for o in os {
            assert_eq!(o.server_index, idx);
            assert_eq!(o.kappa_cs(), kappa_cs);
            assert_eq!(o.mu_cs(), mu_cs);
            assert_eq!(o.block_size(), block_size);
            assert_eq!(o.stored_path_len(), stored_path_len);
            assert_eq!(o.path_index, path_index);
            debug_assert_eq!(o.data.len(), total);
        }

        // One contiguous accumulator. Layout matches `Opening::data` exactly,
        // so we can stream each opening's coefficients straight in. Opening-
        // major loop keeps reads sequential per opening. The i32 accumulator
        // suffices: each input coeff is in (-q/2, q/2] and we sum at most ρ
        // of them; ρ·q/2 fits in i32 for ρ up to ~21000.
        //
        // Backing storage is thread-local; we resize/zero in place so a hot
        // server only pays the ~600 KB malloc once per thread.
        SUM_OPENING_ACC.with(|cell| {
            let mut acc = cell.borrow_mut();
            if acc.len() < total {
                acc.resize(total, [0i32; POLY_N]);
            }
            let acc = &mut acc[..total];
            // Zero only the slice we'll use this call; rest of the cached
            // capacity is left untouched.
            for slot in acc.iter_mut() {
                *slot = [0i32; POLY_N];
            }
            for o in os.iter() {
                for (slot, poly) in o.data.iter().enumerate() {
                    let v = poly.coeffs();
                    let a = &mut acc[slot];
                    // Plain wrapping add; final centering pass below.
                    for k in 0..POLY_N {
                        a[k] = a[k].wrapping_add(v[k]);
                    }
                }
            }

            // Single mod-q + centering pass. Replaces ρ·total per-call
            // centerings in the previous code.
            let q = HVC_MODULUS;
            let half = q / 2;
            let mut data: Vec<HVCPoly> = Vec::with_capacity(total);
            for slot in acc.iter_mut() {
                for k in 0..POLY_N {
                    let mut x = slot[k] % q;
                    if x > half {
                        x -= q;
                    } else if x < -half {
                        x += q;
                    }
                    slot[k] = x;
                }
                data.push(HVCPoly::from_coeffs(*slot));
            }

            Opening {
                server_index: idx,
                path_index,
                kappa_cs,
                mu_cs,
                block_size,
                stored_path_len,
                data: data.into_boxed_slice(),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn rand_share_vec<R: Rng>(rng: &mut R, mu: usize) -> Vec<HVCPoly> {
        (0..mu).map(|_| HVCPoly::rand_poly(rng)).collect()
    }

    fn rand_shares<R: Rng>(rng: &mut R, n: usize, mu: usize) -> Vec<Vec<HVCPoly>> {
        (0..n).map(|_| rand_share_vec(rng, mu)).collect()
    }

    #[test]
    fn commit_verify_mu_1() {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        for n_servers in [1usize, 2, 3, 4, 7] {
            let pp = HidingMerkleCommitment::setup(&mut rng, n_servers);
            assert_eq!(pp.mu_cs, 1);
            let shares = rand_shares(&mut rng, n_servers, pp.mu_cs);
            let (comm, openings) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares);
            assert_eq!(openings.len(), n_servers);
            for o in &openings {
                assert!(HidingMerkleCommitment::verify(&pp, &comm, o));
            }
        }
    }

    #[test]
    fn commit_verify_mu_3() {
        let mut rng = ChaCha20Rng::from_seed([10u8; 32]);
        for n_servers in [1usize, 2, 3, 5, 7] {
            let pp = HidingMerkleCommitment::setup_with_dims(&mut rng, n_servers, 3, 8);
            assert_eq!(pp.mu_cs, 3);
            assert_eq!(pp.block_size(), 4);
            let shares = rand_shares(&mut rng, n_servers, pp.mu_cs);
            let (comm, openings) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares);
            for o in &openings {
                assert!(
                    HidingMerkleCommitment::verify(&pp, &comm, o),
                    "verify failed for n_servers={}",
                    n_servers
                );
            }
        }
    }

    #[test]
    fn commit_verify_mu_4() {
        // The protocol's expected use: μ_cs = κ_kahe = 4.
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        let pp = HidingMerkleCommitment::setup_with_dims(&mut rng, 4, 4, 8);
        assert_eq!(pp.block_size(), 8);
        let shares = rand_shares(&mut rng, 4, 4);
        let (comm, openings) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares);
        for o in &openings {
            assert!(HidingMerkleCommitment::verify(&pp, &comm, o));
        }
    }

    #[test]
    fn high_norm_r_rejected() {
        let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
        let pp = HidingMerkleCommitment::setup(&mut rng, 4);
        let shares = rand_shares(&mut rng, 4, pp.mu_cs);
        let (comm, mut openings) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares);
        openings[1].r_mut()[0] = HVCPoly::rand_poly(&mut rng);
        assert!(!HidingMerkleCommitment::verify(&pp, &comm, &openings[1]));
    }

    #[test]
    fn wrong_s_rejected() {
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let pp = HidingMerkleCommitment::setup(&mut rng, 4);
        let shares = rand_shares(&mut rng, 4, pp.mu_cs);
        let (comm, mut openings) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares);
        let s_mut = openings[0].s_mut();
        s_mut[0] = s_mut[0] + HVCPoly::rand_poly(&mut rng);
        assert!(!HidingMerkleCommitment::verify(&pp, &comm, &openings[0]));
    }

    #[test]
    fn sum_homomorphism_mu_1() {
        let mut rng = ChaCha20Rng::from_seed([5u8; 32]);
        let n_servers = 4;
        let pp = HidingMerkleCommitment::setup(&mut rng, n_servers);
        let shares_a = rand_shares(&mut rng, n_servers, pp.mu_cs);
        let shares_b = rand_shares(&mut rng, n_servers, pp.mu_cs);
        let (comm_a, opens_a) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares_a);
        let (comm_b, opens_b) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares_b);
        let comm_sum = HidingMerkleCommitment::sum_commitments(&[comm_a, comm_b]);
        for i in 0..n_servers {
            let summed = HidingMerkleCommitment::sum_openings(&[&opens_a[i], &opens_b[i]]);
            assert_eq!(summed.s()[0], shares_a[i][0] + shares_b[i][0]);
            assert!(HidingMerkleCommitment::verify(&pp, &comm_sum, &summed));
        }
    }

    #[test]
    fn sum_homomorphism_mu_4() {
        let mut rng = ChaCha20Rng::from_seed([6u8; 32]);
        let n_servers = 4;
        let pp = HidingMerkleCommitment::setup_with_dims(&mut rng, n_servers, 4, 8);
        let shares_a = rand_shares(&mut rng, n_servers, pp.mu_cs);
        let shares_b = rand_shares(&mut rng, n_servers, pp.mu_cs);
        let (comm_a, opens_a) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares_a);
        let (comm_b, opens_b) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares_b);
        let comm_sum = HidingMerkleCommitment::sum_commitments(&[comm_a, comm_b]);
        for i in 0..n_servers {
            let summed = HidingMerkleCommitment::sum_openings(&[&opens_a[i], &opens_b[i]]);
            for k in 0..pp.mu_cs {
                assert_eq!(summed.s()[k], shares_a[i][k] + shares_b[i][k]);
            }
            assert!(HidingMerkleCommitment::verify(&pp, &comm_sum, &summed));
        }
    }
}
