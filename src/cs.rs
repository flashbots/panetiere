//! Commitment scheme: §5.3 hiding-vector-commitment composition.
//!
//! `BDLOP_Leaf(s; r) = (c¹, c²) = (a^T r,  B r + s)` is a hiding commitment
//! to a `μ_cs`-component share vector `s ∈ R_{q_cs}^{μ_cs}` under randomness
//! `r ∈ B^{κ_cs}_{β_cs,q_cs}`. `a` is a length-`κ_cs` row vector, `B` is a
//! `μ_cs × κ_cs` matrix. All BDLOP arithmetic lives on the **CS ring**
//! `R_{q_cs}` (chipmunk's `CsPoly`, q_cs = 147457), decoupled from the HVC
//! Merkle-tree hash ring `R_{q_hvc}` (q_hvc = 40961).
//!
//! **CS → HVC bridge.** A BDLOP leaf is a `CsPoly` whose coefficients span
//! `[-q_cs/2, q_cs/2]`, larger than `q_hvc`. To feed it into the HVC tree hash
//! we base-`(2ζ+1)=69` decompose each leaf element into `HVC_WIDTH = 3`
//! `HVCPoly` digits (`CsPoly::decompose_r_to_hvc`); the digits are tiny
//! (`|·| ≤ ζ = 34 ≪ q_hvc`) so they embed losslessly. The left-inverse
//! `CsPoly::project_r_from_hvc` is linear in the digits, so summed opening
//! digits project to the summed leaf — keeping `sum_openings` correct.
//!
//! The chipmunk Merkle tree is the (non-hiding) vector commitment over the
//! `n_servers` per-server **block roots** (each an `HVCPoly`); their
//! composition is statistically `ρ`-addition hiding + position binding.
//!
//! Each opening stores:
//! 1. `r` (length `κ_cs`) and the share vector `s` (length `μ_cs`), as `CsPoly`.
//! 2. The decomposed block subtree — every node from the leaves up to (but
//!    excluding) the block root. Level 0 (the BDLOP leaves) is decomposed via
//!    the CS→HVC bridge; higher levels via HVC `decompose_r`. That's
//!    `2·block_size − 2` decomposed nodes (`HVC_WIDTH` `HVCPoly` each).
//! 3. The chipmunk Merkle path above the block root — `stored_path_len`
//!    `(left, right)` sibling pairs in decomposed (`HVC`) form.
//!
//! Why store the subtree decomposed?  Decomposition is *non-linear* in raw
//! values, so we cannot recompute decompositions of summed `(r, s)` from
//! scratch. But `hash_separate_inputs` is *linear over decomposed inputs* and
//! the projections are linear, so summing the stored decompositions pointwise
//! keeps `sum_openings` correct.

use chipmunk_code::{
    pointwise_dot_cs, pointwise_sum_polys, CsNTTPoly, CsPoly, HVCHash, HVCPoly, Polynomial, Tree,
    CS_MODULUS, HVC_MODULUS, HVC_WIDTH, N as POLY_N,
};
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
    fn sum_openings(os: &[&Self::Opening]) -> Self::Opening;
}

/// BDLOP matrix-form public parameters. `a`/`B` live on the CS ring.
pub struct CsParams {
    pub a_ntt: Vec<CsNTTPoly>,
    pub b_matrix_ntt: Vec<Vec<CsNTTPoly>>,
    pub mu_cs: usize,
    pub kappa_cs: usize,
    /// Fresh per-opening randomness ∞-norm bound β_cs (sampling radius).
    pub beta_cs: u32,
    /// Aggregated-opening randomness ∞-norm bound β_agg (verify-time check).
    /// `β_agg ≥ ρ·β_cs` ⇒ supports ρ ≤ β_agg/β_cs aggregations.
    pub r_bound: u32,
    /// Aggregated bound on the base-(2η+1) decomposition digits of the HVC
    /// tree opening. Each digit-coefficient ∈ [−η, η] fresh, [−ρη, ρη] after
    /// summing ρ openings, so `β_agg_hvc = ρ_max·η` (= 300·34 = 10200). Checked
    /// in `verify` so a forged opening can't use out-of-range digits.
    pub beta_agg_hvc: u32,
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

    /// Depth of the chipmunk tree over the per-server block roots.
    pub fn total_path_len(&self) -> usize {
        self.n_leaves.trailing_zeros() as usize
    }

    /// The whole tree path above each block root is stored (the block subtree
    /// is held separately, decomposed, in the opening body).
    pub fn stored_path_len(&self) -> usize {
        self.total_path_len()
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
    /// `r` (κ_cs) ‖ `s` (μ_cs), on the CS ring.
    rs: Box<[CsPoly]>,
    /// Decomposed block subtree ‖ decomposed path, on the HVC ring.
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

/// Number of `HVCPoly` in an opening's `data` (block subtree + path).
fn opening_data_polys(block_size: usize, stored_path_len: usize) -> usize {
    block_subtree_node_count(block_size) * HVC_WIDTH + stored_path_len * 2 * HVC_WIDTH
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

    pub fn r(&self) -> &[CsPoly] {
        &self.rs[..self.kappa_cs]
    }
    pub fn r_mut(&mut self) -> &mut [CsPoly] {
        &mut self.rs[..self.kappa_cs]
    }

    /// The committed share vector, length `μ_cs`.
    pub fn s(&self) -> &[CsPoly] {
        &self.rs[self.kappa_cs..self.kappa_cs + self.mu_cs]
    }
    pub fn s_mut(&mut self) -> &mut [CsPoly] {
        &mut self.rs[self.kappa_cs..self.kappa_cs + self.mu_cs]
    }

    /// Decomposed node at block-subtree `level` (0 = leaves), position `idx`.
    pub fn block_node(&self, level: usize, idx: usize) -> &[HVCPoly] {
        let level_offset = block_level_offset(self.block_size, level);
        let start = (level_offset + idx) * HVC_WIDTH;
        &self.data[start..start + HVC_WIDTH]
    }

    pub fn path_node(&self, level: usize) -> (&[HVCPoly], &[HVCPoly]) {
        let base =
            block_subtree_node_count(self.block_size) * HVC_WIDTH + level * 2 * HVC_WIDTH;
        (
            &self.data[base..base + HVC_WIDTH],
            &self.data[base + HVC_WIDTH..base + 2 * HVC_WIDTH],
        )
    }
}

pub struct HidingMerkleCommitment;

/// Compute the `(1 + μ_cs)` raw BDLOP leaf-block elements (on the CS ring):
/// `[c¹, c²_0, c²_1, ..., c²_{μ-1}]` where `c¹ = a^T r`, `c²_k = B[k]·r + s[k]`.
fn leaf_block(
    a_ntt: &[CsNTTPoly],
    b_matrix_ntt: &[Vec<CsNTTPoly>],
    r: &[CsPoly],
    s: &[CsPoly],
) -> Vec<CsPoly> {
    debug_assert_eq!(s.len(), b_matrix_ntt.len());
    let r_ntt: Vec<CsNTTPoly> = r.iter().map(CsNTTPoly::from).collect();
    let mu = b_matrix_ntt.len();
    let mut leaves = Vec::with_capacity(1 + mu);
    leaves.push(CsPoly::from(&pointwise_dot_cs(a_ntt, &r_ntt)));
    for k in 0..mu {
        let leaf = CsPoly::from(&pointwise_dot_cs(&b_matrix_ntt[k], &r_ntt)) + s[k];
        leaves.push(leaf);
    }
    leaves
}

/// Build a server's block subtree from its `1 + μ_cs` raw CS leaves. Appends
/// the decomposed subtree (level 0 via the CS→HVC bridge, higher levels via
/// HVC `decompose_r`) to `data` and returns the block root (an `HVCPoly`).
fn build_block_subtree(
    hasher: &HVCHash,
    raw_leaves: &[CsPoly],
    block_size: usize,
    block_height: usize,
    data: &mut Vec<HVCPoly>,
) -> HVCPoly {
    // Level 0: CS leaves, padded to block_size.
    let leaves_cs: Vec<CsPoly> = raw_leaves
        .iter()
        .copied()
        .chain(std::iter::repeat(CsPoly::default()).take(block_size - raw_leaves.len()))
        .collect();
    // Store level-0 digits (CS→HVC bridge).
    for leaf in &leaves_cs {
        data.extend_from_slice(&leaf.decompose_r_to_hvc());
    }
    // Level 1 (HVCPoly): hash CS-bridged sibling pairs.
    let mut cur: Vec<HVCPoly> = leaves_cs
        .chunks(2)
        .map(|p| hasher.hash_separate_inputs(&p[0].decompose_r_to_hvc(), &p[1].decompose_r_to_hvc()))
        .collect();
    // Levels 1..block_height: store level h (HVC), advance to h+1.
    for _h in 1..block_height {
        for node in &cur {
            data.extend_from_slice(&node.decompose_r());
        }
        cur = cur
            .chunks(2)
            .map(|p| hasher.decom_then_hash(&p[0], &p[1]))
            .collect();
    }
    debug_assert_eq!(cur.len(), 1);
    cur[0]
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
        // β_cs = fresh randomness radius; β_agg = aggregate verify bound,
        // sized so ρ ≤ β_agg/β_cs = 300 (covers the deployment's client count).
        let beta_cs = 122;
        let r_bound = 36600;
        // ρ_max·η = 300·ZETA = 10200, the aggregated HVC digit bound.
        let beta_agg_hvc = 300 * chipmunk_code::ZETA;
        let a_ntt: Vec<CsNTTPoly> = (0..kappa_cs)
            .map(|_| CsNTTPoly::from(&CsPoly::rand_poly(rng)))
            .collect();
        let b_matrix_ntt: Vec<Vec<CsNTTPoly>> = (0..mu_cs)
            .map(|_| {
                (0..kappa_cs)
                    .map(|_| CsNTTPoly::from(&CsPoly::rand_poly(rng)))
                    .collect()
            })
            .collect();
        let hasher = HVCHash::init(rng);
        // Chipmunk tree is over the per-server block roots.
        let n_leaves = num_servers.next_power_of_two().max(2);
        CsParams {
            a_ntt,
            b_matrix_ntt,
            mu_cs,
            kappa_cs,
            beta_cs,
            r_bound,
            beta_agg_hvc,
            hasher,
            n_servers: num_servers,
            n_leaves,
        }
    }
}

impl Cs for HidingMerkleCommitment {
    type Params = CsParams;
    /// A "share" is a `μ_cs`-component vector — the per-server slice of the
    /// committed message, on the CS ring.
    type Secret = Vec<CsPoly>;
    type Commitment = Commitment;
    type Opening = Opening;

    fn setup<R: Rng>(rng: &mut R, num_servers: usize) -> CsParams {
        Self::setup_with_dims(rng, num_servers, 1, 5)
    }

    /// Commit per-server share vectors and emit one opening per server.
    ///
    /// 1. **Per-server BDLOP leaves + block subtree.** Draw fresh randomness
    ///    `r_i ∈ B^{κ_cs}_{β_cs,q_cs}` (uniform `[-β_cs, β_cs]`), compute the
    ///    `1 + μ_cs` raw CS leaves `(a^T r_i, B r_i + s_i)`, then hash them up
    ///    via the CS→HVC bridge to a block root (an `HVCPoly`). The decomposed
    ///    subtree is appended to the opening body.
    /// 2. **Chipmunk Merkle tree over block roots.** Build a standard chipmunk
    ///    `Tree` over `n_leaves = next_pow2(n_servers)` block roots. Its `root`
    ///    is the commitment.
    /// 3. **Opening layout.** `rs = r ‖ s` (CS ring); `data =` decomposed block
    ///    subtree ‖ decomposed tree path above the block root.
    fn commit<R: Rng>(
        rng: &mut R,
        pp: &CsParams,
        shares: &[Vec<CsPoly>],
    ) -> (Commitment, Vec<Opening>) {
        assert_eq!(shares.len(), pp.n_servers);
        for s in shares {
            assert_eq!(s.len(), pp.mu_cs, "each share must have μ_cs components");
        }
        let block_size = pp.block_size();
        let block_height = pp.block_height();
        let stored_path_len = pp.stored_path_len();

        // Step 1: per-server block subtree → block root + decomposed body.
        let mut block_roots = vec![HVCPoly::default(); pp.n_leaves];
        let mut server_block_data: Vec<Vec<HVCPoly>> = Vec::with_capacity(pp.n_servers);
        let mut server_rs: Vec<Vec<CsPoly>> = Vec::with_capacity(pp.n_servers);
        for (i, s_vec) in shares.iter().enumerate() {
            let r: Vec<CsPoly> = (0..pp.kappa_cs)
                .map(|_| CsPoly::rand_mod_p(rng, pp.beta_cs))
                .collect();
            let raw_leaves = leaf_block(&pp.a_ntt, &pp.b_matrix_ntt, &r, s_vec);

            let mut dblock = Vec::with_capacity(block_subtree_node_count(block_size) * HVC_WIDTH);
            let block_root =
                build_block_subtree(&pp.hasher, &raw_leaves, block_size, block_height, &mut dblock);
            block_roots[i] = block_root;
            server_block_data.push(dblock);
            // rs = r ‖ s
            let mut rs = r;
            rs.extend_from_slice(s_vec);
            server_rs.push(rs);
        }

        // Step 2: chipmunk tree over the block roots.
        let tree = Tree::<HVCHash>::new_with_leaf_nodes(&block_roots, &pp.hasher);
        let root = tree.root();

        // Step 3: pack openings (block body already built; append the path).
        let openings: Vec<Opening> = (0..pp.n_servers)
            .map(|i| {
                let mut data = server_block_data[i].clone();
                let raw_path = tree.gen_proof(i);
                debug_assert_eq!(raw_path.nodes.len(), stored_path_len);
                for (l, r) in raw_path.nodes.iter().take(stored_path_len) {
                    data.extend_from_slice(&l.decompose_r());
                    data.extend_from_slice(&r.decompose_r());
                }
                debug_assert_eq!(data.len(), opening_data_polys(block_size, stored_path_len));
                Opening {
                    server_index: i,
                    path_index: raw_path.index,
                    kappa_cs: pp.kappa_cs,
                    mu_cs: pp.mu_cs,
                    block_size,
                    stored_path_len,
                    rs: server_rs[i].clone().into_boxed_slice(),
                    data: data.into_boxed_slice(),
                }
            })
            .collect();

        (Commitment { root }, openings)
    }

    /// Verify reverses `commit`:
    /// 1. Shape + norm checks (`‖r‖∞ ≤ r_bound` on the CS ring).
    /// 2. Recompute the raw CS leaves from `(r, s, a, B)`.
    /// 3. Walk the block subtree: project each stored decomp back (level 0 via
    ///    `project_r_from_hvc` → compare CsPoly; higher levels via
    ///    `projection_r` → compare HVCPoly), hashing stored-decomp sibling
    ///    pairs to the next level. Yields the block root.
    /// 4. Walk the chipmunk path above the block root to the commitment root.
    fn verify(pp: &CsParams, c: &Commitment, o: &Opening) -> bool {
        // Step 1
        if o.kappa_cs() != pp.kappa_cs
            || o.mu_cs() != pp.mu_cs
            || o.block_size() != pp.block_size()
            || o.stored_path_len() != pp.stored_path_len()
        {
            return false;
        }
        if !o.r().iter().all(|p| p.infinity_norm() <= pp.r_bound) {
            return false;
        }
        // Bound the base-(2η+1) HVC decomposition digits (block subtree + path)
        // by β_agg_hvc, so a forged opening can't substitute out-of-range digits
        // that still project/hash correctly. Honest digits are ζ-bounded (ρ·ζ
        // after aggregation), well within β_agg_hvc = ρ_max·ζ.
        if !o.data.iter().all(|p| p.infinity_norm() <= pp.beta_agg_hvc) {
            return false;
        }

        let block_size = pp.block_size();
        let block_height = pp.block_height();

        // Step 2: raw CS leaves, padded.
        let raw = leaf_block(&pp.a_ntt, &pp.b_matrix_ntt, o.r(), o.s());
        let expected_cs: Vec<CsPoly> = raw
            .iter()
            .copied()
            .chain(std::iter::repeat(CsPoly::default()).take(block_size - raw.len()))
            .collect();

        // Step 3a: level 0 (CS ring). Project stored bridge digits, compare.
        for (k, exp) in expected_cs.iter().enumerate() {
            if CsPoly::project_r_from_hvc(o.block_node(0, k)) != *exp {
                return false;
            }
        }
        // Hash level-0 stored digits into level 1 (HVC).
        let mut expected_level: Vec<HVCPoly> = (0..block_size / 2)
            .map(|m| {
                pp.hasher
                    .hash_separate_inputs(o.block_node(0, 2 * m), o.block_node(0, 2 * m + 1))
            })
            .collect();

        // Step 3b: levels 1..block_height (HVC ring).
        for h in 1..block_height {
            let level_size = block_level_size(block_size, h);
            for k in 0..level_size {
                if HVCPoly::projection_r(o.block_node(h, k)) != expected_level[k] {
                    return false;
                }
            }
            let next_size = level_size / 2;
            let mut next: Vec<HVCPoly> = Vec::with_capacity(next_size);
            for m in 0..next_size {
                next.push(
                    pp.hasher
                        .hash_separate_inputs(o.block_node(h, 2 * m), o.block_node(h, 2 * m + 1)),
                );
            }
            expected_level = next;
        }
        debug_assert_eq!(expected_level.len(), 1);
        let leaf_block_root = expected_level[0];

        // Step 4: chipmunk path above the block root.
        let stored_path_len = o.stored_path_len();
        if stored_path_len == 0 {
            return c.root == leaf_block_root;
        }

        let (top_l, top_r) = o.path_node(0);
        if pp.hasher.hash_separate_inputs(top_l, top_r) != c.root {
            return false;
        }

        let leaf_index = o.path_index;
        let pos = position_list(leaf_index, stored_path_len);
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
        stored_running == leaf_block_root
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
        let rs_len = os[0].rs.len();
        for o in os {
            assert_eq!(o.server_index, idx);
            assert_eq!(o.kappa_cs(), kappa_cs);
            assert_eq!(o.mu_cs(), mu_cs);
            assert_eq!(o.block_size(), block_size);
            assert_eq!(o.stored_path_len(), stored_path_len);
            assert_eq!(o.path_index, path_index);
            debug_assert_eq!(o.data.len(), total);
            debug_assert_eq!(o.rs.len(), rs_len);
        }

        // --- CS-ring rs (r ‖ s): small, accumulate coeff-wise then center. ---
        let mut rs_acc: Vec<[i32; POLY_N]> = vec![[0i32; POLY_N]; rs_len];
        for o in os {
            for (acc, poly) in rs_acc.iter_mut().zip(o.rs.iter()) {
                let c = poly.coeffs();
                for k in 0..POLY_N {
                    acc[k] += c[k];
                }
            }
        }
        let cs_half = CS_MODULUS / 2;
        let rs: Vec<CsPoly> = rs_acc
            .into_iter()
            .map(|mut c| {
                for x in c.iter_mut() {
                    let mut v = *x % CS_MODULUS;
                    if v > cs_half {
                        v -= CS_MODULUS;
                    } else if v < -cs_half {
                        v += CS_MODULUS;
                    }
                    *x = v;
                }
                CsPoly::from_coeffs(c)
            })
            .collect();

        // --- HVC-ring data (block subtree + path): optimized SIMD path. ---
        // Allocate the output Box up front and accumulate directly into its
        // coefficients (avoids acc → output transcription). Opening-major loop
        // preserves the HW prefetcher's stream over each opening's contiguous
        // data. i32 accumulator suffices: each input coeff ∈ (-q/2, q/2];
        // summing ρ stays within i32 for ρ up to ~280k.
        let mut data: Vec<HVCPoly> = unsafe {
            let layout = std::alloc::Layout::array::<HVCPoly>(total).unwrap();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut HVCPoly;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Vec::from_raw_parts(ptr, total, total)
        };
        for i in 0..os.len() {
            // Software prefetch opening i+1's leading cachelines into L2.
            #[cfg(target_arch = "x86_64")]
            if i + 1 < os.len() {
                use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T1};
                unsafe {
                    let next = os[i + 1].data.as_ptr() as *const i8;
                    _mm_prefetch(next, _MM_HINT_T1);
                    _mm_prefetch(next.add(64), _MM_HINT_T1);
                    _mm_prefetch(next.add(128), _MM_HINT_T1);
                    _mm_prefetch(next.add(192), _MM_HINT_T1);
                }
            }

            let o = os[i];
            for (slot, poly) in o.data.iter().enumerate() {
                let v = poly.coeffs();
                let a = data[slot].coeffs_mut();
                wrapping_add_avx2(a, v);
            }
        }

        // In-place mod-q centering pass on the output buffer.
        let q = HVC_MODULUS;
        let half = q / 2;
        for poly in data.iter_mut() {
            let coeffs = poly.coeffs_mut();
            for k in 0..POLY_N {
                let mut x = coeffs[k] % q;
                if x > half {
                    x -= q;
                } else if x < -half {
                    x += q;
                }
                coeffs[k] = x;
            }
        }

        Opening {
            server_index: idx,
            path_index,
            kappa_cs,
            mu_cs,
            block_size,
            stored_path_len,
            rs: rs.into_boxed_slice(),
            data: data.into_boxed_slice(),
        }
    }
}

/// SIMD wrapping i32 add: `acc[i] = acc[i].wrapping_add(v[i])`. Dispatches to
/// AVX2 when available (8 i32 lanes/iter); falls back to scalar otherwise.
#[inline(always)]
fn wrapping_add_avx2(acc: &mut [i32; POLY_N], v: &[i32; POLY_N]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { wrapping_add_avx2_impl(acc, v) };
            return;
        }
    }
    for k in 0..POLY_N {
        acc[k] = acc[k].wrapping_add(v[k]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn wrapping_add_avx2_impl(acc: &mut [i32; POLY_N], v: &[i32; POLY_N]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(POLY_N % 8, 0);
    let mut k = 0;
    while k < POLY_N {
        let a_ptr = acc.as_mut_ptr().add(k) as *mut __m256i;
        let av = _mm256_loadu_si256(a_ptr as *const __m256i);
        let bv = _mm256_loadu_si256(v.as_ptr().add(k) as *const __m256i);
        let s = _mm256_add_epi32(av, bv);
        _mm256_storeu_si256(a_ptr, s);
        k += 8;
    }
}

// ============================================================
// Tight bit-packing for Openings (wire-format compaction).
// ============================================================
//
// An opening has 4 regions with different coefficient bounds:
//   r              (κ_cs CS polys, bounded by β_cs fresh / r_bound aggregated)
//   s              (μ_cs CS polys, bounded by q_cs/2)
//   block subtree  ((2·block_size − 2) · HVC_WIDTH HVC polys, decomposed at ZETA)
//   path           (stored_path_len · 2 · HVC_WIDTH HVC polys, decomposed at ZETA)
//
// Tight pack uses just enough bits per coefficient for each region's bound:
//   r:    ⌈log₂(2·r_bound + 1)⌉ bits
//   s:    ⌈log₂(q_cs)⌉ bits (18 at q_cs = 147457)
//   tree: ⌈log₂(2·ZETA + 1)⌉ = ⌈log₂(69)⌉ = 7 bits

/// Number of bits to encode signed values in [-bound, bound].
#[inline]
fn bits_for_signed(bound: u32) -> u32 {
    let n = 2u64 * bound as u64 + 1;
    (64 - n.leading_zeros()).max(1)
}

/// Pack signed values in [-bound, bound] into `out` at `bits` bits each
/// (LSB-first within each byte), offset by `bound` to become unsigned.
fn pack_bits(out: &mut Vec<u8>, values: &[i32], bound: u32, bits: u32) {
    debug_assert!(bits <= 32);
    let offset = bound as i64;
    let max_u: u64 = (1u64 << bits) - 1;
    let mut acc: u64 = 0;
    let mut acc_bits: u32 = 0;
    for &v in values {
        debug_assert!(
            v >= -(bound as i32) && v <= bound as i32,
            "pack_bits: value {} outside [{}, {}]",
            v,
            -(bound as i32),
            bound as i32
        );
        let u = ((v as i64) + offset) as u64;
        debug_assert!(u <= max_u);
        acc |= u << acc_bits;
        acc_bits += bits;
        while acc_bits >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            acc_bits -= 8;
        }
    }
    if acc_bits > 0 {
        out.push(acc as u8);
    }
}

/// Inverse of `pack_bits`. Reads `values.len()` signed integers each `bits`
/// wide from `input` starting at `start_byte`; returns the next byte index.
fn unpack_bits(
    input: &[u8],
    start_byte: usize,
    values: &mut [i32],
    bound: u32,
    bits: u32,
) -> usize {
    let offset = bound as i64;
    let mask: u64 = (1u64 << bits) - 1;
    let mut acc: u64 = 0;
    let mut acc_bits: u32 = 0;
    let mut byte_idx = start_byte;
    for v in values.iter_mut() {
        while acc_bits < bits {
            acc |= (input[byte_idx] as u64) << acc_bits;
            byte_idx += 1;
            acc_bits += 8;
        }
        let u = (acc & mask) as i64;
        *v = (u - offset) as i32;
        acc >>= bits;
        acc_bits -= bits;
    }
    byte_idx
}

/// Tightly bit-packed Opening for wire transport. Stores per-region bit widths
/// so unpacking is self-describing. Layout of `bytes`:
/// `pack(r) ‖ pack(s) ‖ pack(block_subtree) ‖ pack(path)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedOpening {
    pub server_index: u32,
    pub path_index: u32,
    pub kappa_cs: u32,
    pub mu_cs: u32,
    pub block_size: u32,
    pub stored_path_len: u32,
    pub r_bound: u32,
    pub s_bound: u32,
    pub tree_bound: u32,
    pub bytes: Vec<u8>,
}

impl PackedOpening {
    /// Total byte length of the packed payload (header excluded).
    pub fn body_len(&self) -> usize {
        self.bytes.len()
    }
}

impl Opening {
    /// Pack this opening at the tightest bit width for each region.
    ///
    /// - `r_bound`: ∞-norm bound on `r` (`β_cs` for fresh openings,
    ///   `CsParams::r_bound` for aggregated).
    /// - `s_bound`: ∞-norm bound on `s` (q_cs/2 for arbitrary share values).
    /// - `tree_bound`: ∞-norm bound on decomposed nodes (`ZETA` fresh,
    ///   `ρ·ZETA` after summing ρ openings).
    pub fn pack(&self, r_bound: u32, s_bound: u32, tree_bound: u32) -> PackedOpening {
        let r_bits = bits_for_signed(r_bound);
        let s_bits = bits_for_signed(s_bound);
        let tree_bits = bits_for_signed(tree_bound);
        let data_polys = opening_data_polys(self.block_size, self.stored_path_len);
        let est_bytes = ((self.kappa_cs * POLY_N) * r_bits as usize
            + (self.mu_cs * POLY_N) * s_bits as usize
            + (data_polys * POLY_N) * tree_bits as usize
            + 7)
            / 8;
        let mut bytes = Vec::with_capacity(est_bytes);

        for poly in self.r() {
            pack_bits(&mut bytes, poly.coeffs(), r_bound, r_bits);
        }
        for poly in self.s() {
            pack_bits(&mut bytes, poly.coeffs(), s_bound, s_bits);
        }
        for poly in self.data.iter() {
            pack_bits(&mut bytes, poly.coeffs(), tree_bound, tree_bits);
        }
        PackedOpening {
            server_index: self.server_index as u32,
            path_index: self.path_index as u32,
            kappa_cs: self.kappa_cs as u32,
            mu_cs: self.mu_cs as u32,
            block_size: self.block_size as u32,
            stored_path_len: self.stored_path_len as u32,
            r_bound,
            s_bound,
            tree_bound,
            bytes,
        }
    }

    /// Inverse of `pack`. Reconstructs the in-memory `Opening`.
    pub fn from_packed(p: &PackedOpening) -> Opening {
        let kappa_cs = p.kappa_cs as usize;
        let mu_cs = p.mu_cs as usize;
        let block_size = p.block_size as usize;
        let stored_path_len = p.stored_path_len as usize;
        let data_polys = opening_data_polys(block_size, stored_path_len);
        let r_bits = bits_for_signed(p.r_bound);
        let s_bits = bits_for_signed(p.s_bound);
        let tree_bits = bits_for_signed(p.tree_bound);

        let mut rs: Vec<CsPoly> = vec![CsPoly::default(); kappa_cs + mu_cs];
        let mut data: Vec<HVCPoly> = vec![HVCPoly::default(); data_polys];
        let mut byte_idx = 0usize;
        let mut tmp = [0i32; POLY_N];
        // Unpack r (CS)
        for poly in rs[..kappa_cs].iter_mut() {
            byte_idx = unpack_bits(&p.bytes, byte_idx, &mut tmp, p.r_bound, r_bits);
            *poly = CsPoly::from_coeffs(tmp);
        }
        // Unpack s (CS)
        for poly in rs[kappa_cs..kappa_cs + mu_cs].iter_mut() {
            byte_idx = unpack_bits(&p.bytes, byte_idx, &mut tmp, p.s_bound, s_bits);
            *poly = CsPoly::from_coeffs(tmp);
        }
        // Unpack block subtree + path (HVC)
        for poly in data.iter_mut() {
            byte_idx = unpack_bits(&p.bytes, byte_idx, poly.coeffs_mut(), p.tree_bound, tree_bits);
        }
        debug_assert!(byte_idx == p.bytes.len() || byte_idx + 1 == p.bytes.len());

        Opening {
            server_index: p.server_index as usize,
            path_index: p.path_index as usize,
            kappa_cs,
            mu_cs,
            block_size,
            stored_path_len,
            rs: rs.into_boxed_slice(),
            data: data.into_boxed_slice(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn rand_share_vec<R: Rng>(rng: &mut R, mu: usize) -> Vec<CsPoly> {
        (0..mu).map(|_| CsPoly::rand_poly(rng)).collect()
    }

    fn rand_shares<R: Rng>(rng: &mut R, n: usize, mu: usize) -> Vec<Vec<CsPoly>> {
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
            let pp = HidingMerkleCommitment::setup_with_dims(&mut rng, n_servers, 3, 5);
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
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        let pp = HidingMerkleCommitment::setup_with_dims(&mut rng, 4, 4, 5);
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
        // A uniform CS poly has ‖·‖∞ ≈ q_cs/2 = 73728 ≫ r_bound = 36600.
        openings[1].r_mut()[0] = CsPoly::rand_poly(&mut rng);
        assert!(!HidingMerkleCommitment::verify(&pp, &comm, &openings[1]));
    }

    #[test]
    fn high_norm_hvc_digit_rejected() {
        let mut rng = ChaCha20Rng::from_seed([12u8; 32]);
        let pp = HidingMerkleCommitment::setup(&mut rng, 4);
        let shares = rand_shares(&mut rng, 4, pp.mu_cs);
        let (comm, mut openings) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares);
        assert!(HidingMerkleCommitment::verify(&pp, &comm, &openings[0]));
        // Push a single decomposition digit past β_agg_hvc (still < q_hvc/2 so
        // it's a valid HVCPoly coefficient, just out of the honest digit range).
        let over = pp.beta_agg_hvc as i32 + 1;
        openings[0].data[0] = HVCPoly::from_coeffs([over; POLY_N]);
        assert!(!HidingMerkleCommitment::verify(&pp, &comm, &openings[0]));
    }

    #[test]
    fn wrong_s_rejected() {
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let pp = HidingMerkleCommitment::setup(&mut rng, 4);
        let shares = rand_shares(&mut rng, 4, pp.mu_cs);
        let (comm, mut openings) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares);
        let s_mut = openings[0].s_mut();
        s_mut[0] = s_mut[0] + CsPoly::rand_poly(&mut rng);
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
        let pp = HidingMerkleCommitment::setup_with_dims(&mut rng, n_servers, 4, 5);
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

    #[test]
    fn pack_round_trip_fresh_opening() {
        let mut rng = ChaCha20Rng::from_seed([100u8; 32]);
        let n_servers = 4;
        let pp = HidingMerkleCommitment::setup_with_dims(&mut rng, n_servers, 5, 5);
        let shares = rand_shares(&mut rng, n_servers, pp.mu_cs);
        let (_, openings) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares);
        for o in &openings {
            // Fresh r ∈ [-β_cs, β_cs]; s ∈ R_{q_cs} so bound = q_cs/2;
            // decomposed tree nodes bounded by ZETA for fresh openings.
            let packed = o.pack(pp.beta_cs, CS_MODULUS as u32 / 2, chipmunk_code::ZETA);
            let unpacked = Opening::from_packed(&packed);
            assert_eq!(o.r(), unpacked.r());
            assert_eq!(o.s(), unpacked.s());
            assert_eq!(o.server_index, unpacked.server_index);
            assert_eq!(o.path_index, unpacked.path_index);
            assert_eq!(o.block_node(0, 0), unpacked.block_node(0, 0));
            if o.stored_path_len() > 0 {
                let (l0, r0) = o.path_node(0);
                let (l1, r1) = unpacked.path_node(0);
                assert_eq!(l0, l1);
                assert_eq!(r0, r1);
            }
        }
    }

    #[test]
    fn pack_round_trip_aggregated_opening() {
        let mut rng = ChaCha20Rng::from_seed([101u8; 32]);
        let n_servers = 4;
        let pp = HidingMerkleCommitment::setup_with_dims(&mut rng, n_servers, 5, 5);
        let shares_a = rand_shares(&mut rng, n_servers, pp.mu_cs);
        let shares_b = rand_shares(&mut rng, n_servers, pp.mu_cs);
        let (_, opens_a) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares_a);
        let (_, opens_b) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares_b);
        for i in 0..n_servers {
            let agg = HidingMerkleCommitment::sum_openings(&[&opens_a[i], &opens_b[i]]);
            // ρ = 2; tree nodes bounded by ρ·ZETA, r by 2·β_cs.
            let packed = agg.pack(
                2 * pp.beta_cs,
                CS_MODULUS as u32 / 2,
                2 * chipmunk_code::ZETA,
            );
            let unpacked = Opening::from_packed(&packed);
            assert_eq!(agg.r(), unpacked.r());
            assert_eq!(agg.s(), unpacked.s());
            assert_eq!(agg.block_node(0, 0), unpacked.block_node(0, 0));
        }
    }

    #[test]
    fn pack_size_reduction() {
        let mut rng = ChaCha20Rng::from_seed([102u8; 32]);
        let pp = HidingMerkleCommitment::setup_with_dims(&mut rng, 4, 5, 5);
        let shares = rand_shares(&mut rng, 4, pp.mu_cs);
        let (_, openings) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares);
        let o = &openings[0];
        let data_polys = opening_data_polys(o.block_size(), o.stored_path_len());
        let rs_polys = o.kappa_cs() + o.mu_cs();
        let in_mem_bytes = data_polys * std::mem::size_of::<HVCPoly>()
            + rs_polys * std::mem::size_of::<CsPoly>();
        let packed = o.pack(pp.beta_cs, CS_MODULUS as u32 / 2, chipmunk_code::ZETA);
        let packed_bytes = packed.body_len();
        assert!(
            packed_bytes * 4 <= in_mem_bytes,
            "packed {} not ≤ in_mem {} / 4",
            packed_bytes,
            in_mem_bytes
        );
        eprintln!(
            "pack_size_reduction: in_mem={} packed={} ratio={:.2}x",
            in_mem_bytes,
            packed_bytes,
            in_mem_bytes as f64 / packed_bytes as f64
        );
    }
}
