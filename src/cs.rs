//! Commitment scheme: §5.3 hiding-vector-commitment composition.
//!
//! `BDLOP_Leaf(s; r) = (c¹, c²) = (a^T r,  B r + s)` is a hiding commitment
//! to a `μ_cs`-component share vector `s ∈ R_{q_cs}^{μ_cs}` under randomness
//! `r ∈ B^{κ_cs}_{β_cs,q_cs}`. `a` is a length-`κ_cs` row vector, `B` is a
//! `μ_cs × κ_cs` matrix. All BDLOP arithmetic lives on the **CS ring**
//! `R_{q_cs}` (chipmunk's `CsPoly`, q_cs = 139301), decoupled from the HVC
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
//! **Leaf label.** The `ξ = 1 + μ_cs` raw leaf elements are bridged into
//! `u = dec_{q_cs}(m) ∈ R^{ξ·HVC_WIDTH}` and hashed in one shot by [`LeafHash`]:
//! `label = g^T · u`. One flat Ajtai hash, no intermediate levels.
//!
//! The chipmunk Merkle tree is the (non-hiding) vector commitment over the
//! `n_servers` per-server **leaf labels** (each an `HVCPoly`); their
//! composition is statistically `ρ`-addition hiding + position binding.
//!
//! Each opening stores:
//! 1. `r` (length `κ_cs`) and the share vector `s` (length `μ_cs`), as `CsPoly`.
//! 2. `u` — the bridged leaf digits, `ξ·HVC_WIDTH` `HVCPoly`.
//! 3. The chipmunk Merkle path above the leaf label — `stored_path_len`
//!    `(left, right)` sibling pairs in decomposed (`HVC`) form.
//!
//! Why store `u` rather than the raw leaf?  Decomposition is *non-linear* in
//! raw values, so we cannot recompute the decomposition of a summed `(r, s)`
//! from scratch. But `LeafHash::hash` is linear over its digit inputs and the
//! projections are linear, so summing stored digits pointwise keeps
//! `sum_openings` correct.

use chipmunk_code::{
    pointwise_dot as pointwise_dot_hvc, pointwise_dot_cs, pointwise_sum_polys, CsNTTPoly, CsPoly,
    HVCHash, HVCNTTPoly, HVCPoly, Polynomial, Tree, CS_MODULUS, HVC_MODULUS, HVC_WIDTH,
    N as POLY_N,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

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

/// `g ∈ R_{q_hvc}^{ξ·HVC_WIDTH}`: the Ajtai hash labelling one vector-commitment
/// leaf, over `ξ = 1 + μ_cs` elements of `HVC_WIDTH` digits each.
pub struct LeafHash {
    g: Vec<HVCNTTPoly>,
}

impl LeafHash {
    fn init<R: Rng>(rng: &mut R, xi: usize) -> Self {
        LeafHash {
            g: (0..xi * HVC_WIDTH)
                .map(|_| HVCNTTPoly::from(&HVCPoly::rand_poly(rng)))
                .collect(),
        }
    }

    /// `g^T · u`, linear in `u` — that linearity is what lets `sum_openings` add
    /// stored digits pointwise. Defer to SIMD `pointwise_dot`; a hand-rolled
    /// mul/add loop costs ~2×.
    fn hash(&self, u: &[HVCPoly]) -> HVCPoly {
        assert_eq!(u.len(), self.g.len());
        let u_ntt: Vec<HVCNTTPoly> = u
            .iter()
            .map(|x| {
                let mut x = *x;
                x.lift();
                HVCNTTPoly::from(&x)
            })
            .collect();
        HVCPoly::from(&pointwise_dot_hvc(&self.g, &u_ntt))
    }
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
    /// Labels a leaf from its `ξ·HVC_WIDTH` bridged digits.
    pub leaf_hash: LeafHash,
    /// Hashes sibling pairs in the tree above the leaves.
    pub hasher: HVCHash,
    pub n_servers: usize,
    pub n_leaves: usize,
}

impl CsParams {
    /// Leaf vector width: `c¹` plus the `μ_cs` components of `c²`.
    pub fn xi(&self) -> usize {
        1 + self.mu_cs
    }

    /// Depth of the chipmunk tree over the per-server leaf labels.
    pub fn total_path_len(&self) -> usize {
        self.n_leaves.trailing_zeros() as usize
    }

    /// The whole tree path above each leaf label is stored (the leaf digits are
    /// held separately in the opening body).
    pub fn stored_path_len(&self) -> usize {
        self.total_path_len()
    }

    /// Packed byte length of one aggregated `ServerBulletinEntry`'s crypto
    /// (`agg_open` + `agg_share`) over `rho` clients, mirroring
    /// `PackedOpening::to_bytes` and `pack_cs_shares` without materialising them
    /// — for wire-budget planning.
    pub fn aggregated_server_crypto_len(&self, rho: u32) -> usize {
        aggregated_server_crypto_len_for(self.n_servers, self.mu_cs, self.kappa_cs, rho)
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
    stored_path_len: usize,
    /// `r` (κ_cs) ‖ `s` (μ_cs), on the CS ring.
    rs: Box<[CsPoly]>,
    /// Bridged leaf digits `u` ‖ decomposed path, on the HVC ring.
    data: Box<[HVCPoly]>,
}

/// Number of `HVCPoly` in an opening's `data` (leaf digits + path).
fn opening_data_polys(xi: usize, stored_path_len: usize) -> usize {
    xi * HVC_WIDTH + stored_path_len * 2 * HVC_WIDTH
}

impl Opening {
    pub fn kappa_cs(&self) -> usize {
        self.kappa_cs
    }
    pub fn mu_cs(&self) -> usize {
        self.mu_cs
    }
    pub fn xi(&self) -> usize {
        1 + self.mu_cs
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

    /// The bridged leaf digits `u = dec_{q_cs}(m)`, `ξ·HVC_WIDTH` long.
    pub fn u(&self) -> &[HVCPoly] {
        &self.data[..self.xi() * HVC_WIDTH]
    }

    /// Digits of leaf element `k ∈ [0, ξ)`.
    fn u_element(&self, k: usize) -> &[HVCPoly] {
        &self.data[k * HVC_WIDTH..(k + 1) * HVC_WIDTH]
    }

    pub fn path_node(&self, level: usize) -> (&[HVCPoly], &[HVCPoly]) {
        let base = self.xi() * HVC_WIDTH + level * 2 * HVC_WIDTH;
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

/// Bridge the `ξ` raw CS leaf elements into their HVC digits `u = dec_{q_cs}(m)`.
fn bridge_leaf(raw_leaves: &[CsPoly]) -> Vec<HVCPoly> {
    raw_leaves
        .iter()
        .flat_map(|leaf| leaf.decompose_r_to_hvc())
        .collect()
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
        let (beta_cs, r_bound, beta_agg_hvc) = (BETA_CS, R_BOUND, beta_agg_hvc());
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
        let leaf_hash = LeafHash::init(rng, 1 + mu_cs);
        let hasher = HVCHash::init(rng);
        // Chipmunk tree is over the per-server leaf labels.
        let n_leaves = num_servers.next_power_of_two().max(2);
        CsParams {
            a_ntt,
            b_matrix_ntt,
            mu_cs,
            kappa_cs,
            beta_cs,
            r_bound,
            beta_agg_hvc,
            leaf_hash,
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
    /// 1. **Per-server BDLOP leaf + label.** Draw fresh randomness
    ///    `r_i ∈ B^{κ_cs}_{β_cs,q_cs}` (uniform `[-β_cs, β_cs]`), compute the
    ///    `ξ = 1 + μ_cs` raw CS leaf elements `(a^T r_i, B r_i + s_i)`, bridge
    ///    them to digits `u_i`, and label the leaf with the single Ajtai hash
    ///    `g^T · u_i` (an `HVCPoly`). `u_i` is the opening body.
    /// 2. **Chipmunk Merkle tree over leaf labels.** Build a standard chipmunk
    ///    `Tree` over `n_leaves = next_pow2(n_servers)` labels. Its `root`
    ///    is the commitment.
    /// 3. **Opening layout.** `rs = r ‖ s` (CS ring); `data = u ‖` decomposed
    ///    tree path above the leaf label.
    fn commit<R: Rng>(
        rng: &mut R,
        pp: &CsParams,
        shares: &[Vec<CsPoly>],
    ) -> (Commitment, Vec<Opening>) {
        assert_eq!(shares.len(), pp.n_servers);
        for s in shares {
            assert_eq!(s.len(), pp.mu_cs, "each share must have μ_cs components");
        }
        let stored_path_len = pp.stored_path_len();

        // Step 1: per-server leaf digits + label. Independent per server; `r`'s
        // randomness is forked into per-server seeds so the result is
        // deterministic regardless of thread schedule.
        let seeds = crate::fork_seeds(rng, pp.n_servers);
        let per_server: Vec<(HVCPoly, Vec<HVCPoly>, Vec<CsPoly>)> = shares
            .par_iter()
            .zip(seeds.par_iter())
            .map(|(s_vec, seed)| {
                let mut item_rng = ChaCha20Rng::from_seed(*seed);
                let r: Vec<CsPoly> = (0..pp.kappa_cs)
                    .map(|_| CsPoly::rand_mod_p(&mut item_rng, pp.beta_cs))
                    .collect();
                let u = bridge_leaf(&leaf_block(&pp.a_ntt, &pp.b_matrix_ntt, &r, s_vec));
                let label = pp.leaf_hash.hash(&u);
                // rs = r ‖ s
                let mut rs = r;
                rs.extend_from_slice(s_vec);
                (label, u, rs)
            })
            .collect();

        let mut labels = vec![HVCPoly::default(); pp.n_leaves];
        let mut server_u: Vec<Vec<HVCPoly>> = Vec::with_capacity(pp.n_servers);
        let mut server_rs: Vec<Vec<CsPoly>> = Vec::with_capacity(pp.n_servers);
        for (i, (label, u, rs)) in per_server.into_iter().enumerate() {
            labels[i] = label;
            server_u.push(u);
            server_rs.push(rs);
        }

        // Step 2: chipmunk tree over the leaf labels.
        let tree = Tree::<HVCHash>::new_with_leaf_nodes(&labels, &pp.hasher);
        let root = tree.root();

        // Step 3: pack openings (leaf digits already built; append the path).
        let openings: Vec<Opening> = (0..pp.n_servers)
            .into_par_iter()
            .map(|i| {
                let mut data = server_u[i].clone();
                let raw_path = tree.gen_proof(i);
                debug_assert_eq!(raw_path.nodes.len(), stored_path_len);
                for (l, r) in raw_path.nodes.iter().take(stored_path_len) {
                    data.extend_from_slice(&l.decompose_r());
                    data.extend_from_slice(&r.decompose_r());
                }
                debug_assert_eq!(data.len(), opening_data_polys(pp.xi(), stored_path_len));
                Opening {
                    server_index: i,
                    path_index: raw_path.index,
                    kappa_cs: pp.kappa_cs,
                    mu_cs: pp.mu_cs,
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
    /// 2. Recompute the raw CS leaf elements from `(r, s, a, B)` and check that
    ///    each stored digit group projects back to it (`project_r_from_hvc`).
    /// 3. Label the leaf from the stored digits: `g^T · u`.
    /// 4. Walk the chipmunk path above the leaf label to the commitment root.
    fn verify(pp: &CsParams, c: &Commitment, o: &Opening) -> bool {
        // Step 1
        if o.kappa_cs() != pp.kappa_cs
            || o.mu_cs() != pp.mu_cs
            || o.stored_path_len() != pp.stored_path_len()
        {
            return false;
        }
        // Guard backing buffers before the accessors slice into them.
        if o.rs.len() != pp.kappa_cs + pp.mu_cs
            || o.data.len() != opening_data_polys(pp.xi(), pp.stored_path_len())
        {
            return false;
        }
        if !o.r().iter().all(|p| p.infinity_norm() <= pp.r_bound) {
            return false;
        }
        // Bound the base-(2η+1) HVC decomposition digits (leaf digits + path)
        // by β_agg_hvc, so a forged opening can't substitute out-of-range digits
        // that still project/hash correctly. Honest digits are ζ-bounded (ρ·ζ
        // after aggregation), well within β_agg_hvc = ρ_max·ζ.
        if !o.data.iter().all(|p| p.infinity_norm() <= pp.beta_agg_hvc) {
            return false;
        }

        // Step 2: raw CS leaf elements vs. the stored bridge digits.
        let raw = leaf_block(&pp.a_ntt, &pp.b_matrix_ntt, o.r(), o.s());
        for (k, exp) in raw.iter().enumerate() {
            if CsPoly::project_r_from_hvc(o.u_element(k)) != *exp {
                return false;
            }
        }

        // Step 3: one flat Ajtai hash over the stored digits.
        let leaf_label = pp.leaf_hash.hash(o.u());

        // Step 4: chipmunk path above the leaf label.
        let stored_path_len = o.stored_path_len();
        if stored_path_len == 0 {
            return c.root == leaf_label;
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
        stored_running == leaf_label
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
        let stored_path_len = os[0].stored_path_len();
        let path_index = os[0].path_index;
        let total = os[0].data.len();
        let rs_len = os[0].rs.len();
        for o in os {
            assert_eq!(o.server_index, idx);
            assert_eq!(o.kappa_cs(), kappa_cs);
            assert_eq!(o.mu_cs(), mu_cs);
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
//   r            (κ_cs CS polys, bounded by β_cs fresh / r_bound aggregated)
//   s            (μ_cs CS polys, bounded by q_cs/2)
//   leaf digits  (ξ · HVC_WIDTH HVC polys, decomposed at ZETA)
//   path         (stored_path_len · 2 · HVC_WIDTH HVC polys, decomposed at ZETA)
//
// Tight pack uses just enough bits per coefficient for each region's bound:
//   r:    ⌈log₂(2·r_bound + 1)⌉ bits
//   s:    ⌈log₂(q_cs)⌉ bits (18 at q_cs = 139301)
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

/// Bytes one polynomial occupies when bit-packed against `modulus` — each of
/// the `N` coefficients in `⌈log₂(2·modulus+1)⌉` bits.
pub fn poly_packed_len(modulus: i32) -> usize {
    (POLY_N * bits_for_signed(modulus as u32) as usize).div_ceil(8)
}

/// 64-bit variants for the KAHE ring (q ≈ 2^49.3 → 51 bits per coefficient).
#[inline]
fn bits_for_signed64(bound: u64) -> u32 {
    let n = 2u128 * bound as u128 + 1;
    (128 - n.leading_zeros()).max(1)
}

pub fn poly_packed_len64(modulus: i64) -> usize {
    (POLY_N * bits_for_signed64(modulus as u64) as usize).div_ceil(8)
}

fn pack_bits64(out: &mut Vec<u8>, values: &[i64], bound: u64, bits: u32) {
    debug_assert!(bits <= 56, "u64 accumulator supports up to 56-bit symbols");
    let offset = bound as i64;
    let max_u: u64 = (1u64 << bits) - 1;
    let mut acc: u64 = 0;
    let mut acc_bits: u32 = 0;
    for &v in values {
        debug_assert!(
            v >= -(bound as i64) && v <= bound as i64,
            "pack_bits64: value {v} outside [-{bound}, {bound}]"
        );
        let u = (v + offset) as u64;
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

fn unpack_bits64(
    input: &[u8],
    start_byte: usize,
    values: &mut [i64],
    bound: u64,
    bits: u32,
) -> usize {
    let offset = bound as i64;
    let mask: u128 = (1u128 << bits) - 1;
    // u128: acc_bits can reach bits-1+8 = 58 before draining, so byte shifts
    // exceed the u64 range.
    let mut acc: u128 = 0;
    let mut acc_bits: u32 = 0;
    let mut byte_idx = start_byte;
    for v in values.iter_mut() {
        while acc_bits < bits {
            acc |= (input[byte_idx] as u128) << acc_bits;
            byte_idx += 1;
            acc_bits += 8;
        }
        *v = (acc & mask) as i64 - offset;
        acc >>= bits;
        acc_bits -= bits;
    }
    byte_idx
}

/// [`pack_poly`] for i64 coefficient arrays (KAHE ring).
pub(crate) fn pack_poly64(coeffs: &[i64], modulus: i64, out: &mut Vec<u8>) {
    let bound = modulus as u64;
    pack_bits64(out, coeffs, bound, bits_for_signed64(bound));
}

/// [`unpack_poly`] for i64 coefficient arrays (KAHE ring).
pub(crate) fn unpack_poly64(input: &[u8], start: usize, modulus: i64) -> ([i64; POLY_N], usize) {
    let bound = modulus as u64;
    let mut coeffs = [0i64; POLY_N];
    let next = unpack_bits64(input, start, &mut coeffs, bound, bits_for_signed64(bound));
    (coeffs, next)
}

/// Append one polynomial's coefficients to `out`, packed against `modulus`.
/// Like the opening's `s`-region packing, values are offset by the full modulus
/// rather than centered, so any representative in `[-modulus, modulus]` round-
/// trips to the identical `i32` — commitment/equality checks that depend on the
/// exact representative are unaffected.
pub(crate) fn pack_poly(coeffs: &[i32], modulus: i32, out: &mut Vec<u8>) {
    let bound = modulus as u32;
    pack_bits(out, coeffs, bound, bits_for_signed(bound));
}

/// Inverse of [`pack_poly`]: read one polynomial's `N` coefficients starting at
/// `start`, returning them and the next byte index.
pub(crate) fn unpack_poly(input: &[u8], start: usize, modulus: i32) -> ([i32; POLY_N], usize) {
    let bound = modulus as u32;
    let mut coeffs = [0i32; POLY_N];
    let next = unpack_bits(input, start, &mut coeffs, bound, bits_for_signed(bound));
    (coeffs, next)
}

impl Commitment {
    /// Bit-packed wire form: the single HVC root packed against `HVC_MODULUS`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(poly_packed_len(HVC_MODULUS));
        pack_poly(self.root.coeffs(), HVC_MODULUS, &mut out);
        out
    }

    /// Inverse of [`Commitment::to_bytes`]; `None` unless `bytes` is exactly one
    /// packed HVC polynomial.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != poly_packed_len(HVC_MODULUS) {
            return None;
        }
        let (coeffs, _) = unpack_poly(bytes, 0, HVC_MODULUS);
        Some(Commitment { root: HVCPoly::from_coeffs(coeffs) })
    }
}

/// Bit-pack a vector of CS-ring shares (a server's aggregate share) against
/// `CS_MODULUS` — the same width the opening uses for its `s` region.
pub fn pack_cs_shares(shares: &[CsPoly]) -> Vec<u8> {
    let mut out = Vec::with_capacity(shares.len() * poly_packed_len(CS_MODULUS));
    for s in shares {
        pack_poly(s.coeffs(), CS_MODULUS, &mut out);
    }
    out
}

/// Inverse of [`pack_cs_shares`]; `None` unless `bytes` is exactly `count`
/// packed CS polynomials.
pub fn unpack_cs_shares(bytes: &[u8], count: usize) -> Option<Vec<CsPoly>> {
    if bytes.len() != count * poly_packed_len(CS_MODULUS) {
        return None;
    }
    let mut out = Vec::with_capacity(count);
    let mut start = 0;
    for _ in 0..count {
        let (coeffs, next) = unpack_poly(bytes, start, CS_MODULUS);
        start = next;
        out.push(CsPoly::from_coeffs(coeffs));
    }
    Some(out)
}

/// Tightly bit-packed Opening for wire transport. Stores per-region bit widths
/// so unpacking is self-describing. Layout of `bytes`:
/// `pack(r) ‖ pack(s) ‖ pack(leaf digits) ‖ pack(path)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedOpening {
    pub server_index: u32,
    pub path_index: u32,
    pub kappa_cs: u32,
    pub mu_cs: u32,
    pub stored_path_len: u32,
    pub r_bound: u32,
    pub s_bound: u32,
    pub tree_bound: u32,
    pub bytes: Vec<u8>,
}

/// `PackedOpening` header size on the wire: 8 × u32 LE.
pub const PACKED_OPENING_HEADER_LEN: usize = 32;

impl PackedOpening {
    /// Total byte length of the packed payload (header excluded).
    pub fn body_len(&self) -> usize {
        self.bytes.len()
    }

    /// Self-describing wire form: `8 × u32 LE header ‖ bytes`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(PACKED_OPENING_HEADER_LEN + self.bytes.len());
        for v in [
            self.server_index,
            self.path_index,
            self.kappa_cs,
            self.mu_cs,
            self.stored_path_len,
            self.r_bound,
            self.s_bound,
            self.tree_bound,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.bytes);
        out
    }

    /// Inverse of [`Self::to_bytes`]. Header *semantics* are validated later by
    /// [`Opening::from_packed`]; this only checks the length.
    pub fn from_bytes(data: &[u8]) -> Option<PackedOpening> {
        if data.len() < PACKED_OPENING_HEADER_LEN {
            return None;
        }
        let mut f = [0u32; 8];
        for (i, v) in f.iter_mut().enumerate() {
            *v = u32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap());
        }
        Some(PackedOpening {
            server_index: f[0],
            path_index: f[1],
            kappa_cs: f[2],
            mu_cs: f[3],
            stored_path_len: f[4],
            r_bound: f[5],
            s_bound: f[6],
            tree_bound: f[7],
            bytes: data[PACKED_OPENING_HEADER_LEN..].to_vec(),
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum OpeningDecodeError {
    /// Header shape fields are inconsistent or implausibly large.
    BadHeader,
    /// `bytes` is shorter than the declared regions require.
    Truncated,
}

/// Upper bound on any single opening dimension; rejects adversarial headers
/// whose derived buffer sizes would overflow or trigger huge allocations.
const MAX_OPENING_DIM: usize = 1 << 20;

impl Opening {
    /// Pack this opening at the tightest bit width for each region.
    ///
    /// - `r_bound`: ∞-norm bound on `r` (`β_cs` for fresh openings,
    ///   `CsParams::r_bound` for aggregated).
    /// - `s_bound`: ∞-norm bound on `s` (q_cs/2 for arbitrary share values).
    /// - `tree_bound`: ∞-norm bound on digits (`ZETA` fresh, `ρ·ZETA` after
    ///   summing ρ openings).
    pub fn pack(&self, r_bound: u32, s_bound: u32, tree_bound: u32) -> PackedOpening {
        let r_bits = bits_for_signed(r_bound);
        let s_bits = bits_for_signed(s_bound);
        let tree_bits = bits_for_signed(tree_bound);
        let data_polys = opening_data_polys(self.xi(), self.stored_path_len);
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
            stored_path_len: self.stored_path_len as u32,
            r_bound,
            s_bound,
            tree_bound,
            bytes,
        }
    }

    /// Inverse of `pack`. Validates the (untrusted) header before allocating.
    pub fn from_packed(p: &PackedOpening) -> Result<Opening, OpeningDecodeError> {
        let kappa_cs = p.kappa_cs as usize;
        let mu_cs = p.mu_cs as usize;
        let stored_path_len = p.stored_path_len as usize;

        if kappa_cs == 0
            || mu_cs == 0
            || kappa_cs > MAX_OPENING_DIM
            || mu_cs > MAX_OPENING_DIM
            || stored_path_len > MAX_OPENING_DIM
        {
            return Err(OpeningDecodeError::BadHeader);
        }

        let data_polys = opening_data_polys(1 + mu_cs, stored_path_len);
        let r_bits = bits_for_signed(p.r_bound);
        let s_bits = bits_for_signed(p.s_bound);
        let tree_bits = bits_for_signed(p.tree_bound);

        let total_bits = (kappa_cs * POLY_N) * r_bits as usize
            + (mu_cs * POLY_N) * s_bits as usize
            + (data_polys * POLY_N) * tree_bits as usize;
        if p.bytes.len() < total_bits.div_ceil(8) {
            return Err(OpeningDecodeError::Truncated);
        }

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
        // Unpack leaf digits + path (HVC)
        for poly in data.iter_mut() {
            byte_idx = unpack_bits(&p.bytes, byte_idx, poly.coeffs_mut(), p.tree_bound, tree_bits);
        }
        Ok(Opening {
            server_index: p.server_index as usize,
            path_index: p.path_index as usize,
            kappa_cs,
            mu_cs,
            stored_path_len,
            rs: rs.into_boxed_slice(),
            data: data.into_boxed_slice(),
        })
    }
}

/// Pack bounds for a fresh single-client opening (`s` at full modulus, since an
/// aggregated representative isn't always centered).
pub fn fresh_opening_pack_bounds(p: &CsParams) -> (u32, u32, u32) {
    (p.beta_cs, CS_MODULUS as u32, chipmunk_code::ZETA)
}

/// Pack bounds for a server's aggregated opening over `rho` summed openings,
/// capped at the crate's verify bounds. Pass the actual canonical-set size.
pub fn aggregated_opening_pack_bounds(p: &CsParams, rho: u32) -> (u32, u32, u32) {
    aggregated_opening_pack_bounds_raw(p.beta_cs, p.r_bound, p.beta_agg_hvc, rho)
}

fn aggregated_opening_pack_bounds_raw(
    beta_cs: u32,
    r_bound: u32,
    beta_agg_hvc: u32,
    rho: u32,
) -> (u32, u32, u32) {
    let r = rho.saturating_mul(beta_cs).min(r_bound);
    let tree = rho.saturating_mul(chipmunk_code::ZETA).min(beta_agg_hvc);
    (r, CS_MODULUS as u32, tree)
}

/// Fresh per-opening randomness radius.
pub const BETA_CS: u32 = 116;
/// Aggregate verify bound: `ρ_max·β_cs` at `ρ_max = 300`.
pub const R_BOUND: u32 = 34_800;
/// Aggregated HVC digit bound: `ρ_max·ζ`.
pub fn beta_agg_hvc() -> u32 {
    300 * chipmunk_code::ZETA
}

/// [`CsParams::aggregated_server_crypto_len`] from dimensions alone, so a caller
/// sizing a wire budget does not have to sample a CRS to ask.
pub fn aggregated_server_crypto_len_for(
    n_servers: usize,
    mu_cs: usize,
    kappa_cs: usize,
    rho: u32,
) -> usize {
    let n_leaves = n_servers.next_power_of_two().max(2);
    let (r_b, s_b, t_b) = aggregated_opening_pack_bounds_raw(BETA_CS, R_BOUND, beta_agg_hvc(), rho);
    let data_polys = opening_data_polys(1 + mu_cs, n_leaves.trailing_zeros() as usize);
    let body = ((kappa_cs * POLY_N) * bits_for_signed(r_b) as usize
        + (mu_cs * POLY_N) * bits_for_signed(s_b) as usize
        + (data_polys * POLY_N) * bits_for_signed(t_b) as usize)
        .div_ceil(8);
    PACKED_OPENING_HEADER_LEN + body + mu_cs * poly_packed_len(CS_MODULUS)
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
    fn pack_cs_shares_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([9u8; 32]);
        for count in [1usize, 3, 8] {
            let shares = rand_share_vec(&mut rng, count);
            let bytes = pack_cs_shares(&shares);
            assert_eq!(bytes.len(), count * poly_packed_len(CS_MODULUS));
            assert!(bytes.len() < count * POLY_N * 4, "tighter than 4 bytes/coeff");
            assert_eq!(unpack_cs_shares(&bytes, count).unwrap(), shares);
        }
        // Wrong length is rejected, not mis-parsed.
        assert!(unpack_cs_shares(&[0u8; 3], 1).is_none());
    }

    #[test]
    fn commitment_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([10u8; 32]);
        let comm = Commitment { root: HVCPoly::rand_poly(&mut rng) };
        let bytes = comm.to_bytes();
        assert_eq!(bytes.len(), poly_packed_len(HVC_MODULUS));
        assert!(bytes.len() < POLY_N * 4);
        assert_eq!(Commitment::from_bytes(&bytes).unwrap().root, comm.root);
        assert!(Commitment::from_bytes(&[0u8; 3]).is_none());
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
            assert_eq!(pp.xi(), 4);
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
        assert_eq!(pp.xi(), 5);
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
        // A uniform CS poly has ‖·‖∞ ≈ q_cs/2 = 69650 ≫ r_bound = 34800.
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
            let unpacked = Opening::from_packed(&packed).unwrap();
            assert_eq!(o.r(), unpacked.r());
            assert_eq!(o.s(), unpacked.s());
            assert_eq!(o.server_index, unpacked.server_index);
            assert_eq!(o.path_index, unpacked.path_index);
            assert_eq!(o.u(), unpacked.u());
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
            let unpacked = Opening::from_packed(&packed).unwrap();
            assert_eq!(agg.r(), unpacked.r());
            assert_eq!(agg.s(), unpacked.s());
            assert_eq!(agg.u(), unpacked.u());
        }
    }

    /// `sum_openings` needs the label of a sum to be the sum of the labels.
    #[test]
    fn leaf_hash_is_linear() {
        let mut rng = ChaCha20Rng::from_seed([103u8; 32]);
        for xi in [1usize, 2, 5] {
            let lh = LeafHash::init(&mut rng, xi);
            let len = xi * HVC_WIDTH;
            let a: Vec<HVCPoly> = (0..len).map(|_| HVCPoly::rand_poly(&mut rng)).collect();
            let b: Vec<HVCPoly> = (0..len).map(|_| HVCPoly::rand_poly(&mut rng)).collect();
            let summed: Vec<HVCPoly> = a.iter().zip(&b).map(|(x, y)| *x + *y).collect();
            assert_eq!(lh.hash(&summed), lh.hash(&a) + lh.hash(&b), "xi={xi}");
        }
    }

    /// The flat leaf hash must not move the wire size at the deployed μ_cs = 1.
    #[test]
    fn opening_size_unchanged_at_mu_1() {
        for path in 0usize..5 {
            assert_eq!(opening_data_polys(2, path), 6 + 6 * path);
        }
    }

    /// ξ = 1 + μ_cs is no longer rounded up to a power of two, so odd widths are
    /// first-class.
    #[test]
    fn commit_verify_sum_across_widths() {
        let mut rng = ChaCha20Rng::from_seed([104u8; 32]);
        for mu_cs in [1usize, 2, 3, 4, 5] {
            let n_servers = 4;
            let pp = HidingMerkleCommitment::setup_with_dims(&mut rng, n_servers, mu_cs, 5);
            assert_eq!(pp.xi(), 1 + mu_cs);
            let shares_a = rand_shares(&mut rng, n_servers, mu_cs);
            let shares_b = rand_shares(&mut rng, n_servers, mu_cs);
            let (comm_a, opens_a) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares_a);
            let (comm_b, opens_b) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares_b);
            for o in &opens_a {
                assert!(
                    HidingMerkleCommitment::verify(&pp, &comm_a, o),
                    "fresh opening failed at μ_cs={mu_cs}"
                );
                assert_eq!(o.u().len(), pp.xi() * HVC_WIDTH);
            }
            let comm_sum = HidingMerkleCommitment::sum_commitments(&[comm_a, comm_b]);
            for i in 0..n_servers {
                let summed = HidingMerkleCommitment::sum_openings(&[&opens_a[i], &opens_b[i]]);
                for k in 0..mu_cs {
                    assert_eq!(summed.s()[k], shares_a[i][k] + shares_b[i][k]);
                }
                assert!(
                    HidingMerkleCommitment::verify(&pp, &comm_sum, &summed),
                    "aggregate opening failed at μ_cs={mu_cs}"
                );
            }
        }
    }

    /// `N = 2048` is a multiple of 8, so no region pads and the body is exactly
    /// its bit budget — a tighter pin than a ratio, which moves with the region
    /// mix (`s` packs 18/32, digits 7/32).
    #[test]
    fn pack_size_reduction() {
        let mut rng = ChaCha20Rng::from_seed([102u8; 32]);
        for mu_cs in [1usize, 5] {
            let pp = HidingMerkleCommitment::setup_with_dims(&mut rng, 4, mu_cs, 5);
            let shares = rand_shares(&mut rng, 4, pp.mu_cs);
            let (_, openings) = HidingMerkleCommitment::commit(&mut rng, &pp, &shares);
            let o = &openings[0];
            let data_polys = opening_data_polys(o.xi(), o.stored_path_len());
            let rs_polys = o.kappa_cs() + o.mu_cs();
            let in_mem_bytes = data_polys * std::mem::size_of::<HVCPoly>()
                + rs_polys * std::mem::size_of::<CsPoly>();
            let packed = o.pack(pp.beta_cs, CS_MODULUS as u32 / 2, chipmunk_code::ZETA);
            let packed_bytes = packed.body_len();

            let expected = (o.kappa_cs() * POLY_N * bits_for_signed(pp.beta_cs) as usize
                + o.mu_cs() * POLY_N * bits_for_signed(CS_MODULUS as u32 / 2) as usize
                + data_polys * POLY_N * bits_for_signed(chipmunk_code::ZETA) as usize)
                / 8;
            assert_eq!(packed_bytes, expected, "μ_cs={mu_cs}");
            assert!(
                packed_bytes * 3 <= in_mem_bytes,
                "μ_cs={mu_cs}: packed {packed_bytes} not ≤ in_mem {in_mem_bytes} / 3"
            );
        }
    }
}
