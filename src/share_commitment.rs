//! Non-hiding HVC commitment over the RS-coded shares, one leaf per lane.
//!
//! The client commits to the `n` coded shares of its ciphertext; the committed,
//! aggregated, and published objects coincide per lane, so no check ever
//! crosses the Lagrange coefficients. A lane's post — its share-domain sum and
//! the digit-domain sum of the label openings — verifies against the sum of
//! the client-signed roots: the post is its own correctness proof.
//!
//! **Hash, then decompose the hash.** Leaf `j` is not the share but the
//! base-69 digits of its Ajtai hash `A·s_j` over the digest ring — 3 label
//! polys → 30 digit polys, instead of `block_len·10`. Two binding layers:
//!
//! - The HVC tree binds `Σ labels` (digits short by construction, sums
//!   `ρ_max·ζ = 10 200 < q_hvc/2`-bounded, asserted at setup).
//! - `A` binds a lane's posted sum to `Σ labels` **together with shortness**:
//!   systematic sums are gated at `ρ_max·q_kahe/2` (the SIS instance the old
//!   whole-ct digest stood on). Parity sums are full-range and cannot be
//!   gated per lane; a forged parity post that keeps the reconstruction
//!   inside `centered_within_bound` needs `κ` with `A·κ = 0` and `μ·κ` short
//!   for full-range Lagrange `μ` — a short kernel vector of the rotated
//!   matrix `A·μ⁻¹`, i.e. SIS. Out-of-bound forgeries fail reconstruction;
//!   lanes that lie in rounds that still reconstruct are named by re-encoding
//!   the result.
//!
//! Everything is NTT-domain: shares arrive as `DgtNTTPoly`, `A·s` is pointwise
//! dots, and the committed label coefficients are the NTT-domain canonical
//! values (a bijection; `Rs::*` is untouched).

use chipmunk_code::{
    pointwise_dot as pointwise_dot_hvc, pointwise_dot_dgt, DgtNTTPoly, HVCHash, HVCNTTPoly,
    HVCPoly, Polynomial, Tree, DGT_MODULUS, HVC_MODULUS, HVC_WIDTH, KAHE_MODULUS, N as POLY_N,
    TWO_ZETA_PLUS_ONE, ZETA,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use crate::cs::{bits_for_signed, position_list, wrapping_add_avx2};
use crate::rs::Share;

/// `69^10 ≈ 2^61.1 > q_dgt ≈ 2^61`: ten balanced base-69 digits are injective.
pub const DGT_WIDTH: usize = 10;

/// Ajtai-hash output width, sized by the digest ring's SIS argument.
pub const LABEL_POLYS: usize = 3;

const BASE: i64 = TWO_ZETA_PLUS_ONE as i64;
const ZETA_I64: i64 = ZETA as i64;

/// Balanced base-69 digits of the canonical NTT-domain coefficients, centered
/// first. Digit coefficients ∈ [−ζ, ζ], little-endian.
pub fn decompose_share_poly(p: &DgtNTTPoly) -> [HVCPoly; DGT_WIDTH] {
    let q = DGT_MODULUS;
    let half = q / 2;
    let mut out = [HVCPoly::default(); DGT_WIDTH];
    let coeffs = p.coeffs();
    for i in 0..POLY_N {
        let x = coeffs[i];
        let mut v = if x > half { x as i64 - q as i64 } else { x as i64 };
        for digit in out.iter_mut() {
            let mut d = v % BASE;
            if d > ZETA_I64 {
                d -= BASE;
            } else if d < -ZETA_I64 {
                d += BASE;
            }
            digit.coeffs_mut()[i] = d as i32;
            v = (v - d) / BASE;
        }
        debug_assert_eq!(v, 0);
    }
    out
}

/// Left-inverse of [`decompose_share_poly`], linear in the digits. i128
/// Horner: aggregated digits reach `ρ_max·ζ = 10 200`, and `10 200·69⁹ ≈ 2^68`
/// overflows i64. Reduced to canonical mod `q_dgt` — the wire invariant
/// `Rs::*` relies on.
pub fn project_share_poly(digits: &[HVCPoly]) -> DgtNTTPoly {
    debug_assert_eq!(digits.len(), DGT_WIDTH);
    let mut raw = [0u64; POLY_N];
    for (i, r) in raw.iter_mut().enumerate() {
        let mut acc: i128 = digits[DGT_WIDTH - 1].coeffs()[i] as i128;
        for digit in digits.iter().rev().skip(1) {
            acc = acc * BASE as i128 + digit.coeffs()[i] as i128;
        }
        *r = acc.rem_euclid(DGT_MODULUS as i128) as u64;
    }
    DgtNTTPoly::from_raw(&raw)
}

/// `g^T · u` over the `LABEL_POLYS·DGT_WIDTH` label-digit polys. Linear in
/// `u`, so summed digits hash to summed tree labels.
#[derive(Clone)]
struct ShareLeafHash {
    g: Vec<HVCNTTPoly>,
}

impl ShareLeafHash {
    fn init<R: Rng>(rng: &mut R, len: usize) -> Self {
        ShareLeafHash {
            g: (0..len)
                .map(|_| HVCNTTPoly::from(&HVCPoly::rand_poly(rng)))
                .collect(),
        }
    }

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

#[derive(Clone)]
pub struct ShareCommitmentParams {
    /// Polys per share = `mu_kahe.div_ceil(k)`.
    pub block_len: usize,
    pub n_lanes: usize,
    pub n_leaves: usize,
    pub rho_max: usize,
    /// The Ajtai rows `A ∈ R_{q_dgt}^{LABEL_POLYS × block_len}`.
    a_rows: Vec<Vec<DgtNTTPoly>>,
    leaf_hash: ShareLeafHash,
    hasher: HVCHash,
}

impl ShareCommitmentParams {
    /// Deterministic CRS — all parties re-derive it from the session seed.
    pub fn from_seed(seed: [u8; 32], block_len: usize, n_lanes: usize, rho_max: usize) -> Self {
        assert!(block_len >= 1, "need at least one poly per share");
        assert!(n_lanes >= 1, "need at least one lane");
        assert!(rho_max >= 1, "rho_max must be ≥ 1");
        // The exactness condition reconstruction rests on, and the shortness
        // premise binding the systematic lanes.
        let bound = (rho_max as u128) * (KAHE_MODULUS as u128);
        assert!(
            bound < DGT_MODULUS as u128,
            "rho_max = {rho_max} exceeds the digest ring: rho*q_kahe = 2^{:.1} \
             must stay under q_dgt = 2^{:.1}; regenerate the ring with a larger prime",
            (bound as f64).log2(),
            (DGT_MODULUS as f64).log2(),
        );
        // The binding condition of the tree layer: aggregated label digits
        // must not wrap mod q_hvc.
        assert!(
            (rho_max as u64) * (ZETA as u64) < (HVC_MODULUS as u64) / 2,
            "rho_max = {rho_max} exceeds the digit budget: rho*zeta must stay under q_hvc/2",
        );
        let mut s = seed;
        s[0] ^= 0xC0;
        let mut rng = ChaCha20Rng::from_seed(s);
        let a_rows = (0..LABEL_POLYS)
            .map(|_| {
                (0..block_len)
                    .map(|_| DgtNTTPoly::rand_ntt_poly(&mut rng))
                    .collect()
            })
            .collect();
        let leaf_hash = ShareLeafHash::init(&mut rng, LABEL_POLYS * DGT_WIDTH);
        let hasher = HVCHash::init(&mut rng);
        let n_leaves = n_lanes.next_power_of_two().max(2);
        Self {
            block_len,
            n_lanes,
            n_leaves,
            rho_max,
            a_rows,
            leaf_hash,
            hasher,
        }
    }

    pub fn path_len(&self) -> usize {
        self.n_leaves.trailing_zeros() as usize
    }

    /// Aggregated digit bound `ρ_max·ζ` — the tree layer's norm gate.
    pub fn beta_agg(&self) -> u32 {
        self.rho_max as u32 * ZETA
    }

    fn data_polys(&self) -> usize {
        LABEL_POLYS * DGT_WIDTH + self.path_len() * 2 * HVC_WIDTH
    }

    /// `A·s` — pointwise dots over the NTT-resident share.
    pub fn hash_share(&self, share: &[DgtNTTPoly]) -> Vec<DgtNTTPoly> {
        assert_eq!(share.len(), self.block_len);
        self.a_rows
            .iter()
            .map(|row| pointwise_dot_dgt(row, share))
            .collect()
    }
}

fn label_digits(label: &[DgtNTTPoly]) -> Vec<HVCPoly> {
    label.iter().flat_map(decompose_share_poly).collect()
}

/// What travels with each share: the decomposed Merkle path. The label and its
/// digits are recomputable from the share itself.
#[derive(Clone)]
pub struct SharePath {
    pub lane_index: usize,
    /// `path_len` levels × `dec(l) ‖ dec(r)`, level 0 = the root's children.
    pub nodes: Box<[HVCPoly]>,
}

/// A lane's materialized (fresh) or aggregated opening:
/// `data = label digits (LABEL_POLYS·DGT_WIDTH) ‖ path digits (path_len·2·HVC_WIDTH)`.
#[derive(Clone)]
pub struct ShareOpening {
    pub lane_index: usize,
    path_len: usize,
    data: Box<[HVCPoly]>,
}

impl ShareOpening {
    pub fn digits(&self) -> &[HVCPoly] {
        &self.data[..LABEL_POLYS * DGT_WIDTH]
    }

    pub fn path_node(&self, level: usize) -> (&[HVCPoly], &[HVCPoly]) {
        let base = LABEL_POLYS * DGT_WIDTH + level * 2 * HVC_WIDTH;
        (
            &self.data[base..base + HVC_WIDTH],
            &self.data[base + HVC_WIDTH..base + 2 * HVC_WIDTH],
        )
    }

    pub fn data_mut(&mut self) -> &mut [HVCPoly] {
        &mut self.data
    }

    /// Project the digit sums back to `Σ labels` mod q_dgt.
    pub fn project_labels(&self) -> Vec<DgtNTTPoly> {
        (0..LABEL_POLYS)
            .map(|k| project_share_poly(&self.data[k * DGT_WIDTH..(k + 1) * DGT_WIDTH]))
            .collect()
    }
}

/// Commit to all `n_lanes` shares: hash each share to its label, one leaf per
/// lane over the label digits, chipmunk tree above, one decomposed path per
/// lane. The root is what the client signs.
pub fn commit_shares(pp: &ShareCommitmentParams, shares: &[Share]) -> (HVCPoly, Vec<SharePath>) {
    assert_eq!(shares.len(), pp.n_lanes);
    for s in shares {
        assert_eq!(s.len(), pp.block_len, "each share must have block_len polys");
    }
    let lane_labels: Vec<HVCPoly> = shares
        .par_iter()
        .map(|share| pp.leaf_hash.hash(&label_digits(&pp.hash_share(share))))
        .collect();
    let mut labels = vec![HVCPoly::default(); pp.n_leaves];
    labels[..pp.n_lanes].copy_from_slice(&lane_labels);

    let tree = Tree::<HVCHash>::new_with_leaf_nodes(&labels, &pp.hasher);
    let root = tree.root();

    let path_len = pp.path_len();
    let paths: Vec<SharePath> = (0..pp.n_lanes)
        .into_par_iter()
        .map(|j| {
            let raw_path = tree.gen_proof(j);
            debug_assert_eq!(raw_path.nodes.len(), path_len);
            let mut nodes = Vec::with_capacity(path_len * 2 * HVC_WIDTH);
            for (l, r) in raw_path.nodes.iter() {
                nodes.extend_from_slice(&l.decompose_r());
                nodes.extend_from_slice(&r.decompose_r());
            }
            SharePath {
                lane_index: j,
                nodes: nodes.into_boxed_slice(),
            }
        })
        .collect();

    (root, paths)
}

/// Materialize a client's opening without tree hashing: shape and fresh-bound
/// checks plus the `A·s` label. Enough to fold safely — by linearity one hash
/// of the *aggregate* proves the same thing as ρ per-client hashes.
pub fn open_share(
    pp: &ShareCommitmentParams,
    lane: usize,
    share: &[DgtNTTPoly],
    path: &SharePath,
) -> Option<ShareOpening> {
    if lane >= pp.n_lanes || path.lane_index != lane || share.len() != pp.block_len {
        return None;
    }
    if path.nodes.len() != pp.path_len() * 2 * HVC_WIDTH {
        return None;
    }
    // Path digits come from the client, so bound them before they enter a sum
    // that is checked against β_agg.
    if !path.nodes.iter().all(|p| p.infinity_norm() <= ZETA) {
        return None;
    }
    let mut data = label_digits(&pp.hash_share(share));
    data.extend_from_slice(&path.nodes);
    Some(ShareOpening {
        lane_index: lane,
        path_len: pp.path_len(),
        data: data.into_boxed_slice(),
    })
}

/// Verify one client's `(share, path)` against its signed root at the fresh
/// digit bound ζ. `None` rejects the client. This is the attribution path — a
/// lane runs it only when the aggregate fails to open.
pub fn ingest_share(
    pp: &ShareCommitmentParams,
    root: &HVCPoly,
    lane: usize,
    share: &[DgtNTTPoly],
    path: &SharePath,
) -> Option<ShareOpening> {
    let o = open_share(pp, lane, share, path)?;
    if verify_walk(pp, root, &o, ZETA) {
        Some(o)
    } else {
        None
    }
}

/// A lane's post against the sum of signed roots: the digit-domain opening
/// must walk to `summed_root` at bound `ρ_max·ζ`, and the posted share sum
/// must hash to the labels those digits project to. Binding of `share_sum`
/// additionally needs the caller's shortness gate (systematic lanes) or the
/// reconstruction bound (parity lanes).
pub fn verify_aggregated(
    pp: &ShareCommitmentParams,
    summed_root: &HVCPoly,
    share_sum: &[DgtNTTPoly],
    o: &ShareOpening,
) -> bool {
    if share_sum.len() != pp.block_len {
        return false;
    }
    if !verify_walk(pp, summed_root, o, pp.beta_agg()) {
        return false;
    }
    pp.hash_share(share_sum) == o.project_labels()
}

fn verify_walk(pp: &ShareCommitmentParams, root: &HVCPoly, o: &ShareOpening, digit_bound: u32) -> bool {
    if o.lane_index >= pp.n_lanes
        || o.path_len != pp.path_len()
        || o.data.len() != pp.data_polys()
    {
        return false;
    }
    // The norm gate is the tree layer's binding premise: an out-of-range digit
    // hashes to the same mod-q label but projects to a different integer.
    if !o.data.iter().all(|p| p.infinity_norm() <= digit_bound) {
        return false;
    }

    let leaf_label = pp.leaf_hash.hash(o.digits());

    let path_len = o.path_len;
    if path_len == 0 {
        return *root == leaf_label;
    }

    let (top_l, top_r) = o.path_node(0);
    if pp.hasher.hash_separate_inputs(top_l, top_r) != *root {
        return false;
    }

    let pos = position_list(o.lane_index, path_len);
    for i in 1..path_len {
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

    let last = path_len - 1;
    let (last_l, last_r) = o.path_node(last);
    let stored_running = if pos[path_len] {
        HVCPoly::projection_r(last_r)
    } else {
        HVCPoly::projection_r(last_l)
    };
    stored_running == leaf_label
}

fn zeroed_polys(total: usize) -> Vec<HVCPoly> {
    unsafe {
        let layout = std::alloc::Layout::array::<HVCPoly>(total).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut HVCPoly;
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Vec::from_raw_parts(ptr, total, total)
    }
}

/// Running digit-wise sum. Coefficients accumulate as unreduced `i32`
/// (`ρ·q_hvc/2` stays far inside the type) and are centered once by
/// [`ShareOpeningAcc::finish`].
pub struct ShareOpeningAcc {
    lane_index: usize,
    path_len: usize,
    data: Vec<HVCPoly>,
}

impl ShareOpeningAcc {
    pub fn zero(pp: &ShareCommitmentParams, lane: usize) -> Self {
        Self {
            lane_index: lane,
            path_len: pp.path_len(),
            data: zeroed_polys(pp.data_polys()),
        }
    }

    pub fn add(&mut self, o: &ShareOpening) {
        assert_eq!(o.lane_index, self.lane_index);
        assert_eq!(o.path_len, self.path_len);
        debug_assert_eq!(o.data.len(), self.data.len());
        for (slot, poly) in o.data.iter().enumerate() {
            wrapping_add_avx2(self.data[slot].coeffs_mut(), poly.coeffs());
        }
    }

    /// Fold two partial sums — the reduce step of a parallel lane round.
    pub fn merge(&mut self, other: &Self) {
        assert_eq!(other.lane_index, self.lane_index);
        debug_assert_eq!(other.data.len(), self.data.len());
        for (slot, poly) in other.data.iter().enumerate() {
            wrapping_add_avx2(self.data[slot].coeffs_mut(), poly.coeffs());
        }
    }

    pub fn finish(mut self) -> ShareOpening {
        let q = HVC_MODULUS;
        let half = q / 2;
        for poly in self.data.iter_mut() {
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
        ShareOpening {
            lane_index: self.lane_index,
            path_len: self.path_len,
            data: self.data.into_boxed_slice(),
        }
    }
}

/// Digit-wise sum of materialized openings — the batch form of
/// [`ShareOpeningAcc`].
pub fn sum_openings(os: &[&ShareOpening]) -> ShareOpening {
    assert!(!os.is_empty());
    let mut acc = ShareOpeningAcc {
        lane_index: os[0].lane_index,
        path_len: os[0].path_len,
        data: zeroed_polys(os[0].data.len()),
    };
    for o in os {
        acc.add(o);
    }
    acc.finish()
}

/// Wire size of a fresh per-lane path at 7 bits/coeff (digits ζ-bounded).
pub fn fresh_path_packed_len(n_lanes: usize) -> usize {
    let path_len = n_lanes.next_power_of_two().max(2).trailing_zeros() as usize;
    let polys = path_len * 2 * HVC_WIDTH;
    (polys * POLY_N * bits_for_signed(ZETA) as usize).div_ceil(8)
}

/// Wire size of a lane's post: the share-domain sum (flat 8 B/coeff, the RS
/// wire rate) plus the digit-domain opening (aggregated digits ≤ ρ_max·ζ).
pub fn lane_post_packed_len(block_len: usize, n_lanes: usize, rho_max: usize) -> usize {
    let path_len = n_lanes.next_power_of_two().max(2).trailing_zeros() as usize;
    let polys = LABEL_POLYS * DGT_WIDTH + path_len * 2 * HVC_WIDTH;
    block_len * POLY_N * 8
        + (polys * POLY_N * bits_for_signed(rho_max as u32 * ZETA) as usize).div_ceil(8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rs::{Rs, RsParams};
    use chipmunk_code::KahePoly;

    fn rand_share(rng: &mut ChaCha20Rng, len: usize) -> Share {
        (0..len).map(|_| DgtNTTPoly::rand_ntt_poly(rng)).collect()
    }

    fn params(block_len: usize, n_lanes: usize, rho_max: usize) -> ShareCommitmentParams {
        ShareCommitmentParams::from_seed([7u8; 32], block_len, n_lanes, rho_max)
    }

    #[test]
    fn decompose_project_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        for _ in 0..8 {
            let p = DgtNTTPoly::rand_ntt_poly(&mut rng);
            let digits = decompose_share_poly(&p);
            assert!(digits.iter().all(|d| d.infinity_norm() <= ZETA));
            assert_eq!(project_share_poly(&digits), p);
        }
        let embedded = DgtNTTPoly::from_kahe(&KahePoly::rand_poly(&mut rng));
        assert_eq!(
            project_share_poly(&decompose_share_poly(&embedded)),
            embedded
        );
    }

    #[test]
    fn projection_is_linear_and_i128_safe() {
        let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
        let rho = 300;
        let polys: Vec<DgtNTTPoly> = (0..rho)
            .map(|_| DgtNTTPoly::rand_ntt_poly(&mut rng))
            .collect();
        let mut digit_sum = [HVCPoly::default(); DGT_WIDTH];
        for p in &polys {
            for (acc, d) in digit_sum.iter_mut().zip(decompose_share_poly(p)) {
                *acc = *acc + d;
            }
        }
        assert!(digit_sum
            .iter()
            .all(|d| d.infinity_norm() <= rho as u32 * ZETA));
        let want = polys
            .iter()
            .skip(1)
            .fold(polys[0], |acc, p| acc + *p);
        assert_eq!(project_share_poly(&digit_sum), want);

        // Adversarial magnitude: every digit at ±ρ_max·ζ. Wrong under i64.
        let hi = HVCPoly::from_coeffs([10_200i32; POLY_N]);
        let lo = HVCPoly::from_coeffs([-10_200i32; POLY_N]);
        let extreme: Vec<HVCPoly> = (0..DGT_WIDTH)
            .map(|w| if w % 2 == 0 { hi } else { lo })
            .collect();
        let mut acc: i128 = extreme[DGT_WIDTH - 1].coeffs()[0] as i128;
        for d in extreme.iter().rev().skip(1) {
            acc = acc * BASE as i128 + d.coeffs()[0] as i128;
        }
        let want = acc.rem_euclid(DGT_MODULUS as i128) as u64;
        assert_eq!(project_share_poly(&extreme).coeffs()[0], want);
    }

    #[test]
    fn commit_verify_fresh_per_lane() {
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        for n_lanes in [2usize, 5, 6, 16] {
            let block_len = 3;
            let pp = params(block_len, n_lanes, 300);
            let shares: Vec<Share> = (0..n_lanes)
                .map(|_| rand_share(&mut rng, block_len))
                .collect();
            let (root, paths) = commit_shares(&pp, &shares);
            for j in 0..n_lanes {
                assert!(
                    ingest_share(&pp, &root, j, &shares[j], &paths[j]).is_some(),
                    "n_lanes={n_lanes} lane={j}"
                );
            }
            if n_lanes >= 2 {
                assert!(ingest_share(&pp, &root, 1, &shares[0], &paths[0]).is_none());
                let mut bad_share = shares[0].clone();
                bad_share[0] = bad_share[0] + shares[1][0];
                assert!(ingest_share(&pp, &root, 0, &bad_share, &paths[0]).is_none());
                let mut bad_path = paths[0].clone();
                let mut nodes = bad_path.nodes.to_vec();
                let mut c = *nodes[0].coeffs();
                c[0] += 1;
                nodes[0] = HVCPoly::from_coeffs(c);
                bad_path.nodes = nodes.into_boxed_slice();
                assert!(ingest_share(&pp, &root, 0, &shares[0], &bad_path).is_none());
            }
        }
    }

    #[test]
    fn sum_of_openings_verifies_against_summed_roots() {
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let (rho, block_len, n_lanes) = (20usize, 4usize, 16usize);
        let pp = params(block_len, n_lanes, 300);
        let rs = RsParams::new(14, n_lanes);

        let mut roots = Vec::with_capacity(rho);
        let mut per_lane: Vec<Vec<ShareOpening>> = vec![Vec::with_capacity(rho); n_lanes];
        let mut raw_shares: Vec<Vec<Share>> = vec![Vec::with_capacity(rho); n_lanes];
        for _ in 0..rho {
            let ctxt = rand_share(&mut rng, 56);
            let shares = Rs::encode(&rs, &ctxt);
            assert_eq!(shares[0].len(), block_len);
            let (root, paths) = commit_shares(&pp, &shares);
            for j in 0..n_lanes {
                let o = ingest_share(&pp, &root, j, &shares[j], &paths[j]).expect("honest share");
                per_lane[j].push(o);
                raw_shares[j].push(shares[j].clone());
            }
            roots.push(root);
        }
        let summed_root = roots.iter().skip(1).fold(roots[0], |acc, r| acc + *r);

        for j in 0..n_lanes {
            let refs: Vec<&ShareOpening> = per_lane[j].iter().collect();
            let agg = sum_openings(&refs);
            let share_refs: Vec<&[DgtNTTPoly]> =
                raw_shares[j].iter().map(|s| s.as_slice()).collect();
            let share_sum = Rs::sum_shares(&share_refs);
            assert!(
                verify_aggregated(&pp, &summed_root, &share_sum, &agg),
                "lane {j}"
            );
            // A tampered share sum no longer hashes to the projected labels.
            let mut bad = share_sum.clone();
            bad[0] = bad[0] + share_sum[0];
            assert!(!verify_aggregated(&pp, &summed_root, &bad, &agg));
        }
    }

    /// The smuggling attack the norm gate exists for: shifting a digit by
    /// ±q_hvc leaves every hash identical (mod-q residue) but changes the
    /// projected label. Only the ∞-norm bound rejects it.
    #[test]
    fn out_of_range_digit_with_identical_hash_is_rejected() {
        let mut rng = ChaCha20Rng::from_seed([5u8; 32]);
        let (block_len, n_lanes) = (3usize, 16usize);
        let pp = params(block_len, n_lanes, 300);
        let shares: Vec<Share> = (0..n_lanes)
            .map(|_| rand_share(&mut rng, block_len))
            .collect();
        let (root, paths) = commit_shares(&pp, &shares);
        let mut o = ingest_share(&pp, &root, 0, &shares[0], &paths[0]).unwrap();
        assert!(verify_aggregated(&pp, &root, &shares[0], &o));

        let before = o.project_labels();
        let d = &mut o.data_mut()[0];
        let mut c = *d.coeffs();
        c[0] -= HVC_MODULUS;
        *d = HVCPoly::from_coeffs(c);
        assert!(!verify_aggregated(&pp, &root, &shares[0], &o));
        assert_ne!(o.project_labels(), before);
    }

    #[test]
    fn leaf_hash_is_linear() {
        let mut rng = ChaCha20Rng::from_seed([6u8; 32]);
        let pp = params(2, 4, 300);
        let a = label_digits(&pp.hash_share(&rand_share(&mut rng, 2)));
        let b = label_digits(&pp.hash_share(&rand_share(&mut rng, 2)));
        let sum: Vec<HVCPoly> = a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect();
        assert_eq!(
            pp.leaf_hash.hash(&sum),
            pp.leaf_hash.hash(&a) + pp.leaf_hash.hash(&b)
        );
    }

    /// The Ajtai layer is linear over shares: the label of a sum is the sum of
    /// labels, which is what lets one aggregate check stand in for ρ.
    #[test]
    fn share_hash_is_linear() {
        let mut rng = ChaCha20Rng::from_seed([8u8; 32]);
        let pp = params(4, 16, 300);
        let a = rand_share(&mut rng, 4);
        let b = rand_share(&mut rng, 4);
        let sum: Share = a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect();
        let want: Vec<DgtNTTPoly> = pp
            .hash_share(&a)
            .iter()
            .zip(pp.hash_share(&b).iter())
            .map(|(x, y)| *x + *y)
            .collect();
        assert_eq!(pp.hash_share(&sum), want);
    }

    #[test]
    #[should_panic(expected = "exceeds the digest ring")]
    fn rho_beyond_the_ring_is_rejected_at_setup() {
        params(4, 16, 1 << 20);
    }

    #[test]
    #[should_panic(expected = "exceeds the digit budget")]
    fn rho_beyond_the_digit_budget_is_rejected_at_setup() {
        params(4, 16, 603);
    }
}
