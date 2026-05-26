//! Additive Multi-Set Encoding (paper §3, Fig. 1).
//!
//! Matrices `(C, K_0, … , K_{L-1}, V_0, … , V_{ξ-1})` of shape `γ × δ`
//! over `Z_t`. The per-element randomness `r` lives in `Z_{t^L}` and is
//! split into `L` base-`t` limbs; each element is a `ξ`-tuple
//! `(x_0, … , x_{ξ-1})` with each symbol in `Z_t`, so a `payload_symbols
//! · log₂ t`-bit message rides one MSE insert. C and K are shared
//! across symbols — they're determined by `r` alone — so the overhead
//! amortises over ξ.
//!
//! ```text
//!   r ← Z_{t^L}                            (per-element randomness)
//!   r_ℓ := (r / t^ℓ) mod t                 (ℓ ∈ [L])
//!   for i ∈ [γ]:
//!     j := PRF(prf_key, (i, r)) mod δ
//!     C[i,j] += 1; K_ℓ[i,j] += r_ℓ
//!     V_s[i,j] += x_s                      (s ∈ [ξ])
//! ```
//!
//! `Decode` peels cells with `C[i*, j*] = 1`: it reads each limb
//! `r_ℓ := K_ℓ[i*,j*]`, reconstructs `r = Σ r_ℓ · t^ℓ`, reads each
//! payload symbol `x_s := V_s[i*, j*]`, emits the tuple
//! `(x_0, … , x_{ξ-1})`, and subtracts that element's contribution
//! from every row by re-running PRF. Theorem 3 correctness:
//! `2^{-(γ-2) log ρ} + negl(λ)` — independent of ξ (peeling decisions
//! live on C only).
//!
//! Widening L does not require growing C: C is a per-cell hit counter,
//! bounded by `(n/δ)` independent of how `r` is sampled. K-limb cells
//! store base-`t` digits, each accumulating in the same `Z_t` as the
//! single-limb design. So L only multiplies the r-space cardinality,
//! not per-cell magnitudes. Likewise widening ξ adds `ξ` extra `V`
//! matrices but leaves C and K unchanged.
//!
//! `pack` / `unpack` flatten/restore the matrices as a sequence of
//! `KahePoly` coefficient slots (`C, K_0…K_{L-1}, V_0…V_{ξ-1}`, each
//! row-major) so the encoding rides over the flashnet protocol's KAHE
//! ciphertext stream. Sum-of-encodings is pointwise add over the
//! `KahePoly`s.
//!
//! MSE cell arithmetic runs in `Z_t` where `t = T_MODULUS_DEFAULT` is
//! the KAHE plaintext modulus. With `t = 2^18` and `L = K_LIMBS = 2`,
//! `r ∈ Z_{t^2} = Z_{2^36}`, so r-arithmetic fits in a `u64`. If a wider
//! r-space is ever needed, bump `K_LIMBS` (and the `u64` arithmetic with
//! it) — `L = 2` is plenty for the protocol's current operating point.

use chipmunk_code::{KahePoly, Polynomial, N};
use rand::Rng;
use sha2::{Digest, Sha256};

use crate::kahe::T_MODULUS_DEFAULT;

const T: i32 = T_MODULUS_DEFAULT as i32;
const T_OVER_TWO: i32 = T / 2;
/// Number of base-`t` limbs for `r`: fixed at 2. `r ∈ Z_{t^2}`.
pub const K_LIMBS: usize = 2;
/// Bits of payload carried per symbol.
pub const BITS_PER_SYMBOL: usize = T_MODULUS_DEFAULT.trailing_zeros() as usize;

/// Per-row bucket-count layout. `Uniform` is the standard IBLT shape;
/// `Geometric { shrink }` multiplies row `i`'s bucket count by
/// `shrink^i` (rounded, floored at 1) for a smaller structure at the
/// cost of some peeling margin in the later (denser) rows. `shrink =
/// 1.0` reproduces `Uniform`; `shrink = 0.5` halves each row.
#[derive(Clone, Debug, PartialEq)]
pub enum RowLayout {
    Uniform,
    Geometric { shrink: f64 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct MseParams {
    pub gamma: usize,
    /// Number of buckets in row 0. Subsequent rows follow `row_layout`.
    pub delta: usize,
    /// Symbols per element. Each symbol is one `Z_t` value, so total
    /// payload per element is `payload_symbols · BITS_PER_SYMBOL` bits.
    pub payload_symbols: usize,
    pub row_layout: RowLayout,
    pub prf_key: [u8; 32],
}

impl MseParams {
    /// Uniform `γ × δ` layout (standard IBLT shape).
    pub fn new(
        gamma: usize,
        delta: usize,
        payload_symbols: usize,
        prf_key: [u8; 32],
    ) -> Self {
        Self::with_layout(gamma, delta, payload_symbols, RowLayout::Uniform, prf_key)
    }

    pub fn with_layout(
        gamma: usize,
        delta: usize,
        payload_symbols: usize,
        row_layout: RowLayout,
        prf_key: [u8; 32],
    ) -> Self {
        assert!(gamma >= 2, "γ must be ≥ 2 (Theorem 3 needs γ ≥ 2)");
        assert!(delta >= 1, "δ must be ≥ 1");
        assert!(payload_symbols >= 1, "payload_symbols must be ≥ 1");
        Self {
            gamma,
            delta,
            payload_symbols,
            row_layout,
            prf_key,
        }
    }

    /// Convenience: pick `payload_symbols` to fit a message of
    /// `bits` bits.
    pub fn payload_symbols_for_bits(bits: usize) -> usize {
        bits.div_ceil(BITS_PER_SYMBOL)
    }

    /// Buckets in row `row`.
    pub fn row_delta(&self, row: usize) -> usize {
        match self.row_layout {
            RowLayout::Uniform => self.delta,
            RowLayout::Geometric { shrink } => {
                ((self.delta as f64) * shrink.powi(row as i32)).round().max(1.0) as usize
            }
        }
    }

    /// Offset (in cells) of row `row` within a per-matrix flat layout.
    pub fn row_offset(&self, row: usize) -> usize {
        (0..row).map(|i| self.row_delta(i)).sum()
    }

    pub fn total_cells(&self) -> usize {
        (0..self.gamma).map(|i| self.row_delta(i)).sum()
    }

    /// `(1 + K_LIMBS + payload_symbols)` scalars per cell: C, K_0…K_{L-1},
    /// V_0…V_{ξ-1}.
    pub fn total_scalars(&self) -> usize {
        (1 + K_LIMBS + self.payload_symbols) * self.total_cells()
    }

    /// `t^K_LIMBS`, the size of the randomness space. Fits in `u64`
    /// since `T_MODULUS_DEFAULT^2 = 2^36 < 2^64`.
    pub fn r_space(&self) -> u64 {
        let t = T_MODULUS_DEFAULT as u64;
        let mut p: u64 = 1;
        for _ in 0..K_LIMBS {
            p *= t;
        }
        p
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MseEncoding {
    pub params: MseParams,
    /// Counters per cell; row-major `gamma × delta`.
    pub c: Vec<i32>,
    /// Randomness accumulator limbs. `k[ℓ]` has `gamma*delta` entries.
    pub k: Vec<Vec<i32>>,
    /// Payload accumulator symbols. `v[s]` has `gamma*delta` entries.
    pub v: Vec<Vec<i32>>,
}

#[derive(Debug, PartialEq)]
pub enum MseError {
    /// Peeling stalled: cells remain non-zero but no pure cell exists.
    PeelStalled,
    /// `params` of two encodings disagree under `add_assign`.
    ParamsMismatch,
    /// `payload` slice length disagrees with `params.payload_symbols`.
    PayloadArity,
}

#[inline]
fn reduce(x: i64) -> i32 {
    let mut r = x.rem_euclid(T as i64) as i32;
    if r > T_OVER_TWO {
        r -= T;
    }
    r
}

/// Canonical unsigned rep `[0, t)` from centered `[-t/2, t/2]`.
#[inline]
fn to_canonical(x: i32) -> u64 {
    (x as i64).rem_euclid(T as i64) as u64
}

fn prf_bucket(key: &[u8; 32], row: usize, r: u64, delta: usize) -> usize {
    let mut h = Sha256::new();
    h.update(key);
    h.update((row as u32).to_le_bytes());
    h.update(r.to_le_bytes());
    let digest = h.finalize();
    let v = u64::from_le_bytes(digest[..8].try_into().unwrap());
    (v % delta as u64) as usize
}

impl MseEncoding {
    pub fn new(params: MseParams) -> Self {
        let n = params.total_cells();
        let k = (0..K_LIMBS).map(|_| vec![0i32; n]).collect();
        let v = (0..params.payload_symbols).map(|_| vec![0i32; n]).collect();
        Self {
            c: vec![0; n],
            k,
            v,
            params,
        }
    }

    fn idx(&self, row: usize, col: usize) -> usize {
        self.params.row_offset(row) + col
    }

    /// Insert one element. `payload.len()` must equal
    /// `params.payload_symbols`; each entry is reduced mod `t`. Fresh
    /// randomness `r ← Z_{t^L}` is drawn from `rng`.
    pub fn insert<R: Rng>(&mut self, rng: &mut R, payload: &[i32]) {
        let r_space = self.params.r_space();
        let r: u64 = rng.gen::<u64>() % r_space;
        self.insert_with_r(payload, r);
    }

    /// Insert with caller-supplied randomness — useful for deterministic
    /// tests. `r` must be in `[0, t^K_LIMBS)`.
    pub fn insert_with_r(&mut self, payload: &[i32], r: u64) {
        assert_eq!(
            payload.len(),
            self.params.payload_symbols,
            "payload arity must match params.payload_symbols",
        );
        debug_assert!(r < self.params.r_space(), "r must be in [0, t^K_LIMBS)");
        let t = T_MODULUS_DEFAULT as u64;
        let mut limbs = [0i32; K_LIMBS];
        let mut acc = r;
        for limb in limbs.iter_mut() {
            *limb = reduce((acc % t) as i64);
            acc /= t;
        }
        let payload_reduced: Vec<i32> = payload.iter().map(|&x| reduce(x as i64)).collect();
        for row in 0..self.params.gamma {
            let col = prf_bucket(&self.params.prf_key, row, r, self.params.row_delta(row));
            let idx = self.idx(row, col);
            self.c[idx] = reduce(self.c[idx] as i64 + 1);
            for ell in 0..K_LIMBS {
                self.k[ell][idx] = reduce(self.k[ell][idx] as i64 + limbs[ell] as i64);
            }
            for s in 0..self.params.payload_symbols {
                self.v[s][idx] = reduce(self.v[s][idx] as i64 + payload_reduced[s] as i64);
            }
        }
    }

    /// Pointwise add another encoding into self. Both must share `params`.
    pub fn add_assign(&mut self, other: &Self) -> Result<(), MseError> {
        if self.params != other.params {
            return Err(MseError::ParamsMismatch);
        }
        for i in 0..self.c.len() {
            self.c[i] = reduce(self.c[i] as i64 + other.c[i] as i64);
            for ell in 0..K_LIMBS {
                self.k[ell][i] = reduce(self.k[ell][i] as i64 + other.k[ell][i] as i64);
            }
            for s in 0..self.params.payload_symbols {
                self.v[s][i] = reduce(self.v[s][i] as i64 + other.v[s][i] as i64);
            }
        }
        Ok(())
    }

    /// Peel pure cells until exhausted. Returns the recovered multiset
    /// as a `Vec` of payload tuples (each `Vec<i32>` has length
    /// `params.payload_symbols`), sorted lexicographically. Returns
    /// `Err(PeelStalled)` if any cell remains nonzero after no further
    /// pure cell can be found.
    pub fn decode(&self) -> Result<Vec<Vec<i32>>, MseError> {
        let mut c = self.c.clone();
        let mut k = self.k.clone();
        let mut v = self.v.clone();

        let mut queue: Vec<(usize, usize)> = Vec::new();
        for row in 0..self.params.gamma {
            let offset = self.params.row_offset(row);
            for col in 0..self.params.row_delta(row) {
                let idx = offset + col;
                if c[idx] == 1 {
                    queue.push((row, col));
                }
            }
        }

        let mut emitted: Vec<Vec<i32>> = Vec::new();
        let mut limb_signed = [0i32; K_LIMBS];
        let mut x_star_buf = vec![0i32; self.params.payload_symbols];
        let t_u64 = T_MODULUS_DEFAULT as u64;
        // Outer loop terminates because every iteration that hits the
        // `c[idx] == 1` branch strictly reduces the multiset still
        // encoded in the matrices: one element is emitted and subtracted
        // from its γ rows. The queue can only grow with cells that just
        // became pure as a side effect of that subtraction (`c[cell] ==
        // 1` post-decrement), so it never re-enqueues the cell we just
        // peeled. Stale `(row, col)` entries from earlier peels are
        // discarded by the `c[idx] != 1` guard. Total pure-peel
        // iterations ≤ initial multiset size, so progress is bounded.
        while let Some((row, col)) = queue.pop() {
            let idx = self.params.row_offset(row) + col;
            if c[idx] != 1 {
                continue;
            }
            let mut r_star: u64 = 0;
            let mut place: u64 = 1;
            for ell in 0..K_LIMBS {
                let signed = k[ell][idx];
                limb_signed[ell] = signed;
                r_star += to_canonical(signed) * place;
                place *= t_u64;
            }
            for s in 0..self.params.payload_symbols {
                x_star_buf[s] = v[s][idx];
            }
            emitted.push(x_star_buf.clone());
            for i in 0..self.params.gamma {
                let row_d = self.params.row_delta(i);
                let j = prf_bucket(&self.params.prf_key, i, r_star, row_d);
                let cell = self.params.row_offset(i) + j;
                c[cell] = reduce(c[cell] as i64 - 1);
                for ell in 0..K_LIMBS {
                    k[ell][cell] = reduce(k[ell][cell] as i64 - limb_signed[ell] as i64);
                }
                for s in 0..self.params.payload_symbols {
                    v[s][cell] = reduce(v[s][cell] as i64 - x_star_buf[s] as i64);
                }
                if c[cell] == 1 {
                    queue.push((i, j));
                }
            }
        }

        if c.iter().any(|&x| x != 0)
            || k.iter().any(|limb| limb.iter().any(|&x| x != 0))
            || v.iter().any(|sym| sym.iter().any(|&x| x != 0))
        {
            return Err(MseError::PeelStalled);
        }
        emitted.sort();
        Ok(emitted)
    }

    /// Number of `KahePoly`s required to pack this encoding.
    pub fn n_polys(params: &MseParams) -> usize {
        params.total_scalars().div_ceil(N)
    }

    /// Flatten `(C, K_0…K_{L-1}, V_0…V_{ξ-1})` into KahePoly coefficient
    /// slots in that order, each row-major.
    pub fn pack(&self) -> Vec<KahePoly> {
        let total = self.params.total_scalars();
        let n_polys = total.div_ceil(N);
        let mut polys = Vec::with_capacity(n_polys);
        let mut buf = [0i32; N];
        let mut written = 0usize;
        let mut feed = |val: i32, polys: &mut Vec<KahePoly>| {
            buf[written % N] = val;
            written += 1;
            if written % N == 0 {
                polys.push(KahePoly::from_coeffs(buf));
                buf = [0i32; N];
            }
        };
        for &x in &self.c {
            feed(x, &mut polys);
        }
        for limb in &self.k {
            for &x in limb {
                feed(x, &mut polys);
            }
        }
        for sym in &self.v {
            for &x in sym {
                feed(x, &mut polys);
            }
        }
        if written % N != 0 {
            polys.push(KahePoly::from_coeffs(buf));
        }
        debug_assert_eq!(polys.len(), n_polys);
        polys
    }

    /// Inverse of `pack`. `polys.len()` must equal `n_polys(params)`.
    pub fn unpack(params: &MseParams, polys: &[KahePoly]) -> Self {
        let total = params.total_scalars();
        assert_eq!(polys.len(), total.div_ceil(N));
        let mut flat: Vec<i32> = Vec::with_capacity(polys.len() * N);
        for p in polys {
            let mut q = *p;
            q.normalize();
            flat.extend_from_slice(q.coeffs());
        }

        let cells = params.total_cells();
        let c = flat[..cells].to_vec();
        let k: Vec<Vec<i32>> = (0..K_LIMBS)
            .map(|ell| flat[(1 + ell) * cells..(2 + ell) * cells].to_vec())
            .collect();
        let v_base = (1 + K_LIMBS) * cells;
        let v: Vec<Vec<i32>> = (0..params.payload_symbols)
            .map(|s| flat[v_base + s * cells..v_base + (s + 1) * cells].to_vec())
            .collect();
        Self {
            params: params.clone(),
            c,
            k,
            v,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn params_for(
        gamma: usize,
        delta: usize,
        payload_symbols: usize,
        seed: u8,
    ) -> MseParams {
        MseParams::new(gamma, delta, payload_symbols, [seed; 32])
    }

    #[test]
    fn single_element_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        let pp = params_for(4, 16, 1, 1);
        let mut enc = MseEncoding::new(pp.clone());
        enc.insert(&mut rng, &[42]);
        let recovered = enc.decode().expect("decode");
        assert_eq!(recovered, vec![vec![42]]);
    }

    #[test]
    fn multi_element_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        let pp = params_for(4, 64, 1, 2);
        let mut enc = MseEncoding::new(pp);
        let mut elements = vec![1, 7, 13, 19, 100, 200, 300, 400];
        for &x in &elements {
            enc.insert(&mut rng, &[x]);
        }
        let mut recovered: Vec<i32> = enc
            .decode()
            .expect("decode")
            .into_iter()
            .map(|t| t[0])
            .collect();
        elements.sort();
        recovered.sort();
        assert_eq!(recovered, elements);
    }

    #[test]
    fn union_via_pointwise_add() {
        let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
        let pp = params_for(4, 32, 1, 3);
        let mut a = MseEncoding::new(pp.clone());
        let mut b = MseEncoding::new(pp.clone());
        for x in [1, 5, 9, 17] {
            a.insert(&mut rng, &[x]);
        }
        for x in [2, 6, 11, 23] {
            b.insert(&mut rng, &[x]);
        }
        a.add_assign(&b).unwrap();
        let mut recovered: Vec<i32> = a
            .decode()
            .expect("decode")
            .into_iter()
            .map(|t| t[0])
            .collect();
        recovered.sort();
        assert_eq!(recovered, vec![1, 2, 5, 6, 9, 11, 17, 23]);
    }

    #[test]
    fn pack_unpack_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let pp = params_for(4, 32, 1, 5);
        let mut enc = MseEncoding::new(pp.clone());
        for x in [11, 22, 33, 44, 55] {
            enc.insert(&mut rng, &[x]);
        }
        let polys = enc.pack();
        assert_eq!(polys.len(), MseEncoding::n_polys(&pp));
        let restored = MseEncoding::unpack(&pp, &polys);
        assert_eq!(restored, enc);
        let mut got: Vec<i32> = restored
            .decode()
            .expect("decode")
            .into_iter()
            .map(|t| t[0])
            .collect();
        got.sort();
        assert_eq!(got, vec![11, 22, 33, 44, 55]);
    }

    #[test]
    fn pack_sum_unpacks_to_union() {
        let mut rng = ChaCha20Rng::from_seed([6u8; 32]);
        let pp = params_for(4, 32, 1, 7);
        let mut a = MseEncoding::new(pp.clone());
        let mut b = MseEncoding::new(pp.clone());
        for x in [100, 200, 300] {
            a.insert(&mut rng, &[x]);
        }
        for x in [400, 500, 600] {
            b.insert(&mut rng, &[x]);
        }
        let pa = a.pack();
        let pb = b.pack();
        let summed: Vec<KahePoly> = pa
            .iter()
            .zip(pb.iter())
            .map(|(x, y)| *x + *y)
            .collect();
        let unioned = MseEncoding::unpack(&pp, &summed);
        let mut got: Vec<i32> = unioned
            .decode()
            .expect("decode")
            .into_iter()
            .map(|t| t[0])
            .collect();
        got.sort();
        assert_eq!(got, vec![100, 200, 300, 400, 500, 600]);
    }

    /// 4 symbols × 18 bits ≈ 72 bits of payload per element. Verifies
    /// multi-symbol insert/decode and confirms C/K are correctly shared
    /// across the V matrices.
    #[test]
    fn multi_symbol_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([13u8; 32]);
        let pp = params_for(4, 64, 4, 17);
        let mut enc = MseEncoding::new(pp);
        let mut messages: Vec<Vec<i32>> = (0..10)
            .map(|i| vec![i * 11, i * 13 + 1, i * 17 + 2, i * 19 + 3])
            .collect();
        for m in &messages {
            enc.insert(&mut rng, m);
        }
        let mut recovered = enc.decode().expect("decode");
        messages.sort();
        recovered.sort();
        assert_eq!(recovered, messages);
    }

    /// 512-bit message → 29 symbols. Smaller multiset but exercises the
    /// fat-payload code path end-to-end.
    #[test]
    fn five_hundred_twelve_bit_message_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([23u8; 32]);
        let xi = MseParams::payload_symbols_for_bits(512);
        assert_eq!(xi, 29);
        let pp = params_for(4, 32, xi, 31);
        let mut enc = MseEncoding::new(pp);
        let mut messages: Vec<Vec<i32>> = (0..5)
            .map(|i| (0..xi).map(|s| (i * 7 + s * 13) as i32).collect())
            .collect();
        for m in &messages {
            enc.insert(&mut rng, m);
        }
        let mut recovered = enc.decode().expect("decode");
        messages.sort();
        recovered.sort();
        assert_eq!(recovered, messages);
    }

    #[test]
    fn halving_layout_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([37u8; 32]);
        let pp = MseParams::with_layout(
            5,
            128,
            1,
            RowLayout::Geometric { shrink: 0.5 },
            [43u8; 32],
        );
        // Row sizes should be 128, 64, 32, 16, 8 → total 248.
        assert_eq!(
            (0..pp.gamma).map(|i| pp.row_delta(i)).collect::<Vec<_>>(),
            vec![128, 64, 32, 16, 8],
        );
        assert_eq!(pp.total_cells(), 248);
        let mut enc = MseEncoding::new(pp);
        let mut elements: Vec<i32> = (0..20).map(|i| i * 7 + 1).collect();
        for &x in &elements {
            enc.insert(&mut rng, &[x]);
        }
        let mut recovered: Vec<i32> = enc
            .decode()
            .expect("decode")
            .into_iter()
            .map(|t| t[0])
            .collect();
        elements.sort();
        recovered.sort();
        assert_eq!(recovered, elements);
    }

    #[test]
    #[should_panic(expected = "payload arity must match")]
    fn payload_arity_mismatch_panics() {
        let mut rng = ChaCha20Rng::from_seed([29u8; 32]);
        let pp = params_for(4, 32, 3, 41);
        let mut enc = MseEncoding::new(pp);
        enc.insert(&mut rng, &[1, 2]);
    }
}
