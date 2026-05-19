//! Additive Multi-Set Encoding (paper §3, Fig. 1).
//!
//! Three matrices `(C, K, V)` of shape `γ × δ` over `Z_q`. For each element
//! `x ∈ Z_q` in the multiset:
//!
//! ```text
//!   r ← Z_q        (per-element randomness)
//!   for i ∈ [γ]:
//!     j := PRF(prf_key, (i, r)) mod δ
//!     C[i,j] += 1; K[i,j] += r; V[i,j] += x
//! ```
//!
//! `Decode` peels cells with `C[i*, j*] = 1`: it reads `r := K[i*, j*]`,
//! `x := V[i*, j*]`, emits `x`, and subtracts that element's contribution
//! from every row by re-running PRF. Theorem 3 correctness:
//! `2^{-(γ-2) log ρ} + negl(λ)`. The `negl(λ)` term is the PRF advantage; in
//! v1 the PRF is SHA-256 keyed by `prf_key`.
//!
//! v1 fixes the universe to `Z_q` (single-symbol elements); multi-symbol
//! elements (`ξ > 1`) compose by running independent MSE instances at the
//! application layer.
//!
//! `pack` / `unpack` flatten/restore `(C, K, V)` as a sequence of `KahePoly`
//! coefficient slots so the encoding rides over the flashnet protocol's
//! KAHE ciphertext stream. Sum-of-encodings is pointwise add over the
//! `KahePoly`s, matching the paper's group-additive structure.
//!
//! MSE arithmetic runs in `Z_t` where `t = T_MODULUS_DEFAULT` is the KAHE
//! plaintext modulus, since the protocol returns `Σ m mod t`. The MSE values
//! must remain consistent under that reduction.

use chipmunk_code::{KahePoly, Polynomial, N};
use rand::Rng;
use sha2::{Digest, Sha256};

use crate::kahe::T_MODULUS_DEFAULT;

const T: i32 = T_MODULUS_DEFAULT as i32;
const T_OVER_TWO: i32 = T / 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MseParams {
    pub gamma: usize,
    pub delta: usize,
    pub prf_key: [u8; 32],
}

impl MseParams {
    pub fn new(gamma: usize, delta: usize, prf_key: [u8; 32]) -> Self {
        assert!(gamma >= 2, "γ must be ≥ 2 (Theorem 3 needs γ ≥ 2)");
        assert!(delta >= 1, "δ must be ≥ 1");
        Self {
            gamma,
            delta,
            prf_key,
        }
    }

    pub fn total_cells(&self) -> usize {
        self.gamma * self.delta
    }

    pub fn total_scalars(&self) -> usize {
        3 * self.total_cells()
    }

    /// Advisory: largest `N_clients` for which the C matrix stays
    /// representable without mod-q wrap. Peeling is probabilistic regardless;
    /// `decode()` returns `Err(PeelStalled)` when recovery fails for any
    /// reason.
    pub fn max_clients(&self) -> u32 {
        (T as u32).saturating_sub(1)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MseEncoding {
    pub params: MseParams,
    /// Counters per cell; row-major `gamma × delta`.
    pub c: Vec<i32>,
    /// Randomness accumulator per cell.
    pub k: Vec<i32>,
    /// Element accumulator per cell.
    pub v: Vec<i32>,
}

#[derive(Debug, PartialEq)]
pub enum MseError {
    /// Peeling stalled: cells remain non-zero but no pure cell exists.
    PeelStalled,
    /// `params` of two encodings disagree under `add_assign`.
    ParamsMismatch,
}

#[inline]
fn reduce(x: i64) -> i32 {
    let mut r = x.rem_euclid(T as i64) as i32;
    if r > T_OVER_TWO {
        r -= T;
    }
    r
}

fn prf_bucket(key: &[u8; 32], row: usize, r: i32, delta: usize) -> usize {
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
        Self {
            c: vec![0; n],
            k: vec![0; n],
            v: vec![0; n],
            params,
        }
    }

    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.params.delta + col
    }

    /// Insert one element `x ∈ Z_q` with fresh randomness `r ← Z_q`.
    pub fn insert<R: Rng>(&mut self, rng: &mut R, x: i32) {
        let r: i32 = reduce(rng.gen_range(0..T as i64));
        self.insert_with_r(x, r);
    }

    /// Insert with caller-supplied randomness — useful for deterministic tests.
    pub fn insert_with_r(&mut self, x: i32, r: i32) {
        let x = reduce(x as i64);
        let r = reduce(r as i64);
        for row in 0..self.params.gamma {
            let col = prf_bucket(&self.params.prf_key, row, r, self.params.delta);
            let idx = self.idx(row, col);
            self.c[idx] = reduce(self.c[idx] as i64 + 1);
            self.k[idx] = reduce(self.k[idx] as i64 + r as i64);
            self.v[idx] = reduce(self.v[idx] as i64 + x as i64);
        }
    }

    /// Pointwise add another encoding into self. Both must share `params`.
    pub fn add_assign(&mut self, other: &Self) -> Result<(), MseError> {
        if self.params.gamma != other.params.gamma
            || self.params.delta != other.params.delta
            || self.params.prf_key != other.params.prf_key
        {
            return Err(MseError::ParamsMismatch);
        }
        for i in 0..self.c.len() {
            self.c[i] = reduce(self.c[i] as i64 + other.c[i] as i64);
            self.k[i] = reduce(self.k[i] as i64 + other.k[i] as i64);
            self.v[i] = reduce(self.v[i] as i64 + other.v[i] as i64);
        }
        Ok(())
    }

    /// Peel pure cells until exhausted. Returns the recovered multiset
    /// (sorted) or `Err(PeelStalled)` if any cell remains nonzero after no
    /// further pure cell can be found.
    pub fn decode(&self) -> Result<Vec<i32>, MseError> {
        let mut c = self.c.clone();
        let mut k = self.k.clone();
        let mut v = self.v.clone();

        let mut queue: Vec<(usize, usize)> = Vec::new();
        for row in 0..self.params.gamma {
            for col in 0..self.params.delta {
                let idx = row * self.params.delta + col;
                if c[idx] == 1 {
                    queue.push((row, col));
                }
            }
        }

        let mut emitted = Vec::new();
        while let Some((row, col)) = queue.pop() {
            let idx = row * self.params.delta + col;
            if c[idx] != 1 {
                continue;
            }
            let r_star = k[idx];
            let x_star = v[idx];
            emitted.push(x_star);
            for i in 0..self.params.gamma {
                let j = prf_bucket(&self.params.prf_key, i, r_star, self.params.delta);
                let cell = i * self.params.delta + j;
                c[cell] = reduce(c[cell] as i64 - 1);
                k[cell] = reduce(k[cell] as i64 - r_star as i64);
                v[cell] = reduce(v[cell] as i64 - x_star as i64);
                if c[cell] == 1 {
                    queue.push((i, j));
                }
            }
        }

        if c.iter().any(|&x| x != 0) || k.iter().any(|&x| x != 0) || v.iter().any(|&x| x != 0) {
            return Err(MseError::PeelStalled);
        }
        emitted.sort();
        Ok(emitted)
    }

    /// Number of `KahePoly`s required to pack this encoding.
    pub fn n_polys(params: &MseParams) -> usize {
        params.total_scalars().div_ceil(N)
    }

    /// Flatten `(C, K, V)` into KahePoly coefficient slots (C first, then K,
    /// then V; each row-major).
    pub fn pack(&self) -> Vec<KahePoly> {
        let total = self.params.total_scalars();
        let n_polys = total.div_ceil(N);
        let mut polys = Vec::with_capacity(n_polys);
        let mut buf = [0i32; N];
        let mut written = 0usize;
        let feed = |val: i32, buf: &mut [i32; N], written: &mut usize, polys: &mut Vec<KahePoly>| {
            buf[*written % N] = val;
            *written += 1;
            if *written % N == 0 {
                polys.push(KahePoly::from_coeffs(*buf));
                *buf = [0i32; N];
            }
        };
        for &v in &self.c {
            feed(v, &mut buf, &mut written, &mut polys);
        }
        for &v in &self.k {
            feed(v, &mut buf, &mut written, &mut polys);
        }
        for &v in &self.v {
            feed(v, &mut buf, &mut written, &mut polys);
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
        // Lift each poly's coefficients into signed reduced form so summed
        // encodings (over the protocol) come back to canonical Z_q reps.
        let mut flat: Vec<i32> = Vec::with_capacity(polys.len() * N);
        for p in polys {
            let mut q = *p;
            q.normalize();
            flat.extend_from_slice(q.coeffs());
        }

        let cells = params.total_cells();
        let c = flat[..cells].to_vec();
        let k = flat[cells..2 * cells].to_vec();
        let v = flat[2 * cells..3 * cells].to_vec();
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

    fn params_for(gamma: usize, delta: usize, seed: u8) -> MseParams {
        MseParams::new(gamma, delta, [seed; 32])
    }

    #[test]
    fn single_element_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        let pp = params_for(4, 16, 1);
        let mut enc = MseEncoding::new(pp.clone());
        enc.insert(&mut rng, 42);
        let recovered = enc.decode().expect("decode");
        assert_eq!(recovered, vec![42]);
    }

    #[test]
    fn multi_element_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        let pp = params_for(4, 64, 2);
        let mut enc = MseEncoding::new(pp);
        let mut elements = vec![1, 7, 13, 19, 100, 200, 300, 400];
        for &x in &elements {
            enc.insert(&mut rng, x);
        }
        let mut recovered = enc.decode().expect("decode");
        elements.sort();
        recovered.sort();
        assert_eq!(recovered, elements);
    }

    #[test]
    fn union_via_pointwise_add() {
        let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
        let pp = params_for(4, 32, 3);
        let mut a = MseEncoding::new(pp.clone());
        let mut b = MseEncoding::new(pp.clone());
        for x in [1, 5, 9, 17] {
            a.insert(&mut rng, x);
        }
        for x in [2, 6, 11, 23] {
            b.insert(&mut rng, x);
        }
        a.add_assign(&b).unwrap();
        let mut recovered = a.decode().expect("decode");
        recovered.sort();
        assert_eq!(recovered, vec![1, 2, 5, 6, 9, 11, 17, 23]);
    }

    #[test]
    fn pack_unpack_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let pp = params_for(4, 32, 5);
        let mut enc = MseEncoding::new(pp.clone());
        for x in [11, 22, 33, 44, 55] {
            enc.insert(&mut rng, x);
        }
        let polys = enc.pack();
        assert_eq!(polys.len(), MseEncoding::n_polys(&pp));
        let restored = MseEncoding::unpack(&pp, &polys);
        assert_eq!(restored, enc);
        let mut got = restored.decode().expect("decode");
        got.sort();
        assert_eq!(got, vec![11, 22, 33, 44, 55]);
    }

    #[test]
    fn pack_sum_unpacks_to_union() {
        // Pack two encodings, pointwise-sum the packed KahePoly streams, unpack,
        // decode → multiset union. Mirrors the protocol's RingOtp aggregation.
        let mut rng = ChaCha20Rng::from_seed([6u8; 32]);
        let pp = params_for(4, 32, 7);
        let mut a = MseEncoding::new(pp.clone());
        let mut b = MseEncoding::new(pp.clone());
        for x in [100, 200, 300] {
            a.insert(&mut rng, x);
        }
        for x in [400, 500, 600] {
            b.insert(&mut rng, x);
        }
        let pa = a.pack();
        let pb = b.pack();
        let summed: Vec<KahePoly> = pa
            .iter()
            .zip(pb.iter())
            .map(|(x, y)| *x + *y)
            .collect();
        let unioned = MseEncoding::unpack(&pp, &summed);
        let mut got = unioned.decode().expect("decode");
        got.sort();
        assert_eq!(got, vec![100, 200, 300, 400, 500, 600]);
    }
}
