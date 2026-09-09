//! Additive Multi-Set Encoding
//!
//! ```text
//!   r ← Z_{t^L}                            (per-element randomness)
//!   r_ℓ := (r / t^ℓ) mod t                 (ℓ ∈ [L])
//!   for i ∈ [γ]:
//!     j := PRF(prf_key, (i, r)) mod δ
//!     C[i,j] += 1; K_ℓ[i,j] += r_ℓ
//!     V_s[i,j] += x_s                      (s ∈ [ξ])
//! ```

use crate::{KahePoly, N};
use rand::Rng;
use sha2::{Digest, Sha256};

use crate::kahe::T_MODULUS_DEFAULT;

const T: i64 = T_MODULUS_DEFAULT as i64;
const T_OVER_TWO: i64 = T / 2;
/// Number of base-`t` limbs for `r`: fixed at 2. `r ∈ Z_{t^2}`.
pub const K_LIMBS: usize = 2;
/// Bits of payload carried per symbol.
pub const BITS_PER_SYMBOL: usize = T_MODULUS_DEFAULT.trailing_zeros() as usize;

#[derive(Clone, Debug, PartialEq)]
pub struct MseParams {
    pub gamma: usize,
    /// Buckets per row.
    pub delta: usize,
    pub payload_symbols: usize,
    pub prf_key: [u8; 32],
}

impl MseParams {
    /// Uniform `γ × δ` layout (standard IBLT shape).
    pub fn new(gamma: usize, delta: usize, payload_symbols: usize, prf_key: [u8; 32]) -> Self {
        assert!(gamma >= 2, "γ must be ≥ 2 (Theorem 3 needs γ ≥ 2)");
        assert!(delta >= 1, "δ must be ≥ 1");
        assert!(payload_symbols >= 1, "payload_symbols must be ≥ 1");
        Self {
            gamma,
            delta,
            payload_symbols,
            prf_key,
        }
    }

    /// Convenience: pick `payload_symbols` to fit a message of
    /// `bits` bits.
    pub fn payload_symbols_for_bits(bits: usize) -> usize {
        bits.div_ceil(BITS_PER_SYMBOL)
    }

    pub fn total_cells(&self) -> usize {
        self.gamma * self.delta
    }

    /// `(1 + K_LIMBS + payload_symbols)` scalars per cell: C, K_0…K_{L-1},
    /// V_0…V_{ξ-1}.
    pub fn total_scalars(&self) -> usize {
        (1 + K_LIMBS + self.payload_symbols) * self.total_cells()
    }

    /// `t^K_LIMBS`, the size of the randomness space. `t^2 = 2^72`, so u128.
    pub fn r_space(&self) -> u128 {
        let t = T_MODULUS_DEFAULT as u128;
        let mut p: u128 = 1;
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
    pub c: Vec<i64>,
    /// Randomness accumulator limbs. `k[ℓ]` has `gamma*delta` entries.
    pub k: Vec<Vec<i64>>,
    /// Payload accumulator symbols. `v[s]` has `gamma*delta` entries.
    pub v: Vec<Vec<i64>>,
}

#[derive(Debug, PartialEq)]
pub enum MseError {
    PeelStalled,
    ParamsMismatch,
    PayloadArity,
}

#[inline]
fn reduce(x: i64) -> i64 {
    let mut r = x.rem_euclid(T);
    if r > T_OVER_TWO {
        r -= T;
    }
    r
}

/// Canonical unsigned rep `[0, t)` from centered `[-t/2, t/2]`.
#[inline]
fn to_canonical(x: i64) -> u128 {
    x.rem_euclid(T) as u128
}

fn prf_bucket(key: &[u8; 32], row: usize, r: u128, delta: usize) -> usize {
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
        let k = (0..K_LIMBS).map(|_| vec![0i64; n]).collect();
        let v = (0..params.payload_symbols).map(|_| vec![0i64; n]).collect();
        Self {
            c: vec![0; n],
            k,
            v,
            params,
        }
    }

    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.params.delta + col
    }

    /// May be called repeatedly; one item per client is enforced by client code,
    /// not by this multiset encoding.
    pub fn insert<R: Rng>(&mut self, rng: &mut R, payload: &[i64]) {
        let r_space = self.params.r_space();
        let r: u128 = rng.gen::<u128>() % r_space;
        self.insert_with_r(payload, r);
    }

    pub fn insert_with_r(&mut self, payload: &[i64], r: u128) {
        assert_eq!(
            payload.len(),
            self.params.payload_symbols,
            "payload arity must match params.payload_symbols",
        );
        debug_assert!(r < self.params.r_space(), "r must be in [0, t^K_LIMBS)");
        let t = T_MODULUS_DEFAULT as u128;
        let mut limbs = [0i64; K_LIMBS];
        let mut acc = r;
        for limb in limbs.iter_mut() {
            *limb = reduce((acc % t) as i64);
            acc /= t;
        }
        let payload_reduced: Vec<i64> = payload.iter().map(|&x| reduce(x)).collect();
        for row in 0..self.params.gamma {
            let col = prf_bucket(&self.params.prf_key, row, r, self.params.delta);
            let idx = self.idx(row, col);
            self.c[idx] = reduce(self.c[idx] + 1);
            for ell in 0..K_LIMBS {
                self.k[ell][idx] = reduce(self.k[ell][idx] + limbs[ell]);
            }
            for s in 0..self.params.payload_symbols {
                self.v[s][idx] = reduce(self.v[s][idx] + payload_reduced[s]);
            }
        }
    }

    pub fn add_assign(&mut self, other: &Self) -> Result<(), MseError> {
        if self.params != other.params {
            return Err(MseError::ParamsMismatch);
        }
        for i in 0..self.c.len() {
            self.c[i] = reduce(self.c[i] + other.c[i]);
            for ell in 0..K_LIMBS {
                self.k[ell][i] = reduce(self.k[ell][i] + other.k[ell][i]);
            }
            for s in 0..self.params.payload_symbols {
                self.v[s][i] = reduce(self.v[s][i] + other.v[s][i]);
            }
        }
        Ok(())
    }

    pub fn decode(&self) -> Result<Vec<Vec<i64>>, MseError> {
        let mut c = self.c.clone();
        let mut k = self.k.clone();
        let mut v = self.v.clone();

        let mut queue: Vec<(usize, usize)> = Vec::new();
        for row in 0..self.params.gamma {
            for col in 0..self.params.delta {
                if c[row * self.params.delta + col] == 1 {
                    queue.push((row, col));
                }
            }
        }

        let mut emitted: Vec<Vec<i64>> = Vec::new();
        let mut limb_signed = [0i64; K_LIMBS];
        let mut x_star_buf = vec![0i64; self.params.payload_symbols];
        let t_u128 = T_MODULUS_DEFAULT as u128;
        while let Some((row, col)) = queue.pop() {
            let idx = row * self.params.delta + col;
            if c[idx] != 1 {
                continue;
            }
            let mut r_star: u128 = 0;
            let mut place: u128 = 1;
            for ell in 0..K_LIMBS {
                let signed = k[ell][idx];
                limb_signed[ell] = signed;
                r_star += to_canonical(signed) * place;
                place *= t_u128;
            }
            for s in 0..self.params.payload_symbols {
                x_star_buf[s] = v[s][idx];
            }
            emitted.push(x_star_buf.clone());
            for i in 0..self.params.gamma {
                let j = prf_bucket(&self.params.prf_key, i, r_star, self.params.delta);
                let cell = i * self.params.delta + j;
                c[cell] = reduce(c[cell] - 1);
                for ell in 0..K_LIMBS {
                    k[ell][cell] = reduce(k[ell][cell] - limb_signed[ell]);
                }
                for s in 0..self.params.payload_symbols {
                    v[s][cell] = reduce(v[s][cell] - x_star_buf[s]);
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

    pub fn n_polys(params: &MseParams) -> usize {
        params.total_scalars().div_ceil(N)
    }

    /// Cover-traffic message: zero polys
    pub fn cover(params: &MseParams) -> Vec<KahePoly> {
        Self::new(params.clone()).pack()
    }

    pub fn pack(&self) -> Vec<KahePoly> {
        let total = self.params.total_scalars();
        let n_polys = total.div_ceil(N);
        let mut polys = Vec::with_capacity(n_polys);
        let mut buf = [0i64; N];
        let mut written = 0usize;
        let mut feed = |val: i64, polys: &mut Vec<KahePoly>| {
            buf[written % N] = val;
            written += 1;
            if written.is_multiple_of(N) {
                polys.push(KahePoly::from_coeffs(buf));
                buf = [0i64; N];
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
        if !written.is_multiple_of(N) {
            polys.push(KahePoly::from_coeffs(buf));
        }
        debug_assert_eq!(polys.len(), n_polys);
        polys
    }

    pub fn unpack(params: &MseParams, polys: &[KahePoly]) -> Self {
        let total = params.total_scalars();
        assert_eq!(polys.len(), total.div_ceil(N));
        let mut flat: Vec<i64> = Vec::with_capacity(polys.len() * N);
        for p in polys {
            let mut q = *p;
            q.normalize();
            flat.extend_from_slice(q.coeffs());
        }

        let cells = params.total_cells();
        let c = flat[..cells].to_vec();
        let k: Vec<Vec<i64>> = (0..K_LIMBS)
            .map(|ell| flat[(1 + ell) * cells..(2 + ell) * cells].to_vec())
            .collect();
        let v_base = (1 + K_LIMBS) * cells;
        let v: Vec<Vec<i64>> = (0..params.payload_symbols)
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

    fn params_for(gamma: usize, delta: usize, payload_symbols: usize, seed: u8) -> MseParams {
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
        let mut recovered: Vec<i64> = enc
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
        let mut recovered: Vec<i64> = a
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
        let mut got: Vec<i64> = restored
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
        let summed: Vec<KahePoly> = pa.iter().zip(pb.iter()).map(|(x, y)| *x + *y).collect();
        let unioned = MseEncoding::unpack(&pp, &summed);
        let mut got: Vec<i64> = unioned
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
        let mut messages: Vec<Vec<i64>> = (0..10)
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

    /// 512-bit message → 15 symbols at BITS_PER_SYMBOL=35 (t=2^35). Smaller
    /// multiset but exercises the fat-payload code path end-to-end.
    #[test]
    fn five_hundred_twelve_bit_message_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([23u8; 32]);
        let xi = MseParams::payload_symbols_for_bits(512);
        assert_eq!(xi, 15);
        let pp = params_for(4, 32, xi, 31);
        let mut enc = MseEncoding::new(pp);
        let mut messages: Vec<Vec<i64>> = (0..5)
            .map(|i| (0..xi).map(|s| (i * 7 + s * 13) as i64).collect())
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
    #[should_panic(expected = "payload arity must match")]
    fn payload_arity_mismatch_panics() {
        let mut rng = ChaCha20Rng::from_seed([29u8; 32]);
        let pp = params_for(4, 32, 3, 41);
        let mut enc = MseEncoding::new(pp);
        enc.insert(&mut rng, &[1, 2]);
    }
}
