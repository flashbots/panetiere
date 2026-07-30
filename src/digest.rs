//! Placeholder until HVC is implemented

use chipmunk_code::{pointwise_dot_dgt, DgtNTTPoly, KahePoly, DGT_MODULUS, KAHE_MODULUS, N};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

pub const DIGEST_POLYS: usize = 3;

#[derive(Clone)]
pub struct DigestParams {
    rows: Vec<Vec<DgtNTTPoly>>,
    ell: usize,
    rho_max: usize,
}

impl DigestParams {
    pub fn from_seed(seed: [u8; 32], ell: usize, rho_max: usize) -> Self {
        assert!(ell >= 1, "need at least one ciphertext poly");
        assert!(rho_max >= 1, "rho_max must be ≥ 1");
        // The exactness condition the whole construction rests on.
        let bound = (rho_max as u128) * (KAHE_MODULUS as u128);
        assert!(
            bound < DGT_MODULUS as u128,
            "rho_max = {rho_max} exceeds the digest ring: rho*q_kahe = 2^{:.1} \
             must stay under q_dgt = 2^{:.1}; regenerate the ring with a larger prime",
            (bound as f64).log2(),
            (DGT_MODULUS as f64).log2(),
        );
        let mut s = seed;
        s[0] ^= 0xD1;
        let mut rng = ChaCha20Rng::from_seed(s);
        let rows = (0..DIGEST_POLYS)
            .map(|_| (0..ell).map(|_| DgtNTTPoly::rand_ntt_poly(&mut rng)).collect())
            .collect();
        Self { rows, ell, rho_max }
    }

    pub fn ell(&self) -> usize {
        self.ell
    }

    pub fn rho_max(&self) -> usize {
        self.rho_max
    }

    /// `‖·‖∞` the aggregated ciphertext may not exceed. Anything larger is
    /// outside the honest range and would leave the digest unbound.
    pub fn norm_bound(&self) -> i64 {
        self.rho_max as i64 * (KAHE_MODULUS / 2)
    }
}

/// Embed a KAHE ciphertext into the digest ring. One forward NTT per poly; the
/// result is both what gets RS-coded and what gets hashed.
pub fn embed(ctxt: &[KahePoly]) -> Vec<DgtNTTPoly> {
    ctxt.par_iter().map(DgtNTTPoly::from_kahe).collect()
}

/// `A · ct`, one row at a time. Rows are independent; each is an AVX2
/// `pointwise_dot` over `ell` ring elements.
pub fn digest(dp: &DigestParams, ctxt_ntt: &[DgtNTTPoly]) -> Vec<DgtNTTPoly> {
    assert_eq!(ctxt_ntt.len(), dp.ell, "ciphertext width must match the matrix");
    dp.rows
        .par_iter()
        .map(|row| pointwise_dot_dgt(row, ctxt_ntt))
        .collect()
}

/// Recover the aggregated ciphertext as exact integers, rejecting anything
/// outside the honest `ρ·q_kahe/2` range.
///
/// `None` is a hard failure, not a hint: past that bound the reconstruction is
/// not a sum of honest ciphertexts and the digest binds nothing.
pub fn centered_within_bound(dp: &DigestParams, summed_ntt: &[DgtNTTPoly]) -> Option<Vec<[i64; N]>> {
    let bound = dp.norm_bound();
    let out: Vec<[i64; N]> = summed_ntt
        .par_iter()
        .map(DgtNTTPoly::to_centered_coeffs)
        .collect();
    if out
        .par_iter()
        .any(|c| c.iter().any(|&x| x > bound || x < -bound))
    {
        return None;
    }
    Some(out)
}

/// Whether `summed_h` is the digest of `summed_ntt`.
pub fn check(dp: &DigestParams, summed_ntt: &[DgtNTTPoly], summed_h: &[DgtNTTPoly]) -> bool {
    summed_h.len() == DIGEST_POLYS && digest(dp, summed_ntt) == summed_h
}

/// Reduce recovered exact-integer coefficients back into `R_{q_kahe}` for
/// decryption.
pub fn to_kahe(centered: &[[i64; N]]) -> Vec<KahePoly> {
    centered
        .par_iter()
        .map(|c| {
            let mut out = [0i64; N];
            for (o, &x) in out.iter_mut().zip(c.iter()) {
                let mut r = x % KAHE_MODULUS;
                if r > KAHE_MODULUS / 2 {
                    r -= KAHE_MODULUS;
                }
                if r < -(KAHE_MODULUS / 2) {
                    r += KAHE_MODULUS;
                }
                *o = r;
            }
            KahePoly::from_coeffs(out)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn rand_ctxts(rng: &mut ChaCha20Rng, rho: usize, ell: usize) -> Vec<Vec<KahePoly>> {
        (0..rho)
            .map(|_| (0..ell).map(|_| KahePoly::rand_poly(rng)).collect())
            .collect()
    }

    fn sum_ntt(cs: &[Vec<DgtNTTPoly>]) -> Vec<DgtNTTPoly> {
        let ell = cs[0].len();
        (0..ell)
            .map(|j| cs.iter().fold(DgtNTTPoly::default(), |acc, c| acc + c[j]))
            .collect()
    }

    #[test]
    fn homomorphic_over_the_ciphertext_sum() {
        let mut rng = ChaCha20Rng::from_seed([9u8; 32]);
        let (rho, ell) = (17, 5);
        let dp = DigestParams::from_seed([9u8; 32], ell, 300);
        let cts = rand_ctxts(&mut rng, rho, ell);
        let embedded: Vec<Vec<DgtNTTPoly>> = cts.iter().map(|c| embed(c)).collect();

        let per_client: Vec<Vec<DgtNTTPoly>> = embedded.iter().map(|e| digest(&dp, e)).collect();
        let summed_h: Vec<DgtNTTPoly> = (0..DIGEST_POLYS)
            .map(|i| {
                per_client
                    .iter()
                    .fold(DgtNTTPoly::default(), |acc, h| acc + h[i])
            })
            .collect();
        let summed = sum_ntt(&embedded);
        assert!(check(&dp, &summed, &summed_h));
        assert!(centered_within_bound(&dp, &summed).is_some());
    }

    #[test]
    fn catches_a_perturbed_sum() {
        let mut rng = ChaCha20Rng::from_seed([10u8; 32]);
        let (rho, ell) = (8, 4);
        let dp = DigestParams::from_seed([10u8; 32], ell, 300);
        let cts = rand_ctxts(&mut rng, rho, ell);
        let embedded: Vec<Vec<DgtNTTPoly>> = cts.iter().map(|c| embed(c)).collect();
        let summed = sum_ntt(&embedded);
        let h = digest(&dp, &summed);
        assert!(check(&dp, &summed, &h));

        let mut bad = summed.clone();
        let mut c = *bad[2].coeffs();
        c[13] = (c[13] + 1) % DGT_MODULUS;
        bad[2] = DgtNTTPoly::from_raw(&c);
        assert!(!check(&dp, &bad, &h));
    }

    /// Decryption reads the reduced sum, so the exact-integer detour must land
    /// back on the same ring element a mod-q_kahe accumulation would have.
    #[test]
    fn reduces_back_to_the_kahe_sum() {
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        let (rho, ell) = (300, 3);
        let dp = DigestParams::from_seed([11u8; 32], ell, 300);
        let cts = rand_ctxts(&mut rng, rho, ell);
        let summed = sum_ntt(&cts.iter().map(|c| embed(c)).collect::<Vec<_>>());
        let centered = centered_within_bound(&dp, &summed).expect("within bound");
        let got = to_kahe(&centered);

        for j in 0..ell {
            let want = cts.iter().fold(KahePoly::default(), |acc, c| acc + c[j]);
            let (mut a, mut b) = (got[j], want);
            a.normalize();
            b.normalize();
            assert_eq!(a, b, "poly {j}");
        }
    }

    #[test]
    fn rejects_a_sum_past_the_norm_bound() {
        let mut rng = ChaCha20Rng::from_seed([12u8; 32]);
        let dp = DigestParams::from_seed([12u8; 32], 2, 4);
        // 40 clients against a bound sized for 4: outside the honest range.
        let cts = rand_ctxts(&mut rng, 40, 2);
        let summed = sum_ntt(&cts.iter().map(|c| embed(c)).collect::<Vec<_>>());
        assert!(centered_within_bound(&dp, &summed).is_none());
    }

    #[test]
    fn cover_traffic_does_not_stand_out() {
        // Enc(0) is A*sk + t*e, not zero, so a cover client's digest is not a
        // known constant — the reason this can be published in the clear.
        let mut rng = ChaCha20Rng::from_seed([13u8; 32]);
        let dp = DigestParams::from_seed([13u8; 32], 2, 300);
        let cover: Vec<KahePoly> = (0..2)
            .map(|_| {
                let mut c = [0i64; N];
                for x in c.iter_mut() {
                    *x = rng.gen_range(0..KAHE_MODULUS);
                }
                KahePoly::from_coeffs(c)
            })
            .collect();
        let h = digest(&dp, &embed(&cover));
        assert_ne!(h[0], DgtNTTPoly::default());
    }

    #[test]
    #[should_panic(expected = "exceeds the digest ring")]
    fn rho_beyond_the_ring_is_rejected_at_setup() {
        DigestParams::from_seed([14u8; 32], 4, 1 << 20);
    }
}
