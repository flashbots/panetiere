//! Digest-ring bridging for the RS ingress flow: embedding KAHE ciphertexts
//! into the wide ring where coding and commitment happen, and recovering the
//! exact integer sum on the way back out.

use chipmunk_code::{DgtNTTPoly, KahePoly, KAHE_MODULUS, N};
use rayon::prelude::*;

/// Embed a KAHE ciphertext into the digest ring. One forward NTT per poly; the
/// result is both what gets RS-coded and what gets committed to.
pub fn embed(ctxt: &[KahePoly]) -> Vec<DgtNTTPoly> {
    ctxt.par_iter().map(DgtNTTPoly::from_kahe).collect()
}

/// Recover the aggregated ciphertext as exact integers, rejecting anything
/// outside the honest `ρ·q_kahe/2` range.
///
/// `None` is a hard failure, not a hint: past that bound the reconstruction is
/// not a sum of honest ciphertexts.
pub fn centered_within_bound(rho_max: usize, summed_ntt: &[DgtNTTPoly]) -> Option<Vec<[i64; N]>> {
    let bound = rho_max as i64 * (KAHE_MODULUS / 2);
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
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

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

    /// Decryption reads the reduced sum, so the exact-integer detour must land
    /// back on the same ring element a mod-q_kahe accumulation would have.
    #[test]
    fn reduces_back_to_the_kahe_sum() {
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        let (rho, ell) = (300, 3);
        let cts = rand_ctxts(&mut rng, rho, ell);
        let summed = sum_ntt(&cts.iter().map(|c| embed(c)).collect::<Vec<_>>());
        let centered = centered_within_bound(rho, &summed).expect("within bound");
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
        // 40 clients against a bound sized for 4: outside the honest range.
        let cts = rand_ctxts(&mut rng, 40, 2);
        let summed = sum_ntt(&cts.iter().map(|c| embed(c)).collect::<Vec<_>>());
        assert!(centered_within_bound(4, &summed).is_none());
    }
}
