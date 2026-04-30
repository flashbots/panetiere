//! Secret sharing.

use chipmunk_code::{HVCPoly, Polynomial};
use rand::Rng;

pub trait Sss {
    type Secret: Clone;
    type Share: Clone;

    fn share<R: Rng>(rng: &mut R, secret: &Self::Secret, n: usize) -> Vec<Self::Share>;
    fn recover(shares: &[Self::Share]) -> Self::Secret;
}

/// Additive n-of-n sharing over `HVCPoly`. `s_1..s_{n-1}` random; `s_n = secret − Σ s_i`.
pub struct AdditiveSharing;

impl Sss for AdditiveSharing {
    type Secret = HVCPoly;
    type Share = HVCPoly;

    fn share<R: Rng>(rng: &mut R, secret: &Self::Secret, n: usize) -> Vec<Self::Share> {
        assert!(n >= 1);
        let mut shares = Vec::with_capacity(n);
        let mut acc = HVCPoly::default();
        for _ in 0..n - 1 {
            let s = HVCPoly::rand_poly(rng);
            acc = acc + s;
            shares.push(s);
        }
        shares.push(*secret - acc);
        shares
    }

    fn recover(shares: &[Self::Share]) -> Self::Secret {
        shares.iter().copied().fold(HVCPoly::default(), |a, x| a + x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn round_trip() {
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        for n in [1usize, 2, 4, 7] {
            let secret = HVCPoly::rand_poly(&mut rng);
            let shares = AdditiveSharing::share(&mut rng, &secret, n);
            assert_eq!(shares.len(), n);
            assert_eq!(AdditiveSharing::recover(&shares), secret);
        }
    }
}
