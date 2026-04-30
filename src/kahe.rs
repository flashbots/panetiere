//! Key-additive homomorphic encryption.

use chipmunk_code::{HVCPoly, Polynomial};
use rand::Rng;

pub trait Kahe {
    type Key: Clone;
    type Message: Clone;
    type Ciphertext: Clone;

    fn gen<R: Rng>(rng: &mut R) -> Self::Key;
    fn enc(k: &Self::Key, m: &Self::Message) -> Self::Ciphertext;
    fn dec(c: &Self::Ciphertext, k: &Self::Key) -> Self::Message;
    fn agg_ctxt(cs: &[Self::Ciphertext]) -> Self::Ciphertext;
    fn agg_key(ks: &[Self::Key]) -> Self::Key;
}

/// Ring one-time pad over `Z_q[x]/(x^N+1)` with q = HVC_MODULUS.
/// `enc(k,m) = k + m`, `dec(c,k) = c - k`. Both aggregations are pointwise sum.
pub struct RingOtp;

impl Kahe for RingOtp {
    type Key = HVCPoly;
    type Message = HVCPoly;
    type Ciphertext = HVCPoly;

    fn gen<R: Rng>(rng: &mut R) -> Self::Key {
        HVCPoly::rand_poly(rng)
    }

    fn enc(k: &Self::Key, m: &Self::Message) -> Self::Ciphertext {
        *k + *m
    }

    fn dec(c: &Self::Ciphertext, k: &Self::Key) -> Self::Message {
        *c - *k
    }

    fn agg_ctxt(cs: &[Self::Ciphertext]) -> Self::Ciphertext {
        cs.iter().copied().fold(HVCPoly::default(), |acc, x| acc + x)
    }

    fn agg_key(ks: &[Self::Key]) -> Self::Key {
        ks.iter().copied().fold(HVCPoly::default(), |acc, x| acc + x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn round_trip() {
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        let k = RingOtp::gen(&mut rng);
        let m = HVCPoly::rand_poly(&mut rng);
        let c = RingOtp::enc(&k, &m);
        assert_eq!(RingOtp::dec(&c, &k), m);
    }

    #[test]
    fn additive_homomorphism() {
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        let n = 5;
        let keys: Vec<_> = (0..n).map(|_| RingOtp::gen(&mut rng)).collect();
        let msgs: Vec<HVCPoly> = (0..n).map(|_| HVCPoly::rand_poly(&mut rng)).collect();
        let ctxts: Vec<_> = keys.iter().zip(&msgs).map(|(k, m)| RingOtp::enc(k, m)).collect();

        let agg_c = RingOtp::agg_ctxt(&ctxts);
        let agg_k = RingOtp::agg_key(&keys);
        let agg_m_expected = msgs.iter().copied().fold(HVCPoly::default(), |a, x| a + x);

        assert_eq!(RingOtp::dec(&agg_c, &agg_k), agg_m_expected);
    }
}
