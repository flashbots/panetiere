//! P-256 ECDSA over bulletin posts.
//!
//! The RS mode moves the ciphertext off the bulletin, so what a client posts is
//! a commitment plus an encrypted digest. Signing that binds the client to the
//! digest it claims, which is what makes the dispute bisection in
//! [`crate::protocol::dispute`] mean anything: without it a server could
//! attribute a fabricated digest to a client it wants excluded.
//!
//! Same curve as [`crate::pke`], so the dependency set is unchanged.

use p256::ecdsa::signature::{Signer, Verifier};
use rand::{CryptoRng, RngCore};

#[derive(Debug, PartialEq, Eq)]
pub enum SigError {
    BadKey,
    BadSignature,
}

#[derive(Clone)]
pub struct SigningKey(p256::ecdsa::SigningKey);

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VerifyingKey(p256::ecdsa::VerifyingKey);

/// Compressed SEC1 point.
pub const PUBKEY_LEN: usize = 33;
/// Fixed-width `r ‖ s`.
pub const SIG_LEN: usize = 64;

impl SigningKey {
    pub fn generate<R: CryptoRng + RngCore>(rng: &mut R) -> Self {
        Self(p256::ecdsa::SigningKey::random(rng))
    }

    pub fn verifying_key(&self) -> VerifyingKey {
        VerifyingKey(*self.0.verifying_key())
    }

    pub fn sign(&self, msg: &[u8]) -> [u8; SIG_LEN] {
        let sig: p256::ecdsa::Signature = self.0.sign(msg);
        let mut out = [0u8; SIG_LEN];
        out.copy_from_slice(&sig.to_bytes());
        out
    }
}

impl VerifyingKey {
    pub fn to_sec1_bytes(&self) -> [u8; PUBKEY_LEN] {
        let mut out = [0u8; PUBKEY_LEN];
        out.copy_from_slice(self.0.to_encoded_point(true).as_bytes());
        out
    }

    pub fn from_sec1_bytes(b: &[u8]) -> Result<Self, SigError> {
        p256::ecdsa::VerifyingKey::from_sec1_bytes(b)
            .map(VerifyingKey)
            .map_err(|_| SigError::BadKey)
    }

    pub fn verify(&self, msg: &[u8], sig: &[u8; SIG_LEN]) -> Result<(), SigError> {
        let s = p256::ecdsa::Signature::from_slice(sig).map_err(|_| SigError::BadSignature)?;
        self.0.verify(msg, &s).map_err(|_| SigError::BadSignature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn round_trip() {
        let mut rng = ChaCha20Rng::from_seed([5u8; 32]);
        let sk = SigningKey::generate(&mut rng);
        let vk = sk.verifying_key();
        let msg = b"panetiere/rs/v1 payload";
        let sig = sk.sign(msg);
        assert!(vk.verify(msg, &sig).is_ok());
        assert_eq!(vk.verify(b"other", &sig), Err(SigError::BadSignature));

        let encoded = vk.to_sec1_bytes();
        let back = VerifyingKey::from_sec1_bytes(&encoded).unwrap();
        assert_eq!(back, vk);
        assert!(back.verify(msg, &sig).is_ok());
    }

    #[test]
    fn rejects_another_signer() {
        let mut rng = ChaCha20Rng::from_seed([6u8; 32]);
        let a = SigningKey::generate(&mut rng);
        let b = SigningKey::generate(&mut rng);
        let msg = b"bulletin post";
        assert_eq!(
            b.verifying_key().verify(msg, &a.sign(msg)),
            Err(SigError::BadSignature)
        );
    }
}
