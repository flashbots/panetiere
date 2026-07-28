//! ECIES (P-256 ECDH + AES-256-GCM) sealed envelopes.
//!
//! The canonical PKE for sealing per-server openings to a relay's long-lived
//! exchange key. Wire form of a sealed envelope:
//! `ephemeral_pubkey (65 B, SEC1 uncompressed) ‖ nonce (12 B) ‖ ciphertext+tag`.
//! The AEAD associated data is `ephemeral_pubkey ‖ aad`, where the caller's
//! `aad` binds the envelope to its protocol context — see
//! [`crate::protocol::opening_aad`]. Decryption with a different context fails.

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Nonce};
use p256::ecdh::diffie_hellman;
use p256::elliptic_curve::sec1::{FromEncodedPoint, ToEncodedPoint};
use rand::{CryptoRng, RngCore};
use sha2::{Digest, Sha256};

const PUBKEY_LEN: usize = 65;
const NONCE_LEN: usize = 12;
const TAG_LEN: usize = 16;

/// Byte overhead of a sealed envelope over its plaintext.
pub const SEAL_OVERHEAD: usize = PUBKEY_LEN + NONCE_LEN + TAG_LEN;

#[derive(Clone)]
pub struct PrivateKey(p256::SecretKey);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicKey(p256::PublicKey);

impl PrivateKey {
    pub fn generate<R: CryptoRng + RngCore>(rng: &mut R) -> Self {
        Self(p256::SecretKey::random(rng))
    }

    pub fn public(&self) -> PublicKey {
        PublicKey(self.0.public_key())
    }

    /// 32-byte big-endian scalar.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.0.to_bytes().to_vec()
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, PkeError> {
        p256::SecretKey::from_slice(b)
            .map(Self)
            .map_err(|_| PkeError::BadKey)
    }
}

impl PublicKey {
    /// Uncompressed SEC1 encoding (65 bytes).
    pub fn to_sec1_bytes(&self) -> Vec<u8> {
        self.0.to_encoded_point(false).as_bytes().to_vec()
    }

    pub fn from_sec1_bytes(b: &[u8]) -> Result<Self, PkeError> {
        let ep = p256::EncodedPoint::from_bytes(b).map_err(|_| PkeError::BadKey)?;
        Option::<p256::PublicKey>::from(p256::PublicKey::from_encoded_point(&ep))
            .map(Self)
            .ok_or(PkeError::BadKey)
    }
}

fn derive_aes_key(shared_secret: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"panetiere-ecies-v1");
    h.update(shared_secret);
    h.finalize().into()
}

/// Bind the envelope to its ephemeral key and the caller's context.
fn full_aad(eph_pub: &[u8], aad: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(eph_pub.len() + aad.len());
    out.extend_from_slice(eph_pub);
    out.extend_from_slice(aad);
    out
}

/// Seal `plaintext` to `recipient` under a fresh ephemeral key, bound to `aad`.
/// [`decrypt`] must be given the identical `aad` or it returns [`PkeError::Aead`].
pub fn encrypt<R: CryptoRng + RngCore>(
    rng: &mut R,
    recipient: &PublicKey,
    plaintext: &[u8],
    aad: &[u8],
) -> Vec<u8> {
    let ephemeral = p256::SecretKey::random(rng);
    let shared = diffie_hellman(ephemeral.to_nonzero_scalar(), recipient.0.as_affine());
    let key = derive_aes_key(shared.raw_secret_bytes());

    let eph_pub = ephemeral.public_key().to_encoded_point(false);
    let mut nonce = [0u8; NONCE_LEN];
    rng.fill_bytes(&mut nonce);

    let cipher = Aes256Gcm::new((&key).into());
    let ciphertext = cipher
        .encrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: plaintext,
                aad: &full_aad(eph_pub.as_bytes(), aad),
            },
        )
        .expect("AES-GCM encrypt of an in-memory buffer");

    let mut out = Vec::with_capacity(PUBKEY_LEN + NONCE_LEN + ciphertext.len());
    out.extend_from_slice(eph_pub.as_bytes());
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&ciphertext);
    out
}

/// Open an envelope sealed by [`encrypt`] under the same `aad`.
pub fn decrypt(recipient: &PrivateKey, sealed: &[u8], aad: &[u8]) -> Result<Vec<u8>, PkeError> {
    if sealed.len() < SEAL_OVERHEAD {
        return Err(PkeError::TooShort);
    }
    let (eph_bytes, rest) = sealed.split_at(PUBKEY_LEN);
    let (nonce, ciphertext) = rest.split_at(NONCE_LEN);
    let ephemeral = PublicKey::from_sec1_bytes(eph_bytes)?;
    let shared = diffie_hellman(recipient.0.to_nonzero_scalar(), ephemeral.0.as_affine());
    let key = derive_aes_key(shared.raw_secret_bytes());

    let cipher = Aes256Gcm::new((&key).into());
    cipher
        .decrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: ciphertext,
                aad: &full_aad(eph_bytes, aad),
            },
        )
        .map_err(|_| PkeError::Aead)
}

#[derive(Debug, PartialEq, Eq)]
pub enum PkeError {
    TooShort,
    Aead,
    BadKey,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn roundtrip_tamper_and_wrong_key() {
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        let sk = PrivateKey::generate(&mut rng);
        let pk = sk.public();

        let sealed = encrypt(&mut rng, &pk, b"hello panetiere", b"ctx");
        assert_eq!(sealed.len(), b"hello panetiere".len() + SEAL_OVERHEAD);
        assert_eq!(decrypt(&sk, &sealed, b"ctx").unwrap(), b"hello panetiere");

        let mut tampered = sealed.clone();
        *tampered.last_mut().unwrap() ^= 1;
        assert_eq!(decrypt(&sk, &tampered, b"ctx"), Err(PkeError::Aead));

        let other = PrivateKey::generate(&mut rng);
        assert_eq!(decrypt(&other, &sealed, b"ctx"), Err(PkeError::Aead));

        let re = PrivateKey::from_bytes(&sk.to_bytes()).unwrap();
        assert_eq!(decrypt(&re, &sealed, b"ctx").unwrap(), b"hello panetiere");
        let re_pk = PublicKey::from_sec1_bytes(&pk.to_sec1_bytes()).unwrap();
        assert_eq!(re_pk, pk);
    }

    /// Any change to the associated data makes the envelope unopenable — this is
    /// what binds a sealed opening to its (session, client, server) context.
    #[test]
    fn wrong_aad_rejected() {
        let mut rng = ChaCha20Rng::from_seed([19u8; 32]);
        let sk = PrivateKey::generate(&mut rng);
        let pk = sk.public();

        let aad = b"panetiere/opening/v1:sid=1,client=2,server=3";
        let sealed = encrypt(&mut rng, &pk, b"opening bytes", aad);
        assert_eq!(decrypt(&sk, &sealed, aad).unwrap(), b"opening bytes");

        for i in 0..aad.len() {
            let mut other = aad.to_vec();
            other[i] ^= 1;
            assert_eq!(decrypt(&sk, &sealed, &other), Err(PkeError::Aead));
        }
        assert_eq!(decrypt(&sk, &sealed, b""), Err(PkeError::Aead));
        // Empty aad is itself a valid, distinct context.
        let bare = encrypt(&mut rng, &pk, b"opening bytes", b"");
        assert_eq!(decrypt(&sk, &bare, b"").unwrap(), b"opening bytes");
        assert_eq!(decrypt(&sk, &bare, aad), Err(PkeError::Aead));
    }
}
