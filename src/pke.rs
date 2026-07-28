//! ML-KEM-768 + AES-256-GCM sealed envelopes.
//!
//! The canonical PKE for sealing per-server openings to a relay's long-lived
//! encapsulation key. Wire form of a sealed envelope:
//! `kem_ciphertext (1088 B) ‖ ciphertext+tag`.
//!
//! CCA2 comes from the KEM (FIPS 203's Fujisaki-Okamoto transform with implicit
//! rejection), so the AEAD only has to be one-time secure. The KEM ciphertext is
//! bound into the key schedule rather than passed as associated data — a mauled
//! encapsulation yields a different key, not merely a failing tag. The AEAD
//! associated data is the caller's `aad`, which binds the envelope to its
//! protocol context — see [`crate::protocol::opening_aad`]. Decryption with a
//! different context fails.
//!
//! Confidentiality here is post-quantum; [`crate::sig`] is still P-256. That
//! asymmetry is deliberate: a sealed opening carries a Shamir share of a KAHE
//! key, so recording envelopes now and breaking the KEM later would retroactively
//! deanonymise past rounds, whereas breaking the signature later only forges
//! future posts.

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Nonce};
use ml_kem::kem::Decapsulate;
use ml_kem::{B32, KeyExport, Seed};
use rand::{CryptoRng, RngCore};
use sha2::{Digest, Sha512};

type Dk = ml_kem::ml_kem_768::DecapsulationKey;
type Ek = ml_kem::ml_kem_768::EncapsulationKey;
type KemCiphertext = ml_kem::ml_kem_768::Ciphertext;

/// ML-KEM-768 encapsulation (KEM ciphertext) length.
const ENCAPS_LEN: usize = 1088;
/// ML-KEM-768 encapsulation key length.
pub const PUBKEY_LEN: usize = 1184;
/// Seed a [`PrivateKey`] serialises to.
pub const PRIVKEY_LEN: usize = 64;
const NONCE_LEN: usize = 12;
const TAG_LEN: usize = 16;

/// Byte overhead of a sealed envelope over its plaintext.
pub const SEAL_OVERHEAD: usize = ENCAPS_LEN + TAG_LEN;

#[derive(Clone)]
pub struct PrivateKey(Dk);

#[derive(Clone, Debug)]
pub struct PublicKey(Ek);

impl PrivateKey {
    pub fn generate<R: CryptoRng + RngCore>(rng: &mut R) -> Self {
        let mut seed = [0u8; PRIVKEY_LEN];
        rng.fill_bytes(&mut seed);
        Self(Dk::from_seed(Seed::from(seed)))
    }

    pub fn public(&self) -> PublicKey {
        PublicKey(self.0.encapsulation_key().clone())
    }

    /// The 64-byte seed the key was derived from.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.0
            .to_seed()
            .expect("every PrivateKey is built via from_seed")
            .to_vec()
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, PkeError> {
        let seed: [u8; PRIVKEY_LEN] = b.try_into().map_err(|_| PkeError::BadKey)?;
        Ok(Self(Dk::from_seed(Seed::from(seed))))
    }
}

impl PublicKey {
    pub fn to_bytes(&self) -> Vec<u8> {
        self.0.to_bytes().to_vec()
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, PkeError> {
        let encoded = ml_kem::Key::<Ek>::try_from(b).map_err(|_| PkeError::BadKey)?;
        Ek::new(&encoded).map(Self).map_err(|_| PkeError::BadKey)
    }
}

impl PartialEq for PublicKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bytes() == other.0.to_bytes()
    }
}

impl Eq for PublicKey {}

/// Bind the encapsulation into the key schedule, then split off an AEAD key and
/// nonce. The KEM ciphertext is unique per envelope, so the derived nonce is too.
fn derive_key_nonce(kem_ct: &[u8], shared_secret: &[u8]) -> ([u8; 32], [u8; NONCE_LEN]) {
    let mut h = Sha512::new();
    h.update(b"panetiere-mlkem768-aesgcm-v1");
    h.update(kem_ct);
    h.update(shared_secret);
    let out = h.finalize();

    let mut key = [0u8; 32];
    let mut nonce = [0u8; NONCE_LEN];
    key.copy_from_slice(&out[..32]);
    nonce.copy_from_slice(&out[32..32 + NONCE_LEN]);
    (key, nonce)
}

/// Seal `plaintext` to `recipient` under a fresh encapsulation, bound to `aad`.
/// [`decrypt`] must be given the identical `aad` or it returns [`PkeError::Aead`].
pub fn encrypt<R: CryptoRng + RngCore>(
    rng: &mut R,
    recipient: &PublicKey,
    plaintext: &[u8],
    aad: &[u8],
) -> Vec<u8> {
    // Deterministic encapsulation over freshly drawn randomness, so the caller's
    // RNG stays the single source of entropy (`seal_openings` forks seeds per
    // server) and no rand_core version has to be bridged.
    let mut m = [0u8; 32];
    rng.fill_bytes(&mut m);
    let (kem_ct, shared) = recipient.0.encapsulate_deterministic(&B32::from(m));

    let (key, nonce) = derive_key_nonce(&kem_ct, &shared);
    let ciphertext = Aes256Gcm::new((&key).into())
        .encrypt(
            &Nonce::from(nonce),
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .expect("AES-GCM encrypt of an in-memory buffer");

    let mut out = Vec::with_capacity(ENCAPS_LEN + ciphertext.len());
    out.extend_from_slice(&kem_ct);
    out.extend_from_slice(&ciphertext);
    out
}

/// Open an envelope sealed by [`encrypt`] under the same `aad`.
pub fn decrypt(recipient: &PrivateKey, sealed: &[u8], aad: &[u8]) -> Result<Vec<u8>, PkeError> {
    if sealed.len() < SEAL_OVERHEAD {
        return Err(PkeError::TooShort);
    }
    let (kem_bytes, ciphertext) = sealed.split_at(ENCAPS_LEN);
    let kem_ct = KemCiphertext::try_from(kem_bytes).map_err(|_| PkeError::BadKey)?;
    // Implicit rejection: a mauled encapsulation decapsulates to an unrelated
    // shared secret rather than failing, and the AEAD tag catches it below.
    let shared = recipient.0.decapsulate(&kem_ct);

    let (key, nonce) = derive_key_nonce(kem_bytes, &shared);
    Aes256Gcm::new((&key).into())
        .decrypt(
            &Nonce::from(nonce),
            Payload {
                msg: ciphertext,
                aad,
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
        let re_pk = PublicKey::from_bytes(&pk.to_bytes()).unwrap();
        assert_eq!(re_pk, pk);
    }

    /// Mauling the encapsulation must not be distinguishable from any other
    /// forgery — implicit rejection plus the key-schedule binding, not a
    /// decapsulation error.
    #[test]
    fn tampered_encapsulation_rejected() {
        let mut rng = ChaCha20Rng::from_seed([23u8; 32]);
        let sk = PrivateKey::generate(&mut rng);
        let sealed = encrypt(&mut rng, &sk.public(), b"opening bytes", b"ctx");

        for i in [0usize, 17, ENCAPS_LEN - 1] {
            let mut mauled = sealed.clone();
            mauled[i] ^= 1;
            assert_eq!(decrypt(&sk, &mauled, b"ctx"), Err(PkeError::Aead));
        }
        assert_eq!(
            decrypt(&sk, &sealed[..SEAL_OVERHEAD - 1], b"ctx"),
            Err(PkeError::TooShort)
        );
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

    /// The hardcoded lengths must track the parameter set.
    #[test]
    fn wire_lengths_match_parameter_set() {
        let mut rng = ChaCha20Rng::from_seed([31u8; 32]);
        let sk = PrivateKey::generate(&mut rng);
        assert_eq!(sk.to_bytes().len(), PRIVKEY_LEN);
        assert_eq!(sk.public().to_bytes().len(), PUBKEY_LEN);
        assert_eq!(
            encrypt(&mut rng, &sk.public(), b"", b"").len(),
            SEAL_OVERHEAD
        );
    }
}
