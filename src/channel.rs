//! Application payload layer over the multiset encodings.

use crate::KahePoly;
use rand::{CryptoRng, Rng};
use sha2::{Digest as _, Sha256};

use crate::kahe::{SIGMA_E_DEFAULT, SIGMA_S_DEFAULT, T_MODULUS_DEFAULT};
use crate::mse::{self, MseEncoding, MseError, MseParams};
use crate::prony::{PronyError, PronyParams, PronySketch, PRONY_PRIME};
use crate::protocol::ProtocolParams;

/// Bytes packed into one symbol, little-endian.
pub const BYTES_PER_SYMBOL: usize = 4;

/// A symbol must not fill its `Z_t` cell — the slack is the mod-`t` summation
/// headroom. Breaks the build if `t` is ever lowered under the packing.
const _: () = assert!(BYTES_PER_SYMBOL * 8 < mse::BITS_PER_SYMBOL);
/// Prony needs every packed symbol to be a distinct residue mod `p`.
const _: () = assert!((1u64 << (BYTES_PER_SYMBOL * 8)) < PRONY_PRIME);

/// Rows in the peeling structure. 2 is the correctness floor; 4 is the usual
/// operating point.
const GAMMA: usize = 4;

/// Buckets per insertion, across all rows.
const BUCKETS_PER_INSERT: usize = 3;

/// The paper's statistical security target for nonce collisions.
const SECURITY_BITS: usize = 128;

/// `log₂ Q`, the attacker's budget of random oracle queries.
const ORACLE_QUERY_BITS: u32 = 64;

/// Separates the masking oracle from every other use of SHA-256 here.
const ORACLE_DOMAIN: &[u8] = b"panetiere/nonce-mask/v1";

#[derive(Debug, PartialEq)]
pub enum ChannelError {
    /// Peeling stalled: the encoding held more elements than `δ` was sized for,
    /// so the round's payloads are unrecoverable.
    PeelStalled,
    /// The peeling table is not an honest sum: over capacity, or a cell whose
    /// key does not hash to it.
    MalformedEncoding(MseError),
    /// The Prony sketch could not be resolved — over capacity, or a malformed
    /// contribution.
    SketchFailed(PronyError),
    /// Fewer polys than the parameters describe.
    ShortPlaintext { got: usize, need: usize },
    /// Recovered element count disagrees with what the caller expected.
    CountMismatch { got: usize, expected: usize },
    /// Payload exceeds what `ξ` symbols hold.
    PayloadTooLong { got: usize, max: usize },
}

/// The multiset encoding a channel sits on.
#[derive(Clone, Debug, PartialEq)]
enum Encoding {
    Mse(MseParams),
    Prony(PronyParams),
}

/// Sizing and keying for one channel, over either multiset encoding. `Mse`
/// peels; `Prony` is ~3× smaller. Both are all-or-nothing.
///
/// Every element is `(s, m + RO(s))`: a fresh nonce `s`, then the payload
/// masked by a random oracle on that nonce, so no client can make its own
/// element depend on an honest one before the round decodes.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelParams {
    encoding: Encoding,
    /// Leading symbols of each element that hold the nonce.
    nonce_symbols: usize,
}

impl ChannelParams {
    /// Peeling channel for up to `rho` inserted payloads of `message_bytes` each.
    pub fn for_messages(rho: u32, message_bytes: usize, prf_key: [u8; 32]) -> Self {
        Self::for_symbols(rho, payload_symbols(message_bytes), prf_key)
    }

    /// Peeling channel for up to `rho` payloads of `payload_symbols` symbols
    /// each. For channels whose payload is already `Z_t` values rather than
    /// bytes — the byte packing of [`encode_message`] does not apply to those.
    pub fn for_symbols(rho: u32, payload_symbols: usize, prf_key: [u8; 32]) -> Self {
        let rho = rho.max(1) as usize;
        let nonce_symbols = nonce_symbols(rho, mse::BITS_PER_SYMBOL);
        let delta = (BUCKETS_PER_INSERT * rho).div_ceil(GAMMA);
        Self {
            encoding: Encoding::Mse(MseParams::new(
                GAMMA,
                delta,
                rho,
                nonce_symbols + payload_symbols.max(1),
                prf_key,
            )),
            nonce_symbols,
        }
    }

    /// Sketch channel for up to `rho` payloads of `message_bytes` each.
    pub fn prony_for_messages(rho: u32, message_bytes: usize) -> Self {
        Self::prony_for_symbols(rho, payload_symbols(message_bytes))
    }

    /// Sketch channel carrying `payload_bits` per element, sized at the
    /// sketch's own symbol width rather than the peeling structure's.
    pub fn prony_for_bits(rho: u32, payload_bits: usize) -> Self {
        Self::prony_for_symbols(rho, payload_bits.div_ceil(PRONY_PRIME.ilog2() as usize))
    }

    pub fn prony_for_symbols(rho: u32, payload_symbols: usize) -> Self {
        let rho = rho.max(1) as usize;
        let nonce_symbols = nonce_symbols(rho, PRONY_PRIME.ilog2() as usize);
        Self {
            encoding: Encoding::Prony(PronyParams::new(
                rho,
                nonce_symbols + payload_symbols.max(1),
            )),
            nonce_symbols,
        }
    }

    /// The peeling parameters underneath, nonce symbols included.
    pub fn mse(&self) -> Option<&MseParams> {
        match &self.encoding {
            Encoding::Mse(p) => Some(p),
            Encoding::Prony(_) => None,
        }
    }

    /// The sketch parameters underneath, nonce symbols included.
    pub fn prony(&self) -> Option<&PronyParams> {
        match &self.encoding {
            Encoding::Prony(p) => Some(p),
            Encoding::Mse(_) => None,
        }
    }

    /// `KahePoly` width of one packed encoding — every client contributes this
    /// many, real or cover.
    pub fn n_polys(&self) -> usize {
        match &self.encoding {
            Encoding::Mse(p) => MseEncoding::n_polys(p),
            Encoding::Prony(p) => PronySketch::n_polys(p),
        }
    }

    /// Symbols of payload per element, not counting the nonce.
    pub fn payload_symbols(&self) -> usize {
        let element = match &self.encoding {
            Encoding::Mse(p) => p.payload_symbols,
            Encoding::Prony(p) => p.payload_symbols,
        };
        element - self.nonce_symbols
    }

    /// Payload bits one symbol carries: `⌊log₂ t⌋` in the peeling structure's
    /// `Z_t`, `⌊log₂ p⌋` in the sketch's prime field — both are 35.
    pub fn bits_per_symbol(&self) -> usize {
        match &self.encoding {
            Encoding::Mse(_) => mse::BITS_PER_SYMBOL,
            Encoding::Prony(p) => p.bits_per_symbol(),
        }
    }

    /// KAHE plaintext modulus the symbols live in. The sketch needs a prime.
    pub fn plaintext_modulus(&self) -> u64 {
        match &self.encoding {
            Encoding::Mse(_) => T_MODULUS_DEFAULT,
            Encoding::Prony(p) => p.p,
        }
    }

    /// Largest payload [`encode_message`] accepts.
    pub fn max_payload_bytes(&self) -> usize {
        self.payload_symbols() * BYTES_PER_SYMBOL
    }

    /// Protocol parameters whose KAHE width is exactly one packed encoding.
    pub fn protocol_params<R: Rng>(&self, rng: &mut R, n_servers: usize) -> ProtocolParams {
        ProtocolParams::setup_with_kahe_dims_full(
            rng,
            n_servers,
            self.n_polys(),
            SIGMA_S_DEFAULT,
            SIGMA_E_DEFAULT,
            self.plaintext_modulus(),
        )
    }
}

/// Nonce symbols so that `ρ(Q + ρ) / |N| ≤ 2^-SECURITY_BITS`, with `Q`
/// the attacker's random oracle queries and `|N| = 2^(bits_per_symbol · k)`.
fn nonce_symbols(rho: usize, bits_per_symbol: usize) -> usize {
    let rho = rho as u128;
    let numerator = rho * ((1u128 << ORACLE_QUERY_BITS) + rho);
    let numerator_bits = (u128::BITS - (numerator - 1).leading_zeros()) as usize;
    (SECURITY_BITS + numerator_bits).div_ceil(bits_per_symbol)
}

/// `RO(nonce)` expanded to `len` values uniform in `[0, modulus)`: SHA-256 in
/// counter mode, with rejection sampling so a prime modulus stays unbiased.
fn random_oracle(nonce: &[i64], len: usize, modulus: u64) -> Vec<i64> {
    let bits = u64::BITS - (modulus - 1).leading_zeros();
    let mask = (1u64 << bits) - 1;
    let mut prefix = Sha256::new();
    prefix.update(ORACLE_DOMAIN);
    prefix.update((nonce.len() as u64).to_le_bytes());
    for &s in nonce {
        prefix.update(s.to_le_bytes());
    }
    let mut out = Vec::with_capacity(len);
    let mut counter = 0u64;
    while out.len() < len {
        let digest = prefix
            .clone()
            .chain_update(counter.to_le_bytes())
            .finalize();
        for chunk in digest.chunks_exact(8) {
            let v = u64::from_le_bytes(chunk.try_into().expect("chunk is eight bytes")) & mask;
            if v < modulus && out.len() < len {
                out.push(v as i64);
            }
        }
        counter += 1;
    }
    out
}

fn payload_symbols(message_bytes: usize) -> usize {
    message_bytes.div_ceil(BYTES_PER_SYMBOL).max(1)
}

fn bytes_to_symbols(payload: &[u8], xi: usize) -> Vec<i64> {
    let mut buf = payload.to_vec();
    buf.resize(xi * BYTES_PER_SYMBOL, 0);
    buf.chunks(BYTES_PER_SYMBOL)
        .map(|c| u32::from_le_bytes(c.try_into().expect("chunk is BYTES_PER_SYMBOL")) as i64)
        .collect()
}

fn symbols_to_bytes(symbols: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(symbols.len() * BYTES_PER_SYMBOL);
    for &s in symbols {
        out.extend_from_slice(&(s as u32).to_le_bytes());
    }
    out
}

pub fn encode_message<R: CryptoRng + Rng>(
    rng: &mut R,
    p: &ChannelParams,
    payload: &[u8],
) -> Result<Vec<KahePoly>, ChannelError> {
    let max = p.max_payload_bytes();
    if payload.len() > max {
        return Err(ChannelError::PayloadTooLong {
            got: payload.len(),
            max,
        });
    }
    Ok(encode_symbols(
        rng,
        p,
        &bytes_to_symbols(payload, p.payload_symbols()),
    ))
}

/// Insert `(s, payload + RO(s))` for a fresh nonce `s`.
pub fn encode_symbols<R: CryptoRng + Rng>(
    rng: &mut R,
    p: &ChannelParams,
    payload: &[i64],
) -> Vec<KahePoly> {
    let modulus = p.plaintext_modulus();
    let nonce: Vec<i64> = (0..p.nonce_symbols)
        .map(|_| rng.gen_range(0..1i64 << p.bits_per_symbol()))
        .collect();
    let mask = random_oracle(&nonce, payload.len(), modulus);
    let mut element = nonce;
    element.extend(
        payload
            .iter()
            .zip(&mask)
            .map(|(&m, &r)| (m + r).rem_euclid(modulus as i64)),
    );
    match &p.encoding {
        Encoding::Mse(params) => {
            let mut enc = MseEncoding::new(params.clone());
            enc.insert(rng, &element);
            enc.pack()
        }
        Encoding::Prony(params) => {
            let mut sketch = PronySketch::new(params.clone());
            sketch.insert(rng, &element);
            sketch.pack()
        }
    }
}

pub fn cover(p: &ChannelParams) -> Vec<KahePoly> {
    match &p.encoding {
        Encoding::Mse(params) => MseEncoding::cover(params),
        Encoding::Prony(params) => PronySketch::cover(params),
    }
}

pub fn decode_messages(
    p: &ChannelParams,
    plaintext: &[KahePoly],
    expect: Option<usize>,
) -> Result<Vec<Vec<u8>>, ChannelError> {
    Ok(decode_symbols(p, plaintext, expect)?
        .iter()
        .map(|s| symbols_to_bytes(s))
        .collect())
}

/// `expect` optionally checks the total decoded item count, not items per client.
/// Leave it `None` when the active-sender count is unknown (e.g. with cover traffic).
/// MSE residual and Prony slack checks test decoding consistency, not per-client
/// input validity or payload authenticity. Each element's nonce is stripped and
/// its mask removed, so symbols come back in `[0, plaintext_modulus)`.
pub fn decode_symbols(
    p: &ChannelParams,
    plaintext: &[KahePoly],
    expect: Option<usize>,
) -> Result<Vec<Vec<i64>>, ChannelError> {
    let need = p.n_polys();
    if plaintext.len() < need {
        return Err(ChannelError::ShortPlaintext {
            got: plaintext.len(),
            need,
        });
    }
    let elements = match &p.encoding {
        Encoding::Mse(params) => MseEncoding::unpack(params, &plaintext[..need])
            .decode()
            .map_err(|e| match e {
                MseError::OverCapacity | MseError::KeyMismatch => {
                    ChannelError::MalformedEncoding(e)
                }
                _ => ChannelError::PeelStalled,
            })?,
        Encoding::Prony(params) => PronySketch::unpack(params, &plaintext[..need])
            .decode()
            .map_err(ChannelError::SketchFailed)?,
    };
    if let Some(expected) = expect {
        if elements.len() != expected {
            return Err(ChannelError::CountMismatch {
                got: elements.len(),
                expected,
            });
        }
    }
    let modulus = p.plaintext_modulus() as i64;
    Ok(elements
        .iter()
        .map(|element| {
            let (nonce, masked) = element.split_at(p.nonce_symbols);
            let nonce: Vec<i64> = nonce.iter().map(|s| s.rem_euclid(modulus)).collect();
            let mask = random_oracle(&nonce, masked.len(), modulus as u64);
            masked
                .iter()
                .zip(&mask)
                .map(|(&v, &r)| (v - r).rem_euclid(modulus))
                .collect()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    /// Both encodings, same sizing, so every test below runs twice.
    fn variants(rho: u32, bytes: usize) -> Vec<ChannelParams> {
        vec![
            ChannelParams::for_messages(rho, bytes, [0x5C; 32]),
            ChannelParams::prony_for_messages(rho, bytes),
        ]
    }

    /// Sum `n` contributions positionally, as the protocol's aggregation does.
    fn sum(polys: &[Vec<KahePoly>]) -> Vec<KahePoly> {
        let mut acc = polys[0].clone();
        for p in &polys[1..] {
            for (a, b) in acc.iter_mut().zip(p) {
                *a += *b;
            }
        }
        acc
    }

    #[test]
    fn round_trip_at_capacity() {
        for p in variants(8, 64) {
            let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
            assert_eq!(p.max_payload_bytes(), 64);

            let payloads: Vec<Vec<u8>> = (0..8u8)
                .map(|i| {
                    (0..64)
                        .map(|j| i.wrapping_mul(31).wrapping_add(j))
                        .collect()
                })
                .collect();
            let contributions: Vec<Vec<KahePoly>> = payloads
                .iter()
                .map(|m| encode_message(&mut rng, &p, m).unwrap())
                .collect();

            let mut got = decode_messages(&p, &sum(&contributions), Some(8)).unwrap();
            let mut want = payloads;
            got.sort();
            want.sort();
            assert_eq!(got, want, "{p:?}");
        }
    }

    #[test]
    fn payload_one_byte_over_capacity_rejected() {
        for p in variants(8, 64) {
            let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
            let max = p.max_payload_bytes();
            assert!(encode_message(&mut rng, &p, &vec![7u8; max]).is_ok());
            assert_eq!(
                encode_message(&mut rng, &p, &vec![7u8; max + 1]),
                Err(ChannelError::PayloadTooLong { got: max + 1, max })
            );
        }
    }

    #[test]
    fn cover_only_round_decodes_empty() {
        for p in variants(8, 32) {
            let covers: Vec<Vec<KahePoly>> = (0..8).map(|_| cover(&p)).collect();
            assert_eq!(
                decode_messages(&p, &sum(&covers), Some(0)).unwrap().len(),
                0,
                "{p:?}"
            );
        }
    }

    /// The regression this module exists for: an over-subscribed structure must
    /// surface as an error, not as an empty round.
    #[test]
    fn oversubscribed_fails_loudly() {
        for p in variants(4, 16) {
            let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
            // Sized for 4, given 40.
            let contributions: Vec<Vec<KahePoly>> = (0..40u8)
                .map(|i| encode_message(&mut rng, &p, &[i, i, i, i]).unwrap())
                .collect();
            assert!(
                decode_messages(&p, &sum(&contributions), None).is_err(),
                "{p:?}"
            );
        }
    }

    #[test]
    fn count_mismatch_reported() {
        for p in variants(8, 16) {
            let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
            let contributions: Vec<Vec<KahePoly>> = (0..3u8)
                .map(|i| encode_message(&mut rng, &p, &[i; 4]).unwrap())
                .collect();
            let plain = sum(&contributions);
            assert!(decode_messages(&p, &plain, Some(3)).is_ok(), "{p:?}");
            assert_eq!(
                decode_messages(&p, &plain, Some(5)),
                Err(ChannelError::CountMismatch {
                    got: 3,
                    expected: 5
                })
            );
        }
    }

    #[test]
    fn short_plaintext_reported() {
        for p in variants(8, 16) {
            let need = p.n_polys();
            assert_eq!(
                decode_messages(&p, &vec![KahePoly::default(); need - 1], None),
                Err(ChannelError::ShortPlaintext {
                    got: need - 1,
                    need
                })
            );
        }
    }

    #[test]
    fn protocol_params_width_matches_encoding() {
        for p in variants(16, 128) {
            let mut rng = ChaCha20Rng::from_seed([5u8; 32]);
            let pp = p.protocol_params(&mut rng, 4);
            assert_eq!(crate::protocol::message_polys(&pp), p.n_polys(), "{p:?}");
        }
    }

    /// The symbol path carries the channel's full `bits_per_symbol`, so values
    /// above the `BYTES_PER_SYMBOL` byte packing round-trip through it.
    #[test]
    fn wide_symbols_round_trip() {
        // rho=8 for 4 inserts: this test is about symbol width, and the peel
        // stalls on ~8% of seeds when delta is sized exactly to rho.
        for p in [
            ChannelParams::for_symbols(8, 3, [0x77; 32]),
            ChannelParams::prony_for_symbols(4, 3),
        ] {
            let mut rng = ChaCha20Rng::from_seed([6u8; 32]);
            // Quarter of the channel's own plaintext modulus, so the value stays
            // inside `[-t/2, t/2)` whatever `t` the feature set selects.
            let wide = (p.plaintext_modulus() as i64 / 4) + 12_345;
            assert!(wide > (1 << (BYTES_PER_SYMBOL * 8)));
            let payloads: Vec<Vec<i64>> = (0..4)
                .map(|i| vec![wide - i, i + 1, wide / (i + 2)])
                .collect();
            let contributions: Vec<Vec<KahePoly>> = payloads
                .iter()
                .map(|m| encode_symbols(&mut rng, &p, m))
                .collect();

            let mut got = decode_symbols(&p, &sum(&contributions), Some(4)).unwrap();
            let mut want = payloads;
            got.sort();
            want.sort();
            assert_eq!(got, want, "{p:?}");
        }
    }

    /// Same element bits either way, so the sketch needs the extra symbol its
    /// narrower field costs.
    #[test]
    fn prony_for_bits_sizes_at_the_sketch_width() {
        let bits = 911 * mse::BITS_PER_SYMBOL;
        let p = ChannelParams::prony_for_bits(300, bits);
        assert_eq!(p.bits_per_symbol(), 35);
        assert_eq!(p.payload_symbols(), bits.div_ceil(35));
        assert!(p.payload_symbols() * p.bits_per_symbol() >= bits);
    }

    /// 300 clients need about 201 bits of nonce, which is six 35-bit symbols,
    /// and the count stays six far beyond any realistic ρ.
    #[test]
    fn nonce_sized_for_the_security_target() {
        assert_eq!(nonce_symbols(300, 35), 6);
        assert_eq!(nonce_symbols(1, 35), 6);
        assert_eq!(nonce_symbols(1 << 17, 35), 6);
        assert_eq!(nonce_symbols(1 << 19, 35), 7);
        assert_eq!(nonce_symbols(300, 32), 7);
    }

    #[test]
    fn oracle_is_deterministic_and_in_range() {
        for modulus in [T_MODULUS_DEFAULT, PRONY_PRIME] {
            let a = random_oracle(&[1, 2, 3], 1000, modulus);
            assert_eq!(a, random_oracle(&[1, 2, 3], 1000, modulus));
            assert_ne!(a, random_oracle(&[1, 2, 4], 1000, modulus));
            assert!(a.iter().all(|&v| (0..modulus as i64).contains(&v)));
        }
    }

    /// Decoding the encoding underneath, without unmasking, must not reveal the
    /// payload: what the encoding carries is the nonce and the masked payload.
    #[test]
    fn encoding_carries_only_the_masked_payload() {
        for p in variants(4, 64) {
            let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
            let payload = vec![0x42u8; 64];
            let plain = encode_message(&mut rng, &p, &payload).unwrap();
            let raw = match (p.mse(), p.prony()) {
                (Some(params), _) => MseEncoding::unpack(params, &plain).decode().unwrap(),
                (_, Some(params)) => PronySketch::unpack(params, &plain).decode().unwrap(),
                (None, None) => unreachable!(),
            };
            assert_eq!(raw.len(), 1);
            let (_, masked) = raw[0].split_at(p.nonce_symbols);
            let symbols = bytes_to_symbols(&payload, p.payload_symbols());
            assert!(masked.iter().zip(&symbols).all(|(a, b)| a != b), "{p:?}");
            assert_eq!(decode_messages(&p, &plain, Some(1)).unwrap(), vec![payload]);
        }
    }

    /// The sketch is the reason to pick it: same payload, far fewer polys.
    #[test]
    fn sketch_is_smaller_than_peeling() {
        let v = variants(300, 16 * 1024);
        assert!(v[1].n_polys() * 2 < v[0].n_polys(), "{:?}", v[1].n_polys());
    }
}
