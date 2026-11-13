//! Application payload layer over the multiset encodings.

use crate::KahePoly;
use rand::{CryptoRng, Rng};
use sha3::digest::{ExtendableOutput as _, Update as _, XofReader as _};
use sha3::Shake256;

use crate::kahe::{SIGMA_E_DEFAULT, SIGMA_S_DEFAULT, T_MODULUS_DEFAULT};
use crate::mse::{self, MseEncoding, MseError, MseParams};
use crate::prony::{PronyError, PronyParams, PronySketch, PRONY_PRIME};
use crate::protocol::{ProtocolParams, SessionId};

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

/// Separates the masking oracle from every other use of SHAKE256.
const ORACLE_DOMAIN: &[u8] = b"panetiere/ro-mask/v1";

/// Leading payload symbol of a byte message that holds its length, so trailing
/// zero bytes survive the padding.
const LENGTH_HEADER_SYMBOLS: usize = 1;

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
    /// A payload symbol does not fit the channel's `bits_per_symbol`, so
    /// masking it would leave the plaintext modulus.
    SymbolOutOfRange { index: usize, value: i64 },
    /// A decoded message claims more bytes than its element holds.
    BadLength { claimed: u32, max: usize },
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
    pub fn for_messages<R: Rng>(rng: &mut R, rho: u32, message_bytes: usize) -> Self {
        Self::for_symbols(rng, rho, payload_symbols(message_bytes))
    }

    /// Peeling channel for up to `rho` payloads of `payload_symbols` symbols
    /// each. For channels whose payload is already `Z_t` values rather than
    /// bytes — the byte packing of [`encode_message`] does not apply to those.
    pub fn for_symbols<R: Rng>(rng: &mut R, rho: u32, payload_symbols: usize) -> Self {
        let rho = rho.max(1) as usize;
        let nonce_symbols = nonce_symbols(rho, mse::BITS_PER_SYMBOL);
        let delta = (BUCKETS_PER_INSERT * rho).div_ceil(GAMMA);
        Self {
            encoding: Encoding::Mse(MseParams::new(
                rng,
                GAMMA,
                delta,
                rho,
                nonce_symbols + payload_symbols.max(1),
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
        (self.payload_symbols() - LENGTH_HEADER_SYMBOLS) * BYTES_PER_SYMBOL
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

/// `RO(s)`: the first `len · bits_per_symbol` bits of
/// `SHAKE256(domain ‖ sid ‖ s)`, one `bits_per_symbol`-bit value per symbol.
/// The domain, `sid` and nonce all have fixed lengths, so the input is
/// unambiguous.
fn random_oracle(sid: &SessionId, nonce: &[i64], len: usize, bits_per_symbol: usize) -> Vec<i64> {
    let mut hasher = Shake256::default();
    hasher.update(ORACLE_DOMAIN);
    hasher.update(&sid.0);
    for &s in nonce {
        hasher.update(&s.to_le_bytes());
    }
    let mut reader = hasher.finalize_xof();
    let mask = (1u64 << bits_per_symbol) - 1;
    (0..len)
        .map(|_| {
            let mut buf = [0u8; 8];
            reader.read(&mut buf);
            (u64::from_le_bytes(buf) & mask) as i64
        })
        .collect()
}

fn payload_symbols(message_bytes: usize) -> usize {
    LENGTH_HEADER_SYMBOLS + message_bytes.div_ceil(BYTES_PER_SYMBOL)
}

/// The byte length, then the bytes zero-padded to `xi - 1` symbols.
fn bytes_to_symbols(payload: &[u8], xi: usize) -> Vec<i64> {
    let mut buf = (payload.len() as u32).to_le_bytes().to_vec();
    buf.extend_from_slice(payload);
    buf.resize(xi * BYTES_PER_SYMBOL, 0);
    buf.chunks(BYTES_PER_SYMBOL)
        .map(|c| u32::from_le_bytes(c.try_into().expect("chunk is BYTES_PER_SYMBOL")) as i64)
        .collect()
}

fn symbols_to_bytes(symbols: &[i64]) -> Result<Vec<u8>, ChannelError> {
    let (header, body) = symbols.split_at(LENGTH_HEADER_SYMBOLS);
    let claimed = header[0] as u32;
    let max = body.len() * BYTES_PER_SYMBOL;
    if claimed as usize > max {
        return Err(ChannelError::BadLength { claimed, max });
    }
    let mut out = Vec::with_capacity(max);
    for &s in body {
        out.extend_from_slice(&(s as u32).to_le_bytes());
    }
    out.truncate(claimed as usize);
    Ok(out)
}

pub fn encode_message<R: CryptoRng + Rng>(
    rng: &mut R,
    p: &ChannelParams,
    sid: &SessionId,
    payload: &[u8],
) -> Result<Vec<KahePoly>, ChannelError> {
    let max = p.max_payload_bytes();
    if payload.len() > max {
        return Err(ChannelError::PayloadTooLong {
            got: payload.len(),
            max,
        });
    }
    encode_symbols(rng, p, sid, &bytes_to_symbols(payload, p.payload_symbols()))
}

/// Insert `(s, payload ⊕ RO(s))` for a nonce `s` drawn from `rng`. Every
/// symbol must lie in `[0, 2^bits_per_symbol)`.
pub fn encode_symbols<R: CryptoRng + Rng>(
    rng: &mut R,
    p: &ChannelParams,
    sid: &SessionId,
    payload: &[i64],
) -> Result<Vec<KahePoly>, ChannelError> {
    let bits = p.bits_per_symbol();
    if let Some((index, &value)) = payload
        .iter()
        .enumerate()
        .find(|(_, &x)| !(0..1i64 << bits).contains(&x))
    {
        return Err(ChannelError::SymbolOutOfRange { index, value });
    }
    let nonce: Vec<i64> = (0..p.nonce_symbols)
        .map(|_| rng.gen_range(0..1i64 << bits))
        .collect();
    let mask = random_oracle(sid, &nonce, payload.len(), bits);
    let mut element = nonce;
    element.extend(payload.iter().zip(&mask).map(|(&m, &r)| m ^ r));
    Ok(match &p.encoding {
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
    })
}

pub fn cover(p: &ChannelParams) -> Vec<KahePoly> {
    match &p.encoding {
        Encoding::Mse(params) => MseEncoding::cover(params),
        Encoding::Prony(params) => PronySketch::cover(params),
    }
}

/// The round's messages, sorted.
pub fn decode_messages(
    p: &ChannelParams,
    sid: &SessionId,
    plaintext: &[KahePoly],
    expect: Option<usize>,
) -> Result<Vec<Vec<u8>>, ChannelError> {
    let mut messages = decode_symbols(p, sid, plaintext, expect)?
        .iter()
        .map(|s| symbols_to_bytes(s))
        .collect::<Result<Vec<_>, _>>()?;
    messages.sort();
    Ok(messages)
}

/// `expect` optionally checks the total decoded item count, not items per client.
/// Leave it `None` when the active-sender count is unknown (e.g. with cover traffic).
/// MSE residual and Prony slack checks test decoding consistency, not per-client
/// input validity or payload authenticity. Each element's nonce is stripped,
/// its mask removed, and the unmasked payloads come back sorted.
pub fn decode_symbols(
    p: &ChannelParams,
    sid: &SessionId,
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
    let bits = p.bits_per_symbol();
    let mut payloads: Vec<Vec<i64>> = elements
        .iter()
        .map(|element| {
            let canonical: Vec<i64> = element.iter().map(|s| s.rem_euclid(modulus)).collect();
            let (nonce, masked) = canonical.split_at(p.nonce_symbols);
            let mask = random_oracle(sid, nonce, masked.len(), bits);
            masked.iter().zip(&mask).map(|(&v, &r)| v ^ r).collect()
        })
        .collect();
    payloads.sort();
    Ok(payloads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    const SID: SessionId = SessionId([0x11; 32]);

    /// Both encodings, same sizing, so every test below runs twice.
    fn variants(rho: u32, bytes: usize) -> Vec<ChannelParams> {
        let mut rng = ChaCha20Rng::from_seed([0x5C; 32]);
        vec![
            ChannelParams::for_messages(&mut rng, rho, bytes),
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
                .map(|m| encode_message(&mut rng, &p, &SID, m).unwrap())
                .collect();

            let mut got = decode_messages(&p, &SID, &sum(&contributions), Some(8)).unwrap();
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
            assert!(encode_message(&mut rng, &p, &SID, &vec![7u8; max]).is_ok());
            assert_eq!(
                encode_message(&mut rng, &p, &SID, &vec![7u8; max + 1]),
                Err(ChannelError::PayloadTooLong { got: max + 1, max })
            );
        }
    }

    #[test]
    fn cover_only_round_decodes_empty() {
        for p in variants(8, 32) {
            let covers: Vec<Vec<KahePoly>> = (0..8).map(|_| cover(&p)).collect();
            assert_eq!(
                decode_messages(&p, &SID, &sum(&covers), Some(0))
                    .unwrap()
                    .len(),
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
                .map(|i| encode_message(&mut rng, &p, &SID, &[i, i, i, i]).unwrap())
                .collect();
            assert!(
                decode_messages(&p, &SID, &sum(&contributions), None).is_err(),
                "{p:?}"
            );
        }
    }

    #[test]
    fn count_mismatch_reported() {
        for p in variants(8, 16) {
            let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
            let contributions: Vec<Vec<KahePoly>> = (0..3u8)
                .map(|i| encode_message(&mut rng, &p, &SID, &[i; 4]).unwrap())
                .collect();
            let plain = sum(&contributions);
            assert!(decode_messages(&p, &SID, &plain, Some(3)).is_ok(), "{p:?}");
            assert_eq!(
                decode_messages(&p, &SID, &plain, Some(5)),
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
                decode_messages(&p, &SID, &vec![KahePoly::default(); need - 1], None),
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
        let mut key_rng = ChaCha20Rng::from_seed([0x77; 32]);
        for p in [
            ChannelParams::for_symbols(&mut key_rng, 8, 3),
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
                .map(|m| encode_symbols(&mut rng, &p, &SID, m).unwrap())
                .collect();

            let mut got = decode_symbols(&p, &SID, &sum(&contributions), Some(4)).unwrap();
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
    fn oracle_is_deterministic_and_separates_sessions() {
        let a = random_oracle(&SID, &[1, 2, 3], 1000, 35);
        assert_eq!(a, random_oracle(&SID, &[1, 2, 3], 1000, 35));
        assert_ne!(a, random_oracle(&SID, &[1, 2, 4], 1000, 35));
        assert_ne!(
            a,
            random_oracle(&SessionId([0x22; 32]), &[1, 2, 3], 1000, 35)
        );
        assert!(a.iter().all(|&v| (0..1i64 << 35).contains(&v)));
    }

    /// The decoded elements of the encoding underneath, before unmasking.
    fn raw_elements(p: &ChannelParams, plain: &[KahePoly]) -> Vec<Vec<i64>> {
        match (p.mse(), p.prony()) {
            (Some(params), _) => MseEncoding::unpack(params, plain).decode().unwrap(),
            (_, Some(params)) => PronySketch::unpack(params, plain).decode().unwrap(),
            (None, None) => unreachable!("a channel is peeling or sketch"),
        }
    }

    /// The encoding carries the nonce and the masked payload, never the payload.
    #[test]
    fn encoding_carries_only_the_masked_payload() {
        for p in variants(4, 64) {
            let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
            let payload = vec![0x42u8; 64];
            let plain = encode_message(&mut rng, &p, &SID, &payload).unwrap();
            let raw = raw_elements(&p, &plain);
            assert_eq!(raw.len(), 1);
            let (_, masked) = raw[0].split_at(p.nonce_symbols);
            let symbols = bytes_to_symbols(&payload, p.payload_symbols());
            assert!(masked.iter().zip(&symbols).all(|(a, b)| a != b), "{p:?}");
            assert_eq!(
                decode_messages(&p, &SID, &plain, Some(1)).unwrap(),
                vec![payload]
            );
        }
    }

    /// Zero padding alone would eat these; the length header keeps them.
    #[test]
    fn trailing_zero_bytes_and_empty_messages_survive() {
        // rho=8 for 4 inserts, as in `wide_symbols_round_trip`: the peel stalls
        // on some seeds when delta is sized exactly to rho.
        for p in variants(8, 16) {
            let mut rng = ChaCha20Rng::from_seed([8u8; 32]);
            let payloads: Vec<Vec<u8>> = vec![vec![], vec![0], vec![1, 0, 0], vec![0; 16]];
            let contributions: Vec<Vec<KahePoly>> = payloads
                .iter()
                .map(|m| encode_message(&mut rng, &p, &SID, m).unwrap())
                .collect();
            let mut want = payloads;
            want.sort();
            assert_eq!(
                decode_messages(&p, &SID, &sum(&contributions), Some(4)).unwrap(),
                want,
                "{p:?}"
            );
        }
    }

    /// Unmasking under another session's oracle does not give the message back.
    #[test]
    fn wrong_session_does_not_unmask() {
        for p in variants(4, 16) {
            let mut rng = ChaCha20Rng::from_seed([9u8; 32]);
            let plain = encode_symbols(&mut rng, &p, &SID, &[5, 6, 7, 8, 9]).unwrap();
            let other = SessionId([0x22; 32]);
            assert_ne!(
                decode_symbols(&p, &other, &plain, Some(1)).unwrap(),
                vec![vec![5, 6, 7, 8, 9]],
                "{p:?}"
            );
        }
    }

    #[test]
    fn symbol_out_of_range_rejected() {
        for p in variants(4, 4) {
            let mut rng = ChaCha20Rng::from_seed([10u8; 32]);
            let too_wide = 1i64 << p.bits_per_symbol();
            assert_eq!(
                encode_symbols(&mut rng, &p, &SID, &[0, too_wide]),
                Err(ChannelError::SymbolOutOfRange {
                    index: 1,
                    value: too_wide
                })
            );
            assert_eq!(
                encode_symbols(&mut rng, &p, &SID, &[-1, 0]),
                Err(ChannelError::SymbolOutOfRange {
                    index: 0,
                    value: -1
                })
            );
        }
    }

    /// A malicious client can claim any length; decoding refuses one that
    /// overruns its element instead of reading past it.
    #[test]
    fn overlong_length_header_rejected() {
        for p in variants(4, 8) {
            let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
            let plain = encode_symbols(&mut rng, &p, &SID, &[9, 0, 0]).unwrap();
            assert_eq!(
                decode_messages(&p, &SID, &plain, Some(1)),
                Err(ChannelError::BadLength { claimed: 9, max: 8 })
            );
        }
    }

    /// The sketch is the reason to pick it: same payload, far fewer polys.
    #[test]
    fn sketch_is_smaller_than_peeling() {
        let v = variants(300, 16 * 1024);
        assert!(v[1].n_polys() * 2 < v[0].n_polys(), "{:?}", v[1].n_polys());
    }
}
