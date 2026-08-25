//! Application payload layer over the multiset encodings.

use crate::KahePoly;
use rand::Rng;

use crate::kahe::{SIGMA_E_DEFAULT, SIGMA_S_DEFAULT, T_MODULUS_DEFAULT};
use crate::mse::{self, MseEncoding, MseParams};
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

#[derive(Debug, PartialEq)]
pub enum ChannelError {
    /// Peeling stalled: the encoding held more elements than `δ` was sized for,
    /// so the round's payloads are unrecoverable.
    PeelStalled,
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

/// Sizing and keying for one channel, over either multiset encoding. `Mse`
/// peels and degrades gracefully; `Prony` is ~3× smaller but all-or-nothing.
#[derive(Clone, Debug, PartialEq)]
pub enum ChannelParams {
    Mse(MseParams),
    Prony(PronyParams),
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
        let delta = (BUCKETS_PER_INSERT * rho.max(1) as usize).div_ceil(GAMMA);
        ChannelParams::Mse(MseParams::new(
            GAMMA,
            delta,
            payload_symbols.max(1),
            prf_key,
        ))
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
        Self::from_prony(PronyParams::new(
            rho.max(1) as usize,
            payload_symbols.max(1),
        ))
    }

    /// Wrap parameters chosen elsewhere (e.g. a fixed-arity token channel).
    pub fn from_mse(mse: MseParams) -> Self {
        ChannelParams::Mse(mse)
    }

    pub fn from_prony(prony: PronyParams) -> Self {
        assert!(
            (1u64 << (BYTES_PER_SYMBOL * 8)) < prony.p,
            "p must exceed the {BYTES_PER_SYMBOL}-byte symbol packing",
        );
        ChannelParams::Prony(prony)
    }

    pub fn mse(&self) -> Option<&MseParams> {
        match self {
            ChannelParams::Mse(p) => Some(p),
            ChannelParams::Prony(_) => None,
        }
    }

    pub fn prony(&self) -> Option<&PronyParams> {
        match self {
            ChannelParams::Prony(p) => Some(p),
            ChannelParams::Mse(_) => None,
        }
    }

    /// `KahePoly` width of one packed encoding — every client contributes this
    /// many, real or cover.
    pub fn n_polys(&self) -> usize {
        match self {
            ChannelParams::Mse(p) => MseEncoding::n_polys(p),
            ChannelParams::Prony(p) => PronySketch::n_polys(p),
        }
    }

    pub fn payload_symbols(&self) -> usize {
        match self {
            ChannelParams::Mse(p) => p.payload_symbols,
            ChannelParams::Prony(p) => p.payload_symbols,
        }
    }

    /// Payload bits one symbol carries: `⌊log₂ t⌋` in the peeling structure's
    /// `Z_t`, `⌊log₂ p⌋` in the sketch's prime field — 36 against 35 by
    /// default, 35 against 35 under `rns`.
    pub fn bits_per_symbol(&self) -> usize {
        match self {
            ChannelParams::Mse(_) => mse::BITS_PER_SYMBOL,
            ChannelParams::Prony(p) => p.bits_per_symbol(),
        }
    }

    /// KAHE plaintext modulus the symbols live in. The sketch needs a prime.
    pub fn plaintext_modulus(&self) -> u64 {
        match self {
            ChannelParams::Mse(_) => T_MODULUS_DEFAULT,
            ChannelParams::Prony(p) => p.p,
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

pub fn encode_message<R: Rng>(
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

pub fn encode_symbols<R: Rng>(rng: &mut R, p: &ChannelParams, payload: &[i64]) -> Vec<KahePoly> {
    match p {
        ChannelParams::Mse(params) => {
            let mut enc = MseEncoding::new(params.clone());
            enc.insert(rng, payload);
            enc.pack()
        }
        ChannelParams::Prony(params) => {
            let mut sketch = PronySketch::new(params.clone());
            sketch.insert(rng, payload);
            sketch.pack()
        }
    }
}

pub fn cover(p: &ChannelParams) -> Vec<KahePoly> {
    match p {
        ChannelParams::Mse(params) => MseEncoding::cover(params),
        ChannelParams::Prony(params) => PronySketch::cover(params),
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
    let elements = match p {
        ChannelParams::Mse(params) => MseEncoding::unpack(params, &plaintext[..need])
            .decode()
            .map_err(|_| ChannelError::PeelStalled)?,
        ChannelParams::Prony(params) => PronySketch::unpack(params, &plaintext[..need])
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
    Ok(elements)
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

    /// The sketch is the reason to pick it: same payload, far fewer polys.
    #[test]
    fn sketch_is_smaller_than_peeling() {
        let v = variants(300, 16 * 1024);
        assert!(v[1].n_polys() * 2 < v[0].n_polys(), "{:?}", v[1].n_polys());
    }
}
