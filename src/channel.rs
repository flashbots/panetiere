//! Application payload layer over the multiset encoding.
//!
//! A *channel* is one MSE-shaped anonymous broadcast: `ρ` clients each insert at
//! most one byte payload, the protocol recovers `Σ` of their packed encodings,
//! and peeling returns the payloads. This module owns the sizing (`γ`, `δ`, `ξ`),
//! the byte↔symbol packing, and the plaintext→messages direction, so a caller
//! never re-derives them.
//!
//! Payload bytes ride MSE symbols, which live in `Z_t`. [`BYTES_PER_SYMBOL`] is
//! deliberately below the symbol width: the spare bits are the headroom that
//! keeps a summed cell from wrapping.

use chipmunk_code::KahePoly;
use rand::Rng;

use crate::mse::{self, MseEncoding, MseParams};
use crate::protocol::ProtocolParams;

/// Bytes packed into one MSE symbol, little-endian.
pub const BYTES_PER_SYMBOL: usize = 4;

/// A symbol must not fill its `Z_t` cell — the slack is the mod-`t` summation
/// headroom. Breaks the build if `t` is ever lowered under the packing.
const _: () = assert!(BYTES_PER_SYMBOL * 8 < mse::BITS_PER_SYMBOL);

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
    /// Fewer polys than the parameters describe.
    ShortPlaintext { got: usize, need: usize },
    /// Recovered element count disagrees with what the caller expected.
    CountMismatch { got: usize, expected: usize },
    /// Payload exceeds what `ξ` symbols hold.
    PayloadTooLong { got: usize, max: usize },
}

/// Sizing and keying for one channel.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelParams {
    mse: MseParams,
}

impl ChannelParams {
    /// Size for up to `rho` inserted payloads of `message_bytes` each.
    pub fn for_messages(rho: u32, message_bytes: usize, prf_key: [u8; 32]) -> Self {
        Self::for_symbols(
            rho,
            message_bytes.div_ceil(BYTES_PER_SYMBOL).max(1),
            prf_key,
        )
    }

    /// Size for up to `rho` payloads of `payload_symbols` symbols each. For
    /// channels whose payload is already `Z_t` values rather than bytes — the
    /// byte packing of [`encode_message`] does not apply to those.
    pub fn for_symbols(rho: u32, payload_symbols: usize, prf_key: [u8; 32]) -> Self {
        let delta = (BUCKETS_PER_INSERT * rho.max(1) as usize).div_ceil(GAMMA);
        ChannelParams {
            mse: MseParams::new(GAMMA, delta, payload_symbols.max(1), prf_key),
        }
    }

    /// Wrap parameters chosen elsewhere (e.g. a fixed-arity token channel).
    pub fn from_mse(mse: MseParams) -> Self {
        Self { mse }
    }

    pub fn mse(&self) -> &MseParams {
        &self.mse
    }

    /// `KahePoly` width of one packed encoding — every client contributes this
    /// many, real or cover.
    pub fn n_polys(&self) -> usize {
        MseEncoding::n_polys(&self.mse)
    }

    /// Largest payload [`encode_message`] accepts.
    pub fn max_payload_bytes(&self) -> usize {
        self.mse.payload_symbols * BYTES_PER_SYMBOL
    }

    /// Protocol parameters whose KAHE width is exactly one packed encoding.
    pub fn protocol_params<R: Rng>(&self, rng: &mut R, n_servers: usize) -> ProtocolParams {
        ProtocolParams::setup_with_kahe_dims(rng, n_servers, self.n_polys())
    }
}

/// Pack `payload` into `ξ` little-endian symbols, zero-padded.
fn bytes_to_symbols(payload: &[u8], xi: usize) -> Vec<i64> {
    let mut buf = payload.to_vec();
    buf.resize(xi * BYTES_PER_SYMBOL, 0);
    buf.chunks(BYTES_PER_SYMBOL)
        .map(|c| u32::from_le_bytes(c.try_into().expect("chunk is BYTES_PER_SYMBOL")) as i64)
        .collect()
}

/// Inverse of [`bytes_to_symbols`]. Trailing zero padding is left in place —
/// payload framing is the caller's concern.
fn symbols_to_bytes(symbols: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(symbols.len() * BYTES_PER_SYMBOL);
    for &s in symbols {
        out.extend_from_slice(&(s as u32).to_le_bytes());
    }
    out
}

/// Encode one payload as a client's contribution.
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
    let mut enc = MseEncoding::new(p.mse.clone());
    enc.insert(rng, &bytes_to_symbols(payload, p.mse.payload_symbols));
    Ok(enc.pack())
}

/// Cover contribution: adds nothing to the sum, so it occupies no bucket.
pub fn cover(p: &ChannelParams) -> Vec<KahePoly> {
    MseEncoding::cover(&p.mse)
}

/// Peel the recovered plaintext back into payloads.
///
/// `expect = Some(n)` requires exactly `n` elements — the recovered-count check.
/// Pass `None` when the count is unobservable, which is the case whenever cover
/// traffic is indistinguishable from real traffic; a stall is still an `Err`,
/// which is what stops a failed round from looking like an empty one.
pub fn decode_messages(
    p: &ChannelParams,
    plaintext: &[KahePoly],
    expect: Option<usize>,
) -> Result<Vec<Vec<u8>>, ChannelError> {
    let need = p.n_polys();
    if plaintext.len() < need {
        return Err(ChannelError::ShortPlaintext {
            got: plaintext.len(),
            need,
        });
    }
    let elements = MseEncoding::unpack(&p.mse, &plaintext[..need])
        .decode()
        .map_err(|_| ChannelError::PeelStalled)?;
    if let Some(expected) = expect {
        if elements.len() != expected {
            return Err(ChannelError::CountMismatch {
                got: elements.len(),
                expected,
            });
        }
    }
    Ok(elements.iter().map(|s| symbols_to_bytes(s)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn params(rho: u32, bytes: usize) -> ChannelParams {
        ChannelParams::for_messages(rho, bytes, [0x5C; 32])
    }

    /// Sum `n` contributions positionally, as the protocol's aggregation does.
    fn sum(polys: &[Vec<KahePoly>]) -> Vec<KahePoly> {
        let mut acc = polys[0].clone();
        for p in &polys[1..] {
            for (a, b) in acc.iter_mut().zip(p) {
                *a = *a + *b;
            }
        }
        acc
    }

    #[test]
    fn round_trip_at_capacity() {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        let p = params(8, 64);
        assert_eq!(p.max_payload_bytes(), 64);

        let payloads: Vec<Vec<u8>> = (0..8u8)
            .map(|i| (0..64).map(|j| i.wrapping_mul(31).wrapping_add(j)).collect())
            .collect();
        let contributions: Vec<Vec<KahePoly>> = payloads
            .iter()
            .map(|m| encode_message(&mut rng, &p, m).unwrap())
            .collect();

        let got = decode_messages(&p, &sum(&contributions), Some(8)).unwrap();
        let mut got = got;
        let mut want = payloads;
        got.sort();
        want.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn payload_one_byte_over_capacity_rejected() {
        let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
        let p = params(8, 64);
        let max = p.max_payload_bytes();
        assert!(encode_message(&mut rng, &p, &vec![7u8; max]).is_ok());
        assert_eq!(
            encode_message(&mut rng, &p, &vec![7u8; max + 1]),
            Err(ChannelError::PayloadTooLong {
                got: max + 1,
                max
            })
        );
    }

    #[test]
    fn cover_only_round_decodes_empty() {
        let p = params(8, 32);
        let covers: Vec<Vec<KahePoly>> = (0..8).map(|_| cover(&p)).collect();
        assert_eq!(decode_messages(&p, &sum(&covers), Some(0)).unwrap().len(), 0);
    }

    /// The regression this module exists for: an over-subscribed structure must
    /// surface as an error, not as an empty round.
    #[test]
    fn oversubscribed_peel_stalls_loudly() {
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        // Sized for 4, given 40.
        let p = params(4, 16);
        let contributions: Vec<Vec<KahePoly>> = (0..40u8)
            .map(|i| encode_message(&mut rng, &p, &[i, i, i, i]).unwrap())
            .collect();
        assert_eq!(
            decode_messages(&p, &sum(&contributions), None),
            Err(ChannelError::PeelStalled)
        );
    }

    #[test]
    fn count_mismatch_reported() {
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let p = params(8, 16);
        let contributions: Vec<Vec<KahePoly>> = (0..3u8)
            .map(|i| encode_message(&mut rng, &p, &[i; 4]).unwrap())
            .collect();
        let plain = sum(&contributions);
        assert!(decode_messages(&p, &plain, Some(3)).is_ok());
        assert_eq!(
            decode_messages(&p, &plain, Some(5)),
            Err(ChannelError::CountMismatch {
                got: 3,
                expected: 5
            })
        );
    }

    #[test]
    fn short_plaintext_reported() {
        let p = params(8, 16);
        let need = p.n_polys();
        assert_eq!(
            decode_messages(&p, &vec![KahePoly::default(); need - 1], None),
            Err(ChannelError::ShortPlaintext {
                got: need - 1,
                need
            })
        );
    }

    #[test]
    fn protocol_params_width_matches_encoding() {
        let mut rng = ChaCha20Rng::from_seed([5u8; 32]);
        let p = params(16, 128);
        let pp = p.protocol_params(&mut rng, 4);
        assert_eq!(crate::protocol::message_polys(&pp), p.n_polys());
    }
}
