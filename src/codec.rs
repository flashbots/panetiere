//! Byte ↔ polynomial codec for KAHE messages.
//!
//! Layout: little-endian `u32` byte-length header, then the bytes themselves,
//! padded with zeros up to the next even byte count, then packed two bytes per
//! coefficient (little-endian `u16`). Each `KahePoly` carries `N = 2048`
//! coefficients = `4096` bytes; the final poly is zero-padded to fill `N`
//! coefficients. Coefficients land in `[0, 65536) ⊂ [0, q)` so a fresh single-
//! client encode/decode is exact round-trip.
//!
//! **Caveat on aggregation.** Decoding a *sum* of encoded messages is only
//! meaningful for application-defined encodings (e.g. unique non-overlapping
//! slots per client). Generic byte payloads from multiple clients will overflow
//! the per-coefficient budget and the sum is not byte-recoverable. The protocol
//! recovers `Σ m_i` as a polynomial; how to read meaning out of that is the
//! application's choice.

use chipmunk_code::{KahePoly, Polynomial, N};

const BYTES_PER_COEFF: usize = 2;
const BYTES_PER_POLY: usize = N * BYTES_PER_COEFF;
const HEADER_LEN: usize = 4;

#[derive(Debug, PartialEq)]
pub enum CodecError {
    /// `polys` was empty so no header could be read.
    Empty,
    /// Decoded length exceeds available payload bytes.
    LengthOverflow { claimed: u32, available: usize },
    /// A coefficient lies outside `[0, 65536)` after `lift` — typically caused by
    /// decoding a sum of encoded messages whose per-coefficient sums overflowed.
    CoeffOutOfRange { index: usize, value: i32 },
}

/// Encode `bytes` into a sequence of polynomials, padding to a whole number of
/// coefficients but **without** any length header. Used when caller pre-agrees
/// on a fixed buffer size — typical for slot-mode aggregation, where one
/// `decode_raw(sum_of_encoded)` returns the per-slot mixture without the
/// header coefficients overflowing under summation.
pub fn encode_raw(bytes: &[u8]) -> Vec<KahePoly> {
    let mut buf = bytes.to_vec();
    if buf.len() % BYTES_PER_COEFF != 0 {
        buf.push(0);
    }
    // `coeffs_from_bytes` zero-pads short chunks internally, so the
    // final under-filled chunk produces a correctly padded poly.
    buf.chunks(BYTES_PER_POLY).map(coeffs_from_bytes).collect()
}

/// Decode polynomials produced by `encode_raw`. Returns `polys.len() *
/// BYTES_PER_POLY` bytes (caller trims/parses).
pub fn decode_raw(polys: &[KahePoly]) -> Result<Vec<u8>, CodecError> {
    if polys.is_empty() {
        return Err(CodecError::Empty);
    }
    let mut buf = Vec::with_capacity(polys.len() * BYTES_PER_POLY);
    for poly in polys {
        let mut p = *poly;
        p.normalize();
        for (i, &c) in p.coeffs().iter().enumerate() {
            // A symbol is defined mod t = 2^16. It arrives either raw-unsigned
            // `[0, 2^16)` (direct encode) or centered `[-2^15, 2^15)` (KAHE dec,
            // `poly_mod_t`). Accept that union; reject genuine overflow (e.g. a
            // summed-message coefficient ≥ 2^16 in magnitude).
            if !(-(1 << 15)..(1 << 16)).contains(&c) {
                return Err(CodecError::CoeffOutOfRange { index: i, value: c });
            }
            let u = c.rem_euclid(1 << 16);
            buf.push((u & 0xFF) as u8);
            buf.push(((u >> 8) & 0xFF) as u8);
        }
    }
    Ok(buf)
}

fn coeffs_from_bytes(chunk: &[u8]) -> KahePoly {
    let mut coeffs = [0i32; N];
    for (i, pair) in chunk.chunks(BYTES_PER_COEFF).enumerate() {
        let lo = pair[0] as u32;
        let hi = if pair.len() == 2 { pair[1] as u32 } else { 0 };
        coeffs[i] = (lo | (hi << 8)) as i32;
    }
    KahePoly::from_coeffs(coeffs)
}

/// Encode `bytes` into a sequence of polynomials. Always succeeds.
pub fn encode(bytes: &[u8]) -> Vec<KahePoly> {
    let mut buf = Vec::with_capacity(HEADER_LEN + bytes.len() + 1);
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
    if buf.len() % BYTES_PER_COEFF != 0 {
        buf.push(0);
    }

    let n_polys = buf.len().div_ceil(BYTES_PER_POLY).max(1);
    let mut polys = Vec::with_capacity(n_polys);
    for chunk in buf.chunks(BYTES_PER_POLY) {
        polys.push(coeffs_from_bytes(chunk));
    }
    while polys.len() < n_polys {
        polys.push(KahePoly::from_coeffs([0i32; N]));
    }
    polys
}

/// Decode a sequence of polynomials produced by `encode`. Each coefficient must
/// lie in `[0, 65536)` (after `lift`); otherwise `CoeffOutOfRange` is returned.
pub fn decode(polys: &[KahePoly]) -> Result<Vec<u8>, CodecError> {
    if polys.is_empty() {
        return Err(CodecError::Empty);
    }
    let total_bytes = polys.len() * BYTES_PER_POLY;
    let mut buf = Vec::with_capacity(total_bytes);
    for poly in polys {
        let mut p = *poly;
        p.normalize();
        for (i, &c) in p.coeffs().iter().enumerate() {
            // See `decode_raw`: symbols are mod-t = 2^16, raw-unsigned or
            // centered. Accept the union, reject genuine overflow.
            if !(-(1 << 15)..(1 << 16)).contains(&c) {
                return Err(CodecError::CoeffOutOfRange { index: i, value: c });
            }
            let u = c.rem_euclid(1 << 16);
            buf.push((u & 0xFF) as u8);
            buf.push(((u >> 8) & 0xFF) as u8);
        }
    }
    if buf.len() < HEADER_LEN {
        return Err(CodecError::Empty);
    }
    let len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let available = buf.len() - HEADER_LEN;
    if (len as usize) > available {
        return Err(CodecError::LengthOverflow {
            claimed: len,
            available,
        });
    }
    Ok(buf[HEADER_LEN..HEADER_LEN + len as usize].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_empty() {
        let polys = encode(b"");
        assert_eq!(decode(&polys).unwrap(), b"");
    }

    #[test]
    fn round_trip_short() {
        let msg = b"hello panetiere";
        let polys = encode(msg);
        assert_eq!(polys.len(), 1);
        assert_eq!(decode(&polys).unwrap(), msg);
    }

    #[test]
    fn round_trip_exact_one_poly() {
        // header (4) + payload = exactly BYTES_PER_POLY (1024)
        let msg = vec![0xABu8; BYTES_PER_POLY - HEADER_LEN];
        let polys = encode(&msg);
        assert_eq!(polys.len(), 1);
        assert_eq!(decode(&polys).unwrap(), msg);
    }

    #[test]
    fn round_trip_multi_poly() {
        // Span ≥4 polys regardless of BYTES_PER_POLY (4096 at N=2048).
        let msg: Vec<u8> = (0..BYTES_PER_POLY * 3 + 500).map(|i| (i * 37) as u8).collect();
        let polys = encode(&msg);
        assert!(polys.len() >= 4);
        assert_eq!(decode(&polys).unwrap(), msg);
    }

    #[test]
    fn round_trip_odd_length() {
        let msg = b"odd-length-message-with-31-byte";
        assert_eq!(msg.len() % 2, 1);
        let polys = encode(msg);
        assert_eq!(decode(&polys).unwrap().as_slice(), msg);
    }

    #[test]
    fn raw_round_trip() {
        let buf = vec![0xCDu8; BYTES_PER_POLY];
        let polys = encode_raw(&buf);
        assert_eq!(polys.len(), 1);
        assert_eq!(decode_raw(&polys).unwrap(), buf);
    }

    #[test]
    fn raw_sum_of_disjoint_slots_decodes() {
        // Two clients writing into disjoint byte slots inside the same poly.
        // `encode_raw` skips the length header, so coefficient sums stay below
        // 2^16 and `decode_raw` recovers the per-slot bytes.
        let mut buf_a = vec![0u8; BYTES_PER_POLY];
        let mut buf_b = vec![0u8; BYTES_PER_POLY];
        buf_a[0..4].copy_from_slice(b"AAAA");
        buf_b[100..104].copy_from_slice(b"BBBB");

        let pa = encode_raw(&buf_a);
        let pb = encode_raw(&buf_b);
        let summed: Vec<KahePoly> = pa
            .iter()
            .zip(pb.iter())
            .map(|(x, y)| *x + *y)
            .collect();
        let out = decode_raw(&summed).unwrap();
        assert_eq!(&out[0..4], b"AAAA");
        assert_eq!(&out[100..104], b"BBBB");
    }

    #[test]
    fn decode_rejects_empty() {
        assert_eq!(decode(&[]), Err(CodecError::Empty));
    }

    #[test]
    fn decode_rejects_oob_coeff() {
        let mut coeffs = [0i32; N];
        coeffs[0] = 1 << 17; // > 65535
        let polys = vec![KahePoly::from_coeffs(coeffs)];
        match decode(&polys) {
            Err(CodecError::CoeffOutOfRange { .. }) => {}
            other => panic!("expected CoeffOutOfRange, got {:?}", other),
        }
    }
}
