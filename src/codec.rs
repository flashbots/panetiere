//! Byte ↔ polynomial codec for KAHE messages.
//!
//! Layout: little-endian `u32` byte-length header, then the bytes themselves,
//! padded with zeros up to the next 4-byte boundary, then packed four bytes per
//! coefficient (little-endian `u32`). Each `KahePoly` carries `N = 2048`
//! coefficients = `8192` bytes; the final poly is zero-padded to fill `N`
//! coefficients. Coefficients land in `[0, 2^32) ⊂ [0, t = 2^36)` so a fresh
//! single-client encode/decode is exact round-trip, with 2^4 per-coefficient
//! headroom under mod-t summation.

use chipmunk_code::{KahePoly, N};

const BYTES_PER_COEFF: usize = 4;
pub const BYTES_PER_POLY: usize = N * BYTES_PER_COEFF;
const HEADER_LEN: usize = 4;
/// Per-coefficient symbol modulus (what fits in `BYTES_PER_COEFF` bytes).
const SYMBOL_MOD: i64 = 1 << 32;

#[derive(Debug, PartialEq)]
pub enum CodecError {
    Empty,
    LengthOverflow { claimed: u32, available: usize },
    CoeffOutOfRange { index: usize, value: i64 },
}

pub fn encode_raw(bytes: &[u8]) -> Vec<KahePoly> {
    bytes.chunks(BYTES_PER_POLY).map(coeffs_from_bytes).collect()
}

pub fn decode_raw(polys: &[KahePoly]) -> Result<Vec<u8>, CodecError> {
    if polys.is_empty() {
        return Err(CodecError::Empty);
    }
    let mut buf = Vec::with_capacity(polys.len() * BYTES_PER_POLY);
    for poly in polys {
        let mut p = *poly;
        p.normalize();
        for (i, &c) in p.coeffs().iter().enumerate() {
            // Symbols are `[0, 2^32)` both raw (direct encode) and after KAHE
            // dec (`poly_mod_t` centers mod t = 2^36, which leaves values
            // < 2^35 untouched). Anything else is genuine overflow.
            if !(0..SYMBOL_MOD).contains(&c) {
                return Err(CodecError::CoeffOutOfRange { index: i, value: c });
            }
            buf.extend_from_slice(&(c as u32).to_le_bytes());
        }
    }
    Ok(buf)
}

fn coeffs_from_bytes(chunk: &[u8]) -> KahePoly {
    let mut coeffs = [0i64; N];
    for (i, group) in chunk.chunks(BYTES_PER_COEFF).enumerate() {
        let mut bytes = [0u8; 4];
        bytes[..group.len()].copy_from_slice(group);
        coeffs[i] = u32::from_le_bytes(bytes) as i64;
    }
    KahePoly::from_coeffs(coeffs)
}

pub fn encode(bytes: &[u8]) -> Vec<KahePoly> {
    let mut buf = Vec::with_capacity(HEADER_LEN + bytes.len());
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);

    let n_polys = buf.len().div_ceil(BYTES_PER_POLY).max(1);
    let mut polys = Vec::with_capacity(n_polys);
    for chunk in buf.chunks(BYTES_PER_POLY) {
        polys.push(coeffs_from_bytes(chunk));
    }
    while polys.len() < n_polys {
        polys.push(KahePoly::from_coeffs([0i64; N]));
    }
    polys
}

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
            // See `decode_raw`: symbols are `[0, 2^32)`.
            if !(0..SYMBOL_MOD).contains(&c) {
                return Err(CodecError::CoeffOutOfRange { index: i, value: c });
            }
            buf.extend_from_slice(&(c as u32).to_le_bytes());
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

/// Per-round beacon: hash of all `rand` values, order- and duplicate-independent.
pub fn beacon(rands: &[u16]) -> u16 {
    use sha2::{Digest, Sha256};
    let mut sorted: Vec<u16> = rands.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut h = Sha256::new();
    for r in sorted {
        h.update(r.to_le_bytes());
    }
    let d = h.finalize();
    u16::from_le_bytes([d[0], d[1]])
}

pub fn allocate(reservations: &[(u16, usize)], beacon: u16, vector_bytes: usize) -> Vec<Option<usize>> {
    let mut order: Vec<usize> = (0..reservations.len()).collect();
    order.sort_by_key(|&i| (reservations[i].0.wrapping_sub(beacon), reservations[i].1));

    let mut out = vec![None; reservations.len()];
    let mut cursor = 0usize;
    let mut pos = 0usize;
    while pos < order.len() {
        let rand = reservations[order[pos]].0;
        let run_end = order[pos..]
            .iter()
            .position(|&j| reservations[j].0 != rand)
            .map(|k| pos + k)
            .unwrap_or(order.len());
        if run_end - pos > 1 {
            pos = run_end;
            continue;
        }
        let aligned = reservations[order[pos]].1.next_multiple_of(BYTES_PER_COEFF);
        if cursor + aligned <= vector_bytes {
            out[order[pos]] = Some(cursor);
            cursor += aligned;
        }
        pos += 1;
    }
    out
}

pub fn encode_at(offset: usize, vector_bytes: usize, payload: &[u8]) -> Vec<KahePoly> {
    let mut buf = vec![0u8; vector_bytes];
    let end = (offset + payload.len()).min(vector_bytes);
    buf[offset..end].copy_from_slice(&payload[..end - offset]);
    encode_raw(&buf)
}

pub fn decode_ranges(plain: &[KahePoly], ranges: &[(usize, usize)]) -> Result<Vec<Vec<u8>>, CodecError> {
    let buf = decode_raw(plain)?;
    let mut out = Vec::with_capacity(ranges.len());
    for &(offset, size) in ranges {
        let end = offset + size;
        if end > buf.len() {
            return Err(CodecError::LengthOverflow { claimed: end as u32, available: buf.len() });
        }
        out.push(buf[offset..end].to_vec());
    }
    Ok(out)
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
        assert_eq!(msg.len() % BYTES_PER_COEFF, 3);
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
        // symbol modulus and `decode_raw` recovers the per-slot bytes.
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
        let mut coeffs = [0i64; N];
        coeffs[0] = 1 << 33; // ≥ 2^32
        let polys = vec![KahePoly::from_coeffs(coeffs)];
        match decode(&polys) {
            Err(CodecError::CoeffOutOfRange { .. }) => {}
            other => panic!("expected CoeffOutOfRange, got {:?}", other),
        }
    }

    #[test]
    fn beacon_order_independent() {
        assert_eq!(beacon(&[7, 42, 1000]), beacon(&[1000, 7, 42, 42]));
    }

    #[test]
    fn allocate_packs_and_overflows() {
        // Two fit into a 2-poly vector, the third overflows.
        let vector_bytes = 2 * BYTES_PER_POLY;
        let res = vec![(10u16, BYTES_PER_POLY), (20, BYTES_PER_POLY), (30, 16)];
        let offs = allocate(&res, 0, vector_bytes);
        assert_eq!(offs[0], Some(0));
        assert_eq!(offs[1], Some(BYTES_PER_POLY));
        assert_eq!(offs[2], None);
    }

    #[test]
    fn allocate_drops_rand_tie() {
        let offs = allocate(&[(5u16, 16), (5, 16), (9, 16)], 0, BYTES_PER_POLY);
        assert_eq!(offs[0], None);
        assert_eq!(offs[1], None);
        assert!(offs[2].is_some());
    }

    #[test]
    fn slot_sum_round_trips_per_reservation() {
        let vector_bytes = 2 * BYTES_PER_POLY;
        let res = vec![(10u16, 5usize), (20, 7)];
        let b = beacon(&[10, 20]);
        let offs = allocate(&res, b, vector_bytes);
        let (o0, o1) = (offs[0].unwrap(), offs[1].unwrap());

        let pa = encode_at(o0, vector_bytes, b"AAAAA");
        let pb = encode_at(o1, vector_bytes, b"BBBBBBB");
        let summed: Vec<KahePoly> = pa.iter().zip(pb.iter()).map(|(x, y)| *x + *y).collect();

        let out = decode_ranges(&summed, &[(o0, 5), (o1, 7)]).unwrap();
        assert_eq!(out[0], b"AAAAA");
        assert_eq!(out[1], b"BBBBBBB");
    }
}
