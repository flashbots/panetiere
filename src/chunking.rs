//! Bit-level chunking of byte strings into `bits`-wide symbols.
//!
//! The bytes form a little-endian bit stream; symbol `i` is bits
//! `[i·bits, (i+1)·bits)` of it, so every symbol is a value in `[0, 2^bits)`.
//! Chunking is linear over byte strings with disjoint set bits (no carries),
//! which is what lets the slot codec sum byte-granular slots exactly.

pub(crate) fn pack(bytes: &[u8], bits: usize, symbols: usize) -> Vec<u64> {
    debug_assert!((1..=56).contains(&bits));
    let mask = (1u64 << bits) - 1;
    (0..symbols)
        .map(|i| {
            let start = i * bits;
            let mut window = 0u128;
            for k in 0..(start % 8 + bits).div_ceil(8) {
                if let Some(&b) = bytes.get(start / 8 + k) {
                    window |= u128::from(b) << (8 * k);
                }
            }
            (window >> (start % 8)) as u64 & mask
        })
        .collect()
}

pub(crate) fn unpack(symbols: &[u64], bits: usize, n_bytes: usize) -> Vec<u8> {
    debug_assert!((1..=56).contains(&bits));
    let mask = (1u64 << bits) - 1;
    let mut out = vec![0u8; n_bytes];
    for (i, &s) in symbols.iter().enumerate() {
        let start = i * bits;
        let window = u128::from(s & mask) << (start % 8);
        for k in 0..(start % 8 + bits).div_ceil(8) {
            if let Some(b) = out.get_mut(start / 8 + k) {
                *b |= (window >> (8 * k)) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn round_trips_at_every_width_and_length() {
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        for bits in [1usize, 7, 8, 9, 31, 32, 35, 56] {
            for len in [0usize, 1, 2, 3, 7, 8, 31, 100] {
                let bytes: Vec<u8> = (0..len).map(|_| rng.gen()).collect();
                let symbols = pack(&bytes, bits, (len * 8).div_ceil(bits));
                assert!(symbols.iter().all(|&s| s < 1 << bits));
                assert_eq!(unpack(&symbols, bits, len), bytes, "bits={bits} len={len}");
            }
        }
    }

    #[test]
    fn short_input_reads_as_zero_padded() {
        let symbols = pack(&[0xFF], 31, 3);
        assert_eq!(symbols, vec![0xFF, 0, 0]);
    }

    #[test]
    fn disjoint_slots_sum_exactly() {
        let mut a = vec![0u8; 20];
        let mut b = vec![0u8; 20];
        a[..9].copy_from_slice(&[0xA5; 9]);
        b[9..].copy_from_slice(&[0x5A; 11]);
        let (sa, sb) = (pack(&a, 31, 6), pack(&b, 31, 6));
        let summed: Vec<u64> = sa.iter().zip(&sb).map(|(x, y)| x + y).collect();
        let mut both = a.clone();
        both[9..].copy_from_slice(&b[9..]);
        assert_eq!(unpack(&summed, 31, 20), both);
    }
}
