use std::sync::OnceLock;

use chipmunk_code::{HVCPoly, HVC_MODULUS};
use negacyclic_rings::ntt32::{self, Ring32};
use negacyclic_rings::params::{find_psi32, generate_ring32};
use rand::Rng;

use crate::rings::center_canonical_i64;
use crate::N;

fn ring() -> &'static Ring32<N> {
    static RING: OnceLock<Ring32<N>> = OnceLock::new();
    RING.get_or_init(|| {
        let modulus = HVC_MODULUS as u32;
        generate_ring32(modulus, find_psi32::<N>(modulus))
    })
}

fn forward(poly: &HVCPoly, scratch: &mut [u32; N]) {
    ntt32::reduce_centered_i32_into(ring(), poly.coeffs(), scratch);
    ntt32::ntt(ring(), scratch);
}

fn accumulate_product(accumulator: &mut [u64], left: &[u32; N], right: &[u32; N]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        accumulate_product_neon(accumulator, left, right);
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for ((sum, &left), &right) in accumulator.iter_mut().zip(left).zip(right) {
        *sum += left as u64 * right as u64;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn accumulate_product_neon(accumulator: &mut [u64], left: &[u32; N], right: &[u32; N]) {
    use std::arch::aarch64::*;

    debug_assert_eq!(N % 4, 0);
    for offset in (0..N).step_by(4) {
        let left = vld1q_u32(left.as_ptr().add(offset));
        let right = vld1q_u32(right.as_ptr().add(offset));
        let low = vmull_u32(vget_low_u32(left), vget_low_u32(right));
        let high = vmull_u32(vget_high_u32(left), vget_high_u32(right));
        let acc_low = vld1q_u64(accumulator.as_ptr().add(offset));
        let acc_high = vld1q_u64(accumulator.as_ptr().add(offset + 2));
        vst1q_u64(
            accumulator.as_mut_ptr().add(offset),
            vaddq_u64(acc_low, low),
        );
        vst1q_u64(
            accumulator.as_mut_ptr().add(offset + 2),
            vaddq_u64(acc_high, high),
        );
    }
}

#[derive(Clone)]
pub struct StreamingHvcDot {
    bases_ntt: Vec<[u32; N]>,
}

impl StreamingHvcDot {
    pub fn init<R: Rng>(rng: &mut R, len: usize) -> Self {
        let mut scratch = [0u32; N];
        let bases_ntt = (0..len)
            .map(|_| {
                forward(&HVCPoly::rand_poly(rng), &mut scratch);
                scratch
            })
            .collect();
        Self { bases_ntt }
    }

    pub fn hash(&self, inputs: &[HVCPoly]) -> HVCPoly {
        let modulus = HVC_MODULUS as u64;
        let max_product = (modulus - 1) * (modulus - 1);
        assert!(inputs.len() as u64 <= u64::MAX / max_product);
        let mut scratch = [0u32; N];
        let mut accumulator = [0u64; N];
        assert_eq!(inputs.len(), self.bases_ntt.len());

        accumulator.fill(0);
        for (base, input) in self.bases_ntt.iter().zip(inputs) {
            forward(input, &mut scratch);
            accumulate_product(&mut accumulator, base, &scratch);
        }

        for (coefficient, &sum) in scratch.iter_mut().zip(accumulator.iter()) {
            *coefficient = (sum % modulus) as u32;
        }
        ntt32::inv_ntt(ring(), &mut scratch);
        let mut output = [0i32; N];
        for (coefficient, &value) in output.iter_mut().zip(scratch.iter()) {
            *coefficient = center_canonical_i64(value as i64, HVC_MODULUS as i64) as i32;
        }
        HVCPoly::from_coeffs(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chipmunk_code::{pointwise_dot, HVCNTTPoly};
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn streaming_hash_matches_materialized_hash() {
        for width in [6, 24, 30, 63] {
            let mut basis_rng = StdRng::seed_from_u64(0x4856_4300 + width as u64);
            let mut input_rng = StdRng::seed_from_u64(0x494e_5000 + width as u64);
            let streaming = StreamingHvcDot::init(&mut basis_rng, width);

            let mut reference_rng = StdRng::seed_from_u64(0x4856_4300 + width as u64);
            let reference_bases: Vec<_> = (0..width)
                .map(|_| HVCNTTPoly::from(HVCPoly::rand_poly(&mut reference_rng)))
                .collect();
            let inputs: Vec<_> = (0..width)
                .map(|_| HVCPoly::rand_poly(&mut input_rng))
                .collect();
            let materialized_inputs: Vec<_> = inputs.iter().map(HVCNTTPoly::from).collect();
            let expected = HVCPoly::from(pointwise_dot(&reference_bases, &materialized_inputs));
            assert_eq!(streaming.hash(&inputs), expected, "width {width}");
        }
    }
}
