use chipmunk_code::{HVCPoly, HVC_MODULUS};

use crate::rings::center_i64;
use crate::N;

pub fn sum_hvc_polys(polys: &[&HVCPoly]) -> HVCPoly {
    let mut acc = [0i64; N];

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        unsafe { accumulate_avx2(&mut acc, polys) };
        return finish(acc);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        accumulate_neon(&mut acc, polys);
        return finish(acc);
    }

    #[allow(unreachable_code)]
    {
        accumulate_scalar(&mut acc, polys);
        finish(acc)
    }
}

fn accumulate_scalar(acc: &mut [i64; N], polys: &[&HVCPoly]) {
    for poly in polys {
        for (sum, &coefficient) in acc.iter_mut().zip(poly.coeffs()) {
            *sum += coefficient as i64;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_avx2(acc: &mut [i64; N], polys: &[&HVCPoly]) {
    use std::arch::x86_64::*;

    for poly in polys {
        for offset in (0..N).step_by(16) {
            let coefficients = poly.coeffs().as_ptr().add(offset);
            for lane in [0, 4, 8, 12] {
                let sums = _mm256_loadu_si256(acc.as_ptr().add(offset + lane).cast());
                let values = _mm256_cvtepi32_epi64(_mm_loadu_si128(coefficients.add(lane).cast()));
                _mm256_storeu_si256(
                    acc.as_mut_ptr().add(offset + lane).cast(),
                    _mm256_add_epi64(sums, values),
                );
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_neon(acc: &mut [i64; N], polys: &[&HVCPoly]) {
    use std::arch::aarch64::*;

    for poly in polys {
        for offset in (0..N).step_by(8) {
            let coefficients = poly.coeffs().as_ptr().add(offset);
            let values0 = vld1q_s32(coefficients);
            let values1 = vld1q_s32(coefficients.add(4));
            let widened = [
                vmovl_s32(vget_low_s32(values0)),
                vmovl_s32(vget_high_s32(values0)),
                vmovl_s32(vget_low_s32(values1)),
                vmovl_s32(vget_high_s32(values1)),
            ];
            for (lane, values) in widened.into_iter().enumerate() {
                let target = acc.as_mut_ptr().add(offset + lane * 2);
                vst1q_s64(target, vaddq_s64(vld1q_s64(target), values));
            }
        }
    }
}

fn finish(acc: [i64; N]) -> HVCPoly {
    let q = HVC_MODULUS as i64;
    HVCPoly::from_coeffs(acc.map(|sum| center_i64(sum, q) as i32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chipmunk_code::pointwise_sum_polys;

    #[test]
    fn optimized_sum_matches_chipmunk() {
        let q = HVC_MODULUS;
        let polys: Vec<_> = (0..300)
            .map(|seed| {
                HVCPoly::from_coeffs(std::array::from_fn(|i| {
                    let value = ((seed * 257 + i * 17) % (2 * q as usize - 1)) as i32;
                    value - (q - 1)
                }))
            })
            .collect();

        for count in [0, 1, 2, 6, 16, 100, 300] {
            let refs: Vec<_> = polys[..count].iter().collect();
            let expected = pointwise_sum_polys(&refs);
            let actual = sum_hvc_polys(&refs);
            assert_eq!(actual, expected, "sum differs for {count} polynomials");
            assert!(actual
                .coeffs()
                .iter()
                .all(|&x| (-q / 2..=q / 2).contains(&x)));
        }
    }
}
