//! Secret sharing.

use crate::{CsPoly, CS_MODULUS, CS_MODULUS_OVER_TWO, N};
use rand::Rng;

pub trait Sss {
    type Secret: Clone;
    type Share: Clone;

    fn share<R: Rng>(rng: &mut R, secret: &Self::Secret, n: usize) -> Vec<Self::Share>;
    fn recover(shares: &[Self::Share]) -> Self::Secret;
}

/// Additive n-of-n sharing over `CsPoly`. `s_1..s_{n-1}` random; `s_n = secret − Σ s_i`.
pub struct AdditiveSharing;

impl Sss for AdditiveSharing {
    type Secret = CsPoly;
    type Share = CsPoly;

    fn share<R: Rng>(rng: &mut R, secret: &Self::Secret, n: usize) -> Vec<Self::Share> {
        assert!(n >= 1);
        let mut shares = Vec::with_capacity(n);
        let mut acc = CsPoly::default();
        for _ in 0..n - 1 {
            let s = CsPoly::rand_poly(rng);
            acc += s;
            shares.push(s);
        }
        shares.push(*secret - acc);
        shares
    }

    fn recover(shares: &[Self::Share]) -> Self::Secret {
        shares.iter().copied().fold(CsPoly::default(), |a, x| a + x)
    }
}

/// Shamir t-of-n secret sharing over `R_q`.
///
/// Evaluation points are `Z_q*` scalars `1..=n` (their pairwise differences are
/// units, so Lagrange interpolation is well-defined despite `R_q` not being a
/// field). Sharing polynomial `f(X) = secret + Σ_{k=1..t-1} c_k X^k` with
/// `c_k ← R_q` random; `share_i := f(i+1)`. Recovery is Lagrange at X = 0
/// using any subset of `t` shares.
pub struct ShamirSharing;

#[derive(Debug, PartialEq, Eq)]
pub enum SssError {
    NotEnoughShares,
    DuplicateIndex,
    ZeroPoint,
}

#[derive(Clone, Debug)]
pub struct ShamirParams {
    pub t: usize,
    pub n: usize,
}

impl ShamirParams {
    pub fn new(t: usize, n: usize) -> Self {
        assert!(t >= 1 && t <= n, "require 1 ≤ t ≤ n");
        assert!(n < CS_MODULUS as usize, "n must be < q for distinct points");
        Self { t, n }
    }
}

impl ShamirSharing {
    /// Returns `params.n` shares; `out[i]` is the evaluation at point `i+1`.
    pub fn share<R: Rng>(rng: &mut R, params: &ShamirParams, secret: &CsPoly) -> Vec<CsPoly> {
        let t = params.t;
        let n = params.n;
        let coeffs: Vec<CsPoly> = (0..t - 1).map(|_| CsPoly::rand_poly(rng)).collect();
        (0..n)
            .map(|i| {
                let x = (i + 1) as i32;
                let mut x_pow = x;
                let mut acc = *secret;
                for c in &coeffs {
                    acc += scalar_mul(c, x_pow);
                    x_pow = mul_q(x_pow, x);
                }
                acc
            })
            .collect()
    }

    /// Recover the secret from any `t` `(index, share)` samples (0-based
    /// indices into the original `share()` output). Extra samples beyond `t`
    /// are ignored.
    pub fn recover(params: &ShamirParams, samples: &[(usize, CsPoly)]) -> Result<CsPoly, SssError> {
        let t = params.t;
        if samples.len() < t {
            return Err(SssError::NotEnoughShares);
        }
        let xs: Vec<i32> = samples
            .iter()
            .take(t)
            .map(|(idx, _)| reduce(*idx as i64 + 1))
            .collect();
        // Nonzero, pairwise-distinct points mod q ⇒ Lagrange denominators are units.
        for i in 0..t {
            if xs[i] == 0 {
                return Err(SssError::ZeroPoint);
            }
            for j in (i + 1)..t {
                if xs[i] == xs[j] {
                    return Err(SssError::DuplicateIndex);
                }
            }
        }
        let lagrange: Vec<i32> = (0..t)
            .map(|i| {
                let xi = xs[i];
                let mut num = 1i32;
                let mut den = 1i32;
                for j in 0..t {
                    if j == i {
                        continue;
                    }
                    let xj = xs[j];
                    num = mul_q(num, sub_q(0, xj));
                    den = mul_q(den, sub_q(xi, xj));
                }
                mul_q(num, inv_q(den))
            })
            .collect();
        let mut acc = CsPoly::default();
        for (slot, (_, share)) in samples.iter().take(t).enumerate() {
            acc += scalar_mul(share, lagrange[slot]);
        }
        Ok(acc)
    }
}

#[inline]
fn reduce(x: i64) -> i32 {
    let mut r = x.rem_euclid(CS_MODULUS as i64) as i32;
    if r > CS_MODULUS_OVER_TWO {
        r -= CS_MODULUS;
    }
    r
}

#[inline]
fn mul_q(a: i32, b: i32) -> i32 {
    reduce(a as i64 * b as i64)
}

#[inline]
fn sub_q(a: i32, b: i32) -> i32 {
    reduce(a as i64 - b as i64)
}

fn pow_q(base: i32, mut exp: i32) -> i32 {
    let mut result = 1i32;
    let mut b = reduce(base as i64);
    while exp > 0 {
        if exp & 1 == 1 {
            result = mul_q(result, b);
        }
        b = mul_q(b, b);
        exp >>= 1;
    }
    result
}

#[inline]
fn inv_q(x: i32) -> i32 {
    debug_assert!(reduce(x as i64) != 0, "inv of zero");
    pow_q(x, CS_MODULUS - 2)
}

fn scalar_mul(p: &CsPoly, c: i32) -> CsPoly {
    let mut out = [0i32; N];
    let src = p.coeffs();
    for i in 0..N {
        out[i] = mul_q(src[i], c);
    }
    CsPoly::from_coeffs(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn round_trip() {
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        for n in [1usize, 2, 4, 7] {
            let secret = CsPoly::rand_poly(&mut rng);
            let shares = AdditiveSharing::share(&mut rng, &secret, n);
            assert_eq!(shares.len(), n);
            assert_eq!(AdditiveSharing::recover(&shares), secret);
        }
    }

    #[test]
    fn shamir_round_trip() {
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        for &(t, n) in &[(1usize, 1), (1, 4), (2, 3), (3, 5), (4, 7), (5, 7)] {
            let params = ShamirParams::new(t, n);
            let secret = CsPoly::rand_poly(&mut rng);
            let shares = ShamirSharing::share(&mut rng, &params, &secret);
            assert_eq!(shares.len(), n);
            // Recover from the first t shares.
            let samples: Vec<_> = shares.iter().take(t).copied().enumerate().collect();
            assert_eq!(ShamirSharing::recover(&params, &samples).unwrap(), secret);
        }
    }

    #[test]
    fn shamir_any_subset() {
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        let (t, n) = (3, 5);
        let params = ShamirParams::new(t, n);
        let secret = CsPoly::rand_poly(&mut rng);
        let shares = ShamirSharing::share(&mut rng, &params, &secret);
        // Try every t-subset of {0..n}.
        for a in 0..n {
            for b in (a + 1)..n {
                for c in (b + 1)..n {
                    let samples = vec![(a, shares[a]), (b, shares[b]), (c, shares[c])];
                    assert_eq!(ShamirSharing::recover(&params, &samples).unwrap(), secret);
                }
            }
        }
    }

    #[test]
    fn shamir_homomorphic_sum() {
        // Σ_j recover(shares_j[I]) == recover(Σ_j shares_j at I)
        let mut rng = ChaCha20Rng::from_seed([13u8; 32]);
        let (t, n) = (3, 5);
        let params = ShamirParams::new(t, n);
        let secrets: Vec<CsPoly> = (0..4).map(|_| CsPoly::rand_poly(&mut rng)).collect();
        let all_shares: Vec<Vec<CsPoly>> = secrets
            .iter()
            .map(|s| ShamirSharing::share(&mut rng, &params, s))
            .collect();
        let indices = [0usize, 2, 4];
        // sum the shares per index across clients
        let summed: Vec<CsPoly> = indices
            .iter()
            .map(|&i| all_shares.iter().fold(CsPoly::default(), |a, sh| a + sh[i]))
            .collect();
        let samples: Vec<_> = indices
            .iter()
            .copied()
            .zip(summed.iter().copied())
            .collect();
        let recovered = ShamirSharing::recover(&params, &samples).unwrap();
        let expected = secrets.iter().fold(CsPoly::default(), |a, s| a + *s);
        assert_eq!(recovered, expected);
    }

    #[test]
    fn shamir_recover_rejects_bad_samples() {
        let mut rng = ChaCha20Rng::from_seed([17u8; 32]);
        let (t, n) = (3, 5);
        let params = ShamirParams::new(t, n);
        let secret = CsPoly::rand_poly(&mut rng);
        let shares = ShamirSharing::share(&mut rng, &params, &secret);

        let dup = vec![(0, shares[0]), (0, shares[0]), (2, shares[2])];
        assert_eq!(
            ShamirSharing::recover(&params, &dup),
            Err(SssError::DuplicateIndex)
        );

        let short = vec![(0, shares[0]), (1, shares[1])];
        assert_eq!(
            ShamirSharing::recover(&params, &short),
            Err(SssError::NotEnoughShares)
        );

        // Point idx+1 ≡ 0 mod q.
        let zero = vec![
            (CS_MODULUS as usize - 1, shares[0]),
            (1, shares[1]),
            (2, shares[2]),
        ];
        assert_eq!(
            ShamirSharing::recover(&params, &zero),
            Err(SssError::ZeroPoint)
        );
    }
}
