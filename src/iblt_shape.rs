//! IBLT shape selection: the smallest `θ × δ` table whose provable decoding
//! failure probability (the stopping-set union bound of the paper, Theorem
//! `thm:multisetencoding`) is at most the target.
//!
//! For capacity `ρ`, `θ` rows and `δ` cells per row, the bound is
//!
//! ```text
//!   Σ_{s=2}^{ρ} C(ρ, s) · φ_δ(s)^θ
//! ```
//!
//! where `φ_δ(s)` is the probability that `s` balls thrown uniformly into `δ`
//! bins leave no bin with exactly one ball. `φ` is evaluated in floating point
//! without cancellation: splitting the bins into two groups of `a` and `b`,
//! the balls in the first group are `Binomial(s, a/(a+b))` and both groups
//! must be singleton-free, so `φ_{a+b}(s) = Σ_k Bin(s, a/(a+b))(k) φ_a(k)
//! φ_b(s−k)`. `φ_δ` is assembled from `φ_1, φ_2, φ_4, …` along the binary
//! digits of `δ`. This mirrors `scripts/iblt_common.py` in the paper
//! repository and agrees with the exact integer bound to about `1e-12` in
//! `log2`.

/// Stopping-set budget: `2^-20`, less the `1/16` the paper reserves for key
/// collisions.
fn target_log2() -> f64 {
    -20.0 + (15.0f64 / 16.0).log2()
}

/// Largest row count tried; beyond this the tables only grow.
const MAX_THETA: usize = 15;
const MIN_THETA: usize = 3;

/// `(θ, δ)` minimising `θ·δ` for capacity `rho` under the target bound.
pub(crate) fn iblt_shape(rho: usize) -> (usize, usize) {
    let rho = rho.max(2);
    let target = target_log2();
    let mut table = Bound::new(rho);
    let pair_log2 = log2_binomial(&table.log_factorial, rho, 2);
    let mut best: Option<(usize, usize)> = None;
    for theta in (MIN_THETA..=MAX_THETA).rev() {
        let d_lo = ((pair_log2 - target) / theta as f64).exp2().floor().max(2.0) as usize;
        if best.is_some_and(|(bt, bd)| theta * d_lo >= bt * bd) {
            continue;
        }
        let mut d_hi = d_lo;
        while table.union_bound_log2(theta, d_hi) > target {
            d_hi *= 2;
            if best.is_some_and(|(bt, bd)| theta * d_hi >= 2 * bt * bd) || d_hi > 1 << 24 {
                break;
            }
        }
        if table.union_bound_log2(theta, d_hi) > target {
            continue;
        }
        let (mut lo, mut hi) = (d_lo, d_hi);
        while lo < hi {
            let mid = lo.midpoint(hi);
            if table.union_bound_log2(theta, mid) <= target {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        if best.is_none_or(|(bt, bd)| theta * lo < bt * bd) {
            best = Some((theta, lo));
        }
    }
    best.expect("some shape satisfies the bound: delta doubles until it does")
}

struct Bound {
    rho: usize,
    log_factorial: Vec<f64>,
    /// `φ_{2^k}(s)` for `s = 0..=ρ`, filled on demand.
    chain: Vec<Vec<f64>>,
}

impl Bound {
    fn new(rho: usize) -> Self {
        let mut log_factorial = Vec::with_capacity(rho + 1);
        log_factorial.push(0.0);
        for i in 1..=rho {
            log_factorial.push(log_factorial[i - 1] + (i as f64).ln());
        }
        let mut p1 = vec![1.0; rho + 1];
        p1[1] = 0.0;
        Self {
            rho,
            log_factorial,
            chain: vec![p1],
        }
    }

    /// `φ_{a+b}(s)` from `φ_a` and `φ_b`.
    fn combine(&self, pa: &[f64], a: usize, pb: &[f64], b: usize) -> Vec<f64> {
        let r = a as f64 / (a + b) as f64;
        let (ln_r, ln_not_r) = (r.ln(), (-r).ln_1p());
        (0..=self.rho)
            .map(|s| {
                (0..=s)
                    .map(|k| {
                        let log_c = self.log_factorial[s]
                            - self.log_factorial[k]
                            - self.log_factorial[s - k];
                        (log_c + k as f64 * ln_r + (s - k) as f64 * ln_not_r).exp() * pa[k] * pb[s - k]
                    })
                    .sum()
            })
            .collect()
    }

    fn probs(&mut self, delta: usize) -> Vec<f64> {
        let mut acc: Option<(Vec<f64>, usize)> = None;
        let (mut d, mut k) = (delta, 0);
        while d != 0 {
            while self.chain.len() <= k {
                let last = self.chain.len() - 1;
                let next = self.combine(&self.chain[last], 1 << last, &self.chain[last], 1 << last);
                self.chain.push(next);
            }
            if d & 1 == 1 {
                let pw_n = 1 << k;
                acc = Some(match acc {
                    None => (self.chain[k].clone(), pw_n),
                    Some((a, a_n)) => (self.combine(&a, a_n, &self.chain[k], pw_n), a_n + pw_n),
                });
            }
            d >>= 1;
            k += 1;
        }
        acc.expect("delta >= 1").0
    }

    /// `log2 Σ_{s=2}^{ρ} C(ρ,s) φ_δ(s)^θ`.
    fn union_bound_log2(&mut self, theta: usize, delta: usize) -> f64 {
        let p = self.probs(delta);
        let terms: Vec<f64> = (2..=self.rho)
            .filter(|&s| p[s] > 0.0)
            .map(|s| log2_binomial(&self.log_factorial, self.rho, s) + theta as f64 * p[s].log2())
            .collect();
        let Some(m) = terms.iter().copied().reduce(f64::max) else {
            return f64::NEG_INFINITY;
        };
        m + terms.iter().map(|t| (t - m).exp2()).sum::<f64>().log2()
    }
}

fn log2_binomial(log_factorial: &[f64], n: usize, k: usize) -> f64 {
    (log_factorial[n] - log_factorial[k] - log_factorial[n - k]) / std::f64::consts::LN_2
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shapes from the paper's exact-integer search
    /// (`generate_params.find_iblt_shape`) at the same target.
    #[test]
    fn matches_paper_shapes() {
        for (rho, theta, delta) in [
            (2, 13, 3),
            (5, 15, 3),
            (10, 15, 4),
            (20, 10, 8),
            (50, 8, 18),
            (100, 7, 34),
            (200, 6, 68),
            (300, 6, 101),
        ] {
            assert_eq!(iblt_shape(rho), (theta, delta), "rho = {rho}");
        }
    }

    #[test]
    fn bound_at_paper_shape() {
        let mut b = Bound::new(300);
        let v = b.union_bound_log2(6, 101);
        assert!((v - -23.92).abs() < 0.05, "{v}");
    }

    #[test]
    fn tiny_capacity_clamps_to_two() {
        assert_eq!(iblt_shape(0), iblt_shape(2));
        assert_eq!(iblt_shape(1), iblt_shape(2));
    }
}
