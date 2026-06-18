//! MSE parameter sweep. Estimates `P[decode fails]` over a grid of
//! (γ, δ, n) and writes a CSV for plotting / parameter selection.
//!
//! δ is parameterised by load factor `α = n / δ` so the same α sweeps
//! across `n`. Trials are Monte-Carlo with a fresh PRF key and ChaCha
//! rng per trial; failures are Wilson-95 % bracketed.
//!
//! Env knobs (all comma-separated lists of usize/float):
//!   SWEEP_GAMMAS   default "3,4,5,6,8"
//!   SWEEP_ALPHAS   default "0.5,0.75,1.0,1.25,1.5,2.0"
//!   SWEEP_DELTAS   if set, overrides SWEEP_ALPHAS and sweeps δ directly
//!                  (same δ list applied at every n; α derived as n/δ).
//!   SWEEP_NS       default "32,64,128,256,512,1024"
//!   SWEEP_XI       default "1" — payload symbols per element (ξ).
//!                  Each symbol carries ~18 bits, so ξ=29 ≈ 512-bit message.
//!   SWEEP_SHRINKS  default "1.0" — per-row shrink factor. 1.0 = uniform
//!                  layout; 0.5 = halving rows (row `i` has δ·0.5^i buckets).
//!   SWEEP_FAIL_MAX default 2 — a cross-section's "success" threshold is
//!                  `n_fail < SWEEP_FAIL_MAX`. Reported as the
//!                  `min_cells_under_threshold` column.
//!   SWEEP_TRIALS   default 10000
//!   SWEEP_OUT      default "mse_sweep.csv"
//!   SWEEP_THREADS  default 8 (caps rayon)
//!
//! Run:
//!   RAYON_NUM_THREADS=8 cargo run --release --example mse_sweep
//!
//! The CSV columns are documented at the top of the output.

use panetiere::mse::{MseEncoding, MseParams, RowLayout};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

fn parse_list<T: std::str::FromStr>(var: &str, default: &str) -> Vec<T> {
    env::var(var)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|s| s.trim().parse().ok().expect("parse list element"))
        .collect()
}

/// Wilson 95 % CI for a binomial proportion.
fn wilson(successes: u64, trials: u64) -> (f64, f64, f64) {
    let n = trials as f64;
    let p = successes as f64 / n;
    let z = 1.96f64;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / denom;
    let half = z * ((p * (1.0 - p) / n) + z2 / (4.0 * n * n)).sqrt() / denom;
    (p, (centre - half).max(0.0), (centre + half).min(1.0))
}

fn one_trial(
    gamma: usize,
    delta: usize,
    xi: usize,
    shrink: f64,
    n: usize,
    seed: u64,
) -> bool {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut prf_key = [0u8; 32];
    rng.fill(&mut prf_key);
    let layout = if shrink >= 1.0 {
        RowLayout::Uniform
    } else {
        RowLayout::Geometric { shrink }
    };
    let pp = MseParams::with_layout(gamma, delta, xi, layout, prf_key);
    let mut enc = MseEncoding::new(pp);
    let mut payload = vec![0i32; xi];
    for _ in 0..n {
        for s in payload.iter_mut() {
            *s = rng.gen_range(0..1_000_000);
        }
        enc.insert(&mut rng, &payload);
    }
    enc.decode().is_ok()
}

fn main() {
    let gammas: Vec<usize> = parse_list("SWEEP_GAMMAS", "3,4,5,6,8");
    let ns: Vec<usize> = parse_list("SWEEP_NS", "32,64,128,256,512,1024");
    let xis: Vec<usize> = parse_list("SWEEP_XI", "1");
    let shrinks: Vec<f64> = parse_list("SWEEP_SHRINKS", "1.0");
    let fail_max: u64 = env::var("SWEEP_FAIL_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let deltas_override: Option<Vec<usize>> =
        env::var("SWEEP_DELTAS").ok().map(|s| {
            s.split(',')
                .map(|x| x.trim().parse().expect("parse SWEEP_DELTAS"))
                .collect()
        });
    let alphas: Vec<f64> = if deltas_override.is_some() {
        Vec::new()
    } else {
        parse_list("SWEEP_ALPHAS", "0.5,0.75,1.0,1.25,1.5,2.0")
    };
    let trials: u64 = env::var("SWEEP_TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let out_path = env::var("SWEEP_OUT").unwrap_or_else(|_| "mse_sweep.csv".into());
    let threads: usize = env::var("SWEEP_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .ok();

    let inner_axis_len = deltas_override
        .as_ref()
        .map(|d| d.len())
        .unwrap_or(alphas.len());
    let n_cells = gammas.len() * inner_axis_len * ns.len() * xis.len() * shrinks.len();
    eprintln!(
        "sweep: γ={:?} {} n={:?} ξ={:?} shrinks={:?} trials={} cells={}",
        gammas,
        match &deltas_override {
            Some(d) => format!("δ={:?}", d),
            None => format!("α={:?}", alphas),
        },
        ns,
        xis,
        shrinks,
        trials,
        n_cells,
    );
    eprintln!("threads={} writing aggregate to {}", threads, out_path);

    let file = File::create(&out_path).expect("open out");
    let mut w = BufWriter::new(file);
    // One row per (γ, L, ξ, shrink, n) cross-section. Columns:
    //   min_cells_under_threshold — smallest total-cells (over swept δ)
    //                               with n_fail < SWEEP_FAIL_MAX. Empty
    //                               if no swept δ met the criterion.
    //   min_cells_under_1e_minus_3 — same with p_fail ≤ 1e-3.
    //   best_p_fail / best_p_fail_hi — minimum observed p_fail and its
    //                                  Wilson upper across all δ.
    //   best_total_cells — total cells of the cell achieving best_p_fail.
    writeln!(
        w,
        "gamma,xi,shrink,n,trials,fail_max,n_deltas,\
         min_cells_under_threshold,min_cells_under_1e_minus_3,\
         best_p_fail,best_p_fail_hi,best_total_cells"
    )
    .unwrap();

    let mut cell_idx = 0usize;
    let total_start = Instant::now();

    // Loop order: (ξ, shrink, γ, n) outer → δ innermost so each
    // (γ, ξ, shrink, n) cross-section is a contiguous run we can
    // aggregate without storing all measurements.
    for &xi in &xis {
        for &shrink in &shrinks {
            for &gamma in &gammas {
                for &n in &ns {
                        let inner: Vec<(usize, f64)> = match &deltas_override {
                            Some(deltas) => deltas
                                .iter()
                                .map(|&d| (d.max(1), n as f64 / d as f64))
                                .collect(),
                            None => alphas
                                .iter()
                                .map(|&a| (((n as f64 / a).ceil() as usize).max(1), a))
                                .collect(),
                        };

                        let mut min_cells_thresh: Option<usize> = None;
                        let mut min_cells_1e3: Option<usize> = None;
                        let mut best_p: f64 = f64::INFINITY;
                        let mut best_p_hi: f64 = f64::INFINITY;
                        let mut best_cells: usize = 0;

                        for &(delta, alpha) in &inner {
                            cell_idx += 1;
                            let t0 = Instant::now();
                            let fails: u64 = (0..trials)
                                .into_par_iter()
                                .map(|t| {
                                    let seed = ((cell_idx as u64) << 32) ^ t;
                                    if one_trial(gamma, delta, xi, shrink, n, seed) {
                                        0
                                    } else {
                                        1
                                    }
                                })
                                .sum();
                            let elapsed = t0.elapsed();
                            let (p, lo, hi) = wilson(fails, trials);

                            // Compute actual total_cells from layout.
                            let total_cells_here: usize = if shrink >= 1.0 {
                                gamma * delta
                            } else {
                                (0..gamma)
                                    .map(|i| {
                                        ((delta as f64) * shrink.powi(i as i32))
                                            .round()
                                            .max(1.0) as usize
                                    })
                                    .sum()
                            };

                            if fails < fail_max
                                && min_cells_thresh.map_or(true, |c| total_cells_here < c)
                            {
                                min_cells_thresh = Some(total_cells_here);
                            }
                            if p <= 1e-3 && min_cells_1e3.map_or(true, |c| total_cells_here < c) {
                                min_cells_1e3 = Some(total_cells_here);
                            }
                            if p < best_p {
                                best_p = p;
                                best_p_hi = hi;
                                best_cells = total_cells_here;
                            }

                            eprintln!(
                                "[{:>4}/{:>4}] γ={} δ={:>5} ξ={:>2} s={:.2} n={:>5} α={:>4.2}  cells={:>6} p={:.4e} ({:.2e}..{:.2e})  {:>5} ms",
                                cell_idx,
                                n_cells,
                                gamma,
                                delta,
                                xi,
                                shrink,
                                n,
                                alpha,
                                total_cells_here,
                                p,
                                lo,
                                hi,
                                elapsed.as_millis(),
                            );
                        }

                        writeln!(
                            w,
                            "{},{},{:.4},{},{},{},{},{},{},{:.6e},{:.6e},{}",
                            gamma,
                            xi,
                            shrink,
                            n,
                            trials,
                            fail_max,
                            inner.len(),
                            min_cells_thresh.map_or(String::new(), |c| c.to_string()),
                            min_cells_1e3.map_or(String::new(), |c| c.to_string()),
                            best_p,
                            best_p_hi,
                            best_cells,
                        )
                        .unwrap();
                        w.flush().ok();
                }
            }
        }
    }

    eprintln!("done in {:.1?}", total_start.elapsed());
}
