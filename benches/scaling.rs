//! Scaling bench: sweeps `(S, N)` × `(μ_kahe, κ_kahe)`.
//! Times the public protocol API (client / server / verify) only.
//!
//! Per cell runs ONE protocol round at a payload of exactly `μ` polys (so
//! `n_rounds = 1`). Reported wall = `client_round + server_round +
//! verify_round` for that round, and an extrapolated 1 MB total =
//! `⌈1024 / μ⌉ · per-round`.
//!
//! Outer loop is `(S, N)` so each cell prints all (μ, κ) variants together —
//! direct comparison without waiting through the full sweep.
//!
//! `κ_kahe` chosen so Lemma 8 holds at λ=128, ρ ≤ 2²⁰, β=64 with the chipmunk
//! HVC modulus q=202753, n=512.
//!
//! Run with:
//!   RAYON_NUM_THREADS=8 cargo bench -j 8 --bench scaling
//! Override budget (seconds):
//!   BENCH_BUDGET_SECS=600 cargo bench --bench scaling

use std::time::{Duration, Instant};

use chipmunk_code::{HVCPoly, Polynomial};
use flashnet::protocol::client::run_client_round;
use flashnet::protocol::message::{ClientId, ServerId};
use flashnet::protocol::server::{run_server_round, ServerInbox};
use flashnet::protocol::verify::aggregate_and_decrypt;
use flashnet::protocol::ProtocolParams;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const CELLS: &[(usize, usize)] = &[(8, 300)];

/// (μ_kahe, κ_kahe, β). Each row satisfies Lemma 8 at λ=128, ρ ≤ 2²⁰.
/// Two operating points retained from a wider sweep:
///   - (μ=16, κ=31, β=1024): best 1 MB throughput (block_size=32, 64 rounds)
///   - (μ=8,  κ=15, β=2048): best per-round latency (block_size=16 cliff, 128 rounds)
const KAHE_DIMS: &[(usize, usize, u32)] = &[
    (16, 31, 1024),
    ( 8, 15, 2048),
];

const TARGET_PAYLOAD_KB: usize = 1024;

fn time_us<F: FnOnce() -> R, R>(f: F) -> (f64, R) {
    let t = Instant::now();
    let r = f();
    (t.elapsed().as_secs_f64() * 1e6, r)
}

fn fmt_us(us: f64) -> String {
    if us < 1000.0 {
        format!("{:.2}us", us)
    } else if us < 1_000_000.0 {
        format!("{:.3}ms", us / 1e3)
    } else {
        format!("{:.2}s", us / 1e6)
    }
}

struct Row {
    s: usize,
    n: usize,
    mu: usize,
    kappa: usize,
    beta: u32,
    client_us: f64,
    server_us: f64,
    verify_us: f64,
}

impl Row {
    fn per_round_us(&self) -> f64 {
        self.client_us + self.server_us + self.verify_us
    }
    /// Extrapolated wall time to broadcast a 1 MB payload (≈ 1024 polys),
    /// `n_rounds = ⌈1024 / μ⌉`.
    fn extrap_1mb_us(&self) -> f64 {
        let n_rounds = TARGET_PAYLOAD_KB.div_ceil(self.mu) as f64;
        n_rounds * self.per_round_us()
    }
}

fn run_cell(s: usize, n: usize, mu: usize, kappa: usize, beta: u32) -> Row {
    let mut seed = [0u8; 32];
    seed[0] = s as u8;
    seed[1] = n as u8;
    seed[2] = (n >> 8) as u8;
    seed[3] = mu as u8;
    seed[4] = kappa as u8;
    seed[5] = (kappa >> 8) as u8;
    seed[6] = beta as u8;
    seed[7] = (beta >> 8) as u8;
    let mut rng = ChaCha20Rng::from_seed(seed);

    let pp = ProtocolParams::setup_with_kahe_dims_beta(&mut rng, s, mu, kappa, beta);
    let server_ids: Vec<ServerId> = (0..s as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n as u32).map(ClientId).collect();
    let messages: Vec<Vec<HVCPoly>> = (0..n)
        .map(|_| (0..mu).map(|_| HVCPoly::rand_poly(&mut rng)).collect())
        .collect();

    let mut publics = Vec::with_capacity(n);
    let mut inboxes: Vec<ServerInbox> = server_ids
        .iter()
        .map(|&sid| ServerInbox {
            server_id: sid,
            items: vec![],
        })
        .collect();
    for (i, &cid) in client_ids.iter().enumerate() {
        let round = run_client_round(&mut rng, &pp, cid, messages[i].clone(), &server_ids);
        publics.push((round.client_id, round.public));
        for (idx, (sid, ops)) in round.private.into_iter().enumerate() {
            let _ = sid;
            inboxes[idx].items.push((cid, ops));
        }
    }
    let canonical = client_ids.clone();
    let outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).unwrap())
        .collect();

    let (client_us, _) = time_us(|| {
        run_client_round(&mut rng, &pp, ClientId(0), messages[0].clone(), &server_ids)
    });
    let (server_us, _) = time_us(|| run_server_round(&inboxes[0], &canonical).unwrap());
    let (verify_us, _) =
        time_us(|| aggregate_and_decrypt(&pp, &canonical, &publics, &outputs).unwrap());

    Row {
        s,
        n,
        mu,
        kappa,
        beta,
        client_us,
        server_us,
        verify_us,
    }
}

fn print_row(r: &Row) {
    let n_rounds = TARGET_PAYLOAD_KB.div_ceil(r.mu);
    let bytes = TARGET_PAYLOAD_KB as f64 * 1024.0;
    let mb_per_s = bytes / 1_000_000.0 / (r.extrap_1mb_us() / 1e6);
    println!(
        "S={:>2} N={:>4} (μ={:>3}, κ={:>3}, β={:>5}) | client {:>9} server {:>9} verify {:>9}  per-round {:>9}  ⇒ 1MB ({} rounds) {:>9}  ({:.4} MB/s)",
        r.s,
        r.n,
        r.mu,
        r.kappa,
        r.beta,
        fmt_us(r.client_us),
        fmt_us(r.server_us),
        fmt_us(r.verify_us),
        fmt_us(r.per_round_us()),
        n_rounds,
        fmt_us(r.extrap_1mb_us()),
        mb_per_s,
    );
}

fn main() {
    let _ = std::env::args();
    let budget_secs: u64 = std::env::var("BENCH_BUDGET_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    let budget = Duration::from_secs(budget_secs);
    let start = Instant::now();

    println!("flashnet scaling bench  (budget: {}s)", budget_secs);
    println!("wall times assume parallel deployment.");
    println!("per-round wall = client_round + server_round + verify_round.");
    println!(
        "1MB extrapolation = ⌈{}/μ⌉ · per-round (one round timed per cell).",
        TARGET_PAYLOAD_KB
    );
    println!();

    'outer: for &(s, n) in CELLS {
        for &(mu, kappa, beta) in KAHE_DIMS {
            if start.elapsed() >= budget {
                println!(
                    "(budget exhausted after {:.1}s; remaining cells skipped)",
                    start.elapsed().as_secs_f64()
                );
                break 'outer;
            }
            let cell_start = Instant::now();
            let r = run_cell(s, n, mu, kappa, beta);
            print_row(&r);
            if start.elapsed() + cell_start.elapsed() > budget {
                println!(
                    "(stopping early: cell took {:.1}s; elapsed {:.1}s)",
                    cell_start.elapsed().as_secs_f64(),
                    start.elapsed().as_secs_f64()
                );
                break 'outer;
            }
        }
        println!();
    }
    println!("done in {:.1}s", start.elapsed().as_secs_f64());
}
