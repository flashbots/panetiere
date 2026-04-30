//! Progressive scaling bench. Runs (|S|, N_clients) cells in increasing cost
//! order, printing a single row per cell, until a wall-clock budget elapses.
//!
//! Each row reports: per-stage timings (client_round, server_round, verify) and
//! the full end-to-end-one-poly cycle, plus the resulting MB/s for the
//! 1024-byte payload that one HVCPoly carries via `codec::encode_raw`.
//! Time-to-1MB at this cell = 1 (MB) / (reported MB/s).
//!
//! Run with:
//!   RAYON_NUM_THREADS=8 cargo bench -j 8 --bench scaling
//! Override budget (seconds):
//!   BENCH_BUDGET_SECS=120 cargo bench --bench scaling

use std::time::{Duration, Instant};

use chipmunk_code::{HVCPoly, Polynomial};
use flashnet::cs::{Cs, HidingMerkleCommitment};
use flashnet::protocol::client::run_client_round;
use flashnet::protocol::message::{ClientId, ServerId};
use flashnet::protocol::server::{run_server_round, ServerInbox};
use flashnet::protocol::verify::aggregate_and_decrypt;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// (|S|, N_clients) cells in increasing estimated cost. Stop early when the
/// wall budget is reached, so smaller cells always print regardless of grid.
const CELLS: &[(usize, usize)] = &[
    (4, 1),
    (8, 1),
    (12, 1),
    (4, 10),
    (8, 10),
    (12, 10),
    (4, 100),
    (8, 100),
    (12, 100),
    (4, 1000),
    (8, 1000),
    (12, 1000),
];

const BYTES_PER_POLY: f64 = 1024.0;

struct Row {
    s: usize,
    n: usize,
    client_ms: f64,
    server_ms: f64,
    verify_ms: f64,
    e2e_ms: f64,
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn time_one<F: FnMut() -> R, R>(mut f: F) -> f64 {
    let t = Instant::now();
    let _ = std::hint::black_box(f());
    t.elapsed().as_secs_f64() * 1e3
}

fn run_cell(s: usize, n: usize) -> Row {
    let mut rng = ChaCha20Rng::from_seed([(s as u8).wrapping_mul(13).wrapping_add(n as u8); 32]);
    let pp = HidingMerkleCommitment::setup(&mut rng, s);
    let server_ids: Vec<ServerId> = (0..s as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n as u32).map(ClientId).collect();
    let messages: Vec<HVCPoly> = (0..n).map(|_| HVCPoly::rand_poly(&mut rng)).collect();

    // Build a fixture for server_round + verify by running one full client pass.
    let mut publics = Vec::with_capacity(n);
    let mut inboxes: Vec<ServerInbox> = server_ids
        .iter()
        .map(|&sid| ServerInbox {
            server_id: sid,
            items: vec![],
        })
        .collect();
    for (i, &cid) in client_ids.iter().enumerate() {
        let round = run_client_round(&mut rng, &pp, cid, messages[i], &server_ids);
        publics.push((round.client_id, round.public));
        for (idx, (sid, op, sh)) in round.private.into_iter().enumerate() {
            inboxes[idx].items.push((cid, op, sh));
            let _ = sid;
        }
    }
    let canonical = client_ids.clone();
    let outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).unwrap())
        .collect();

    // Per-stage timings (median of 3).
    let mut t_client = [0.0; 3];
    for slot in &mut t_client {
        let m = HVCPoly::rand_poly(&mut rng);
        *slot = time_one(|| run_client_round(&mut rng, &pp, ClientId(0), m, &server_ids));
    }
    let mut t_server = [0.0; 3];
    for slot in &mut t_server {
        *slot = time_one(|| run_server_round(&inboxes[0], &canonical).unwrap());
    }
    let mut t_verify = [0.0; 3];
    for slot in &mut t_verify {
        *slot = time_one(|| aggregate_and_decrypt(&pp, &canonical, &publics, &outputs).unwrap());
    }

    // End-to-end one poly: full pipeline timed once (it dominates wall time).
    let t_e2e = time_one(|| run_full_cycle(&mut rng, &pp, &server_ids, &client_ids, &messages));

    Row {
        s,
        n,
        client_ms: median(&mut t_client),
        server_ms: median(&mut t_server),
        verify_ms: median(&mut t_verify),
        e2e_ms: t_e2e,
    }
}

fn run_full_cycle<R: rand::Rng>(
    rng: &mut R,
    pp: &<HidingMerkleCommitment as Cs>::Params,
    server_ids: &[ServerId],
    client_ids: &[ClientId],
    messages: &[HVCPoly],
) -> HVCPoly {
    let mut publics = Vec::with_capacity(client_ids.len());
    let mut inboxes: Vec<ServerInbox> = server_ids
        .iter()
        .map(|&sid| ServerInbox {
            server_id: sid,
            items: vec![],
        })
        .collect();
    for (i, &cid) in client_ids.iter().enumerate() {
        let round = run_client_round(rng, pp, cid, messages[i], server_ids);
        publics.push((round.client_id, round.public));
        for (idx, (sid, op, sh)) in round.private.into_iter().enumerate() {
            inboxes[idx].items.push((cid, op, sh));
            let _ = sid;
        }
    }
    let canonical = client_ids.to_vec();
    let outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).unwrap())
        .collect();
    aggregate_and_decrypt(pp, &canonical, &publics, &outputs).unwrap()
}

fn print_header() {
    println!(
        "{:>3} {:>5}  {:>10}  {:>10}  {:>10}  {:>12}  {:>10}  {:>10}",
        "S", "N", "client_ms", "server_ms", "verify_ms", "e2e_ms", "MB/s", "1MB_s"
    );
    println!("{}", "-".repeat(88));
}

fn print_row(r: &Row) {
    let mb_per_s = BYTES_PER_POLY / 1_000_000.0 / (r.e2e_ms / 1e3);
    let s_for_1mb = (1_000_000.0 / BYTES_PER_POLY) * (r.e2e_ms / 1e3);
    println!(
        "{:>3} {:>5}  {:>10.3}  {:>10.3}  {:>10.3}  {:>12.1}  {:>10.4}  {:>10.2}",
        r.s, r.n, r.client_ms, r.server_ms, r.verify_ms, r.e2e_ms, mb_per_s, s_for_1mb
    );
}

fn main() {
    // criterion harness compatibility: ignore --bench / --quick / etc. flags.
    let _ = std::env::args();

    let budget_secs: u64 = std::env::var("BENCH_BUDGET_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let budget = Duration::from_secs(budget_secs);
    let start = Instant::now();

    println!("flashnet scaling bench  (budget: {}s)", budget_secs);
    println!("each row times one HVCPoly's worth of broadcast (1024 bytes payload)");
    println!();
    print_header();

    for &(s, n) in CELLS {
        let elapsed = start.elapsed();
        if elapsed >= budget {
            println!(
                "\n(budget exhausted after {:.1}s; remaining cells skipped)",
                elapsed.as_secs_f64()
            );
            return;
        }

        // Forecast end-to-end for this cell from the previous cells' largest e2e
        // and skip if it would exceed the remaining budget plus a small slack.
        let cell_start = Instant::now();
        let r = run_cell(s, n);
        let cell_elapsed = cell_start.elapsed().as_secs_f64();
        print_row(&r);
        if start.elapsed() + Duration::from_secs_f64(cell_elapsed) > budget {
            // Likely the next cell is more expensive; print partial and stop.
            println!(
                "\n(stopping early: cell took {:.1}s; budget {} s; elapsed {:.1}s)",
                cell_elapsed,
                budget_secs,
                start.elapsed().as_secs_f64()
            );
            return;
        }
    }

    println!(
        "\nall cells done in {:.1}s",
        start.elapsed().as_secs_f64()
    );
}
