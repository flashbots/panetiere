//! Progressive IBLT-over-flashnet scaling bench. Each row is one
//! (n_servers, n_clients, message_slots) cell; cells run in increasing cost
//! order until a wall budget elapses.
//!
//! For each cell: every client builds an IBLT with one chunk, packs it into
//! HVCPolys, the protocol runs once per poly index, recovered polys unpack
//! into the union IBLT, peeling recovers each client's chunk. We time:
//!   pack_ms       — total per-cell pack time across all clients
//!   protocol_ms   — sum of run_full_cycle times across all poly indices
//!   unpack_ms     — unpack of recovered polys
//!   recover_ms    — IBLT peeling
//!   e2e_ms        — pack + protocol + unpack + recover
//!   MB/s          — (n_polys · 1024 bytes) / e2e_ms
//!
//! Run:
//!   RAYON_NUM_THREADS=8 cargo bench -j 8 --bench iblt_scaling
//! Override budget:
//!   BENCH_BUDGET_SECS=120 cargo bench -j 8 --bench iblt_scaling

use std::time::{Duration, Instant};

use chipmunk_code::{HVCPoly, HVC_MODULUS};
use flashnet::cs::{Cs, HidingMerkleCommitment};
use flashnet::iblt::{IbltParams, IbltVector, IBLT_CHUNK_BYTES};
use flashnet::protocol::client::run_client_round;
use flashnet::protocol::message::{ClientId, ServerId};
use flashnet::protocol::server::{run_server_round, ServerInbox};
use flashnet::protocol::verify::aggregate_and_decrypt;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// (n_servers, n_clients, message_slots), in increasing cost order.
const CELLS: &[(usize, usize, u32)] = &[
    (4, 4, 16),
    (8, 4, 16),
    (12, 4, 16),
    (4, 16, 32),
    (8, 16, 32),
    (12, 16, 32),
    (4, 64, 64),
    (8, 64, 64),
    (12, 64, 64),
    (4, 100, 100),
    (8, 100, 100),
    (12, 100, 100),
    (4, 256, 256),
    (8, 256, 256),
];

const BYTES_PER_POLY: f64 = 1024.0;

/// Largest `base_bits ∈ [4, 12]` for which `n_clients · (2^b − 1) < q`.
fn pick_base_bits(n_clients: usize) -> u32 {
    for b in (4..=12u32).rev() {
        let base = 1i64 << b;
        let max = (HVC_MODULUS as i64) / (base - 1);
        if max >= n_clients as i64 {
            return b;
        }
    }
    panic!("n_clients={} exceeds capacity at base_bits=4", n_clients);
}

struct Row {
    s: usize,
    n: usize,
    slots: u32,
    base_bits: u32,
    n_polys: usize,
    pack_ms: f64,
    protocol_ms: f64,
    unpack_ms: f64,
    recover_ms: f64,
    e2e_ms: f64,
    ok: bool,
}

fn rand_chunk<R: Rng>(rng: &mut R) -> [u8; IBLT_CHUNK_BYTES] {
    let mut c = [0u8; IBLT_CHUNK_BYTES];
    rng.fill(&mut c[..]);
    c
}

fn run_cell(s: usize, n: usize, slots: u32) -> Row {
    let base_bits = pick_base_bits(n);
    let params = IbltParams {
        message_slots: slots,
        base_bits,
    };
    let n_polys = IbltVector::n_polys(&params);

    let mut rng = ChaCha20Rng::from_seed([
        s as u8,
        (n & 0xff) as u8,
        ((n >> 8) & 0xff) as u8,
        slots as u8,
        (slots >> 8) as u8,
        (slots >> 16) as u8,
        base_bits as u8,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);

    let pp = HidingMerkleCommitment::setup(&mut rng, s);
    let server_ids: Vec<ServerId> = (0..s as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n as u32).map(ClientId).collect();
    let canonical = client_ids.clone();

    // Per-client chunk + IBLT.
    let chunks: Vec<[u8; IBLT_CHUNK_BYTES]> = (0..n).map(|_| rand_chunk(&mut rng)).collect();

    // ---- Pack ----
    let t = Instant::now();
    let client_polys: Vec<Vec<HVCPoly>> = chunks
        .iter()
        .map(|c| {
            let mut iblt = IbltVector::new(params.clone());
            iblt.insert_chunk(*c);
            iblt.pack()
        })
        .collect();
    let pack_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- Protocol: one full round per poly index ----
    let t = Instant::now();
    let mut recovered_polys: Vec<HVCPoly> = Vec::with_capacity(n_polys);
    for k in 0..n_polys {
        let mut publics = Vec::with_capacity(n);
        let mut inboxes: Vec<ServerInbox> = server_ids
            .iter()
            .map(|&sid| ServerInbox {
                server_id: sid,
                items: vec![],
            })
            .collect();
        for (i, &cid) in client_ids.iter().enumerate() {
            let m = client_polys[i][k];
            let round = run_client_round(&mut rng, &pp, cid, m, &server_ids);
            publics.push((round.client_id, round.public));
            for (idx, (sid, op, sh)) in round.private.into_iter().enumerate() {
                debug_assert_eq!(sid, server_ids[idx]);
                inboxes[idx].items.push((cid, op, sh));
            }
        }
        let outputs: Vec<_> = inboxes
            .iter()
            .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
            .collect();
        let recovered = aggregate_and_decrypt(&pp, &canonical, &publics, &outputs)
            .unwrap_or_else(|e| panic!("verify failed at poly {}: {:?}", k, e));
        recovered_polys.push(recovered);
    }
    let protocol_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- Unpack ----
    let t = Instant::now();
    let union = IbltVector::unpack(&params, &recovered_polys);
    let unpack_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- Recover (peeling) ----
    let t = Instant::now();
    let recovered_chunks = union.recover();
    let recover_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- Sanity ----
    let ok = match recovered_chunks {
        Ok(mut got) => {
            got.sort();
            let mut expected = chunks.clone();
            expected.sort();
            got == expected
        }
        Err(_) => false,
    };

    let e2e_ms = pack_ms + protocol_ms + unpack_ms + recover_ms;
    Row {
        s,
        n,
        slots,
        base_bits,
        n_polys,
        pack_ms,
        protocol_ms,
        unpack_ms,
        recover_ms,
        e2e_ms,
        ok,
    }
}

fn print_header() {
    println!(
        "{:>3} {:>5} {:>6} {:>3} {:>5}  {:>9}  {:>11}  {:>9}  {:>10}  {:>9}  {:>8}  {:>3}",
        "S",
        "N",
        "slots",
        "b",
        "polys",
        "pack_ms",
        "protocol_ms",
        "unpack_ms",
        "recover_ms",
        "e2e_ms",
        "MB/s",
        "ok"
    );
    println!("{}", "-".repeat(120));
}

fn print_row(r: &Row) {
    let mb_per_s =
        (r.n_polys as f64 * BYTES_PER_POLY) / 1_000_000.0 / (r.e2e_ms / 1e3);
    println!(
        "{:>3} {:>5} {:>6} {:>3} {:>5}  {:>9.2}  {:>11.1}  {:>9.2}  {:>10.2}  {:>9.1}  {:>8.4}  {:>3}",
        r.s,
        r.n,
        r.slots,
        r.base_bits,
        r.n_polys,
        r.pack_ms,
        r.protocol_ms,
        r.unpack_ms,
        r.recover_ms,
        r.e2e_ms,
        mb_per_s,
        if r.ok { "yes" } else { "NO" }
    );
}

fn main() {
    let _ = std::env::args();

    let budget_secs: u64 = std::env::var("BENCH_BUDGET_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let budget = Duration::from_secs(budget_secs);
    let start = Instant::now();

    println!("flashnet IBLT scaling bench  (budget: {}s)", budget_secs);
    println!("each row: per-client 1-chunk IBLT, packed, sent through flashnet, unpacked, peeled");
    println!();
    print_header();

    for &(s, n, slots) in CELLS {
        if start.elapsed() >= budget {
            println!(
                "\n(budget exhausted after {:.1}s; remaining cells skipped)",
                start.elapsed().as_secs_f64()
            );
            return;
        }
        let cell_start = Instant::now();
        let row = run_cell(s, n, slots);
        let cell_elapsed = cell_start.elapsed().as_secs_f64();
        print_row(&row);
        if start.elapsed() + Duration::from_secs_f64(cell_elapsed) > budget {
            println!(
                "\n(stopping early: cell took {:.1}s; elapsed {:.1}s)",
                cell_elapsed,
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
