//! Progressive scaling bench. Runs (|S|, N_clients) cells in increasing cost
//! order with sub-stage breakdowns inside client/server/verify.
//!
//! Reported wall time **assumes parallel deployment**: clients run on their
//! own machines, servers on theirs, the verifier is one party. Per-poly wall
//! time = `client_total + server_total + verify_total`. Sub-stages within each
//! role are sequential on the same machine, so they sum within their role.
//!
//! Run with:
//!   RAYON_NUM_THREADS=8 cargo bench -j 8 --bench scaling
//! Override budget (seconds):
//!   BENCH_BUDGET_SECS=120 cargo bench --bench scaling

use std::time::{Duration, Instant};

use chipmunk_code::{HVCPoly, Polynomial};
use flashnet::cs::{Cs, HidingMerkleCommitment};
use flashnet::kahe::{Kahe, RingOtp};
use flashnet::protocol::client::run_client_round;
use flashnet::protocol::message::{ClientId, ServerId};
use flashnet::protocol::server::{run_server_round, ServerInbox};
use flashnet::protocol::verify::aggregate_and_decrypt;
use flashnet::sss::{AdditiveSharing, Sss};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

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

#[derive(Default)]
struct ClientTimes {
    gen_us: f64,
    enc_us: f64,
    share_us: f64,
    commit_us: f64,
}
impl ClientTimes {
    fn total(&self) -> f64 {
        self.gen_us + self.enc_us + self.share_us + self.commit_us
    }
}

#[derive(Default)]
struct ServerTimes {
    pick_us: f64,
    sum_open_us: f64,
    sum_share_us: f64,
}
impl ServerTimes {
    fn total(&self) -> f64 {
        self.pick_us + self.sum_open_us + self.sum_share_us
    }
}

#[derive(Default)]
struct VerifyTimes {
    sum_ctxt_us: f64,
    sum_comm_us: f64,
    cs_verify_us: f64,
    share_check_us: f64,
    decrypt_us: f64,
}
impl VerifyTimes {
    fn total(&self) -> f64 {
        self.sum_ctxt_us
            + self.sum_comm_us
            + self.cs_verify_us
            + self.share_check_us
            + self.decrypt_us
    }
}

struct Row {
    s: usize,
    n: usize,
    client: ClientTimes,
    server: ServerTimes,
    verify: VerifyTimes,
}

impl Row {
    fn e2e_us(&self) -> f64 {
        self.client.total() + self.server.total() + self.verify.total()
    }
}

fn time_client_stages<R: rand::Rng>(
    rng: &mut R,
    pp: &<HidingMerkleCommitment as Cs>::Params,
    server_ids: &[ServerId],
) -> ClientTimes {
    let msg = HVCPoly::rand_poly(rng);
    let (gen_us, key) = time_us(|| RingOtp::gen(rng));
    let (enc_us, _) = time_us(|| RingOtp::enc(&key, &msg));
    let (share_us, shares) = time_us(|| AdditiveSharing::share(rng, &key, server_ids.len()));
    let (commit_us, _) = time_us(|| HidingMerkleCommitment::commit(rng, pp, &shares));
    ClientTimes {
        gen_us,
        enc_us,
        share_us,
        commit_us,
    }
}

fn time_server_stages(
    inbox: &ServerInbox,
    canonical: &[ClientId],
) -> ServerTimes {
    // Mirrors run_server_round: index inbox by ClientId once, then O(1) lookups.
    use std::collections::HashMap;
    let (pick_us, (openings, shares)) = time_us(|| {
        let index: HashMap<ClientId, usize> = inbox
            .items
            .iter()
            .enumerate()
            .map(|(i, (cid, _))| (*cid, i))
            .collect();
        let mut openings: Vec<&flashnet::cs::Opening> = Vec::with_capacity(canonical.len());
        let mut shares = Vec::with_capacity(canonical.len());
        for cid in canonical {
            let i = *index.get(cid).unwrap();
            let (_, op) = &inbox.items[i];
            openings.push(op);
            shares.push(*op.s());
        }
        (openings, shares)
    });
    let (sum_open_us, _) = time_us(|| HidingMerkleCommitment::sum_openings(&openings));
    let (sum_share_us, _) = time_us(|| AdditiveSharing::recover(&shares));
    ServerTimes {
        pick_us,
        sum_open_us,
        sum_share_us,
    }
}

fn time_verify_stages(
    pp: &<HidingMerkleCommitment as Cs>::Params,
    canonical: &[ClientId],
    publics: &[(ClientId, flashnet::bulletin::ClientPublic)],
    outputs: &[flashnet::bulletin::ServerPublic],
) -> VerifyTimes {
    use std::collections::HashMap;
    // Mirrors aggregate_and_decrypt: O(1) index by ClientId.
    let pub_index: HashMap<ClientId, usize> = publics
        .iter()
        .enumerate()
        .map(|(i, (cid, _))| (*cid, i))
        .collect();
    let (sum_ctxt_us, summed_ctxt) = time_us(|| {
        let ctxts: Vec<_> = canonical
            .iter()
            .map(|cid| publics[pub_index[cid]].1.ctxt)
            .collect();
        RingOtp::agg_ctxt(&ctxts)
    });
    let (sum_comm_us, summed_comm) = time_us(|| {
        let comms: Vec<_> = canonical
            .iter()
            .map(|cid| publics[pub_index[cid]].1.comm.clone())
            .collect();
        HidingMerkleCommitment::sum_commitments(&comms)
    });
    // Step 2: per-server CS::verify across all S servers.
    let (cs_verify_us, _) = time_us(|| {
        for sp in outputs {
            assert!(HidingMerkleCommitment::verify(
                pp,
                &summed_comm,
                &sp.agg_open
            ));
        }
    });
    // Step 3: agg_share == agg_open.s checks across S servers.
    let (share_check_us, _) = time_us(|| {
        for sp in outputs {
            assert_eq!(sp.agg_share, *sp.agg_open.s());
        }
    });
    // Step 4: aggregate keys, decrypt.
    let (decrypt_us, _) = time_us(|| {
        let agg_key =
            RingOtp::agg_key(&outputs.iter().map(|s| s.agg_share).collect::<Vec<_>>());
        RingOtp::dec(&summed_ctxt, &agg_key)
    });
    VerifyTimes {
        sum_ctxt_us,
        sum_comm_us,
        cs_verify_us,
        share_check_us,
        decrypt_us,
    }
}

fn run_cell(s: usize, n: usize) -> Row {
    let mut rng = ChaCha20Rng::from_seed([(s as u8).wrapping_mul(13).wrapping_add(n as u8); 32]);
    let pp = HidingMerkleCommitment::setup(&mut rng, s);
    let server_ids: Vec<ServerId> = (0..s as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n as u32).map(ClientId).collect();
    let messages: Vec<HVCPoly> = (0..n).map(|_| HVCPoly::rand_poly(&mut rng)).collect();

    // Build a fixture: one full sequential pass to populate inboxes/outputs.
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
        for (idx, (sid, op)) in round.private.into_iter().enumerate() {
            inboxes[idx].items.push((cid, op));
            let _ = sid;
        }
    }
    let canonical = client_ids.clone();
    let outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).unwrap())
        .collect();

    let client = time_client_stages(&mut rng, &pp, &server_ids);
    let server = time_server_stages(&inboxes[0], &canonical);
    let verify = time_verify_stages(&pp, &canonical, &publics, &outputs);

    Row {
        s,
        n,
        client,
        server,
        verify,
    }
}

fn print_row(r: &Row) {
    let total_us = r.e2e_us();
    let mb_per_s = BYTES_PER_POLY / 1_000_000.0 / (total_us / 1e6);
    println!(
        "S={:>2} N={:>4} | client {:>9} = gen {:>8} + enc {:>8} + share {:>8} + commit {:>9}",
        r.s,
        r.n,
        fmt_us(r.client.total()),
        fmt_us(r.client.gen_us),
        fmt_us(r.client.enc_us),
        fmt_us(r.client.share_us),
        fmt_us(r.client.commit_us),
    );
    println!(
        "                 | server {:>9} = pick {:>8} + sumO  {:>8} + sumS   {:>8}",
        fmt_us(r.server.total()),
        fmt_us(r.server.pick_us),
        fmt_us(r.server.sum_open_us),
        fmt_us(r.server.sum_share_us),
    );
    println!(
        "                 | verify {:>9} = sumC {:>8} + sumK  {:>8} + cs_v   {:>8} + chk {:>8} + dec {:>8}",
        fmt_us(r.verify.total()),
        fmt_us(r.verify.sum_ctxt_us),
        fmt_us(r.verify.sum_comm_us),
        fmt_us(r.verify.cs_verify_us),
        fmt_us(r.verify.share_check_us),
        fmt_us(r.verify.decrypt_us),
    );
    println!(
        "                 | TOTAL  {:>9}   ({:.4} MB/s, {:.2} s/MB)",
        fmt_us(total_us),
        mb_per_s,
        1.0 / mb_per_s.max(f64::MIN_POSITIVE),
    );
    println!();
}

fn main() {
    let _ = std::env::args();
    let budget_secs: u64 = std::env::var("BENCH_BUDGET_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let budget = Duration::from_secs(budget_secs);
    let start = Instant::now();

    println!("flashnet scaling bench  (budget: {}s)", budget_secs);
    println!("wall times assume parallel deployment: clients run in parallel on N machines,");
    println!("servers in parallel on S machines, the verifier is one party.");
    println!("per-poly wall = client_total + server_total + verify_total. payload = 1024 bytes.");
    println!();

    for &(s, n) in CELLS {
        if start.elapsed() >= budget {
            println!(
                "(budget exhausted after {:.1}s; remaining cells skipped)",
                start.elapsed().as_secs_f64()
            );
            return;
        }
        let cell_start = Instant::now();
        let r = run_cell(s, n);
        print_row(&r);
        if start.elapsed() + cell_start.elapsed() > budget {
            println!(
                "(stopping early: cell took {:.1}s; elapsed {:.1}s)",
                cell_start.elapsed().as_secs_f64(),
                start.elapsed().as_secs_f64()
            );
            return;
        }
    }
    println!("all cells done in {:.1}s", start.elapsed().as_secs_f64());
}
