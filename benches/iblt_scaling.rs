//! Progressive IBLT-over-flashnet scaling bench with sub-stage breakdowns.
//!
//! Reported wall time **assumes parallel deployment**: clients run on their own
//! machines, servers on theirs, the verifier is one party. Per-poly wall time
//! is `client_total + server_total + verify_total`. Sub-stages within each
//! role are sequential on the same machine, so they sum within their role.
//! Per-IBLT-round wall = pack + n_polys · per_poly + unpack + recover.
//!
//! Run:
//!   RAYON_NUM_THREADS=8 cargo bench -j 8 --bench iblt_scaling
//! Override budget:
//!   BENCH_BUDGET_SECS=120 cargo bench -j 8 --bench iblt_scaling

use std::time::{Duration, Instant};

use chipmunk_code::{HVCPoly, Polynomial, HVC_MODULUS};
use flashnet::cs::{Cs, HidingMerkleCommitment};
use flashnet::iblt::{IbltParams, IbltVector, IBLT_CHUNK_BYTES};
use flashnet::kahe::{Kahe, RingOtp};
use flashnet::protocol::client::run_client_round;
use flashnet::protocol::message::{ClientId, ServerId};
use flashnet::protocol::server::{run_server_round, ServerInbox};
use flashnet::protocol::verify::aggregate_and_decrypt;
use flashnet::sss::{AdditiveSharing, Sss};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

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
    (12, 256, 256),
    (4, 1000, 256),
    (8, 1000, 256),
    (12, 1000, 256),
];

const BYTES_PER_POLY: f64 = 1024.0;

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
    slots: u32,
    base_bits: u32,
    n_polys: usize,
    pack_us: f64,
    unpack_us: f64,
    recover_us: f64,
    client: ClientTimes,
    server: ServerTimes,
    verify: VerifyTimes,
    ok: bool,
}

impl Row {
    fn per_poly_us(&self) -> f64 {
        self.client.total() + self.server.total() + self.verify.total()
    }
    fn total_us(&self) -> f64 {
        self.pack_us + self.n_polys as f64 * self.per_poly_us() + self.unpack_us + self.recover_us
    }
}

fn rand_chunk<R: Rng>(rng: &mut R) -> [u8; IBLT_CHUNK_BYTES] {
    let mut c = [0u8; IBLT_CHUNK_BYTES];
    rng.fill(&mut c[..]);
    c
}

fn time_client_stages<R: Rng>(
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
    let (cs_verify_us, _) = time_us(|| {
        for sp in outputs {
            assert!(HidingMerkleCommitment::verify(
                pp,
                &summed_comm,
                &sp.agg_open
            ));
        }
    });
    let (share_check_us, _) = time_us(|| {
        for sp in outputs {
            assert_eq!(sp.agg_share, *sp.agg_open.s());
        }
    });
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

fn run_cell(s: usize, n: usize, slots: u32) -> Row {
    let base_bits = pick_base_bits(n);
    let params = IbltParams::new(slots, base_bits);
    let n_polys = IbltVector::n_polys(&params);

    let mut seed = [0u8; 32];
    seed[0] = s as u8;
    seed[1] = n as u8;
    seed[2] = (n >> 8) as u8;
    seed[3] = slots as u8;
    seed[4] = (slots >> 8) as u8;
    seed[5] = base_bits as u8;
    let mut rng = ChaCha20Rng::from_seed(seed);

    let pp = HidingMerkleCommitment::setup(&mut rng, s);
    let server_ids: Vec<ServerId> = (0..s as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n as u32).map(ClientId).collect();
    let canonical = client_ids.clone();

    let chunks: Vec<[u8; IBLT_CHUNK_BYTES]> = (0..n).map(|_| rand_chunk(&mut rng)).collect();

    // Pack: time once (clients pack in parallel in deployment).
    let (pack_us, _) = time_us(|| {
        let mut iblt = IbltVector::new(params.clone());
        iblt.insert_chunk(chunks[0]);
        iblt.pack()
    });
    let client_polys: Vec<Vec<HVCPoly>> = chunks
        .iter()
        .map(|c| {
            let mut iblt = IbltVector::new(params.clone());
            iblt.insert_chunk(*c);
            iblt.pack()
        })
        .collect();

    // One-poly fixture for sub-stage timings.
    let mut publics_fix = Vec::with_capacity(n);
    let mut inboxes_fix: Vec<ServerInbox> = server_ids
        .iter()
        .map(|&sid| ServerInbox {
            server_id: sid,
            items: vec![],
        })
        .collect();
    for (i, &cid) in client_ids.iter().enumerate() {
        let m = client_polys[i][0];
        let round = run_client_round(&mut rng, &pp, cid, m, &server_ids);
        publics_fix.push((round.client_id, round.public));
        for (idx, (sid, op)) in round.private.into_iter().enumerate() {
            debug_assert_eq!(sid, server_ids[idx]);
            inboxes_fix[idx].items.push((cid, op));
        }
    }
    let outputs_fix: Vec<_> = inboxes_fix
        .iter()
        .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
        .collect();

    let client = time_client_stages(&mut rng, &pp, &server_ids);
    let server = time_server_stages(&inboxes_fix[0], &canonical);
    let verify = time_verify_stages(&pp, &canonical, &publics_fix, &outputs_fix);

    // Full pipeline once for correctness.
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
            for (idx, (sid, op)) in round.private.into_iter().enumerate() {
                debug_assert_eq!(sid, server_ids[idx]);
                inboxes[idx].items.push((cid, op));
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

    let (unpack_us, union) = time_us(|| IbltVector::unpack(&params, &recovered_polys));
    let (recover_us, recovered_chunks) = time_us(|| union.recover());
    let ok = match recovered_chunks {
        Ok(mut got) => {
            got.sort();
            let mut expected = chunks.clone();
            expected.sort();
            got == expected
        }
        Err(_) => false,
    };

    Row {
        s,
        n,
        slots,
        base_bits,
        n_polys,
        pack_us,
        unpack_us,
        recover_us,
        client,
        server,
        verify,
        ok,
    }
}

fn print_row(r: &Row) {
    let bytes = r.n_polys as f64 * BYTES_PER_POLY;
    let mb_per_s = bytes / 1_000_000.0 / (r.total_us() / 1e6);
    let s_for_1mb = (1_000_000.0 / bytes) * (r.total_us() / 1e6);
    println!(
        "S={:>2} N={:>4} slots={:>4} b={:>2} polys={:>4}  | client {:>9} = gen {:>8} + enc {:>8} + share {:>8} + commit {:>9}",
        r.s,
        r.n,
        r.slots,
        r.base_bits,
        r.n_polys,
        fmt_us(r.client.total()),
        fmt_us(r.client.gen_us),
        fmt_us(r.client.enc_us),
        fmt_us(r.client.share_us),
        fmt_us(r.client.commit_us),
    );
    println!(
        "                                       | server {:>9} = pick {:>8} + sumO  {:>8} + sumS   {:>8}",
        fmt_us(r.server.total()),
        fmt_us(r.server.pick_us),
        fmt_us(r.server.sum_open_us),
        fmt_us(r.server.sum_share_us),
    );
    println!(
        "                                       | verify {:>9} = sumC {:>8} + sumK  {:>8} + cs_v   {:>8} + chk {:>8} + dec {:>8}",
        fmt_us(r.verify.total()),
        fmt_us(r.verify.sum_ctxt_us),
        fmt_us(r.verify.sum_comm_us),
        fmt_us(r.verify.cs_verify_us),
        fmt_us(r.verify.share_check_us),
        fmt_us(r.verify.decrypt_us),
    );
    println!(
        "                                       | per-poly {:>7}, IBLT-round {:>9} = pack {:>8} + {} polys + unpack {:>8} + recover {:>8}   ({:.4} MB/s, {:.2} s/MB, ok={})",
        fmt_us(r.per_poly_us()),
        fmt_us(r.total_us()),
        fmt_us(r.pack_us),
        r.n_polys,
        fmt_us(r.unpack_us),
        fmt_us(r.recover_us),
        mb_per_s,
        s_for_1mb,
        if r.ok { "yes" } else { "NO" },
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

    println!(
        "flashnet IBLT scaling bench  (budget: {}s)",
        budget_secs
    );
    println!("wall times assume parallel deployment: clients on N machines, servers on S machines.");
    println!("per-poly = client_total + server_total + verify_total.");
    println!("IBLT-round = pack + n_polys · per-poly + unpack + recover.");
    println!();

    for &(s, n, slots) in CELLS {
        if start.elapsed() >= budget {
            println!(
                "(budget exhausted after {:.1}s; remaining cells skipped)",
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
                "(stopping early: cell took {:.1}s; elapsed {:.1}s)",
                cell_elapsed,
                start.elapsed().as_secs_f64()
            );
            return;
        }
    }

    println!(
        "all cells done in {:.1}s",
        start.elapsed().as_secs_f64()
    );
}
