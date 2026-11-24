//! Benchmark of the protocol as the paper describes it.
//!
//! Direct mode only: every client posts its KAHE ciphertext and key-share
//! commitment to the bulletin and sends one sealed opening to each server; each
//! server posts one aggregated opening; the recipient downloads everything,
//! sums, verifies, decrypts and decodes.
//!
//! Everything reported is either measured on this machine (time per role and
//! phase) or the exact length of what goes on the wire. Nothing is modelled:
//! there is no cost model and no network simulation.
//!
//! Per cell, all ρ clients' posts and sealed openings are generated first,
//! untimed, and kept only as wire bytes; then every server runs its round
//! once, untimed, and its entry is kept as wire bytes too. Then
//! - the client is timed on one client's whole round (every client does the
//!   same work), from its message to its serialized post;
//! - the server is timed on server 0's round over all ρ clients, from its
//!   sealed openings and the parsed commitments to its serialized entry. A
//!   server only needs the commitment part of each post, but the public API
//!   parses commitments only as part of a whole post, so they are parsed once,
//!   untimed, and the server's download is counted as the commitment part of
//!   each post;
//! - the recipient is timed from the bytes of all ρ posts and the γ server
//!   entries to the decoded messages. It runs the library's recipient,
//!   `recover_unverified_direct_from_bytes`, which parses and sums the posts,
//!   applies the client-set rule, verifies the servers' openings, and
//!   decrypts; the benchmark only parses the server entries before it and
//!   decodes the messages after it.
//!
//! Every phase calls the library's public API; the benchmark only generates
//! the inputs and moves bytes between the parties.
//!
//! Environment (all optional):
//!   PAPER_ENCODINGS  "ng,iblt"                  encodings to run
//!   PAPER_CLIENTS    "100x100,300x300,300x100"  ρ x C pairs (clients x capacity);
//!                                               C clients send, ρ − C send cover
//!   PAPER_SERVERS    "8,16,32"                  server counts γ
//!   PAPER_MSG_BYTES  "4096,16384"               message sizes
//!   PAPER_REPS       5                          timed repetitions per role
//!   PAPER_CSV        paper_bench.csv            output file
//!   RAYON_NUM_THREADS                           threads (recorded in the output)

use std::fmt::Write as _;
use std::time::Instant;

use panetiere::bulletin::{ClientBulletinEntry, ServerBulletinEntry};
use panetiere::channel::{self, ChannelParams};
use panetiere::pke;
use panetiere::protocol::client::run_client_round;
use panetiere::protocol::recipient::{recover_unverified_direct_from_bytes, SetPolicy};
use panetiere::protocol::server::{run_server_round, unseal_openings, ServerInbox};
use panetiere::protocol::{
    round_wire_sizes, ClientId, Commitment, ProtocolParams, ServerId, SessionId,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use sha2::{Digest, Sha256};

// ── settings ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Encoding {
    NewtonGirard,
    Iblt,
}

impl Encoding {
    fn label(self) -> &'static str {
        match self {
            Self::NewtonGirard => "ng",
            Self::Iblt => "iblt",
        }
    }

    fn channel(self, rng: &mut ChaCha20Rng, capacity: usize, msg_bytes: usize) -> ChannelParams {
        match self {
            Self::NewtonGirard => ChannelParams::prony_for_messages(capacity as u32, msg_bytes),
            Self::Iblt => ChannelParams::for_messages(rng, capacity as u32, msg_bytes),
        }
    }
}

struct Cell {
    encoding: Encoding,
    rho: usize,
    capacity: usize,
    gamma: usize,
    msg_bytes: usize,
}

fn env_list(key: &str, default: &str) -> Vec<String> {
    std::env::var(key)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn parse<T: std::str::FromStr>(key: &str, s: &str) -> T {
    s.parse()
        .unwrap_or_else(|_| panic!("{key}: cannot parse {s:?}"))
}

fn cells() -> Vec<Cell> {
    let encodings: Vec<Encoding> = env_list("PAPER_ENCODINGS", "ng,iblt")
        .iter()
        .map(|e| match e.as_str() {
            "ng" => Encoding::NewtonGirard,
            "iblt" => Encoding::Iblt,
            other => panic!("PAPER_ENCODINGS: unknown encoding {other:?} (want ng or iblt)"),
        })
        .collect();
    let clients: Vec<(usize, usize)> = env_list("PAPER_CLIENTS", "100x100,300x300,300x100")
        .iter()
        .map(|c| {
            let (rho, cap) = c
                .split_once('x')
                .unwrap_or_else(|| panic!("PAPER_CLIENTS: want RHOxC, got {c:?}"));
            let (rho, cap) = (parse("PAPER_CLIENTS", rho), parse("PAPER_CLIENTS", cap));
            assert!(
                (1..=rho).contains(&cap),
                "PAPER_CLIENTS: need 1 ≤ C ≤ ρ in {c:?}"
            );
            (rho, cap)
        })
        .collect();
    let servers: Vec<usize> = env_list("PAPER_SERVERS", "8,16,32")
        .iter()
        .map(|s| parse("PAPER_SERVERS", s))
        .collect();
    let sizes: Vec<usize> = env_list("PAPER_MSG_BYTES", "4096,16384")
        .iter()
        .map(|s| parse("PAPER_MSG_BYTES", s))
        .collect();
    let mut out = Vec::new();
    for &encoding in &encodings {
        for &(rho, capacity) in &clients {
            for &gamma in &servers {
                for &msg_bytes in &sizes {
                    out.push(Cell {
                        encoding,
                        rho,
                        capacity,
                        gamma,
                        msg_bytes,
                    });
                }
            }
        }
    }
    out
}

// ── timing ──────────────────────────────────────────────────────────────────

/// Process CPU time (user + system over all threads), milliseconds.
#[cfg(unix)]
fn cpu_ms() -> f64 {
    let mut ru = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: `getrusage` only writes the struct it is handed.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, ru.as_mut_ptr()) } != 0 {
        return f64::NAN;
    }
    // SAFETY: a zero return means the struct was fully written.
    let ru = unsafe { ru.assume_init() };
    (ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) as f64 * 1e3
        + (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) as f64 / 1e3
}

#[cfg(not(unix))]
fn cpu_ms() -> f64 {
    f64::NAN
}

/// Wall and CPU time of one phase over the repetitions, milliseconds.
#[derive(Default, Clone)]
struct Samples {
    wall: Vec<f64>,
    cpu: Vec<f64>,
}

impl Samples {
    fn time<T>(&mut self, f: impl FnOnce() -> T) -> T {
        let (c0, t0) = (cpu_ms(), Instant::now());
        let out = f();
        self.wall.push(t0.elapsed().as_secs_f64() * 1e3);
        self.cpu.push(cpu_ms() - c0);
        out
    }

    fn median(v: &[f64]) -> f64 {
        let mut v = v.to_vec();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    fn wall_med(&self) -> f64 {
        Self::median(&self.wall)
    }

    fn cpu_med(&self) -> f64 {
        Self::median(&self.cpu)
    }

    fn wall_min(&self) -> f64 {
        self.wall.iter().copied().fold(f64::INFINITY, f64::min)
    }

    fn wall_max(&self) -> f64 {
        self.wall.iter().copied().fold(0.0, f64::max)
    }

    /// Per-repetition sum of several phases.
    fn total(parts: &[&Samples]) -> Samples {
        let reps = parts[0].wall.len();
        Samples {
            wall: (0..reps)
                .map(|i| parts.iter().map(|p| p.wall[i]).sum())
                .collect(),
            cpu: (0..reps)
                .map(|i| parts.iter().map(|p| p.cpu[i]).sum())
                .collect(),
        }
    }
}

// ── one cell ────────────────────────────────────────────────────────────────

struct CellResult {
    // setting
    encoding: Encoding,
    rho: usize,
    capacity: usize,
    gamma: usize,
    msg_bytes: usize,
    // parameters
    threshold: usize,
    mu: usize,
    t: u64,
    q: u64,
    sigma: f64,
    payload_symbols: usize,
    // bytes
    post_b: usize,
    opening_b: usize,
    comm_b: usize,
    server_entry_b: usize,
    // time
    phases: Vec<(&'static str, &'static str, Samples)>,
    recovered: bool,
    peak_rss_mib: f64,
}

fn cell_seed(c: &Cell) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"panetiere/paper-bench/v1");
    for v in [c.rho, c.capacity, c.gamma, c.msg_bytes] {
        h.update((v as u64).to_le_bytes());
    }
    h.update(c.encoding.label());
    h.finalize().into()
}

fn sub_seed(seed: &[u8; 32], label: &[u8], index: usize) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(seed);
    h.update(label);
    h.update((index as u64).to_le_bytes());
    h.finalize().into()
}

/// One client's wire output: the post and one sealed opening per server.
struct ClientWire {
    post: Vec<u8>,
    sealed: Vec<Vec<u8>>,
}

fn run_cell(c: &Cell, reps: usize) -> CellResult {
    let seed = cell_seed(c);
    let mut rng = ChaCha20Rng::from_seed(seed);

    // Setup: channel, CRS, server keys, session.
    let ch = c.encoding.channel(&mut rng, c.capacity, c.msg_bytes);
    assert!(ch.max_payload_bytes() >= c.msg_bytes);
    let pp: ProtocolParams = ch.protocol_params(&mut rng, c.gamma);
    let server_keys: Vec<pke::PrivateKey> = (0..c.gamma)
        .map(|_| pke::PrivateKey::generate(&mut rng))
        .collect();
    let servers: Vec<(ServerId, pke::PublicKey)> = server_keys
        .iter()
        .enumerate()
        .map(|(i, k)| (ServerId::new(i as u32), k.public()))
        .collect();
    let sid = SessionId::new(rng.gen());
    let canonical: Vec<ClientId> = (0..c.rho as u32).map(ClientId::new).collect();
    let messages: Vec<Vec<u8>> = (0..c.capacity)
        .map(|_| (0..c.msg_bytes).map(|_| rng.gen()).collect())
        .collect();

    // Untimed: every client's round, kept only as wire bytes. Clients 0..C
    // send a message, the rest send cover.
    let wire: Vec<ClientWire> = (0..c.rho)
        .into_par_iter()
        .map(|i| {
            let mut r = ChaCha20Rng::from_seed(sub_seed(&seed, b"client", i));
            let plain = if i < c.capacity {
                channel::encode_message(&mut r, &ch, &sid, &messages[i]).expect("message fits")
            } else {
                channel::cover(&ch)
            };
            let (post, sealed) =
                run_client_round(&mut r, &pp, &sid, ClientId::new(i as u32), plain, &servers)
                    .into_parts();
            ClientWire {
                post: post.to_bytes(),
                sealed: sealed.into_iter().map(|(_, s)| s).collect(),
            }
        })
        .collect();
    let sizes = round_wire_sizes(c.gamma, pp.message_polys(), c.rho as u32);
    let (post_b, opening_b) = (wire[0].post.len(), wire[0].sealed[0].len());
    assert_eq!(post_b, sizes.client_post());
    // The commitment part of a post: a post without ciphertext.
    let comm_b = round_wire_sizes(c.gamma, 0, c.rho as u32).client_post();
    let mut posts: Vec<Vec<u8>> = Vec::with_capacity(c.rho);
    let mut inboxes: Vec<Vec<(ClientId, Vec<u8>)>> = vec![Vec::with_capacity(c.rho); c.gamma];
    for (i, w) in wire.into_iter().enumerate() {
        assert_eq!(w.post.len(), post_b, "every post has the same length");
        posts.push(w.post);
        for (inbox, sealed) in inboxes.iter_mut().zip(w.sealed) {
            assert_eq!(
                sealed.len(),
                opening_b,
                "every sealed opening has the same length"
            );
            inbox.push((ClientId::new(i as u32), sealed));
        }
    }

    // Untimed: the commitments, parsed once from the posts.
    let commitments: Vec<(ClientId, Commitment)> = posts
        .iter()
        .zip(&canonical)
        .map(|(b, &cid)| {
            let post = ClientBulletinEntry::try_from_bytes(b).expect("post parses");
            (cid, post.commitment().clone())
        })
        .collect();

    // The server's work, from its sealed openings and the commitments.
    let server_round = |s: usize| -> ServerBulletinEntry {
        let server_id = ServerId::new(s as u32);
        let items: Vec<(ClientId, _)> =
            unseal_openings(&server_keys[s], &sid, server_id, &inboxes[s])
                .into_iter()
                .zip(&canonical)
                .map(|(o, &cid)| (cid, o.expect("opening unseals")))
                .collect();
        run_server_round(
            &pp,
            &ServerInbox::new(server_id, items),
            &canonical,
            &commitments,
        )
        .expect("honest server round")
    };

    // Untimed: every server's entry, as the bytes the recipient downloads.
    let entry_bytes: Vec<Vec<u8>> = (0..c.gamma)
        .map(|s| server_round(s).to_bytes(&pp))
        .collect();
    let server_entry_b = entry_bytes[0].len();
    assert!(entry_bytes.iter().all(|b| b.len() == server_entry_b));
    assert_eq!(server_entry_b, 8 + 4 * c.rho + sizes.server_entry());

    // Timed: client 0, from its message to its serialized post and openings.
    let (mut c_enc, mut c_round, mut c_ser) =
        (Samples::default(), Samples::default(), Samples::default());
    for rep in 0..reps {
        let mut r = ChaCha20Rng::from_seed(sub_seed(&seed, b"timed-client", rep));
        let plain =
            c_enc.time(|| channel::encode_message(&mut r, &ch, &sid, &messages[0]).unwrap());
        let out =
            c_round.time(|| run_client_round(&mut r, &pp, &sid, ClientId::new(0), plain, &servers));
        let bytes = c_ser.time(|| out.post().to_bytes());
        assert_eq!(bytes.len(), post_b);
    }

    // Timed: server 0, from its sealed openings and the parsed commitments to
    // its serialized entry.
    let (mut s_unseal, mut s_round, mut s_ser) =
        (Samples::default(), Samples::default(), Samples::default());
    for _ in 0..reps {
        let server_id = ServerId::new(0);
        let opened =
            s_unseal.time(|| unseal_openings(&server_keys[0], &sid, server_id, &inboxes[0]));
        let items = opened
            .into_iter()
            .zip(&canonical)
            .map(|(o, &cid)| (cid, o.unwrap()))
            .collect();
        let entry = s_round.time(|| {
            run_server_round(
                &pp,
                &ServerInbox::new(server_id, items),
                &canonical,
                &commitments,
            )
            .unwrap()
        });
        let bytes = s_ser.time(|| entry.to_bytes(&pp));
        assert_eq!(bytes.len(), server_entry_b);
    }

    // Timed: the recipient, from the bytes of the posts and the server entries
    // to the messages.
    let mut expected = messages.clone();
    expected.sort();
    let policy = SetPolicy::anchored(&canonical, 1, pp.max_clients());
    let (mut r_entries, mut r_recover, mut r_decode) =
        (Samples::default(), Samples::default(), Samples::default());
    let mut recovered = true;
    for _ in 0..reps {
        let entries: Vec<ServerBulletinEntry> = r_entries.time(|| {
            entry_bytes
                .iter()
                .map(|b| ServerBulletinEntry::try_from_bytes(b).expect("server entry parses"))
                .collect()
        });
        let rec = r_recover.time(|| {
            recover_unverified_direct_from_bytes(
                &pp,
                &policy,
                |cid: ClientId| posts.get(cid.get() as usize).map(Vec::as_slice),
                &entries,
            )
        });
        let decoded = r_decode.time(|| {
            rec.ok()
                .and_then(|r| channel::decode_messages(&ch, &sid, r.plaintext(), None).ok())
        });
        recovered &= decoded.as_ref() == Some(&expected);
    }

    let client_total = Samples::total(&[&c_enc, &c_round, &c_ser]);
    let server_total = Samples::total(&[&s_unseal, &s_round, &s_ser]);
    let recipient_total = Samples::total(&[&r_entries, &r_recover, &r_decode]);
    CellResult {
        encoding: c.encoding,
        rho: c.rho,
        capacity: c.capacity,
        gamma: c.gamma,
        msg_bytes: c.msg_bytes,
        threshold: pp.threshold(),
        mu: pp.message_polys(),
        t: pp.plaintext_modulus(),
        q: pp.ciphertext_modulus(),
        sigma: pp.key_sigma(),
        payload_symbols: ch.payload_symbols(),
        post_b,
        opening_b,
        comm_b,
        server_entry_b,
        phases: vec![
            ("client", "encode_message", c_enc),
            ("client", "encrypt_share_commit_seal", c_round),
            ("client", "serialize_post", c_ser),
            ("client", "total", client_total),
            ("server", "unseal_openings", s_unseal),
            ("server", "verify_and_sum_openings", s_round),
            ("server", "serialize_entry", s_ser),
            ("server", "total", server_total),
            ("recipient", "parse_entries", r_entries),
            ("recipient", "recover", r_recover),
            ("recipient", "decode_messages", r_decode),
            ("recipient", "total", recipient_total),
        ],
        recovered,
        peak_rss_mib: peak_rss_mib(),
    }
}

// ── output ──────────────────────────────────────────────────────────────────

/// Highest resident set of the process so far (`VmHWM`), MiB; 0 where
/// `/proc` is unavailable.
fn peak_rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("VmHWM:"))
                .and_then(|v| v.split_whitespace().next()?.parse::<f64>().ok())
        })
        .map_or(0.0, |kib| kib / 1024.0)
}

fn mib(b: f64) -> String {
    format!("{:.2} MiB", b / (1024.0 * 1024.0))
}

fn kib(b: f64) -> String {
    format!("{:.1} KiB", b / 1024.0)
}

fn print_result(r: &CellResult) {
    let (rho, gamma) = (r.rho as f64, r.gamma as f64);
    println!(
        "{} | ρ = {}, C = {}, γ = {}, {} B messages | t_s = {}, µ = {}, ξ = {} | t = {}, q = {}, σ = {} | recovered: {}",
        r.encoding.label(),
        r.rho,
        r.capacity,
        r.gamma,
        r.msg_bytes,
        r.threshold,
        r.mu,
        r.payload_symbols,
        r.t,
        r.q,
        r.sigma,
        if r.recovered { "yes" } else { "NO" },
    );
    println!(
        "  {:<10} {:<26} {:>12} {:>12} {:>12} {:>12}",
        "role", "phase", "wall med ms", "min", "max", "cpu med ms"
    );
    for (role, phase, s) in &r.phases {
        println!(
            "  {:<10} {:<26} {:>12.2} {:>12.2} {:>12.2} {:>12.2}",
            role,
            phase,
            s.wall_med(),
            s.wall_min(),
            s.wall_max(),
            s.cpu_med()
        );
    }
    println!(
        "  bytes: client → bulletin {} | client → each server {} | client upload {} | server download {} | server → bulletin {} | recipient download {}",
        mib(r.post_b as f64),
        kib(r.opening_b as f64),
        mib(r.post_b as f64 + gamma * r.opening_b as f64),
        mib(rho * (r.opening_b + r.comm_b) as f64),
        kib(r.server_entry_b as f64),
        mib(rho * r.post_b as f64 + gamma * r.server_entry_b as f64),
    );
    println!("  peak resident set so far: {:.0} MiB", r.peak_rss_mib);
    println!();
}

fn csv(results: &[CellResult], header_note: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# {header_note}");
    out.push_str(
        "encoding,rho,capacity,gamma,msg_bytes,threshold,mu,t,q,sigma,payload_symbols,recovered,\
         post_b,opening_b,comm_b,server_entry_b,client_upload_b,server_download_b,recipient_download_b",
    );
    for (role, phase, _) in &results[0].phases {
        let _ = write!(
            out,
            ",{role}_{phase}_wall_med_ms,{role}_{phase}_wall_min_ms,{role}_{phase}_wall_max_ms,{role}_{phase}_cpu_med_ms"
        );
    }
    out.push_str(",peak_rss_mib\n");
    for r in results {
        let _ = write!(
            out,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            r.encoding.label(),
            r.rho,
            r.capacity,
            r.gamma,
            r.msg_bytes,
            r.threshold,
            r.mu,
            r.t,
            r.q,
            r.sigma,
            r.payload_symbols,
            r.recovered,
            r.post_b,
            r.opening_b,
            r.comm_b,
            r.server_entry_b,
            r.post_b + r.gamma * r.opening_b,
            r.rho * (r.opening_b + r.comm_b),
            r.rho * r.post_b + r.gamma * r.server_entry_b,
        );
        for (_, _, s) in &r.phases {
            let _ = write!(
                out,
                ",{:.3},{:.3},{:.3},{:.3}",
                s.wall_med(),
                s.wall_min(),
                s.wall_max(),
                s.cpu_med()
            );
        }
        let _ = writeln!(out, ",{:.0}", r.peak_rss_mib);
    }
    out
}

fn main() {
    // The setup runs nested parallel work: each client's round is parallel
    // inside the parallel loop over clients. A worker that waits inside one
    // round can steal and run another on top of it, so a worker's stack can
    // hold several rounds' frames; Rust's default 2 MiB per thread is not
    // always enough. Only the stacks grow; timings are unaffected.
    rayon::ThreadPoolBuilder::new()
        .stack_size(64 << 20)
        .build_global()
        .expect("configure the thread pool");
    let reps: usize = std::env::var("PAPER_REPS")
        .map(|s| parse("PAPER_REPS", &s))
        .unwrap_or(5);
    assert!(reps >= 1, "PAPER_REPS must be at least 1");
    let csv_path = std::env::var("PAPER_CSV").unwrap_or_else(|_| "paper_bench.csv".into());
    let note = format!(
        "panetiere paper_bench | revision {} | threads {} | reps {}",
        env!("PANETIERE_REVISION"),
        rayon::current_num_threads(),
        reps,
    );
    println!("{note}");
    println!("measured: wall and process CPU time (all threads) per phase, median over reps; bytes are exact wire lengths");
    println!();

    let mut results = Vec::new();
    for cell in cells() {
        let r = run_cell(&cell, reps);
        print_result(&r);
        results.push(r);
        // Rewritten after every cell, so a run cut short keeps what it measured.
        std::fs::write(&csv_path, csv(&results, &note)).expect("write csv");
    }
    if results.iter().any(|r| !r.recovered) {
        eprintln!("some cells did not recover their messages");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One tiny cell per encoding runs every phase and recovers its messages,
    /// with consistent byte counts.
    #[test]
    fn a_small_cell_recovers() {
        for encoding in [Encoding::NewtonGirard, Encoding::Iblt] {
            let cell = Cell {
                encoding,
                rho: 3,
                capacity: 2,
                gamma: 3,
                msg_bytes: 16,
            };
            let r = run_cell(&cell, 1);
            assert!(r.recovered, "{}", encoding.label());
            assert_eq!(r.phases.len(), 12);
            assert!(r.comm_b < r.post_b);
            print_result(&r);
            let report = csv(std::slice::from_ref(&r), "test");
            assert_eq!(report.lines().count(), 3);
        }
    }

    /// Without the `PAPER_*` environment variables, the grid is the paper's: two
    /// encodings, three (ρ, C) pairs, three server counts and two message
    /// sizes.
    #[test]
    fn the_default_grid() {
        let cells = cells();
        assert_eq!(cells.len(), 2 * 3 * 3 * 2);
        assert!(cells.iter().all(|c| c.capacity <= c.rho));
    }
}
