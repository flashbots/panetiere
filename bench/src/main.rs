//! Benchmark of the protocol as the paper describes it.
//!
//! Direct mode only: every client posts its KAHE ciphertext and key-share
//! commitment to the bulletin and sends one sealed opening to each server; each
//! server posts one aggregated opening; the recipient downloads everything,
//! sums, verifies, decrypts and decodes.
//!
//! Everything reported is either measured on this machine (time per role) or
//! the exact length of what goes on the wire. Nothing is modelled: there is no
//! cost model and no network simulation.
//!
//! Per cell, all ρ clients' outputs are generated first, untimed, and kept only
//! as wire bytes; then every server runs its round once, untimed, and its entry
//! is kept as wire bytes too. Then each role is timed as a whole:
//! - the client, on one client's round (every client does the same work), from
//!   its message to its post and sealed openings as bytes;
//! - the server, on server 0's round over all ρ clients, from its sealed
//!   openings and the parsed commitments to its entry as bytes. A server only
//!   needs the commitment part of each post, but the public API parses
//!   commitments only as part of a whole post, so they are parsed once,
//!   untimed;
//! - the recipient, from the bytes of all ρ posts and the γ server entries to
//!   the messages.
//!
//! Every timed step calls the library's public API. Until the roles take and
//! return wire bytes themselves, the benchmark also does what a caller of the
//! current API has to: it encodes the message before the client's round, routes
//! the sealed openings to the servers, names the client set for the servers and
//! the recipient, and decodes the recipient's plaintext without a count check,
//! since the API has no check for |X| ≤ |I| with cover traffic.
//!
//! The settings are environment variables, documented in TESTING.md.

use std::env::VarError;
use std::fmt::Write as _;
use std::time::Instant;

use cpu_time::ProcessTime;
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
        let capacity = u32::try_from(capacity).expect("the capacity should fit in a u32");
        match self {
            Self::NewtonGirard => ChannelParams::prony_for_messages(capacity, msg_bytes),
            Self::Iblt => ChannelParams::for_messages(rng, capacity, msg_bytes),
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

/// The value of an environment variable, or `None` if it is not set. A value
/// that is not valid UTF-8 is an error, not a missing value.
fn env_var(key: &str) -> Option<String> {
    match std::env::var(key) {
        Ok(v) => Some(v),
        Err(VarError::NotPresent) => None,
        Err(VarError::NotUnicode(_)) => panic!("{key}: the value should be valid UTF-8"),
    }
}

fn env_list(key: &str, default: &str) -> Vec<String> {
    env_var(key)
        .unwrap_or_else(|| default.into())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn parse<T: std::str::FromStr>(key: &str, s: &str) -> T {
    s.parse()
        .unwrap_or_else(|_| panic!("{key}: {s:?} should be a number"))
}

fn cells() -> Vec<Cell> {
    let encodings: Vec<Encoding> = env_list("PAPER_ENCODINGS", "ng,iblt")
        .iter()
        .map(|e| match e.as_str() {
            "ng" => Encoding::NewtonGirard,
            "iblt" => Encoding::Iblt,
            other => panic!("PAPER_ENCODINGS: {other:?} should be ng or iblt"),
        })
        .collect();
    let clients: Vec<(usize, usize)> = env_list("PAPER_CLIENTS", "100,300,300x100")
        .iter()
        .map(|c| {
            let (rho, cap) = match c.split_once('x') {
                Some((rho, cap)) => (parse("PAPER_CLIENTS", rho), parse("PAPER_CLIENTS", cap)),
                None => {
                    let rho = parse("PAPER_CLIENTS", c);
                    (rho, rho)
                }
            };
            assert!(
                (1..=rho).contains(&cap),
                "PAPER_CLIENTS: {c:?} should have 1 ≤ C ≤ ρ"
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

/// Wall and process CPU time (all threads) of one role over the repetitions,
/// milliseconds.
#[derive(Default)]
struct Samples {
    wall: Vec<f64>,
    cpu: Vec<f64>,
}

impl Samples {
    fn time<T>(&mut self, f: impl FnOnce() -> T) -> T {
        let (c0, t0) = (ProcessTime::now(), Instant::now());
        let out = f();
        self.wall.push(t0.elapsed().as_secs_f64() * 1e3);
        self.cpu.push(c0.elapsed().as_secs_f64() * 1e3);
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
        self.wall.iter().copied().fold(f64::NAN, f64::min)
    }

    fn wall_max(&self) -> f64 {
        self.wall.iter().copied().fold(f64::NAN, f64::max)
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
    client: Samples,
    server: Samples,
    recipient: Samples,
}

const ROLES: [&str; 3] = ["client", "server", "recipient"];

impl CellResult {
    fn roles(&self) -> [(&'static str, &Samples); 3] {
        [
            (ROLES[0], &self.client),
            (ROLES[1], &self.server),
            (ROLES[2], &self.recipient),
        ]
    }
}

fn cell_seed(c: &Cell, seed: u64) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"panetiere/paper-bench/v1/cell");
    for v in [c.rho, c.capacity, c.gamma, c.msg_bytes] {
        h.update((v as u64).to_le_bytes());
    }
    h.update(seed.to_le_bytes());
    h.update(c.encoding.label());
    h.finalize().into()
}

fn sub_seed(seed: &[u8; 32], label: &[u8], index: usize) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"panetiere/paper-bench/v1/sub-seed");
    h.update(seed);
    h.update((index as u64).to_le_bytes());
    h.update(label);
    h.finalize().into()
}

/// One client's wire output: the post, and its sealed openings with the
/// server each one is for.
struct ClientWire {
    post: Vec<u8>,
    sealed: Vec<(ServerId, Vec<u8>)>,
}

fn run_cell(c: &Cell, reps: usize, run_seed: u64) -> CellResult {
    let seed = cell_seed(c, run_seed);
    let mut rng = ChaCha20Rng::from_seed(seed);

    // Setup: channel, CRS, server keys, session.
    let ch = c.encoding.channel(&mut rng, c.capacity, c.msg_bytes);
    assert!(
        ch.max_payload_bytes() >= c.msg_bytes,
        "the encoding should fit {}-byte messages",
        c.msg_bytes
    );
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

    // A client's work, from its message (or none, for cover) to its post and
    // sealed openings as bytes.
    let client = |r: &mut ChaCha20Rng, cid: ClientId, message: Option<&Vec<u8>>| -> ClientWire {
        let plain = match message {
            Some(m) => channel::encode_message(r, &ch, &sid, m).expect("the message should fit"),
            None => channel::cover(&ch),
        };
        let (post, sealed) = run_client_round(r, &pp, &sid, cid, plain, &servers).into_parts();
        ClientWire {
            post: post.to_bytes(),
            sealed,
        }
    };

    // Untimed: every client's round, kept only as wire bytes. Clients 0..C
    // send a message, the rest send cover.
    let wire: Vec<ClientWire> = (0..c.rho)
        .into_par_iter()
        .map(|i| {
            let mut r = ChaCha20Rng::from_seed(sub_seed(&seed, b"client", i));
            client(&mut r, ClientId::new(i as u32), messages.get(i))
        })
        .collect();
    let post_b = wire[0].post.len();
    let opening_b = wire[0].sealed[0].1.len();
    // The commitment part of a post, computed as a post without ciphertext,
    // until the library can encode the commitment on its own.
    let comm_b = round_wire_sizes(c.gamma, 0, c.rho as u32).client_post();

    // The bulletin's posts in client order, and each server's sealed
    // openings, routed by the server they are for.
    let mut posts: Vec<Vec<u8>> = Vec::with_capacity(c.rho);
    let mut inboxes: Vec<Vec<(ClientId, Vec<u8>)>> =
        (0..c.gamma).map(|_| Vec::with_capacity(c.rho)).collect();
    for (i, w) in wire.into_iter().enumerate() {
        posts.push(w.post);
        for (server, sealed) in w.sealed {
            inboxes[server.get() as usize].push((ClientId::new(i as u32), sealed));
        }
    }

    // Untimed: the commitments, parsed once from the posts.
    let commitments: Vec<(ClientId, Commitment)> = posts
        .iter()
        .zip(&canonical)
        .map(|(b, &cid)| {
            let post = ClientBulletinEntry::try_from_bytes(b).expect("every post should parse");
            (cid, post.commitment().clone())
        })
        .collect();

    // A server's work, from its sealed openings and the commitments to its
    // entry as bytes.
    let server = |s: usize| -> Vec<u8> {
        let server_id = ServerId::new(s as u32);
        let items: Vec<(ClientId, _)> =
            unseal_openings(&server_keys[s], &sid, server_id, &inboxes[s])
                .into_iter()
                .zip(&inboxes[s])
                .map(|(o, (cid, _))| (*cid, o.expect("every opening should unseal")))
                .collect();
        run_server_round(
            &pp,
            &ServerInbox::new(server_id, items),
            &canonical,
            &commitments,
        )
        .expect("the server's round should succeed")
        .to_bytes(&pp)
    };

    // Untimed: every server's entry, as the bytes the recipient downloads.
    let entry_bytes: Vec<Vec<u8>> = (0..c.gamma).map(&server).collect();
    let server_entry_b = entry_bytes[0].len();

    // Timed: client 0, with fresh randomness in every repetition.
    let mut client_t = Samples::default();
    for rep in 0..reps {
        let mut r = ChaCha20Rng::from_seed(sub_seed(&seed, b"timed-client", rep));
        client_t.time(|| client(&mut r, ClientId::new(0), messages.first()));
    }

    // Timed: server 0.
    let mut server_t = Samples::default();
    for _ in 0..reps {
        server_t.time(|| server(0));
    }

    // Timed: the recipient, from the bytes of the posts and the server entries
    // to the messages.
    let mut expected = messages.clone();
    expected.sort();
    let policy = SetPolicy::anchored(&canonical, 1, pp.max_clients());
    let mut recipient_t = Samples::default();
    for _ in 0..reps {
        let decoded = recipient_t.time(|| {
            let entries: Vec<ServerBulletinEntry> = entry_bytes
                .iter()
                .map(|b| {
                    ServerBulletinEntry::try_from_bytes(b).expect("every server entry should parse")
                })
                .collect();
            let recovered = recover_unverified_direct_from_bytes(
                &pp,
                &policy,
                |cid: ClientId| posts.get(cid.get() as usize).map(Vec::as_slice),
                &entries,
            )
            .expect("recovery should succeed");
            channel::decode_messages(&ch, &sid, recovered.plaintext(), None)
                .expect("decoding should succeed")
        });
        assert_eq!(
            decoded, expected,
            "the recipient should output the messages that were sent"
        );
    }

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
        sigma: pp.sigma(),
        payload_symbols: ch.payload_symbols(),
        post_b,
        opening_b,
        comm_b,
        server_entry_b,
        client: client_t,
        server: server_t,
        recipient: recipient_t,
    }
}

// ── output ──────────────────────────────────────────────────────────────────

/// A byte count in the largest binary unit that keeps it at least 1, with
/// three significant digits; plain bytes below 1 KiB.
fn size(b: usize) -> String {
    const UNITS: [&str; 4] = ["KiB", "MiB", "GiB", "TiB"];
    if b < 1024 {
        return format!("{b} B");
    }
    let mut v = b as f64 / 1024.0;
    let mut unit = 0;
    while v >= 1024.0 && unit + 1 < UNITS.len() {
        v /= 1024.0;
        unit += 1;
    }
    let decimals = if v >= 100.0 {
        0
    } else if v >= 10.0 {
        1
    } else {
        2
    };
    format!("{v:.decimals$} {}", UNITS[unit])
}

/// Bytes on each link of one round, from the sizes of its messages.
fn link_bytes(r: &CellResult) -> [(&'static str, usize); 6] {
    [
        ("client → bulletin", r.post_b),
        ("client → each server", r.opening_b),
        ("client upload", r.post_b + r.gamma * r.opening_b),
        ("server download", r.rho * (r.opening_b + r.comm_b)),
        ("server → bulletin", r.server_entry_b),
        (
            "recipient download",
            r.rho * r.post_b + r.gamma * r.server_entry_b,
        ),
    ]
}

fn print_result(r: &CellResult) {
    println!(
        "{} | ρ = {}, C = {}, γ = {}, {} B messages | t_s = {}, µ = {}, ξ = {} | t = {}, q = {}, σ = {}",
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
    );
    println!(
        "  {:<10} {:>12} {:>12} {:>12} {:>12}",
        "role", "wall med ms", "min", "max", "cpu med ms"
    );
    for (role, s) in r.roles() {
        println!(
            "  {:<10} {:>12.2} {:>12.2} {:>12.2} {:>12.2}",
            role,
            s.wall_med(),
            s.wall_min(),
            s.wall_max(),
            s.cpu_med()
        );
    }
    let links: Vec<String> = link_bytes(r)
        .iter()
        .map(|(link, b)| format!("{link} {}", size(*b)))
        .collect();
    println!("  bytes: {}", links.join(" | "));
    println!();
}

fn csv(results: &[CellResult], header_note: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# {header_note}");
    out.push_str(
        "encoding,rho,capacity,gamma,msg_bytes,threshold,mu,t,q,sigma,payload_symbols,\
         post_b,opening_b,comm_b,server_entry_b,client_upload_b,server_download_b,recipient_download_b",
    );
    for role in ROLES {
        let _ = write!(
            out,
            ",{role}_wall_med_ms,{role}_wall_min_ms,{role}_wall_max_ms,{role}_cpu_med_ms"
        );
    }
    out.push('\n');
    for r in results {
        let links = link_bytes(r);
        let _ = write!(
            out,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
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
            r.post_b,
            r.opening_b,
            r.comm_b,
            r.server_entry_b,
            links[2].1,
            links[3].1,
            links[5].1,
        );
        for (_, s) in r.roles() {
            let _ = write!(
                out,
                ",{:.3},{:.3},{:.3},{:.3}",
                s.wall_med(),
                s.wall_min(),
                s.wall_max(),
                s.cpu_med()
            );
        }
        out.push('\n');
    }
    out
}

/// The value after `key` in a `key: value` line of a file under `/proc`, or
/// `unknown` where there is none (on systems other than Linux, for example).
fn proc_field(path: &str, key: &str) -> String {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| {
            s.lines()
                .filter_map(|l| l.split_once(':'))
                .find(|(k, _)| k.trim() == key)
                .map(|(_, v)| v.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
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
        .expect("the thread pool should not exist yet");
    let reps: usize = env_var("PAPER_REPS")
        .map(|s| parse("PAPER_REPS", &s))
        .unwrap_or(5);
    assert!(reps >= 1, "PAPER_REPS should be at least 1");
    let seed: u64 = env_var("PAPER_SEED")
        .map(|s| parse("PAPER_SEED", &s))
        .unwrap_or(0);
    let csv_path = env_var("PAPER_CSV").unwrap_or_else(|| "paper_bench.csv".into());
    let note = format!(
        "panetiere paper_bench | revision {} | {} | cpu {} | allowed cpus {} | threads {} | reps {} | seed {}",
        env!("PANETIERE_REVISION"),
        env!("PANETIERE_RUSTC"),
        proc_field("/proc/cpuinfo", "model name"),
        proc_field("/proc/self/status", "Cpus_allowed_list"),
        rayon::current_num_threads(),
        reps,
        seed,
    );
    println!("{note}");
    println!("measured: wall and process CPU time (all threads) per role, median over reps; bytes are exact wire lengths");
    println!();

    let mut results = Vec::new();
    for cell in cells() {
        let r = run_cell(&cell, reps, seed);
        print_result(&r);
        results.push(r);
        // Rewritten after every cell, so a run cut short keeps what it measured.
        std::fs::write(&csv_path, csv(&results, &note)).expect("the CSV file should be writable");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One tiny cell per encoding runs every role, recovers its messages (run_cell
    /// asserts that), and gives one CSV row.
    #[test]
    fn a_small_cell_runs() {
        for encoding in [Encoding::NewtonGirard, Encoding::Iblt] {
            let cell = Cell {
                encoding,
                rho: 3,
                capacity: 2,
                gamma: 3,
                msg_bytes: 16,
            };
            let r = run_cell(&cell, 1, 0);
            print_result(&r);
            let report = csv(std::slice::from_ref(&r), "test");
            assert_eq!(report.lines().count(), 3);
        }
    }

    /// Without the `PAPER_*` environment variables, the grid is the paper's: two
    /// encodings, three client counts, three server counts and two message
    /// sizes.
    #[test]
    fn the_default_grid() {
        let cells = cells();
        assert_eq!(cells.len(), 2 * 3 * 3 * 2);
        let clients: Vec<(usize, usize)> = cells.iter().map(|c| (c.rho, c.capacity)).collect();
        assert!(clients.contains(&(100, 100)));
        assert!(clients.contains(&(300, 100)));
    }

    #[test]
    fn env_var_distinguishes_set_and_unset() {
        assert!(env_var("PATH").is_some());
        assert_eq!(env_var("PAPER_BENCH_TEST_SURELY_UNSET"), None);
    }

    #[test]
    fn proc_field_reads_a_value_or_unknown() {
        let path = std::env::temp_dir().join("paper_bench_proc_field_test");
        std::fs::write(&path, "first: 1\nmodel name\t: Some CPU \n").unwrap();
        let path = path.to_str().unwrap();
        assert_eq!(proc_field(path, "model name"), "Some CPU");
        assert_eq!(proc_field(path, "missing"), "unknown");
        assert_eq!(
            proc_field("/nonexistent/paper_bench", "model name"),
            "unknown"
        );
    }

    #[test]
    fn sizes_have_three_significant_digits() {
        assert_eq!(size(512), "512 B");
        assert_eq!(size(52_332), "51.1 KiB");
        assert_eq!(size(6_682_114), "6.37 MiB");
        assert_eq!(size(5_351_000_000), "4.98 GiB");
        assert_eq!(size(150 << 20), "150 MiB");
    }
}
