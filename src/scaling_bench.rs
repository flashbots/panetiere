//! End-to-end scaling bench: sweeps `S × (ρ, active)` × MSE/KAHE configs.
//!
//! Headline metric: **useful multiset throughput** — bits/sec of recovered
//! application payload through the full pipeline. Output is a set of
//! provenance-tagged tables (one row per cell) plus `scaling_sweep.csv`
//! with every number including min/max spread:
//!
//!   [M] measured  — per-phase CPU, median of REPS one-party runs through the
//!                   same APIs the protocol uses; plus wire sizes read off
//!                   real packed bytes (sealed opening envelope, server entry)
//!   [D] derived   — exact byte arithmetic from packing formulas
//!   [C] composed  — cost model `wall = fixed + payload` over [M] medians;
//!                   one client → one server → verifier, summed as a serial
//!                   pipeline (one party of each kind, not ρ of them)
//!   [P] projected — synthetic network sim
//!
//! Each cell runs ONE live round; every active client encodes a distinct
//! (element, r). Recovery is attempted and reported, never asserted — a decode
//! to garbage still yields timings; a verify that Errs on every rep records
//! zeros for the verifier phases and warns on stderr.
//!
//! ──────────────────────────────────────────────────────────────────────
//! INPUTS (knobs — sweep via env, or edit the defaults below)
//! ──────────────────────────────────────────────────────────────────────
//!  Sweep order is clients → servers → protocol, protocol innermost, so a
//!  budget cut-off leaves complete protocol comparisons at every (client,
//!  server) point reached rather than one protocol across all of them.
//!  SERVERS: S                              server counts to sweep
//!  CLIENTS: (clients_total, clients_active) clients_total−active send cover;
//!                                           IBLT is sized to active, not total.
//!  CONFIGS: ξ             payload symbols/client (useful bits = ξ·log₂ t)
//!  SCHED_MESSAGE_BYTES:   message-vector sizes for the scheduled flow
//!  NETWORKS: per-profile one-way latency + uniform jitter, and Mbit/s for each
//!            of the three link classes (client, server, bulletin/recipient)
//!  env:     SWEEP_SERVERS          "8,16,64"          overrides SERVERS
//!           SWEEP_CLIENTS          "100x100,300x300"  overrides CLIENTS
//!           SWEEP_PAYLOAD_SYMBOLS  "2048,10240"       overrides CONFIGS ξ
//!           SWEEP_SCHED_BYTES      "409600,2097152"   overrides SCHED_MESSAGE_BYTES
//!                                  ("" = skip the scheduled section)
//!           SWEEP_APPEND=1         append to the csv instead of overwriting
//!                                  (dedup on the param columns, not `id`)
//!           BENCH_BUDGET_SECS      default 300
//!           RAYON_NUM_THREADS      recommend 8
//!
//! ──────────────────────────────────────────────────────────────────────
//! DERIVED FROM INPUTS (computed per cell — not knobs)
//! ──────────────────────────────────────────────────────────────────────
//!  δ              = ⌈3·active/γ⌉ IBLT buckets per MSE row
//!  μ_kahe         = n_polys — one key covers the whole plaintext
//!  total_cells    = γ · δ
//!  Shamir threshold, κ_cs, β, β_agg, block_size — chosen by
//!  `ProtocolParams::setup_with_kahe_dims_full(s, μ_kahe, κ_kahe, …)`
//!
//! ──────────────────────────────────────────────────────────────────────
//! CONSTANTS (pinned upstream — the sources are authoritative)
//! ──────────────────────────────────────────────────────────────────────
//!  src/rings.rs: N = 2048, q_cs = 139_301, q_kahe = 347_280_875_347_969
//!  chipmunk param.rs: q_hvc = 40_961, HVC_WIDTH = 3
//!  src/kahe.rs:  t = T_MODULUS_DEFAULT = 2^35, σ_s = σ_e = 15.72
//!                correctness needs q ≥ tρ + tσ√(8ρ(ln2 − ln(1 − (1−2^−ε)^(1/nμ))))
//!                — reported per cell as `eps`, the exponent that condition
//!                actually buys at that (ρ, μ). It falls with √ρ and with
//!                log(nμ), so it is a per-cell number, not a global ρ ceiling.
//!  src/mse.rs:   BITS_PER_SYMBOL = 36, K_LIMBS = 2
//!  bench-internal: MSE PRF key = [0xAA; 32]; per-cell ChaCha20 seed = SHA-256
//!                  over (S, ρ, active, ξ, flow, message_bytes), so cells never
//!                  share a stream
//!  protocol coupling (asserted at runtime): μ_cs = κ_kahe
//!  asserted per cell: agg_open + agg_share == CsParams::aggregated_server_crypto_len
//!
//! Run with:
//!   RAYON_NUM_THREADS=8 cargo bench -j 8 --bench protocol_sweep
//!   BENCH_BUDGET_SECS=600 cargo bench --bench protocol_sweep

use std::fmt::Write as _;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::bulletin::rs_poly_packed_len;
use crate::bulletin::{RsClientBulletinEntry, RsNodeBulletinEntry};
use crate::channel::{self, ChannelParams};
use crate::codec::{self, BYTES_PER_COEFF, BYTES_PER_POLY};
use crate::cs::{
    aggregated_opening_pack_bounds, pack_cs_shares, poly_packed_len, poly_packed_len64, Commitment,
    Cs, HidingMerkleCommitment,
};
use crate::kahe::{Kahe, KaheScheme, SIGMA_E_DEFAULT, SIGMA_S_DEFAULT};
use crate::mse::BITS_PER_SYMBOL;
use crate::pke;
use crate::protocol::aggregator::run_aggregator_round;
use crate::protocol::client::{
    cs_commit, kahe_encrypt, kahe_keygen, run_client_round, run_client_round_rs, seal_openings,
    shamir_share,
};
use crate::protocol::server::{
    run_rs_node_round, run_server_round, unseal_opening, unseal_openings, RsNodeInbox, ServerInbox,
};
use crate::protocol::verify::{
    aggregate_and_decrypt_rs, aggregate_and_decrypt_unverified_timed, decrypt_unverified_aggregate, RsVerifyTimings,
    VerifyError, VerifyTimings,
};
use crate::protocol::ProtocolParams;
use crate::protocol::{ClientId, NodeId, ServerId, SessionId};
use crate::rs::Rs;
use crate::share_commitment::{
    commit_shares, fresh_path_packed_len, ingest_share, lane_post_packed_len, open_share,
};
use crate::sig::SigningKey;
use crate::{KahePoly, KAHE_MODULUS, N as POLY_N};
use chipmunk_code::{HVCPoly, HVC_MODULUS, HVC_WIDTH};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

const SERVERS: &[usize] = &[8, 16, 64];

// (clients_total, clients_active): explicit pairs, not a cross product.
const CLIENTS: &[(usize, usize)] = &[(100, 100), (300, 300), (300, 100)];

struct Config {
    label: String,
    payload_symbols: usize, // ξ: per-client element size = ξ·log₂t bits
}

/// Repetitions per measured phase; tables show the median, CSV keeps min/max.
const REPS: usize = 5;

/// One cell is one protocol execution.
const SESSION: SessionId = SessionId([0x5C; 32]);

/// Scheduling token = (rand u16, size u16) → 2 MSE symbols.
const SCHED_TOKEN_SYMBOLS: usize = 2;
/// Messaging payload sizes swept in the scheduled (two-round) section.
/// The scheduled codec ignores ξ (token IBLT is fixed at 2 symbols), so the
/// scheduled sweep runs once per message size, reusing CONFIGS[0].
const SCHED_MESSAGE_BYTES: &[usize] = &[409600, 2097152];

// RS-sharded ingress: (k, n) pairs run live, one extra round per pair per cell
// (so the default is a single point). The pair fixes the redundancy `n−k`; `n`
// and `k` both rise together when a cell has more threshold servers. The
// k-sweep in the `[D] rs sizing` table is exact arithmetic and costs nothing,
// so only measure what measurement actually settles.
const RS_PLANS: &[(usize, usize)] = &[(14, 16)];

/// k values priced (not measured) in the sizing table, at n = k+2.
const RS_SIZING_K: &[usize] = &[4, 8, 14, 32, 64];

/// Aggregation ceiling for the RNS share commitment.
const RS_RHO_MAX: usize = 300;

// Per-client element size = ξ symbols · 36 bits: 911 ⇒ 4 KiB, 3641 ⇒ 16 KiB.
const DEFAULT_PAYLOAD_SYMBOLS: &[usize] = &[911, 3641];

// ── env sweep knobs (fall back to the consts above) ─────────────────────────

fn env_list<T: std::str::FromStr>(key: &str) -> Option<Vec<T>> {
    let raw = std::env::var(key).ok()?;
    if raw.trim().is_empty() {
        return Some(vec![]);
    }
    let parsed: Result<Vec<T>, _> = raw.split(',').map(|s| s.trim().parse()).collect();
    match parsed {
        Ok(v) => Some(v),
        Err(_) => panic!("bad {key}: {raw:?}"),
    }
}

fn sweep_clients() -> Vec<(usize, usize)> {
    match std::env::var("SWEEP_CLIENTS") {
        Err(_) => CLIENTS.to_vec(),
        Ok(raw) => {
            raw.split(',')
                .map(|c| {
                    let (total, active) = c.trim().split_once('x').unwrap_or_else(|| {
                        panic!("bad SWEEP_CLIENTS pair {c:?} (want TOTALxACTIVE)")
                    });
                    (
                        total
                            .parse()
                            .unwrap_or_else(|_| panic!("bad clients_total in {c:?}")),
                        active
                            .parse()
                            .unwrap_or_else(|_| panic!("bad clients_active in {c:?}")),
                    )
                })
                .collect()
        }
    }
}

fn sweep_configs() -> Vec<Config> {
    env_list::<usize>("SWEEP_PAYLOAD_SYMBOLS")
        .unwrap_or_else(|| DEFAULT_PAYLOAD_SYMBOLS.to_vec())
        .into_iter()
        .map(|xi| Config {
            label: format!("{}KB", xi * BITS_PER_SYMBOL / 8 / 1024),
            payload_symbols: xi,
        })
        .collect()
}

/// Network sim profile: per-message one-way latency = `lat_ms + U[0, jitter_ms)`
/// (parallel arrivals take the max draw). Three link classes: client links
/// (residential, the slow side), server links (relays and aggregators), and the
/// bulletin/recipient link — the one endpoint that ingests every client's post
/// and is read in full by the verifier, so it saturates first. Each endpoint
/// serializes its own bytes at its class rate, both directions.
struct NetProfile {
    label: &'static str,
    lat_ms: f64,
    jitter_ms: f64,
    client_mbps: f64,
    server_mbps: f64,
    bulletin_mbps: f64,
}

// The `bul*` profiles hold the servers at 1 Gbit and vary only the bulletin, so
// the delta against `1Gbit` is exactly what a faster recipient link buys.
const NETWORKS: &[NetProfile] = &[
    NetProfile {
        label: "1Gbit",
        lat_ms: 60.0,
        jitter_ms: 10.0,
        client_mbps: 100.0,
        server_mbps: 1000.0,
        bulletin_mbps: 1000.0,
    },
    NetProfile {
        label: "4Gbit",
        lat_ms: 60.0,
        jitter_ms: 10.0,
        client_mbps: 100.0,
        server_mbps: 4000.0,
        bulletin_mbps: 4000.0,
    },
    NetProfile {
        label: "bul8",
        lat_ms: 60.0,
        jitter_ms: 10.0,
        client_mbps: 100.0,
        server_mbps: 1000.0,
        bulletin_mbps: 8000.0,
    },
    NetProfile {
        label: "bul20",
        lat_ms: 60.0,
        jitter_ms: 10.0,
        client_mbps: 100.0,
        server_mbps: 1000.0,
        bulletin_mbps: 20000.0,
    },
    // Client uplink at 1 Gbit instead of 100 Mbit, across the same bulletin
    // range. Pairs with `bul8`/`bul20` to separate the two links: same bulletin,
    // only the client rate differs.
    NetProfile {
        label: "cl1Gb4",
        lat_ms: 60.0,
        jitter_ms: 10.0,
        client_mbps: 1000.0,
        server_mbps: 1000.0,
        bulletin_mbps: 4000.0,
    },
    NetProfile {
        label: "cl1Gb8",
        lat_ms: 60.0,
        jitter_ms: 10.0,
        client_mbps: 1000.0,
        server_mbps: 1000.0,
        bulletin_mbps: 8000.0,
    },
    NetProfile {
        label: "cl1Gb20",
        lat_ms: 60.0,
        jitter_ms: 10.0,
        client_mbps: 1000.0,
        server_mbps: 1000.0,
        bulletin_mbps: 20000.0,
    },
];

#[derive(Clone, Copy, Default)]
struct Stat {
    med: f64,
    min: f64,
    max: f64,
}

fn stat(mut v: Vec<f64>) -> Stat {
    v.sort_by(f64::total_cmp);
    Stat {
        med: v[v.len() / 2],
        min: v[0],
        max: *v.last().unwrap(),
    }
}

fn measure<R>(mut f: impl FnMut() -> R) -> (Stat, R) {
    let mut times = Vec::with_capacity(REPS);
    let mut out = None;
    for _ in 0..REPS {
        let t = Instant::now();
        let r = f();
        times.push(t.elapsed().as_secs_f64() * 1e6);
        // Assign after stopping the clock: dropping the previous rep's result
        // (a multi-MB ciphertext at large ξ) must not land inside the window.
        out = Some(r);
    }
    (stat(times), out.unwrap())
}

fn fmt_us(us: f64) -> String {
    if us < 1e3 {
        format!("{:.2}us", us)
    } else if us < 1e6 {
        format!("{:.3}ms", us / 1e3)
    } else {
        format!("{:.2}s", us / 1e6)
    }
}

fn fmt_bytes(b: f64) -> String {
    if b < 1024.0 {
        format!("{:.0} B", b)
    } else if b < 1024.0 * 1024.0 {
        format!("{:.2} KiB", b / 1024.0)
    } else {
        format!("{:.2} MiB", b / (1024.0 * 1024.0))
    }
}

/// `2^-eps` failure probability, or `-` where the parameters give no bound.
fn fmt_eps(eps: f64) -> String {
    if eps <= 0.0 {
        "-".into()
    } else if eps.is_infinite() {
        "exact".into()
    } else {
        format!("{:.0}", eps)
    }
}

fn cell_id(i: usize) -> String {
    let c = (b'A' + (i % 26) as u8) as char;
    if i < 26 {
        c.to_string()
    } else {
        format!("{}{}", c, i / 26)
    }
}

// One aggregation layer: `n_l1` groups of `group_size`, leader sums the rest.
struct AggPlan {
    group_size: usize,
    n_l1: usize,
    agg_cpu: Stat,
    leader: Stat,
    recovered: bool,
}

/// One live RS-sharded round. `reconstruct` is the identity-set path (nodes
/// `0..k` hold the blocks, so it is a concatenation); `reconstruct_lagrange` is
/// the same with the sample set shifted by one, i.e. what a lane outage costs.
struct RsPlan {
    k: usize,
    n_nodes: usize,
    share_b: usize,
    /// One lane's Merkle path, sent alongside its share.
    path_b: usize,
    /// A lane's digit-domain post: the summed opening plus its path.
    lane_post_b: usize,
    bulletin_b: usize,
    rns_encrypt: Stat,
    share_commit: Stat,
    rs_enc: Stat,
    client_sign: Stat,
    /// Full lane round: ρ opens, the digit-wise sum, one aggregate proof.
    node_sum: Stat,
    /// Breakdown of `node_sum` — per-client decompose + bound checks.
    lane_ingest: Stat,
    /// Off the happy path: ρ per-client verifications to name a culprit.
    lane_attribute: Stat,
    v_sig: Stat,
    v_open: Stat,
    v_interp: Stat,
    v_lane_open: Stat,
    v_crosscheck: Stat,
    v_reconstruct: Stat,
    v_reconstruct_lagrange: Stat,
    v_kahe_dec: Stat,
    recovered: bool,
    /// Which mechanism rejected a tampered lane post.
    lane_lie_caught: &'static str,
}

struct Row {
    s: usize,
    n: usize, // ρ — total clients = anonymity set
    active: usize,
    cover: usize,
    iblt_cells: usize,
    flow: &'static str,      // "mse" | "prony" | "sched"
    payload_client_b: usize, // per-client useful bytes (element / msg slot)
    mu_kahe: usize,
    delta: usize,
    xi: usize,
    n_polys: usize, // actual message length; ≤ μ
    /// Where this row's timings were taken: `host`, `tdx`, `host-rest`, or
    /// `tdx+host` once merged.
    env: String,
    /// Rayon pool width and CPU affinity mask. A merge across differing thread
    /// counts is not meaningful, so it is refused.
    threads: usize,
    affinity: String,
    recovered_ok: bool,
    // [M] CPU, per-round totals for one party.
    enc_app: Stat,
    kahe_keygen: Stat,
    kahe_enc: Stat,
    share: Stat,
    cs: Stat,
    seal: Stat,
    unseal: Stat,
    server: Stat,
    v_agg_ctxt: Stat,
    v_sum_comm: Stat,
    v_open: Stat,
    v_interp: Stat,
    v_kahe_dec: Stat,
    dec_app: Stat,
    /// One fresh opening against one commitment — the per-opening `VC.Vf` unit,
    /// as distinct from `v_open`, which times the S-way parallel verify.
    v_one_open: Stat,
    // [M] wire: real ML-KEM-sealed opening envelope (client → one server).
    open_env_b: usize,
    // [D] wire components (bytes).
    comm_client_b: usize, // one commitment root
    ctxt_client_b: usize, // one client's full ciphertext (n_polys polys)
    kahe_key_b: usize,    // one KAHE key (κ_kahe = 1 ring element)
    plaintext_b: usize,   // KAHE plaintext, n_polys elements of R_t
    agg_open_b: usize,    // server-posted aggregate opening
    agg_share_b: usize,   // server-posted agg_share (one CS poly)
    /// Correctness exponent the KAHE condition yields at this (ρ, n_polys):
    /// a round decodes with probability ≥ 1 − 2^−eps. `0` = no guarantee.
    eps_correct: f64,
    // [D] round totals across all parties.
    useful_b: f64,
    wire_ctxt_b: f64,
    wire_comm_b: f64,
    wire_opening_b: f64,
    wire_server_b: f64,
    agg: AggPlan,
    rs_plans: Vec<RsPlan>,
}

/// Group size balancing both fan-ins at ⌈√ρ⌉. Pure in `n`, so a row rebuilt
/// from CSV reproduces the same topology as the run that wrote it.
fn agg_group_size(n: usize) -> usize {
    ((n as f64).sqrt().ceil() as usize).max(2)
}

/// Which phases a run's timings are authoritative for. Setup and the live round
/// are unaffected — this only gates whether a phase's `Stat` is recorded, so a
/// split run and a whole run walk identical code.
#[derive(Clone, Copy, PartialEq)]
struct Roles {
    client: bool,
    server: bool,
    verifier: bool,
    agg: bool,
    rs: bool,
}

impl Roles {
    /// `BENCH_ROLES=client`, `=server,verifier,agg,rs`, or absent for everything.
    fn from_env() -> Self {
        match std::env::var("BENCH_ROLES") {
            Err(_) => Self::all(),
            Ok(raw) => Self::from_env_str(&raw),
        }
    }

    fn from_env_str(raw: &str) -> Self {
        if raw.trim().is_empty() || raw.trim() == "all" {
            return Self::all();
        }
        let mut roles = Roles {
            client: false,
            server: false,
            verifier: false,
            agg: false,
            rs: false,
        };
        for name in raw.split(',').map(str::trim) {
            match name {
                "client" => roles.client = true,
                "server" => roles.server = true,
                "verifier" => roles.verifier = true,
                "agg" => roles.agg = true,
                "rs" => roles.rs = true,
                other => panic!(
                    "bad BENCH_ROLES entry {other:?} \
                     (want client|server|verifier|agg|rs|all)"
                ),
            }
        }
        roles
    }

    fn all() -> Self {
        Roles {
            client: true,
            server: true,
            verifier: true,
            agg: true,
            rs: true,
        }
    }
}

/// Where this run is executing, for the `env` column. Cannot be inferred from the
/// role set — a client-only run is equally plausible on the host as a baseline or
/// in a VM as the real measurement — so the caller states it. Default `host`.
fn env_tag() -> String {
    std::env::var("BENCH_ENV").unwrap_or_else(|_| "host".into())
}

/// `Stat` only when this run owns the phase, otherwise zero — so a merge can tell
/// "not measured here" from "measured as fast".
fn owned(keep: bool, s: Stat) -> Stat {
    if keep {
        s
    } else {
        Stat::default()
    }
}

/// The CPU set this process may run on, e.g. `0-7`. Records whether a run was
/// pinned, which decides whether its timings are comparable to another's.
fn affinity() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
                .map(|v| v.trim().to_string())
        })
        .unwrap_or_else(|| "?".into())
}

/// Largest `ε` for which KAHE addition correctness holds at `rho` aggregations
/// over a `mu`-element ciphertext: a round decodes with probability ≥ 1 − 2^−ε.
///
/// Inverts `q ≥ tρ + tσ√(8ρ(ln2 − ln(1 − (1−2^−ε)^(1/nμ))))`. Returns 0 when the
/// modulus leaves no room for noise at all, and `∞` when every coefficient fits
/// deterministically.
fn correctness_epsilon(rho: usize, mu: usize, t_modulus: u64) -> f64 {
    let q = KAHE_MODULUS as f64;
    let t = t_modulus as f64;
    let sigma = SIGMA_E_DEFAULT;
    let (rho_f, nmu) = (rho as f64, (POLY_N * mu) as f64);
    // Headroom left for the error term once the ρ plaintexts have their share.
    let slack = q / t - rho_f;
    if slack <= 0.0 {
        return 0.0;
    }
    let l = slack * slack / (8.0 * rho_f * sigma * sigma);
    // 2e^−L is the per-coefficient tail mass; ln_1p/exp_m1 keep the union bound
    // over nμ coefficients accurate when it is tiny.
    let tail = 2.0 * (-l).exp();
    if tail >= 1.0 {
        return 0.0;
    }
    let p_fail = -((nmu * (-tail).ln_1p()).exp_m1());
    if p_fail <= 0.0 {
        return f64::INFINITY;
    }
    -p_fail.log2()
}

#[cfg(test)]
mod csv_tests {
    use super::*;

    /// Every phase gets a distinct value so a swapped column shows up as a
    /// mismatch rather than passing by coincidence.
    fn sample_row() -> Row {
        let mut r = Row {
            s: 8,
            n: 300,
            active: 100,
            cover: 200,
            iblt_cells: 300,
            flow: "mse",
            payload_client_b: 4099,
            mu_kahe: 134,
            delta: 75,
            xi: 911,
            n_polys: 134,
            env: "host".into(),
            threads: 8,
            affinity: "0-7".into(),
            recovered_ok: true,
            enc_app: Stat::default(),
            kahe_keygen: Stat::default(),
            kahe_enc: Stat::default(),
            share: Stat::default(),
            cs: Stat::default(),
            seal: Stat::default(),
            unseal: Stat::default(),
            server: Stat::default(),
            v_agg_ctxt: Stat::default(),
            v_sum_comm: Stat::default(),
            v_open: Stat::default(),
            v_interp: Stat::default(),
            v_kahe_dec: Stat::default(),
            dec_app: Stat::default(),
            v_one_open: Stat::default(),
            open_env_b: 57862,
            comm_client_b: 4352,
            ctxt_client_b: 1715200,
            kahe_key_b: 12800,
            plaintext_b: 1234560,
            agg_open_b: 106240,
            agg_share_b: 4864,
            eps_correct: 35.9,
            useful_b: 409900.0,
            wire_ctxt_b: 5.0e8,
            wire_comm_b: 1.3e6,
            wire_opening_b: 1.4e8,
            wire_server_b: 8.9e5,
            agg: AggPlan {
                group_size: agg_group_size(300),
                n_l1: 300usize.div_ceil(agg_group_size(300)),
                agg_cpu: Stat::default(),
                leader: Stat::default(),
                recovered: true,
            },
            rs_plans: vec![RsPlan {
                k: 14,
                n_nodes: 16,
                share_b: 1884160,
                path_b: 86016,
                lane_post_b: 4423680,
                bulletin_b: 4194,
                rns_encrypt: Stat::default(),
                share_commit: Stat::default(),
                rs_enc: Stat::default(),
                client_sign: Stat::default(),
                node_sum: Stat::default(),
                lane_ingest: Stat::default(),
                lane_attribute: Stat::default(),
                v_sig: Stat::default(),
                v_open: Stat::default(),
                v_interp: Stat::default(),
                v_lane_open: Stat::default(),
                v_crosscheck: Stat::default(),
                v_reconstruct: Stat::default(),
                v_reconstruct_lagrange: Stat::default(),
                v_kahe_dec: Stat::default(),
                recovered: true,
                lane_lie_caught: "lane-proof",
            }],
        };
        for (i, p) in PHASES.iter().enumerate() {
            let base = (i as f64 + 1.0) * 100.0;
            (p.set)(
                &mut r,
                Stat {
                    med: base,
                    min: base - 10.0,
                    max: base + 10.0,
                },
            );
        }
        r
    }

    /// Round-trip through the real writer and reader, so a column added to one and
    /// not the other fails here.
    #[test]
    fn row_survives_csv_round_trip() {
        let row = sample_row();
        let dir = std::env::temp_dir().join(format!("panetiere-csv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rt.csv");
        std::env::set_var("BENCH_CSV", &path);
        std::env::remove_var("SWEEP_APPEND");
        write_csv(std::slice::from_ref(&row));

        let back = read_rows(path.to_str().unwrap()).unwrap();
        assert_eq!(back.len(), 1);
        let b = &back[0];

        assert_eq!(
            (b.s, b.n, b.active, b.cover),
            (row.s, row.n, row.active, row.cover)
        );
        assert_eq!(b.flow, row.flow);
        assert_eq!(b.n_polys, row.n_polys);
        assert_eq!(b.env, row.env);
        assert_eq!(b.threads, row.threads);
        assert_eq!(b.affinity, row.affinity);
        assert_eq!(b.open_env_b, row.open_env_b);
        assert_eq!(b.agg_open_b, row.agg_open_b);
        assert_eq!(b.plaintext_b, row.plaintext_b);
        assert!((b.eps_correct - row.eps_correct).abs() < 0.01);
        assert_eq!(b.rs_plans[0].k, 14);
        assert_eq!(b.rs_plans[0].lane_lie_caught, "lane-proof");
        assert_eq!(b.rs_plans[0].path_b, row.rs_plans[0].path_b);
        assert_eq!(b.rs_plans[0].lane_post_b, row.rs_plans[0].lane_post_b);
        // Topology is recomputed, not stored — it must land on the same values.
        assert_eq!(b.agg.group_size, row.agg.group_size);
        assert_eq!(b.agg.n_l1, row.agg.n_l1);

        for p in PHASES {
            let (want, got) = ((p.get)(&row), (p.get)(b));
            assert!(
                (want.med - got.med).abs() < 0.01
                    && (want.min - got.min).abs() < 0.01
                    && (want.max - got.max).abs() < 0.01,
                "{}: wrote {:?} read {:?}",
                p.name,
                (want.med, want.min, want.max),
                (got.med, got.min, got.max)
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The role split has to partition the phases: a phase owned by nobody would
    /// silently never be recorded by any run.
    #[test]
    fn every_phase_has_an_owner_reachable_from_roles() {
        let all = Roles::all();
        for p in PHASES {
            let owned_by_all = match p.owner {
                Owner::Client => all.client,
                Owner::Server => all.server,
                Owner::Verifier => all.verifier,
                Owner::Agg => all.agg,
                Owner::Rs => all.rs,
            };
            assert!(owned_by_all, "{} unreachable", p.name);
        }
        let client_only = Roles::from_env_str("client");
        assert!(PHASES.iter().any(|p| p.owner == Owner::Client));
        assert!(!client_only.server && !client_only.verifier);
    }
}

/// Paper-table role totals: the phases a single party of each kind runs per
/// round. The three sum to the `[C]` serial wall.
fn client_us(r: &Row) -> f64 {
    r.enc_app.med + r.kahe_keygen.med + r.kahe_enc.med + r.share.med + r.cs.med + r.seal.med
}

fn server_us(r: &Row) -> f64 {
    r.unseal.med + r.server.med
}

fn recipient_us(r: &Row) -> f64 {
    r.v_agg_ctxt.med
        + r.v_sum_comm.med
        + r.v_open.med
        + r.v_interp.med
        + r.v_kahe_dec.med
        + r.dec_app.med
}

/// The S sealed openings, addressed to the servers. Depends on S alone — not on
/// the message.
fn client_to_server_b(r: &Row) -> usize {
    r.s * r.open_env_b
}

/// The bulletin post the recipient reads: commitment plus ciphertext. Depends on
/// the plaintext width alone — not on S.
fn client_to_recipient_b(r: &Row) -> usize {
    r.comm_client_b + r.ctxt_client_b
}

/// Everything one client puts on the wire.
fn client_post_b(r: &Row) -> usize {
    client_to_server_b(r) + client_to_recipient_b(r)
}

fn server_entry_b(r: &Row) -> usize {
    r.agg_open_b + r.agg_share_b
}

/// Per-cell RNG seed: domain-separated hash over every parameter that defines
/// the cell, including the flow. Hashing (not byte-packing) keeps distinct
/// cells on distinct streams regardless of magnitude.
fn cell_seed(
    s: usize,
    n: usize,
    active: usize,
    xi: usize,
    flow: &str,
    message_bytes: usize,
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"panetiere-scaling-cell");
    for v in [
        s as u64,
        n as u64,
        active as u64,
        xi as u64,
        message_bytes as u64,
    ] {
        h.update(v.to_le_bytes());
    }
    h.update(flow.as_bytes());
    h.finalize().into()
}

fn run_cell(
    s: usize,
    n: usize,
    active: usize,
    cfg: &Config,
    flow: &'static str,
    message_bytes: usize,
    roles: Roles,
) -> Row {
    let cover = n - active;
    let seed = cell_seed(s, n, active, cfg.payload_symbols, flow, message_bytes);
    let mut rng = ChaCha20Rng::from_seed(seed);
    let sched = flow == "sched";

    // Channel sized to active (cover adds nothing). `mse` inserts the ξ-element;
    // `prony` re-derives ξ so its element carries the same bytes at 35 rather
    // than 36 bits per symbol; `sched` inserts a (rand,size) reservation
    // token and appends a message vector to the same plaintext.
    let ch = match flow {
        "mse" => ChannelParams::for_symbols(active as u32, cfg.payload_symbols, [0xAA; 32]),
        "prony" => {
            ChannelParams::prony_for_bits(active as u32, cfg.payload_symbols * BITS_PER_SYMBOL)
        }
        "sched" => ChannelParams::for_symbols(active as u32, SCHED_TOKEN_SYMBOLS, [0xAA; 32]),
        other => panic!("unknown flow {other:?}"),
    };
    let ch_polys = ch.n_polys();
    // For `prony` the structural width is the Vandermonde column count, reported
    // in place of the IBLT's cell count.
    let (iblt_cells, delta) = match &ch {
        ChannelParams::Mse(p) => (p.total_cells(), p.delta),
        ChannelParams::Prony(p) => (p.cols(), p.cols()),
    };
    // Slot alignment matches the codec's packing width, so `active` slots fill
    // the vector exactly and `allocate` never rounds one past the end.
    let slot_bytes = if sched {
        (message_bytes / active.max(1)).next_multiple_of(BYTES_PER_COEFF)
    } else {
        0
    };
    let msg_vector_bytes = active * slot_bytes;
    let msg_polys = msg_vector_bytes.div_ceil(BYTES_PER_POLY);
    let n_polys = ch_polys + msg_polys;
    // One key covers the whole joint plaintext.
    let mu_kahe = n_polys;
    let t_modulus = ch.plaintext_modulus();

    let pp = ProtocolParams::setup_with_kahe_dims_full(
        &mut rng,
        s,
        mu_kahe,
        SIGMA_S_DEFAULT,
        SIGMA_E_DEFAULT,
        t_modulus,
    );
    let server_ids: Vec<ServerId> = (0..s as u32).map(ServerId).collect();
    let server_keys: Vec<pke::PrivateKey> = (0..s)
        .map(|_| pke::PrivateKey::generate(&mut rng))
        .collect();
    let servers: Vec<(ServerId, pke::PublicKey)> = server_ids
        .iter()
        .map(|&sid| (sid, server_keys[sid.0 as usize].public()))
        .collect();
    let client_ids: Vec<ClientId> = (0..n as u32).map(ClientId).collect();

    // Active clients carry a distinct payload; cover clients contribute zero.
    // Mse/Prony: a ξ-symbol element. Scheduled: the (rand, size) reservation
    // token each client posts, then the slot the allocator gives it.
    let reservations: Vec<(u16, usize)> = if sched {
        (0..active)
            .map(|_| (rng.gen::<u16>(), slot_bytes))
            .collect()
    } else {
        vec![]
    };
    let symbol_payloads: Vec<Vec<i64>> = if sched {
        reservations
            .iter()
            .map(|&(rand, size)| vec![rand as i64, size as i64])
            .collect()
    } else {
        (0..active)
            .map(|i| {
                (0..ch.payload_symbols())
                    .map(|j| i as i64 + j as i64 + 1)
                    .collect()
            })
            .collect()
    };
    // The allocation the recipient actually performs: peel the tokens, derive
    // the beacon from their rands, assign offsets. Two clients that drew the
    // same rand both lose their slot, so what a round delivers is `allocated`,
    // not `active`.
    let rands: Vec<u16> = reservations.iter().map(|&(r, _)| r).collect();
    let offsets = codec::allocate(&reservations, codec::beacon(&rands), msg_vector_bytes);
    let allocated: Vec<(usize, usize)> = offsets
        .iter()
        .enumerate()
        .filter_map(|(i, off)| off.map(|o| (i, o)))
        .collect();
    let byte_payloads: Vec<Vec<u8>> = (0..active)
        .map(|i| vec![(i as u8).wrapping_add(1); slot_bytes])
        .collect();
    let ranges: Vec<(usize, usize)> = allocated.iter().map(|&(_, o)| (o, slot_bytes)).collect();
    let range_payloads: Vec<Vec<u8>> = allocated
        .iter()
        .map(|&(i, _)| byte_payloads[i].clone())
        .collect();
    let (payload_bits, delivered) = if sched {
        (slot_bytes * 8, allocated.len())
    } else {
        (ch.payload_symbols() * ch.bits_per_symbol(), active)
    };
    let client_polys: Vec<Vec<KahePoly>> = (0..n)
        .map(|i| {
            let mut polys = if i < active {
                channel::encode_symbols(&mut rng, &ch, &symbol_payloads[i])
            } else {
                channel::cover(&ch)
            };
            polys.resize(ch_polys, KahePoly::default());
            if let Some(off) = offsets.get(i).copied().flatten() {
                polys.extend(codec::encode_at(off, msg_vector_bytes, &byte_payloads[i]));
            }
            polys.resize(n_polys, KahePoly::default());
            polys
        })
        .collect();

    let mut client_entries = Vec::with_capacity(n);
    let mut inboxes: Vec<ServerInbox> = server_ids
        .iter()
        .map(|&sid| ServerInbox {
            server_id: sid,
            items: vec![],
        })
        .collect();
    let mut sealed_inbox0: Vec<(ClientId, Vec<u8>)> = Vec::with_capacity(n);
    for (i, &cid) in client_ids.iter().enumerate() {
        let round = run_client_round(
            &mut rng,
            &pp,
            &SESSION,
            cid,
            client_polys[i].clone(),
            &servers,
        );
        client_entries.push((round.client_id, round.encrypted_message));
        for (idx, (sid, sealed)) in round.sealed_openings.into_iter().enumerate() {
            assert_eq!(sid, servers[idx].0);
            let opening = unseal_opening(&server_keys[idx], &SESSION, cid, sid, &sealed).unwrap();
            if idx == 0 {
                sealed_inbox0.push((cid, sealed));
            }
            inboxes[idx].items.push((cid, opening));
        }
    }
    let canonical = client_ids.clone();
    let outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).unwrap())
        .collect();

    // Wire ledger. Ciphertext + commitment (root) are uniform-mod-q → tight at
    // ⌈log₂ q⌉. Openings are decomposed digits + small r/s, packed at
    // per-region bit widths; the client→server form is the REAL ML-KEM-sealed
    // envelope, measured off the wire bytes.
    let open_env_b = sealed_inbox0[0].1.len();
    // Server-posted aggregate, at the same bounds the servers would pack with.
    let rho = canonical.len() as u32;
    let (r_b, s_b, t_b) = aggregated_opening_pack_bounds(&pp.cs, rho);
    let agg_open_b = outputs[0].agg_open.pack(r_b, s_b, t_b).to_bytes().len();
    let agg_share_b = pack_cs_shares(std::slice::from_ref(&outputs[0].agg_share)).len();
    // Pins the wire-planning formula to what packing actually emits.
    assert_eq!(
        agg_open_b + agg_share_b,
        pp.cs.aggregated_server_crypto_len(rho)
    );
    let comm_client_b = poly_packed_len(HVC_MODULUS);
    let ctxt_client_b = n_polys * poly_packed_len64(KAHE_MODULUS);

    // [M] per-phase CPU, median of REPS one-party runs each.
    let (enc_app, _) = measure(|| {
        let mut p = channel::encode_symbols(&mut rng, &ch, &symbol_payloads[0]);
        if let Some(&(_, off)) = allocated.first() {
            p.extend(codec::encode_at(off, msg_vector_bytes, &byte_payloads[0]));
        }
        p
    });
    let (kahe_keygen_t, key0) = measure(|| kahe_keygen(&mut rng, &pp));
    let (kahe_enc, _ctxt0) = measure(|| kahe_encrypt(&mut rng, &pp, &key0, &client_polys[0]));
    let (share, shares0) = measure(|| shamir_share(&mut rng, &pp, &key0));
    let (cs, (comm0, openings0)) = measure(|| cs_commit(&mut rng, &pp, &shares0));
    // One position of one commitment — the `VC.Vf` unit. `v_open` below times the
    // verifier's S-way parallel pass instead, so the two are not interchangeable.
    let (v_one_open, one_open_ok) =
        measure(|| HidingMerkleCommitment::verify(&pp.cs, &comm0, &openings0[0]));
    assert!(one_open_ok, "fresh opening must verify");
    let (seal, _) =
        measure(|| seal_openings(&mut rng, &pp, &SESSION, client_ids[0], &openings0, &servers));
    let (unseal, _) =
        measure(|| unseal_openings(&server_keys[0], &SESSION, servers[0].0, &sealed_inbox0));
    let (server, _) = measure(|| run_server_round(&inboxes[0], &canonical).unwrap());

    // Verify: REPS full runs, field-wise medians over the returned timings.
    // Recovery is ATTEMPTED, not asserted: a decode to garbage still yields
    // timings; an Err on every rep leaves the verifier phases at zero.
    let mut vts: Vec<VerifyTimings> = Vec::with_capacity(REPS);
    let mut recovered: Option<Vec<KahePoly>> = None;
    for _ in 0..REPS {
        if let Ok((rec, vt)) =
            aggregate_and_decrypt_unverified_timed(&pp, &canonical, &client_entries, &outputs)
        {
            vts.push(vt);
            recovered = Some(rec);
        }
    }
    if vts.is_empty() {
        eprintln!("  verify errored on all {REPS} reps; verifier phases recorded as zero");
    }
    let vstat = |f: fn(&VerifyTimings) -> f64| {
        if vts.is_empty() {
            Stat::default()
        } else {
            stat(vts.iter().map(f).collect())
        }
    };
    let v_agg_ctxt = vstat(|v| v.agg_ctxt_us);
    let v_sum_comm = vstat(|v| v.sum_comm_us);
    let v_open = vstat(|v| v.opening_verify_us);
    let v_interp = vstat(|v| v.interpolation_us);
    let v_kahe_dec = vstat(|v| v.kahe_dec_us);

    // `decode` returns the multiset sorted lexicographically; compare against
    // the inserted payloads by value, not just by count.
    let expected_payloads = {
        let mut p = symbol_payloads.clone();
        p.sort();
        p
    };
    let check_decode = |rec: &[KahePoly]| -> bool {
        let elements_ok = channel::decode_symbols(&ch, &rec[..ch_polys], None)
            .map(|d| d == expected_payloads)
            .unwrap_or(false);
        if sched {
            elements_ok
                && codec::decode_ranges(&rec[ch_polys..n_polys], &ranges)
                    .map(|d| d == range_payloads)
                    .unwrap_or(false)
        } else {
            elements_ok
        }
    };
    let recovered_ok = recovered.as_deref().map(check_decode).unwrap_or(false);

    let dec_app = match &recovered {
        None => Stat::default(),
        // May Err (PeelStalled) on unrecoverable params — we time the attempt.
        Some(rec) => {
            measure(|| {
                let _ = channel::decode_symbols(&ch, &rec[..ch_polys], None);
                if sched {
                    let _ = codec::decode_ranges(&rec[ch_polys..n_polys], &ranges);
                }
            })
            .0
        }
    };

    // Aggregated flow, measured live: one layer of ⌈√ρ⌉-sized groups, then the
    // leader sums the group aggregates and decrypts. Recovery is re-checked
    // end-to-end through the tree.
    let agg = {
        let g = agg_group_size(n);
        let n_l1 = n.div_ceil(g);
        let l1_groups: Vec<Vec<_>> = (0..n_l1)
            .map(|grp| {
                client_entries
                    .iter()
                    .filter(|(cid, _)| cid.0 as usize % n_l1 == grp)
                    .cloned()
                    .collect()
            })
            .collect();
        let (agg_cpu, _) = measure(|| run_aggregator_round(&l1_groups[0]));
        let l1: Vec<_> = l1_groups.iter().map(|g| run_aggregator_round(g)).collect();
        let ctxts: Vec<Vec<KahePoly>> = l1.iter().map(|a| a.summed_ctxt.clone()).collect();
        let comms: Vec<Commitment> = l1.iter().map(|a| a.summed_comm.clone()).collect();

        let (leader, res) = measure(|| {
            let total_ctxt = Kahe::agg_ctxt(&ctxts);
            let total_comm = HidingMerkleCommitment::sum_commitments(&comms);
            decrypt_unverified_aggregate(&pp, &total_ctxt, &total_comm, &outputs)
        });
        let recovered = res.map(|rec| check_decode(&rec)).unwrap_or(false);

        AggPlan {
            group_size: g,
            n_l1,
            agg_cpu,
            leader,
            recovered,
        }
    };

    // RS-sharded ingress, measured live: its own `pp` (μ = n_polys + 1, the
    // extra slot holding Enc(H(m))), its own client rounds, one node round per
    // lane, and the full verify. The ciphertext never reaches the bulletin, so
    // the comparison against the direct/aggregated flows is per-role wire, not
    // just CPU.
    let rs_specs = rs_specs();
    if !rs_specs.is_empty() {
        assert!(
            n <= RS_RHO_MAX,
            "RS lane posts and digest params are budgeted for rho <= {RS_RHO_MAX}, got {n}"
        );
    }
    let rs_plans: Vec<RsPlan> = rs_specs
        .iter()
        .map(|&(spec_k, spec_n)| {
            // Every threshold server is also a lane, so n ≥ S; the pair fixes
            // n−k, not k.
            let n_nodes = spec_n.max(s);
            let k = n_nodes - (spec_n - spec_k);
            let mut rrng = ChaCha20Rng::from_seed(rs_seed(&seed, k, n_nodes));
            let pp_rs = ProtocolParams::setup_rs_mode(
                &mut rrng, s, n_polys, k, n_nodes, t_modulus, RS_RHO_MAX, [0x42; 32],
            );
            let rs_keys: Vec<pke::PrivateKey> = (0..s)
                .map(|_| pke::PrivateKey::generate(&mut rrng))
                .collect();
            let rs_servers: Vec<(ServerId, pke::PublicKey)> = server_ids
                .iter()
                .map(|&sid| (sid, rs_keys[sid.0 as usize].public()))
                .collect();

            let rounds: Vec<_> = client_ids
                .iter()
                .enumerate()
                .map(|(i, &cid)| {
                    let sk = SigningKey::generate(&mut rrng);
                    run_client_round_rs(
                        &mut rrng,
                        &pp_rs,
                        &SESSION,
                        cid,
                        client_polys[i].clone(),
                        &rs_servers,
                        &sk,
                    )
                })
                .collect();
            let entries: Vec<(ClientId, RsClientBulletinEntry)> = rounds
                .iter()
                .map(|r| (r.client_id, r.bulletin.clone()))
                .collect();
            // Pin the [D] RS wire formulas to the real objects.
            assert_eq!(
                entries[0].1.to_bytes().len(),
                RsClientBulletinEntry::packed_len()
            );

            let sign_key = SigningKey::generate(&mut rrng);
            let (client_sign, _) = measure(|| {
                sign_key.sign(&RsClientBulletinEntry::signing_bytes(
                    &SESSION,
                    client_ids[0],
                    &entries[0].1.comm,
                    &entries[0].1.share_root,
                ))
            });

            let rs_outputs: Vec<_> = (0..s)
                .map(|j| {
                    let items = rounds
                        .iter()
                        .map(|r| {
                            let (sid, sealed) = &r.sealed_openings[j];
                            let op =
                                unseal_opening(&rs_keys[j], &SESSION, r.client_id, *sid, sealed)
                                    .unwrap();
                            (r.client_id, op)
                        })
                        .collect();
                    run_server_round(
                        &ServerInbox {
                            server_id: server_ids[j],
                            items,
                        },
                        &canonical,
                    )
                    .unwrap()
                })
                .collect();

            let scp = pp_rs.share_comm.as_ref().unwrap();
            let roots: Vec<(ClientId, HVCPoly)> = entries
                .iter()
                .map(|(cid, e)| (*cid, e.share_root))
                .collect();
            let node_inboxes: Vec<RsNodeInbox> = (0..n_nodes)
                .map(|j| RsNodeInbox {
                    node_id: NodeId(j as u32),
                    items: rounds
                        .iter()
                        .map(|r| {
                            (
                                r.client_id,
                                r.rs_shares[j].clone(),
                                r.share_paths[j].clone(),
                            )
                        })
                        .collect(),
                })
                .collect();
            let (node_sum, _) =
                measure(|| run_rs_node_round(scp, &node_inboxes[0], &canonical, &roots).unwrap());
            // The per-client half of the lane round: decompose + bound-check,
            // no hashing. Same rayon shape, so it is a breakdown of `node_sum`.
            let (lane_ingest, _) = measure(|| {
                node_inboxes[0]
                    .items
                    .par_iter()
                    .filter(|(_, share, path)| open_share(scp, 0, share, path).is_none())
                    .count()
            });
            // What naming a culprit costs, charged only when the aggregate fails.
            let (lane_attribute, _) = measure(|| {
                node_inboxes[0]
                    .items
                    .par_iter()
                    .filter(|(cid, share, path)| {
                        let root = roots.iter().find(|(c, _)| c == cid).unwrap().1;
                        ingest_share(scp, &root, 0, share, path).is_none()
                    })
                    .count()
            });
            let node_outputs: Vec<RsNodeBulletinEntry> = node_inboxes
                .iter()
                .map(|inb| run_rs_node_round(scp, inb, &canonical, &roots).unwrap())
                .collect();
            assert_eq!(node_outputs[0].share_sum.len(), scp.block_len);
            assert_eq!(
                rounds[0].share_paths[0].nodes.len(),
                scp.path_len() * 2 * HVC_WIDTH
            );

            // Full encryption directly into persistent NTT channels. Encoding
            // is (n−k)·n_polys·N modmuls.
            let key0 = kahe_keygen(&mut rrng, &pp_rs);
            let (rns_encrypt, embedded0) =
                measure(|| Kahe::enc_ntt(&mut rrng, &pp_rs.kahe, &key0, &client_polys[0]));
            let rs_params = pp_rs.rs.as_ref().unwrap();
            let (rs_enc, shares0) = measure(|| Rs::encode(rs_params, &embedded0));
            let (share_commit, _) = measure(|| commit_shares(scp, &shares0));
            let share_b = shares0[0].len() * rs_poly_packed_len();

            let run_verify = |nodes: &[RsNodeBulletinEntry]| {
                aggregate_and_decrypt_rs(&pp_rs, &SESSION, &canonical, &entries, &rs_outputs, nodes)
            };
            let mut rvts: Vec<RsVerifyTimings> = Vec::with_capacity(REPS);
            let mut rs_recovered: Option<Vec<KahePoly>> = None;
            for _ in 0..REPS {
                if let Ok((rec, vt)) = run_verify(&node_outputs) {
                    rvts.push(vt);
                    rs_recovered = Some(rec);
                }
            }
            if rvts.is_empty() {
                eprintln!(
                    "  rs verify (k={k}/n={n_nodes}) errored on all {REPS} reps; \
                     verifier phases recorded as zero"
                );
            }
            let rstat = |f: fn(&RsVerifyTimings) -> f64| {
                if rvts.is_empty() {
                    Stat::default()
                } else {
                    stat(rvts.iter().map(f).collect())
                }
            };

            // Lane outage: shift the sample set off the identity so the
            // reconstruction actually interpolates.
            let shifted: Vec<RsNodeBulletinEntry> = node_outputs[1..].to_vec();
            let mut lag: Vec<f64> = Vec::with_capacity(REPS);
            for _ in 0..REPS {
                if let Ok((_, vt)) = run_verify(&shifted) {
                    lag.push(vt.reconstruct_us);
                }
            }

            // A tampered lane post must be attributed, not just detected: a
            // wrong share_sum no longer hashes to the labels its opening
            // projects to, so the Ajtai check names the lane.
            let mut lying = node_outputs.clone();
            lying[0].share_sum[0] = lying[0].share_sum[0] + lying[0].share_sum[0];
            let lane_lie_caught = match run_verify(&lying) {
                Err(VerifyError::LaneOpeningFailed(ref v)) if v == &[NodeId(0)] => "lane-proof",
                Err(_) => "other",
                Ok(_) => "MISSED",
            };

            RsPlan {
                k,
                n_nodes,
                share_b,
                path_b: fresh_path_packed_len(n_nodes),
                lane_post_b: lane_post_packed_len(scp.block_len, n_nodes, RS_RHO_MAX),
                bulletin_b: RsClientBulletinEntry::packed_len(),
                rns_encrypt,
                share_commit,
                rs_enc,
                client_sign,
                node_sum,
                lane_ingest,
                lane_attribute,
                v_sig: rstat(|v| v.sig_verify_us),
                v_open: rstat(|v| v.opening_verify_us),
                v_interp: rstat(|v| v.interpolation_us),
                v_lane_open: rstat(|v| v.lane_open_us),
                v_crosscheck: rstat(|v| v.crosscheck_us),
                v_reconstruct: rstat(|v| v.reconstruct_us),
                v_reconstruct_lagrange: if lag.is_empty() {
                    Stat::default()
                } else {
                    stat(lag)
                },
                v_kahe_dec: rstat(|v| v.kahe_dec_us),
                recovered: rs_recovered.as_deref().map(check_decode).unwrap_or(false),
                lane_lie_caught,
            }
        })
        .collect();

    Row {
        s,
        n,
        active,
        cover,
        iblt_cells,
        flow,
        payload_client_b: payload_bits / 8,
        mu_kahe,
        delta,
        xi: ch.payload_symbols(),
        n_polys,
        env: env_tag(),
        threads: rayon::current_num_threads(),
        affinity: affinity(),
        recovered_ok,
        // Every phase is measured; only the ones this run owns are recorded, so a
        // split run and a whole run execute identical code.
        enc_app: owned(roles.client, enc_app),
        kahe_keygen: owned(roles.client, kahe_keygen_t),
        kahe_enc: owned(roles.client, kahe_enc),
        share: owned(roles.client, share),
        cs: owned(roles.client, cs),
        seal: owned(roles.client, seal),
        unseal: owned(roles.server, unseal),
        server: owned(roles.server, server),
        v_agg_ctxt: owned(roles.verifier, v_agg_ctxt),
        v_sum_comm: owned(roles.verifier, v_sum_comm),
        v_open: owned(roles.verifier, v_open),
        v_interp: owned(roles.verifier, v_interp),
        v_kahe_dec: owned(roles.verifier, v_kahe_dec),
        dec_app: owned(roles.verifier, dec_app),
        v_one_open: owned(roles.verifier, v_one_open),
        open_env_b,
        comm_client_b,
        ctxt_client_b,
        kahe_key_b: poly_packed_len64(KAHE_MODULUS),
        plaintext_b: n_polys * POLY_N * BITS_PER_SYMBOL / 8,
        agg_open_b,
        agg_share_b,
        // ρ = every client's ciphertext is summed, cover included.
        eps_correct: correctness_epsilon(n, n_polys, t_modulus),
        useful_b: (delivered * payload_bits) as f64 / 8.0,
        wire_ctxt_b: n as f64 * ctxt_client_b as f64,
        wire_comm_b: n as f64 * poly_packed_len(HVC_MODULUS) as f64,
        wire_opening_b: (n * s) as f64 * open_env_b as f64,
        wire_server_b: s as f64 * (agg_open_b + agg_share_b) as f64,
        agg: AggPlan {
            agg_cpu: owned(roles.agg, agg.agg_cpu),
            leader: owned(roles.agg, agg.leader),
            ..agg
        },
        rs_plans: rs_plans
            .into_iter()
            .map(|mut p| {
                for st in [
                    &mut p.rns_encrypt,
                    &mut p.share_commit,
                    &mut p.rs_enc,
                    &mut p.client_sign,
                    &mut p.node_sum,
                    &mut p.lane_ingest,
                    &mut p.lane_attribute,
                    &mut p.v_sig,
                    &mut p.v_open,
                    &mut p.v_interp,
                    &mut p.v_lane_open,
                    &mut p.v_crosscheck,
                    &mut p.v_reconstruct,
                    &mut p.v_reconstruct_lagrange,
                    &mut p.v_kahe_dec,
                ] {
                    *st = owned(roles.rs, *st);
                }
                p
            })
            .collect(),
    }
}

/// `SWEEP_RS="14x16,32x34"`; empty string skips the RS section entirely.
fn rs_specs() -> Vec<(usize, usize)> {
    match std::env::var("SWEEP_RS") {
        Err(_) => RS_PLANS.to_vec(),
        Ok(raw) if raw.trim().is_empty() => vec![],
        Ok(raw) => raw
            .split(',')
            .map(|c| {
                let (k, n) = c
                    .trim()
                    .split_once('x')
                    .unwrap_or_else(|| panic!("bad SWEEP_RS pair {c:?} (want KxN)"));
                (
                    k.parse().unwrap_or_else(|_| panic!("bad k in {c:?}")),
                    n.parse().unwrap_or_else(|_| panic!("bad n in {c:?}")),
                )
            })
            .collect(),
    }
}

fn rs_seed(cell: &[u8; 32], k: usize, n_nodes: usize) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"panetiere/bench/rs");
    h.update(cell);
    h.update((k as u64).to_le_bytes());
    h.update((n_nodes as u64).to_le_bytes());
    h.finalize().into()
}

// ── [C] cost model over [M] medians ─────────────────────────────────────────

struct Model {
    fixed_us: f64,
    /// Payload-proportional work: everything that scales with the plaintext width.
    payload_us: f64,
    wall_us: f64,
    wire_total_b: f64,
    efficiency: f64,
    post_b: f64,        // one client bulletin post = comm + ctxt
    client_open_b: f64, // one client's S sealed openings
    agg_cpu_us: f64,
    rs_cpu_us: Vec<f64>, // per RsPlan
}

fn model(r: &Row) -> Model {
    // fixed = paid once per round (key-related); payload = scales with the width.
    let fixed_us = r.kahe_keygen.med
        + r.share.med
        + r.cs.med
        + r.seal.med
        + r.unseal.med
        + r.server.med
        + r.v_open.med
        + r.v_interp.med
        + r.v_sum_comm.med;
    let payload_us =
        r.enc_app.med + r.kahe_enc.med + r.v_agg_ctxt.med + r.v_kahe_dec.med + r.dec_app.med;
    let wall_us = fixed_us + payload_us;
    let wire_total_b = r.wire_ctxt_b + r.wire_comm_b + r.wire_opening_b + r.wire_server_b;
    // Aggregated flow replaces the leader's direct ingest phases with the tree.
    let cpu_base_agg_us = (fixed_us - r.v_open.med - r.v_interp.med - r.v_sum_comm.med)
        + (payload_us - r.v_agg_ctxt.med - r.v_kahe_dec.med);
    let agg_cpu_us = cpu_base_agg_us + r.agg.leader.med;
    // RS flow: client pays app-encode + encrypt + RS-encode + key work; one
    // node sums its lane; the verifier no longer sums ciphertexts at all.
    let rs_cpu_us = r
        .rs_plans
        .iter()
        .map(|p| {
            let encryption_us = p.rns_encrypt.med;
            r.enc_app.med
                + r.kahe_keygen.med
                + encryption_us
                + p.rs_enc.med
                + p.share_commit.med
                + p.client_sign.med
                + r.share.med
                + r.cs.med
                + r.seal.med
                + r.unseal.med
                + r.server.med
                + p.node_sum.med
                + p.v_sig.med
                + r.v_sum_comm.med
                + p.v_open.med
                + p.v_interp.med
                + p.v_lane_open.med
                + p.v_crosscheck.med
                + p.v_reconstruct.med
                + p.v_kahe_dec.med
                + r.dec_app.med
        })
        .collect();
    Model {
        fixed_us,
        payload_us,
        wall_us,
        wire_total_b,
        efficiency: r.useful_b / wire_total_b,
        post_b: (r.wire_comm_b + r.wire_ctxt_b) / r.n as f64,
        client_open_b: r.wire_opening_b / r.n as f64,
        agg_cpu_us,
        rs_cpu_us,
    }
}

// ── [P] network sim ─────────────────────────────────────────────────────────
// Three sequential wire phases per round, kept separate from CPU wall:
//   A  client upload — N parallel uplinks; each endpoint serializes its own
//      bytes at its class rate (a client's post and openings share its NIC).
//      Gated by the slowest of: a client uplink, a server downlink (ingesting
//      N openings), the bulletin ingest (N posts).
//   B  server post — S entries onto the bulletin; gated by the slower of a
//      server's uplink and the bulletin ingest.
//   C  verifier read — the full bulletin over the verifier's downlink.
// No compute/transfer overlap is modelled, so e2e = CPU wall + net is the
// conservative end of pipelined reality. Deterministic jitter seed per row.

struct NetPoint {
    direct_net_us: f64,
    direct_e2e_us: f64,
    agg: (f64, f64),     // (net_us, e2e_us)
    rs: Vec<(f64, f64)>, // (net_us, e2e_us) per RsPlan
}

fn net_sim(r: &Row, m: &Model, prof: &NetProfile, nrng: &mut ChaCha20Rng) -> NetPoint {
    // Mbit/s → bytes/µs, one closure per link class.
    let xfer_cl = |b: f64| b / (prof.client_mbps / 8.0);
    let xfer_srv = |b: f64| b / (prof.server_mbps / 8.0);
    let xfer_bul = |b: f64| b / (prof.bulletin_mbps / 8.0);
    let mut maxlat = |k: usize| -> f64 {
        (0..k)
            .map(|_| prof.lat_ms + nrng.gen::<f64>() * prof.jitter_ms)
            .fold(0.0, f64::max)
            * 1e3
    };

    let a = (maxlat(r.n) + xfer_cl(m.post_b + m.client_open_b)) // client uplink
        .max(maxlat(r.n) + xfer_srv(r.wire_opening_b / r.s as f64)) // server downlink
        .max(maxlat(r.n) + xfer_bul(r.n as f64 * m.post_b)); // bulletin ingest
    let b = (maxlat(r.s) + xfer_srv(r.wire_server_b / r.s as f64)) // server uplink
        .max(maxlat(r.s) + xfer_bul(r.wire_server_b)); // bulletin ingest
    let c = maxlat(1) + xfer_bul(r.wire_comm_b + r.wire_server_b + r.wire_ctxt_b);
    let direct_net_us = a + b + c;

    // Aggregated flow: ctxt+comm go to aggregators (not broadcast), which sum g
    // posts each and forward one aggregate to the leader. Openings→servers and
    // the server post are unchanged. Conservative (no overlap).
    let agg = {
        let g = r.agg.group_size as f64;
        let a = (maxlat(r.n) + xfer_cl(m.post_b + m.client_open_b)) // client uplink
            .max(maxlat(r.n) + xfer_srv(r.wire_opening_b / r.s as f64)) // server downlink
            .max(maxlat(r.agg.n_l1) + xfer_srv(g * m.post_b)); // aggregator ingest
        let hop = r.agg.agg_cpu.med
            + (maxlat(r.agg.n_l1) + xfer_srv(m.post_b)) // aggregator uplink
                .max(maxlat(1) + xfer_bul(r.agg.n_l1 as f64 * m.post_b)); // leader ingest
        let b = (maxlat(r.s) + xfer_srv(r.wire_server_b / r.s as f64)) // server uplink
            .max(maxlat(r.s) + xfer_bul(r.wire_server_b)); // bulletin ingest
        let c = maxlat(1) + xfer_bul(r.wire_server_b);
        let net_us = a + hop + b + c;
        (net_us, m.agg_cpu_us + net_us)
    };

    // RS flow: the client uplinks n coded shares (total (n/k)·C) plus its S
    // openings; the bulletin only ever sees the constant-size post. A node
    // ingests ρ shares of C/k. The verifier reads every lane post (k
    // reconstruct, all carry a proof), the server entries and every client's
    // bulletin post (it must check the signatures and sum the digests).
    let rs = r
        .rs_plans
        .iter()
        .zip(&m.rs_cpu_us)
        .map(|(p, &rs_cpu_us)| {
            // Each share travels with its Merkle path, so the lane can verify
            // it against the client's signed root before summing.
            let lane_item_b = (p.share_b + p.path_b) as f64;
            let client_shares_b = p.n_nodes as f64 * lane_item_b;
            let node_in_b = r.n as f64 * lane_item_b;
            let client_out_b = client_shares_b + p.bulletin_b as f64 + m.client_open_b;
            let a = (maxlat(r.n) + xfer_cl(client_out_b)) // client uplink
                .max(maxlat(r.n) + xfer_srv(r.wire_opening_b / r.s as f64)) // server downlink
                .max(maxlat(r.n) + xfer_srv(node_in_b)) // one lane's ingest
                .max(maxlat(r.n) + xfer_bul(r.n as f64 * p.bulletin_b as f64)); // bulletin ingest
            let lane_posts_b = p.n_nodes as f64 * p.lane_post_b as f64;
            let b = (maxlat(r.s) + xfer_srv(r.wire_server_b / r.s as f64)) // server uplink
                .max(maxlat(p.n_nodes) + xfer_srv(p.lane_post_b as f64)) // lane uplink
                .max(maxlat(p.n_nodes) + xfer_bul(lane_posts_b + r.wire_server_b)); // bulletin ingest
            let c = maxlat(1)
                + xfer_bul(lane_posts_b + r.wire_server_b + r.n as f64 * p.bulletin_b as f64);
            let net_us = a + b + c;
            (net_us, rs_cpu_us + net_us)
        })
        .collect();

    NetPoint {
        direct_net_us,
        direct_e2e_us: m.wall_us + direct_net_us,
        agg,
        rs,
    }
}

// Deterministic per row, and reseeded per profile so every profile sees the
// same latency draws — profiles then differ only by bandwidth. Tables and CSV
// call this independently and get identical numbers.
fn net_all(r: &Row, m: &Model) -> Vec<NetPoint> {
    NETWORKS
        .iter()
        .map(|prof| net_sim(r, m, prof, &mut ChaCha20Rng::from_seed([0x5E; 32])))
        .collect()
}

// ── tables ──────────────────────────────────────────────────────────────────

// The composite cells are pre-formatted so the outer width specifiers pad them.
#[allow(clippy::format_in_format_args)]
fn print_tables(rows: &[Row]) {
    let models: Vec<Model> = rows.iter().map(model).collect();

    println!("── cells ───────────────────────────────────────────────────────────────────");
    println!(
        "{:<4}{:>4}{:>6}{:>6}{:>6}  {:<6}{:>12}{:>6}{:>7}{:>8}{:>7}{:>7}  recovered",
        "id", "S", "ρ", "act", "cov", "flow", "payload/cl", "δ", "ξ", "polys", "μ", "cells"
    );
    for (i, r) in rows.iter().enumerate() {
        println!(
            "{:<4}{:>4}{:>6}{:>6}{:>6}  {:<6}{:>12}{:>6}{:>7}{:>8}{:>7}{:>7}  {}",
            cell_id(i),
            r.s,
            r.n,
            r.active,
            r.cover,
            r.flow,
            fmt_bytes(r.payload_client_b as f64),
            r.delta,
            r.xi,
            r.n_polys,
            r.mu_kahe,
            r.iblt_cells,
            if r.recovered_ok { "yes" } else { "NO" },
        );
    }
    println!();

    println!("── per role, one party of each kind ([M] cpu, [D] wire, eps = correctness) ──");
    println!(
        "{:<4}{:>11}{:>10}{:>6}  {:>11}{:>11}{:>12}  {:>12}{:>12}{:>7}",
        "id",
        "msg/cl",
        "ρ/act",
        "S",
        "client",
        "server",
        "recipient",
        "client →",
        "server →",
        "eps"
    );
    for (i, r) in rows.iter().enumerate() {
        println!(
            "{:<4}{:>11}{:>10}{:>6}  {:>11}{:>11}{:>12}  {:>12}{:>12}{:>7}",
            cell_id(i),
            fmt_bytes(r.payload_client_b as f64),
            format!("{}/{}", r.n, r.active),
            r.s,
            fmt_us(client_us(r)),
            fmt_us(server_us(r)),
            fmt_us(recipient_us(r)),
            fmt_bytes(client_post_b(r) as f64),
            fmt_bytes(server_entry_b(r) as f64),
            fmt_eps(r.eps_correct),
        );
    }
    println!("    (client → = comm + ctxt + S·open_env; server → = agg_open + agg_share)");
    println!();

    println!("── components: vector commitment | KAHE ([M] times, [D] sizes) ──────────────");
    println!(
        "{:4}{:─^42} {:─^57}",
        "", " vector commitment (γ = S) ", " KAHE "
    );
    println!(
        "{:<4}{:>11}{:>11}{:>10}{:>10} {:>12}{:>12}{:>11}{:>11}{:>11}",
        "id", "comm", "opening", "com", "vf", "plaintext", "ctxt", "key", "enc", "dec"
    );
    for (i, r) in rows.iter().enumerate() {
        println!(
            "{:<4}{:>11}{:>11}{:>10}{:>10} {:>12}{:>12}{:>11}{:>11}{:>11}",
            cell_id(i),
            fmt_bytes(r.comm_client_b as f64),
            fmt_bytes((r.open_env_b - pke::SEAL_OVERHEAD) as f64),
            fmt_us(r.cs.med),
            fmt_us(r.v_one_open.med),
            fmt_bytes(r.plaintext_b as f64),
            fmt_bytes(r.ctxt_client_b as f64),
            fmt_bytes(r.kahe_key_b as f64),
            fmt_us(r.kahe_enc.med),
            fmt_us(r.v_kahe_dec.med),
        );
    }
    println!(
        "    (opening is the packed body, sealed adds {} B of ML-KEM; `com` commits all S \
         positions, `vf` opens one)",
        pke::SEAL_OVERHEAD
    );
    println!();

    println!(
        "── [M] cpu — per-round totals for one party, median of {} (min/max in csv) ────",
        REPS
    );
    println!(
        "{:4}{:─^60} {:─^20} {:─^60}",
        "", " client ", " server ", " verifier "
    );
    println!(
        "{:<4}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10} {:>10}{:>10} {:>10}{:>10}{:>10}{:>10}{:>10}{:>10}",
        "id",
        "enc_app",
        "keygen",
        "kahe_enc",
        "share",
        "cs",
        "seal",
        "unseal",
        "round",
        "agg_ctxt",
        "sum_comm",
        "open",
        "interp",
        "kahe_dec",
        "dec_app"
    );
    for (i, r) in rows.iter().enumerate() {
        println!(
            "{:<4}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10} {:>10}{:>10} {:>10}{:>10}{:>10}{:>10}{:>10}{:>10}",
            cell_id(i),
            fmt_us(r.enc_app.med),
            fmt_us(r.kahe_keygen.med),
            fmt_us(r.kahe_enc.med),
            fmt_us(r.share.med),
            fmt_us(r.cs.med),
            fmt_us(r.seal.med),
            fmt_us(r.unseal.med),
            fmt_us(r.server.med),
            fmt_us(r.v_agg_ctxt.med),
            fmt_us(r.v_sum_comm.med),
            fmt_us(r.v_open.med),
            fmt_us(r.v_interp.med),
            fmt_us(r.v_kahe_dec.med),
            fmt_us(r.dec_app.med),
        );
    }
    println!();

    println!("── [M] cpu — aggregated flow (per aggregator / leader) ──────────────────────");
    println!(
        "{:<4}{:>4}  {:<16}{:>11}{:>13}  recovered",
        "id", "g", "fan-in", "agg", "leader"
    );
    for (i, r) in rows.iter().enumerate() {
        let p = &r.agg;
        println!(
            "{:<4}{:>4}  {:<16}{:>11}{:>13}  {}",
            cell_id(i),
            p.group_size,
            format!("{}→{}→1", r.n, p.n_l1),
            fmt_us(p.agg_cpu.med),
            fmt_us(p.leader.med),
            if p.recovered { "yes" } else { "NO" },
        );
    }
    println!();

    if rows.iter().any(|r| !r.rs_plans.is_empty()) {
        println!("── [M] cpu — rs-sharded ingress (client commit / lane round / verifier) ─────");
        println!(
            "{:<4}{:>8}  {:>10}{:>10}{:>11}{:>10}{:>12}{:>11}{:>12} {:>10}{:>10}{:>11}{:>12}{:>12}{:>10}  {:<10}lane lie",
            "id", "k/n", "embed", "rs_enc", "commit", "sign", "node_sum", "open+fold", "attribute",
            "sig", "open", "lane_open", "crosscheck", "reconstruct", "kahe_dec", "recovered"
        );
        for (i, r) in rows.iter().enumerate() {
            for p in &r.rs_plans {
                println!(
                    "{:<4}{:>8}  {:>10}{:>10}{:>11}{:>10}{:>12}{:>11}{:>12} {:>10}{:>10}{:>11}{:>12}{:>12}{:>10}  {:<10}{}",
                    cell_id(i),
                    format!("{}/{}", p.k, p.n_nodes),
                    fmt_us(p.rns_encrypt.med),
                    fmt_us(p.rs_enc.med),
                    fmt_us(p.share_commit.med),
                    fmt_us(p.client_sign.med),
                    fmt_us(p.node_sum.med),
                    fmt_us(p.lane_ingest.med),
                    fmt_us(p.lane_attribute.med),
                    fmt_us(p.v_sig.med),
                    fmt_us(p.v_open.med),
                    fmt_us(p.v_lane_open.med),
                    fmt_us(p.v_crosscheck.med),
                    fmt_us(p.v_reconstruct.med),
                    fmt_us(p.v_kahe_dec.med),
                    if p.recovered { "yes" } else { "NO" },
                    p.lane_lie_caught,
                );
            }
        }
        println!(
            "    (reconstruct is the identity sample set — a concatenation. With one lane down it\n     interpolates instead: {})",
            rows.iter()
                .flat_map(|r| r.rs_plans.iter())
                .map(|p| fmt_us(p.v_reconstruct_lagrange.med))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!();

        println!("── [D] rs sizing — measured at the run k/n, priced across k at n − k = 2 ────");
        println!("    (k clamped to ≥ S−2: every threshold server is a lane, so n ≥ S)");
        println!(
            "{:<4}{:>5}{:>5}{:>14}{:>10}{:>12}{:>12}{:>16}{:>15}{:>14}  gate",
            "id",
            "k",
            "n",
            "share = C/k",
            "path",
            "lane post",
            "bulletin",
            "client egress",
            "node ingress",
            "verifier in"
        );
        for (i, r) in rows.iter().enumerate() {
            if r.rs_plans.is_empty() {
                continue;
            }
            let bulletin_b = r.rs_plans[0].bulletin_b as f64;
            let open_b = (r.open_env_b * r.s) as f64;
            let ctxt_b = r.ctxt_client_b as f64;
            let mut priced: Vec<usize> = Vec::new();
            for &k_req in RS_SIZING_K {
                let n_nodes = (k_req + 2).max(r.s);
                let k = n_nodes - 2;
                if priced.contains(&k) {
                    continue;
                }
                let block_len = r.n_polys.div_ceil(k);
                let share_b = (block_len * rs_poly_packed_len()) as f64;
                // The path rides with the share; the lane's post is its proof.
                let path_b = fresh_path_packed_len(n_nodes) as f64;
                let lane_post_b = lane_post_packed_len(block_len, n_nodes, RS_RHO_MAX) as f64;
                let egress = n_nodes as f64 * (share_b + path_b) + open_b + bulletin_b;
                let node_in = r.n as f64 * (share_b + path_b);
                let verifier_in =
                    n_nodes as f64 * lane_post_b + r.n as f64 * bulletin_b + r.wire_server_b;
                let gate = if egress / NETWORKS[0].client_mbps > node_in / NETWORKS[0].server_mbps {
                    "client NIC"
                } else {
                    "node ingest"
                };
                println!(
                    "{:<4}{:>5}{:>5}{:>14}{:>10}{:>12}{:>12}{:>16}{:>15}{:>14}  {}",
                    if priced.is_empty() {
                        cell_id(i)
                    } else {
                        String::new()
                    },
                    k,
                    n_nodes,
                    fmt_bytes(share_b),
                    fmt_bytes(path_b),
                    fmt_bytes(lane_post_b),
                    fmt_bytes(bulletin_b),
                    fmt_bytes(egress),
                    fmt_bytes(node_in),
                    fmt_bytes(verifier_in),
                    gate,
                );
                priced.push(k);
            }
            println!(
                "    (broadcast baseline: client egress {}, per-node ingress {})",
                fmt_bytes(ctxt_b + open_b + r.comm_client_b as f64),
                fmt_bytes(r.n as f64 * ctxt_b),
            );
        }
        println!();
    }

    println!("── wire components, per emitter ([M] off the wire except comm/ctxt, [D]) ────");
    println!(
        "{:<4}{:>10}{:>15}   {:<30}srv_entry = agg_open + agg_share",
        "id", "comm/cl", "ctxt/cl", "open_env →1srv (×S /cl)"
    );
    for (i, r) in rows.iter().enumerate() {
        println!(
            "{:<4}{:>10}{:>15}   {:<30}{}",
            cell_id(i),
            fmt_bytes(r.comm_client_b as f64),
            fmt_bytes(r.ctxt_client_b as f64),
            format!(
                "{} (×{} = {})",
                fmt_bytes(r.open_env_b as f64),
                r.s,
                fmt_bytes((r.open_env_b * r.s) as f64)
            ),
            format!(
                "{} = {} + {}",
                fmt_bytes((r.agg_open_b + r.agg_share_b) as f64),
                fmt_bytes(r.agg_open_b as f64),
                fmt_bytes(r.agg_share_b as f64)
            ),
        );
    }
    println!();

    println!("── [D] per-role wire, one round (← in / → out) ──────────────────────────────");
    println!(
        "{:<4}{:<44}{:<24}{:<24}leader ← direct / agg-1",
        "id", "client → (comm + ctxt + S·open)", "server ← / →", "L1-agg ← / →"
    );
    for (i, r) in rows.iter().enumerate() {
        let m = &models[i];
        let client_ctxt = r.wire_ctxt_b / r.n as f64;
        let client_out = r.comm_client_b as f64 + client_ctxt + m.client_open_b;
        let l1_in = r.agg.group_size as f64 * m.post_b;
        let leader_agg = r.wire_server_b + r.agg.n_l1 as f64 * m.post_b;
        println!(
            "{:<4}{:<44}{:<24}{:<24}{}",
            cell_id(i),
            format!(
                "{} ({} + {} + {})",
                fmt_bytes(client_out),
                fmt_bytes(r.comm_client_b as f64),
                fmt_bytes(client_ctxt),
                fmt_bytes(m.client_open_b)
            ),
            format!(
                "{} / {}",
                fmt_bytes(r.wire_opening_b / r.s as f64),
                fmt_bytes(r.wire_server_b / r.s as f64)
            ),
            format!("{} / {}", fmt_bytes(l1_in), fmt_bytes(m.post_b)),
            format!(
                "{} / {}",
                fmt_bytes(r.wire_comm_b + r.wire_server_b + r.wire_ctxt_b),
                fmt_bytes(leader_agg)
            ),
        );
    }
    println!();

    println!("── [C] model: wall = fixed + payload (client → server → verifier, serial) ───");
    println!(
        "{:<4}{:>10}{:>10}{:>10}{:>12}{:>13}{:>10}{:>12}",
        "id", "fixed", "payload", "wall", "useful", "wire/round", "eff", "agg1-wall"
    );
    for (i, r) in rows.iter().enumerate() {
        let m = &models[i];
        println!(
            "{:<4}{:>10}{:>10}{:>10}{:>12}{:>13}{:>10.2e}{:>12}",
            cell_id(i),
            fmt_us(m.fixed_us),
            fmt_us(m.payload_us),
            fmt_us(m.wall_us),
            fmt_bytes(r.useful_b),
            fmt_bytes(m.wire_total_b),
            m.efficiency,
            fmt_us(m.agg_cpu_us),
        );
    }
    println!();

    println!("── [P] network sim: e2e = [C] wall + synthetic net; MB/s = useful/e2e ───────");
    println!(
        "{:<4}{:<9}{:>12}{:>13}{:>12}{:>12}{}",
        "id",
        "net",
        "direct e2e",
        "direct MB/s",
        "agg-1 e2e",
        "agg-1 MB/s",
        if rows.iter().any(|r| !r.rs_plans.is_empty()) {
            format!("{:>12}{:>12}", "rs e2e", "rs MB/s")
        } else {
            String::new()
        },
    );
    for (i, r) in rows.iter().enumerate() {
        let nets = net_all(r, &models[i]);
        for (prof, np) in NETWORKS.iter().zip(&nets) {
            let (_, agg_e2e) = np.agg;
            let rs = np.rs.first().map_or(String::new(), |(_, e2e)| {
                format!(
                    "{:>12}{:>12.3}",
                    fmt_us(*e2e),
                    r.useful_b / (e2e / 1e6) / 1e6
                )
            });
            println!(
                "{:<4}{:<9}{:>12}{:>13.3}{:>12}{:>12.3}{}",
                cell_id(i),
                prof.label,
                fmt_us(np.direct_e2e_us),
                r.useful_b / (np.direct_e2e_us / 1e6) / 1e6,
                fmt_us(agg_e2e),
                r.useful_b / (agg_e2e / 1e6) / 1e6,
                rs,
            );
        }
    }
    println!(
        "    ({}+U[0,{})ms … per profile; client/server/bulletin Mbps: {})",
        NETWORKS[0].lat_ms,
        NETWORKS[0].jitter_ms,
        NETWORKS
            .iter()
            .map(|p| format!(
                "{} {}/{}/{}",
                p.label, p.client_mbps, p.server_mbps, p.bulletin_mbps
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

// ── csv ─────────────────────────────────────────────────────────────────────

const CSV_PATH_DEFAULT: &str = "scaling_sweep.csv";

/// Output path. Overridable so a split run can keep its halves apart — a host
/// run, a TDX run and their merge cannot share one file.
fn csv_path() -> String {
    std::env::var("BENCH_CSV").unwrap_or_else(|_| CSV_PATH_DEFAULT.to_string())
}

fn rs0(r: &Row) -> Option<&RsPlan> {
    r.rs_plans.first()
}

/// Which role owns a phase — the same grouping `client_us`/`server_us`/
/// `recipient_us` use, so gating and composing cannot disagree.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Owner {
    Client,
    Server,
    Verifier,
    Agg,
    Rs,
}

/// One timing column: its name, its owning role, and how to read and write it on
/// a `Row`. The CSV writer, the CSV reader and `merge` all drive off this table,
/// so a column cannot exist in one and not the others.
struct PhaseCol {
    name: &'static str,
    owner: Owner,
    get: fn(&Row) -> Stat,
    set: fn(&mut Row, Stat),
}

/// Setters for plan-nested phases no-op when the plan is absent, which is the
/// case whenever that flow was skipped for the run.
const PHASES: &[PhaseCol] = &[
    PhaseCol {
        name: "enc_app",
        owner: Owner::Client,
        get: |r| r.enc_app,
        set: |r, s| r.enc_app = s,
    },
    PhaseCol {
        name: "kahe_keygen",
        owner: Owner::Client,
        get: |r| r.kahe_keygen,
        set: |r, s| r.kahe_keygen = s,
    },
    PhaseCol {
        name: "kahe_enc",
        owner: Owner::Client,
        get: |r| r.kahe_enc,
        set: |r, s| r.kahe_enc = s,
    },
    PhaseCol {
        name: "share",
        owner: Owner::Client,
        get: |r| r.share,
        set: |r, s| r.share = s,
    },
    PhaseCol {
        name: "cs_commit",
        owner: Owner::Client,
        get: |r| r.cs,
        set: |r, s| r.cs = s,
    },
    PhaseCol {
        name: "seal",
        owner: Owner::Client,
        get: |r| r.seal,
        set: |r, s| r.seal = s,
    },
    PhaseCol {
        name: "unseal",
        owner: Owner::Server,
        get: |r| r.unseal,
        set: |r, s| r.unseal = s,
    },
    PhaseCol {
        name: "server_round",
        owner: Owner::Server,
        get: |r| r.server,
        set: |r, s| r.server = s,
    },
    PhaseCol {
        name: "verify_agg_ctxt",
        owner: Owner::Verifier,
        get: |r| r.v_agg_ctxt,
        set: |r, s| r.v_agg_ctxt = s,
    },
    PhaseCol {
        name: "verify_sum_comm",
        owner: Owner::Verifier,
        get: |r| r.v_sum_comm,
        set: |r, s| r.v_sum_comm = s,
    },
    PhaseCol {
        name: "verify_open",
        owner: Owner::Verifier,
        get: |r| r.v_open,
        set: |r, s| r.v_open = s,
    },
    PhaseCol {
        name: "verify_interp",
        owner: Owner::Verifier,
        get: |r| r.v_interp,
        set: |r, s| r.v_interp = s,
    },
    PhaseCol {
        name: "verify_kahe_dec",
        owner: Owner::Verifier,
        get: |r| r.v_kahe_dec,
        set: |r, s| r.v_kahe_dec = s,
    },
    PhaseCol {
        name: "verify_one_open",
        owner: Owner::Verifier,
        get: |r| r.v_one_open,
        set: |r, s| r.v_one_open = s,
    },
    PhaseCol {
        name: "dec_app",
        owner: Owner::Verifier,
        get: |r| r.dec_app,
        set: |r, s| r.dec_app = s,
    },
    PhaseCol {
        name: "agg1_level",
        owner: Owner::Agg,
        get: |r| r.agg.agg_cpu,
        set: |r, s| r.agg.agg_cpu = s,
    },
    PhaseCol {
        name: "agg1_leader",
        owner: Owner::Agg,
        get: |r| r.agg.leader,
        set: |r, s| r.agg.leader = s,
    },
    PhaseCol {
        // Stable CSV name retained for pre-RNS sweep compatibility.
        name: "rs_dgt_embed",
        owner: Owner::Rs,
        get: |r| rs0(r).map_or(Stat::default(), |p| p.rns_encrypt),
        set: |r, s| {
            if let Some(p) = r.rs_plans.first_mut() {
                p.rns_encrypt = s
            }
        },
    },
    PhaseCol {
        name: "rs_enc",
        owner: Owner::Rs,
        get: |r| rs0(r).map_or(Stat::default(), |p| p.rs_enc),
        set: |r, s| {
            if let Some(p) = r.rs_plans.first_mut() {
                p.rs_enc = s
            }
        },
    },
    PhaseCol {
        name: "rs_client_sign",
        owner: Owner::Rs,
        get: |r| rs0(r).map_or(Stat::default(), |p| p.client_sign),
        set: |r, s| {
            if let Some(p) = r.rs_plans.first_mut() {
                p.client_sign = s
            }
        },
    },
    PhaseCol {
        name: "rs_share_commit",
        owner: Owner::Rs,
        get: |r| rs0(r).map_or(Stat::default(), |p| p.share_commit),
        set: |r, s| {
            if let Some(p) = r.rs_plans.first_mut() {
                p.share_commit = s
            }
        },
    },
    PhaseCol {
        name: "rs_node_sum",
        owner: Owner::Rs,
        get: |r| rs0(r).map_or(Stat::default(), |p| p.node_sum),
        set: |r, s| {
            if let Some(p) = r.rs_plans.first_mut() {
                p.node_sum = s
            }
        },
    },
    PhaseCol {
        name: "rs_lane_ingest",
        owner: Owner::Rs,
        get: |r| rs0(r).map_or(Stat::default(), |p| p.lane_ingest),
        set: |r, s| {
            if let Some(p) = r.rs_plans.first_mut() {
                p.lane_ingest = s
            }
        },
    },
    PhaseCol {
        name: "rs_lane_attribute",
        owner: Owner::Rs,
        get: |r| rs0(r).map_or(Stat::default(), |p| p.lane_attribute),
        set: |r, s| {
            if let Some(p) = r.rs_plans.first_mut() {
                p.lane_attribute = s
            }
        },
    },
    PhaseCol {
        name: "rs_verify_sig",
        owner: Owner::Rs,
        get: |r| rs0(r).map_or(Stat::default(), |p| p.v_sig),
        set: |r, s| {
            if let Some(p) = r.rs_plans.first_mut() {
                p.v_sig = s
            }
        },
    },
    PhaseCol {
        name: "rs_verify_open",
        owner: Owner::Rs,
        get: |r| rs0(r).map_or(Stat::default(), |p| p.v_open),
        set: |r, s| {
            if let Some(p) = r.rs_plans.first_mut() {
                p.v_open = s
            }
        },
    },
    PhaseCol {
        name: "rs_verify_interp",
        owner: Owner::Rs,
        get: |r| rs0(r).map_or(Stat::default(), |p| p.v_interp),
        set: |r, s| {
            if let Some(p) = r.rs_plans.first_mut() {
                p.v_interp = s
            }
        },
    },
    PhaseCol {
        name: "rs_reconstruct",
        owner: Owner::Rs,
        get: |r| rs0(r).map_or(Stat::default(), |p| p.v_reconstruct),
        set: |r, s| {
            if let Some(p) = r.rs_plans.first_mut() {
                p.v_reconstruct = s
            }
        },
    },
    PhaseCol {
        name: "rs_reconstruct_lagrange",
        owner: Owner::Rs,
        get: |r| rs0(r).map_or(Stat::default(), |p| p.v_reconstruct_lagrange),
        set: |r, s| {
            if let Some(p) = r.rs_plans.first_mut() {
                p.v_reconstruct_lagrange = s
            }
        },
    },
    PhaseCol {
        name: "rs_verify_kahe_dec",
        owner: Owner::Rs,
        get: |r| rs0(r).map_or(Stat::default(), |p| p.v_kahe_dec),
        set: |r, s| {
            if let Some(p) = r.rs_plans.first_mut() {
                p.v_kahe_dec = s
            }
        },
    },
    PhaseCol {
        name: "rs_verify_lane_open",
        owner: Owner::Rs,
        get: |r| rs0(r).map_or(Stat::default(), |p| p.v_lane_open),
        set: |r, s| {
            if let Some(p) = r.rs_plans.first_mut() {
                p.v_lane_open = s
            }
        },
    },
    PhaseCol {
        name: "rs_verify_crosscheck",
        owner: Owner::Rs,
        get: |r| rs0(r).map_or(Stat::default(), |p| p.v_crosscheck),
        set: |r, s| {
            if let Some(p) = r.rs_plans.first_mut() {
                p.v_crosscheck = s
            }
        },
    },
];

fn write_csv(rows: &[Row]) {
    let phases = PHASES;

    let mut header = String::new();
    header.push_str("id,s,rho,active,cover,flow,payload_client_b,delta,xi,n_polys,mu_kahe,iblt_cells,env,m_threads,m_affinity,recovered,agg1_recovered,rs_k,rs_n,rs_recovered,rs_lane_lie");
    for p in phases {
        write!(header, ",m_{0}_us_med,m_{0}_us_min,m_{0}_us_max", p.name).unwrap();
    }
    header.push_str(
        ",m_open_env_b\
         ,d_comm_client_b,d_ctxt_client_b,d_agg_open_b,d_agg_share_b\
         ,d_open_body_b,d_kahe_key_b,d_plaintext_b\
         ,d_client_to_server_b,d_client_to_recipient_b,d_client_post_b,d_server_entry_b,d_epsilon\
         ,d_wire_ctxt_b,d_wire_comm_b,d_wire_opening_b,d_wire_server_b,d_useful_b\
         ,d_rs_share_b,d_rs_path_b,d_rs_lane_post_b,d_rs_bulletin_b,d_rs_client_egress_b,d_rs_node_ingress_b\
         ,c_client_us,c_server_us,c_recipient_us\
         ,c_fixed_us,c_payload_us,c_wall_us,c_efficiency,c_agg1_wall_us,c_rs_wall_us",
    );
    for prof in NETWORKS {
        write!(
            header,
            ",p_{0}_direct_net_us,p_{0}_direct_e2e_us,p_{0}_direct_mbps,p_{0}_agg1_net_us,p_{0}_agg1_e2e_us,p_{0}_agg1_mbps,p_{0}_rs_net_us,p_{0}_rs_e2e_us,p_{0}_rs_mbps",
            prof.label
        )
        .unwrap();
    }

    // Param columns s..env (skipping the run-local `id`) identify a cell for
    // cross-run dedup in append mode. `env` is part of the key so a tdx row and a
    // host row for the same cell coexist instead of evicting each other.
    let param_key = |line: &str| -> String {
        line.split(',')
            .skip(1)
            .take(12)
            .collect::<Vec<_>>()
            .join(",")
    };

    let mut out = String::new();
    for (i, r) in rows.iter().enumerate() {
        let m = model(r);
        write!(
            out,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            cell_id(i),
            r.s,
            r.n,
            r.active,
            r.cover,
            r.flow,
            r.payload_client_b,
            r.delta,
            r.xi,
            r.n_polys,
            r.mu_kahe,
            r.iblt_cells,
            r.env,
            r.threads,
            r.affinity,
            r.recovered_ok,
            r.agg.recovered,
            rs0(r).map_or(0, |p| p.k),
            rs0(r).map_or(0, |p| p.n_nodes),
            rs0(r).is_some_and(|p| p.recovered),
            rs0(r).map_or("-", |p| p.lane_lie_caught),
        )
        .unwrap();
        for p in phases {
            let st = (p.get)(r);
            write!(out, ",{:.3},{:.3},{:.3}", st.med, st.min, st.max).unwrap();
        }
        write!(
            out,
            ",{},{},{},{},{},{},{},{},{},{},{},{},{:.2},{:.0},{:.0},{:.0},{:.0},{:.0}",
            r.open_env_b,
            r.comm_client_b,
            r.ctxt_client_b,
            r.agg_open_b,
            r.agg_share_b,
            r.open_env_b - pke::SEAL_OVERHEAD,
            r.kahe_key_b,
            r.plaintext_b,
            client_to_server_b(r),
            client_to_recipient_b(r),
            client_post_b(r),
            server_entry_b(r),
            r.eps_correct,
            r.wire_ctxt_b,
            r.wire_comm_b,
            r.wire_opening_b,
            r.wire_server_b,
            r.useful_b,
        )
        .unwrap();
        let (rs_share_b, rs_path_b, rs_lane_post_b, rs_bulletin_b, rs_egress_b, rs_node_in_b) =
            match rs0(r) {
                None => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                Some(p) => {
                    let item_b = (p.share_b + p.path_b) as f64;
                    (
                        p.share_b as f64,
                        p.path_b as f64,
                        p.lane_post_b as f64,
                        p.bulletin_b as f64,
                        p.n_nodes as f64 * item_b
                            + (r.open_env_b * r.s) as f64
                            + p.bulletin_b as f64,
                        r.n as f64 * item_b,
                    )
                }
            };
        write!(
            out,
            ",{:.0},{:.0},{:.0},{:.0},{:.0},{:.0}",
            rs_share_b, rs_path_b, rs_lane_post_b, rs_bulletin_b, rs_egress_b, rs_node_in_b
        )
        .unwrap();
        write!(
            out,
            ",{:.3},{:.3},{:.3}",
            client_us(r),
            server_us(r),
            recipient_us(r),
        )
        .unwrap();
        write!(
            out,
            ",{:.3},{:.3},{:.3},{:.6e},{:.3},{:.3}",
            m.fixed_us,
            m.payload_us,
            m.wall_us,
            m.efficiency,
            m.agg_cpu_us,
            m.rs_cpu_us.first().copied().unwrap_or(0.0),
        )
        .unwrap();
        let mbps = |e2e_us: f64| {
            if e2e_us > 0.0 {
                r.useful_b / (e2e_us / 1e6) / 1e6
            } else {
                0.0
            }
        };
        for np in net_all(r, &m) {
            let (agg_net, agg_e2e) = np.agg;
            let (rs_net, rs_e2e) = np.rs.first().copied().unwrap_or((0.0, 0.0));
            write!(
                out,
                ",{:.3},{:.3},{:.4},{:.3},{:.3},{:.4},{:.3},{:.3},{:.4}",
                np.direct_net_us,
                np.direct_e2e_us,
                mbps(np.direct_e2e_us),
                agg_net,
                agg_e2e,
                mbps(agg_e2e),
                rs_net,
                rs_e2e,
                mbps(rs_e2e),
            )
            .unwrap();
        }
        out.push('\n');
    }

    // SWEEP_APPEND=1 merges into an existing csv: keep prior rows whose param
    // key isn't re-run here, renumber ids. A header mismatch (schema changed)
    // discards the old file.
    let append = std::env::var("SWEEP_APPEND").is_ok_and(|v| v == "1");
    let mut lines: Vec<String> = Vec::new();
    if append {
        if let Ok(old) = std::fs::read_to_string(csv_path()) {
            let mut it = old.lines();
            if it.next() == Some(header.as_str()) {
                let new_keys: Vec<String> = out.lines().map(param_key).collect();
                lines.extend(
                    it.filter(|l| !new_keys.contains(&param_key(l)))
                        .map(String::from),
                );
            } else {
                eprintln!("csv schema changed; overwriting {}", csv_path());
            }
        }
    }
    lines.extend(out.lines().map(String::from));
    let mut merged = header;
    merged.push('\n');
    for (i, line) in lines.iter().enumerate() {
        let rest = line.split_once(',').unwrap().1;
        merged.push_str(&cell_id(i));
        merged.push(',');
        merged.push_str(rest);
        merged.push('\n');
    }
    std::fs::write(csv_path(), merged).unwrap();
}

// ── reading rows back, and splicing two environments ────────────────────────

/// Rebuild a `Row` from one CSV line. Only the fields the CSV carries are
/// restored; the aggregator topology is recomputed from `ρ` via
/// [`agg_group_size`] rather than stored, and is therefore identical to the run
/// that wrote the line.
fn row_from_csv(header: &[&str], line: &str) -> Result<Row, String> {
    let cells: Vec<&str> = line.split(',').collect();
    if cells.len() != header.len() {
        return Err(format!(
            "row has {} fields, header has {}",
            cells.len(),
            header.len()
        ));
    }
    let at = |name: &str| -> Result<&str, String> {
        header
            .iter()
            .position(|h| *h == name)
            .map(|i| cells[i])
            .ok_or_else(|| format!("missing column {name}"))
    };
    let num = |name: &str| -> Result<f64, String> {
        at(name)?.parse::<f64>().map_err(|e| format!("{name}: {e}"))
    };
    let idx = |name: &str| -> Result<usize, String> { Ok(num(name)? as usize) };
    let flag = |name: &str| -> Result<bool, String> { Ok(at(name)? == "true") };
    let stat = |name: &str| -> Result<Stat, String> {
        Ok(Stat {
            med: num(&format!("m_{name}_us_med"))?,
            min: num(&format!("m_{name}_us_min"))?,
            max: num(&format!("m_{name}_us_max"))?,
        })
    };

    let n = idx("rho")?;
    // `flow` and `lane_lie_caught` are `&'static str` in `Row`; map the parsed
    // text back onto the fixed set rather than leaking a string.
    let flow = match at("flow")? {
        "mse" => "mse",
        "prony" => "prony",
        "sched" => "sched",
        other => return Err(format!("unknown flow {other:?}")),
    };

    let g = agg_group_size(n);
    let agg = AggPlan {
        group_size: g,
        n_l1: n.div_ceil(g),
        agg_cpu: stat("agg1_level")?,
        leader: stat("agg1_leader")?,
        recovered: flag("agg1_recovered")?,
    };

    let rs_k = idx("rs_k")?;
    let rs_plans = if rs_k == 0 {
        vec![]
    } else {
        vec![RsPlan {
            k: rs_k,
            n_nodes: idx("rs_n")?,
            share_b: num("d_rs_share_b")? as usize,
            path_b: num("d_rs_path_b")? as usize,
            lane_post_b: num("d_rs_lane_post_b")? as usize,
            bulletin_b: num("d_rs_bulletin_b")? as usize,
            rns_encrypt: stat("rs_dgt_embed")?,
            share_commit: stat("rs_share_commit")?,
            rs_enc: stat("rs_enc")?,
            client_sign: stat("rs_client_sign")?,
            node_sum: stat("rs_node_sum")?,
            lane_ingest: stat("rs_lane_ingest")?,
            lane_attribute: stat("rs_lane_attribute")?,
            v_sig: stat("rs_verify_sig")?,
            v_open: stat("rs_verify_open")?,
            v_interp: stat("rs_verify_interp")?,
            v_lane_open: stat("rs_verify_lane_open")?,
            v_crosscheck: stat("rs_verify_crosscheck")?,
            v_reconstruct: stat("rs_reconstruct")?,
            v_reconstruct_lagrange: stat("rs_reconstruct_lagrange")?,
            v_kahe_dec: stat("rs_verify_kahe_dec")?,
            recovered: flag("rs_recovered")?,
            lane_lie_caught: match at("rs_lane_lie")? {
                "lane-proof" => "lane-proof",
                "other" => "other",
                "MISSED" => "MISSED",
                _ => "-",
            },
        }]
    };

    let mut row = Row {
        s: idx("s")?,
        n,
        active: idx("active")?,
        cover: idx("cover")?,
        iblt_cells: idx("iblt_cells")?,
        flow,
        payload_client_b: idx("payload_client_b")?,
        mu_kahe: idx("mu_kahe")?,
        delta: idx("delta")?,
        xi: idx("xi")?,
        n_polys: idx("n_polys")?,
        env: at("env")?.to_string(),
        threads: idx("m_threads")?,
        affinity: at("m_affinity")?.to_string(),
        recovered_ok: flag("recovered")?,
        enc_app: Stat::default(),
        kahe_keygen: Stat::default(),
        kahe_enc: Stat::default(),
        share: Stat::default(),
        cs: Stat::default(),
        seal: Stat::default(),
        unseal: Stat::default(),
        server: Stat::default(),
        v_agg_ctxt: Stat::default(),
        v_sum_comm: Stat::default(),
        v_open: Stat::default(),
        v_interp: Stat::default(),
        v_kahe_dec: Stat::default(),
        dec_app: Stat::default(),
        v_one_open: Stat::default(),
        open_env_b: idx("m_open_env_b")?,
        comm_client_b: idx("d_comm_client_b")?,
        ctxt_client_b: idx("d_ctxt_client_b")?,
        kahe_key_b: idx("d_kahe_key_b")?,
        plaintext_b: idx("d_plaintext_b")?,
        agg_open_b: idx("d_agg_open_b")?,
        agg_share_b: idx("d_agg_share_b")?,
        eps_correct: num("d_epsilon")?,
        useful_b: num("d_useful_b")?,
        wire_ctxt_b: num("d_wire_ctxt_b")?,
        wire_comm_b: num("d_wire_comm_b")?,
        wire_opening_b: num("d_wire_opening_b")?,
        wire_server_b: num("d_wire_server_b")?,
        agg,
        rs_plans,
    };
    // Plain phases go through the same table the writer used, so the two cannot
    // drift. Plan-nested ones were already set above.
    for p in PHASES {
        if matches!(p.owner, Owner::Agg | Owner::Rs) {
            continue;
        }
        (p.set)(&mut row, stat(p.name)?);
    }
    Ok(row)
}

/// Cell identity for splicing: the param columns, minus `env`, so a `tdx` row and
/// a `host` row for the same cell match each other.
fn cell_key(r: &Row) -> String {
    format!(
        "{},{},{},{},{},{},{},{},{},{},{}",
        r.s,
        r.n,
        r.active,
        r.cover,
        r.flow,
        r.payload_client_b,
        r.delta,
        r.xi,
        r.n_polys,
        r.mu_kahe,
        r.iblt_cells
    )
}

/// Human-readable cell label for the overhead table.
fn cell_key_short(r: &Row) -> String {
    format!(
        "S{} {}/{} {} {}B",
        r.s, r.n, r.active, r.flow, r.payload_client_b
    )
}

fn read_rows(path: &str) -> Result<Vec<Row>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| format!("{path}: empty"))?
        .split(',')
        .collect();
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| row_from_csv(&header, l).map_err(|e| format!("{path}: {e}")))
        .collect()
}

/// Splice a client-side run into an otherwise host-measured one.
///
/// Only the client-owned timings move; everything else comes from the host row.
/// The composed and projected columns are then recomputed from the spliced
/// timings by the normal `model`/`net_all` path, which is why this works at the
/// `Row` level rather than by copying columns.
///
/// Cells present in only one input are carried through untouched, so a reduced
/// client sweep still produces a usable merge.
pub fn merge(client_csv: &str, host_csv: &str, out_csv: &str) {
    let client_rows = read_rows(client_csv).unwrap_or_else(|e| panic!("{e}"));
    let host_rows = read_rows(host_csv).unwrap_or_else(|e| panic!("{e}"));

    let mut merged: Vec<Row> = Vec::with_capacity(host_rows.len());
    let mut spliced = 0usize;
    let mut client_only: Vec<String> = Vec::new();
    let mut host_only: Vec<String> = Vec::new();
    // Per-phase (client, host) medians, captured before the splice overwrites the
    // host's own client timings — after that the ratio is unrecoverable.
    let mut overhead: Vec<(String, Vec<(f64, f64)>)> = Vec::new();

    for mut host in host_rows {
        let key = cell_key(&host);
        match client_rows.iter().find(|c| cell_key(c) == key) {
            None => {
                host_only.push(key);
                merged.push(host);
            }
            Some(client) => {
                if client.threads != host.threads {
                    panic!(
                        "cell {key}: client run used {} threads, host run {} — \
                         timings are not comparable",
                        client.threads, host.threads
                    );
                }
                // Sizes are exact arithmetic in both runs; a mismatch means these
                // are not the same cell and the splice would be meaningless.
                for (what, a, b) in [
                    ("open_env_b", client.open_env_b, host.open_env_b),
                    ("ctxt_client_b", client.ctxt_client_b, host.ctxt_client_b),
                    ("agg_open_b", client.agg_open_b, host.agg_open_b),
                ] {
                    assert_eq!(a, b, "cell {key}: {what} differs between runs");
                }
                let mut pairs = Vec::new();
                for p in PHASES {
                    if p.owner == Owner::Client {
                        let (c, h) = ((p.get)(client), (p.get)(&host));
                        pairs.push((c.med, h.med));
                        (p.set)(&mut host, c);
                    }
                }
                // A host `rest`-only run has no client timings to compare against,
                // so only record the ratio when the host row actually has them.
                if pairs.iter().any(|(_, h)| *h > 0.0) {
                    overhead.push((cell_key_short(&host), pairs));
                }
                // Both halves named, so a host+host splice never reads as a TDX one.
                host.env = format!("{}+{}", client.env, host.env);
                spliced += 1;
                merged.push(host);
            }
        }
    }
    for client in &client_rows {
        let key = cell_key(client);
        if !merged.iter().any(|m| cell_key(m) == key) {
            client_only.push(key);
        }
    }

    println!(
        "merged {spliced} cells; {} host-only, {} client-only (not in output)",
        host_only.len(),
        client_only.len()
    );
    for k in &host_only {
        println!("  host-only, client timings absent: {k}");
    }
    for k in &client_only {
        println!("  client-only, dropped (no host round to splice into): {k}");
    }

    if !overhead.is_empty() {
        println!();
        println!("── [M] client-side overhead (client run / host run, per phase) ──────────────");
        let names: Vec<&str> = PHASES
            .iter()
            .filter(|p| p.owner == Owner::Client)
            .map(|p| p.name)
            .collect();
        print!("{:<26}", "cell");
        for n in &names {
            print!("{:>12}", n);
        }
        println!("{:>12}", "total");
        for (key, pairs) in &overhead {
            print!("{:<26}", key);
            for (c, h) in pairs {
                print!(
                    "{:>12}",
                    if *h > 0.0 {
                        format!("{:.2}x", c / h)
                    } else {
                        "-".into()
                    }
                );
            }
            let (ct, ht): (f64, f64) = pairs
                .iter()
                .fold((0.0, 0.0), |(a, b), (c, h)| (a + c, b + h));
            println!(
                "{:>12}",
                if ht > 0.0 {
                    format!("{:.2}x", ct / ht)
                } else {
                    "-".into()
                }
            );
        }
    }

    print_tables(&merged);
    // `write_csv` reads the destination from the environment.
    std::env::set_var("BENCH_CSV", out_csv);
    write_csv(&merged);
    println!();
    println!("csv: {} ({} cells)", out_csv, merged.len());
}

/// Re-derive the composed and projected columns of an existing CSV from its
/// measured/derived ones. After a model or net-sim change the [M]/[D] numbers
/// stay valid; only `c_*`/`p_*` need recomputing.
pub fn recompute(in_csv: &str, out_csv: &str) {
    let rows = read_rows(in_csv).unwrap_or_else(|e| panic!("{e}"));
    std::env::set_var("BENCH_CSV", out_csv);
    std::env::remove_var("SWEEP_APPEND");
    write_csv(&rows);
    println!("csv: {} ({} cells)", out_csv, rows.len());
}

pub fn run() {
    let budget = Duration::from_secs(
        std::env::var("BENCH_BUDGET_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300),
    );
    let start = Instant::now();

    println!(
        "Panetière scaling bench  (budget: {}s, {} reps/phase)",
        budget.as_secs(),
        REPS
    );
    println!(
        "ring: HVC {} bits/coef ({} B/poly) | KAHE {} bits/coef ({} B/poly) | t bits/symbol {}",
        poly_packed_len(HVC_MODULUS) * 8 / POLY_N,
        poly_packed_len(HVC_MODULUS),
        poly_packed_len64(KAHE_MODULUS) * 8 / POLY_N,
        poly_packed_len64(KAHE_MODULUS),
        BITS_PER_SYMBOL,
    );
    let roles = Roles::from_env();
    println!(
        "env: {} | threads {} | cpus {}{}",
        env_tag(),
        rayon::current_num_threads(),
        affinity(),
        if roles == Roles::all() {
            String::new()
        } else {
            format!(
                " | timing only: {}",
                [
                    ("client", roles.client),
                    ("server", roles.server),
                    ("verifier", roles.verifier),
                    ("agg", roles.agg),
                    ("rs", roles.rs),
                ]
                .iter()
                .filter(|(_, on)| *on)
                .map(|(n, _)| *n)
                .collect::<Vec<_>>()
                .join(",")
            )
        }
    );
    println!(
        "provenance: [M] measured   — median of {} one-party reps (spread in csv)",
        REPS
    );
    println!("            [D] derived    — exact byte arithmetic from packing formulas");
    println!("            [C] composed   — model wall = fixed + payload over [M] medians");
    println!("            [P] projected  — extrapolation / synthetic network sim");
    println!("useful = delivered · ξ · log₂(t) bits/round (cover clients carry no payload, and");
    println!("         nor does a scheduled client whose reservation lost a rand collision)");
    println!();

    let servers = env_list::<usize>("SWEEP_SERVERS").unwrap_or_else(|| SERVERS.to_vec());
    let clients = sweep_clients();
    let configs = sweep_configs();
    let sched_bytes =
        env_list::<usize>("SWEEP_SCHED_BYTES").unwrap_or_else(|| SCHED_MESSAGE_BYTES.to_vec());

    // The protocol sweep, flattened once. Scheduled ignores ξ (its token IBLT
    // is fixed at 2 symbols), so it varies over sched_bytes with configs[0].
    let mut variants: Vec<(&Config, &'static str, usize, String)> = configs
        .iter()
        .flat_map(|cfg| {
            [
                (cfg, "mse", 0, format!("mse {}", cfg.label)),
                (cfg, "prony", 0, format!("prony {}", cfg.label)),
            ]
        })
        .collect();
    match configs.first() {
        Some(cfg0) => variants.extend(sched_bytes.iter().map(|&mb| {
            (
                cfg0,
                "sched",
                mb,
                format!("sched msg={}", fmt_bytes(mb as f64)),
            )
        })),
        None if !sched_bytes.is_empty() => {
            eprintln!("(no payload configs; scheduled section skipped)")
        }
        None => {}
    }

    let mut rows: Vec<Row> = Vec::new();
    let mut skipped = false;

    // Nesting is clients → servers → protocol, protocol innermost, so a budget
    // cut-off leaves *complete* protocol comparisons at every (client, server)
    // point it reached rather than one protocol across all of them.
    'sweep: for &(clients_total, clients_active) in &clients {
        for &s in &servers {
            for (cfg, flow, message_bytes, desc) in &variants {
                if start.elapsed() >= budget {
                    skipped = true;
                    break 'sweep;
                }
                eprintln!(
                    "running cell {}: clients={clients_total}x{clients_active} S={s} {desc}",
                    cell_id(rows.len())
                );
                rows.push(run_cell(
                    s,
                    clients_total,
                    clients_active,
                    cfg,
                    flow,
                    *message_bytes,
                    roles,
                ));
            }
        }
    }
    if skipped {
        println!("(budget exhausted; remaining cells skipped)");
    }

    if rows.is_empty() {
        println!("no cells completed within budget");
        return;
    }
    print_tables(&rows);
    write_csv(&rows);
    println!();
    println!("csv: {} ({} cells)", csv_path(), rows.len());
    println!("done in {:.1}s", start.elapsed().as_secs_f64());
}
