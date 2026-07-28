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
//!   [C] composed  — cost model `wall = fixed + l·chunk` over [M] medians;
//!                   one client → one server → verifier, summed as a serial
//!                   pipeline (one party of each kind, not ρ of them)
//!   [P] projected — synthetic network sim
//!
//! Each cell runs ONE live round; every active client encodes a distinct
//! (element, r). Recovery is attempted and reported, never asserted —
//! irrecoverable parameter sets still yield timings.
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
//!  NETWORKS: per-profile one-way latency + uniform jitter, link Mbit/s
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
//!  l              = ⌈n_polys/MU_KAHE⌉ KAHE chunks
//!  total_cells    = γ · δ
//!  Shamir threshold, κ_cs, β, β_agg, block_size — chosen by
//!  `ProtocolParams::setup_with_kahe_dims_full(s, μ_kahe, κ_kahe, l, …)`
//!
//! ──────────────────────────────────────────────────────────────────────
//! CONSTANTS (pinned upstream — the sources are authoritative)
//! ──────────────────────────────────────────────────────────────────────
//!  chipmunk param.rs: N = 2048, q_hvc = 40_961, q_cs = 139_301,
//!                     q_kahe = 347_280_875_347_969, HVC_WIDTH = 3
//!  src/kahe.rs:  t = T_MODULUS_DEFAULT = 2^36, σ_s = σ_e = 15.72
//!                noise budget t·8σ_e·√ρ + ρ·t/2 < q_kahe/2 (ρ ≲ 349)
//!  src/mse.rs:   BITS_PER_SYMBOL = 36, K_LIMBS = 2
//!  bench-internal: MSE PRF key = [0xAA; 32]; per-cell ChaCha20 seed = SHA-256
//!                  over (S, ρ, active, ξ, codec), so cells never share a stream
//!  protocol coupling (asserted at runtime): μ_cs = κ_kahe
//!  asserted per cell: agg_open + agg_share == CsParams::aggregated_server_crypto_len
//!
//! Run with:
//!   RAYON_NUM_THREADS=8 cargo bench -j 8 --bench scaling
//!   BENCH_BUDGET_SECS=600 cargo bench --bench scaling

use std::fmt::Write as _;
use std::time::{Duration, Instant};

use chipmunk_code::{DgtNTTPoly, KahePoly, HVC_MODULUS, KAHE_MODULUS, N as POLY_N};
use crate::codec;
use crate::cs::{
    aggregated_opening_pack_bounds, pack_cs_shares, poly_packed_len, poly_packed_len64, Commitment,
    Cs, HidingMerkleCommitment,
};
use crate::kahe::{Kahe, KaheScheme, SIGMA_E_DEFAULT, SIGMA_S_DEFAULT, T_MODULUS_DEFAULT};
use crate::mse::{MseEncoding, MseParams, BITS_PER_SYMBOL};
use crate::pke;
use crate::prony::{PronyParams, PronySketch, PRONY_PRIME};
use crate::bulletin::{RsClientBulletinEntry, RsNodeBulletinEntry};
use crate::bulletin::dgt_packed_len;
use crate::digest::{digest, embed, DIGEST_POLYS};
use crate::protocol::aggregator::run_aggregator_round;
use crate::protocol::client::{
    cs_commit, kahe_encrypt, kahe_keygen, run_client_round, run_client_round_rs, seal_openings,
    shamir_share,
};
use crate::protocol::server::{
    run_node_round, run_server_round, unseal_opening, unseal_openings, RsNodeInbox, ServerInbox,
};
use crate::protocol::verify::{
    aggregate_and_decrypt_rs, aggregate_and_decrypt_timed, decrypt_aggregate, RsVerifyTimings,
    VerifyError, VerifyTimings,
};
use crate::protocol::ProtocolParams;
use crate::protocol::{ClientId, NodeId, ServerId, SessionId};
use crate::rs::Rs;
use crate::sig::SigningKey;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

const SERVERS: &[usize] = &[8, 16, 64];

// (clients_total, clients_active): explicit pairs, not a cross product.
const CLIENTS: &[(usize, usize)] = &[(100, 100), (300, 300), (300, 100)];

// γ (MSE rows) is fixed, not swept.
const GAMMA: usize = 4;

struct Config {
    label: String,
    payload_symbols: usize, // ξ: per-client element size = ξ·log₂t bits
}

// MU_KAHE — the KAHE encryption chunk size, in polys (µ from the KAHE param
// table, sized so one key covers a 20 KiB-element IBLT at ρ=300). A larger
// message's IBLT encoding spans l = ⌈n_polys/MU_KAHE⌉ chunks, each under its
// own matrix A_i but the same key sk, so the Shamir+CS pass amortizes over
// all l chunks.
const MU_KAHE: usize = 2002;

/// Codec bytes per poly (chipmunk `N` coefficients × 4 bytes), matching `codec`.
const BYTES_PER_POLY: usize = POLY_N * 4;

/// Repetitions per measured phase; tables show the median, CSV keeps min/max.
const REPS: usize = 5;

/// One cell is one protocol execution.
const SESSION: SessionId = SessionId([0x5C; 32]);

/// `Mse` = single-round IBLT. `Prony` = single-round Vandermonde sketch at the
/// same element size, no coding blowup, run at `t = PRONY_PRIME`. `Scheduled`
/// = one joint plaintext, reservation IBLT ‖ message vector of `message_bytes`,
/// under one key (openings paid once).
enum AppCodec {
    Mse,
    Prony,
    Scheduled { message_bytes: usize },
}

/// `⌊log₂ PRONY_PRIME⌋ = 35`, one below MSE's 36, so ξ grows by that ratio at
/// equal element bytes.
fn prony_bits() -> usize {
    PRONY_PRIME.ilog2() as usize
}

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

/// Aggregation ceiling the digest ring is sized for (see `digest::DigestParams`).
const RS_RHO_MAX: usize = 300;

// Aggregation depths to evaluate. Group size per depth is balanced so the
// largest per-hop fan-in is ρ^{1/(layers+1)} (see `AggPlan`): 1-layer → ⌈√ρ⌉,
// 2-layer → ⌈∛ρ⌉.
const AGG_LAYERS: &[usize] = &[1];

// Per-client element size = ξ symbols · 36 bits: 911 ⇒ 4 KiB, 4552 ⇒ 20 KiB.
const DEFAULT_PAYLOAD_SYMBOLS: &[usize] = &[911, 4552];

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
        Ok(raw) => raw
            .split(',')
            .map(|c| {
                let (total, active) = c
                    .trim()
                    .split_once('x')
                    .unwrap_or_else(|| panic!("bad SWEEP_CLIENTS pair {c:?} (want TOTALxACTIVE)"));
                (
                    total.parse().unwrap_or_else(|_| panic!("bad clients_total in {c:?}")),
                    active.parse().unwrap_or_else(|_| panic!("bad clients_active in {c:?}")),
                )
            })
            .collect(),
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
/// (parallel arrivals take the max draw). Two link classes: client links
/// (residential, the slow side) and server-side links (servers, bulletin,
/// verifier — cloud, ~Gbit). Each endpoint serializes its own bytes at its
/// class rate, both directions.
struct NetProfile {
    label: &'static str,
    lat_ms: f64,
    jitter_ms: f64,
    client_mbps: f64,
    server_mbps: f64,
}

const NETWORKS: &[NetProfile] = &[
    NetProfile {
        label: "1Gbit",
        lat_ms: 60.0,
        jitter_ms: 10.0,
        client_mbps: 100.0,
        server_mbps: 1000.0,
    },
    NetProfile {
        label: "4Gbit",
        lat_ms: 60.0,
        jitter_ms: 10.0,
        client_mbps: 100.0,
        server_mbps: 4000.0,
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

fn cell_id(i: usize) -> String {
    let c = (b'A' + (i % 26) as u8) as char;
    if i < 26 {
        c.to_string()
    } else {
        format!("{}{}", c, i / 26)
    }
}

// `layer_counts`: aggregator count per level bottom-up; leader ingests the last.
struct AggPlan {
    layers: usize,
    group_size: usize,
    layer_counts: Vec<usize>,
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
    bulletin_b: usize,
    dgt_embed: Stat,
    dgt_hash: Stat,
    rs_enc: Stat,
    node_sum: Stat,
    v_sig: Stat,
    v_open: Stat,
    v_interp: Stat,
    v_reconstruct: Stat,
    v_reconstruct_lagrange: Stat,
    v_syndrome: Stat,
    v_kahe_dec: Stat,
    v_digest: Stat,
    recovered: bool,
    /// Which mechanism rejected a single flipped coefficient in one lane's sum.
    lane_lie_caught: &'static str,
}

struct Row {
    s: usize,
    n: usize, // ρ — total clients = anonymity set
    active: usize,
    cover: usize,
    iblt_cells: usize,
    flow: &'static str,          // "mse" | "sched"
    payload_client_b: usize,     // per-client useful bytes (element / msg slot)
    mu_kahe: usize,
    delta: usize,
    xi: usize,
    l: usize,
    n_polys: usize, // actual message length; ≤ μ·l (final chunk may be partial)
    recovered_ok: bool,
    // [M] CPU, per-round totals for one party (the ×l chunk work included).
    enc_app: Stat,
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
    // [M] wire: real ML-KEM-sealed opening envelope (client → one server).
    open_env_b: usize,
    // [D] wire components (bytes).
    comm_client_b: usize,      // one commitment root
    ctxt_client_b: usize,      // one client's full ciphertext (n_polys polys)
    agg_open_b: usize,          // server-posted aggregate opening
    agg_share_b: usize,         // server-posted agg_share (one CS poly)
    // [D] round totals across all parties.
    useful_b: f64,
    wire_ctxt_b: f64,
    wire_comm_b: f64,
    wire_opening_b: f64,
    wire_server_b: f64,
    agg_plans: Vec<AggPlan>,
    rs_plans: Vec<RsPlan>,
}

/// Per-cell RNG seed: domain-separated hash over every parameter that defines
/// the cell, including the codec. Hashing (not byte-packing) keeps distinct
/// cells on distinct streams regardless of magnitude.
fn cell_seed(s: usize, n: usize, active: usize, cfg: &Config, codec: &AppCodec) -> [u8; 32] {
    let (codec_tag, codec_arg) = match codec {
        AppCodec::Mse => (0u64, 0u64),
        AppCodec::Scheduled { message_bytes } => (1, *message_bytes as u64),
        AppCodec::Prony => (2, 0),
    };
    let mut h = Sha256::new();
    h.update(b"panetiere-scaling-cell-v1");
    for v in [
        s as u64,
        n as u64,
        active as u64,
        cfg.payload_symbols as u64,
        codec_tag,
        codec_arg,
    ] {
        h.update(v.to_le_bytes());
    }
    h.finalize().into()
}

fn run_cell(s: usize, n: usize, active: usize, cfg: &Config, codec: &AppCodec) -> Row {
    let cover = n - active;
    let seed = cell_seed(s, n, active, cfg, codec);
    let mut rng = ChaCha20Rng::from_seed(seed);

    // IBLT sized to active (cover adds nothing), ~3 buckets/insertion. `Mse`
    // packs a ξ-element into it; `Scheduled` packs a (rand,size) token and
    // appends a message vector — each active client an equal slot within it.
    let delta = (3 * active).div_ceil(GAMMA);
    // Capacity = active; ξ re-derived so the element carries the same bytes as
    // the MSE cell at 35 rather than 36 bits per coefficient.
    let prony_params = match codec {
        AppCodec::Prony => Some(PronyParams::new(
            active,
            (cfg.payload_symbols * BITS_PER_SYMBOL).div_ceil(prony_bits()),
        )),
        _ => None,
    };
    let (mse_params, sched_polys, slot_bytes, msg_vector_bytes) = match codec {
        AppCodec::Mse => {
            let p = MseParams::new(GAMMA, delta, cfg.payload_symbols, [0xAA; 32]);
            let sp = MseEncoding::n_polys(&p);
            (p, sp, 0, 0)
        }
        // No IBLT in this flow; `mse_params` stays only to keep the shared Row
        // fields well-formed and is never encoded into.
        AppCodec::Prony => {
            let p = MseParams::new(GAMMA, delta, cfg.payload_symbols, [0xAA; 32]);
            let sp = PronySketch::n_polys(prony_params.as_ref().unwrap());
            (p, sp, 0, 0)
        }
        AppCodec::Scheduled { message_bytes } => {
            let p = MseParams::new(GAMMA, delta, SCHED_TOKEN_SYMBOLS, [0xAA; 32]);
            let sp = MseEncoding::n_polys(&p);
            let slot = (message_bytes / active.max(1)).next_multiple_of(2);
            (p, sp, slot, active * slot)
        }
    };
    let msg_polys = msg_vector_bytes.div_ceil(BYTES_PER_POLY);
    let n_polys = sched_polys + msg_polys;
    // KAHE encrypts the whole joint plaintext in l chunks of the fixed MU_KAHE.
    let mu_kahe = MU_KAHE;
    let l = n_polys.div_ceil(mu_kahe);

    let pp = ProtocolParams::setup_with_kahe_dims_full(
        &mut rng,
        s,
        mu_kahe,
        l,
        SIGMA_S_DEFAULT,
        SIGMA_E_DEFAULT,
        match codec {
            AppCodec::Prony => PRONY_PRIME,
            _ => T_MODULUS_DEFAULT,
        },
    );
    let server_ids: Vec<ServerId> = (0..s as u32).map(ServerId).collect();
    let server_keys: Vec<pke::PrivateKey> =
        (0..s).map(|_| pke::PrivateKey::generate(&mut rng)).collect();
    let servers: Vec<(ServerId, pke::PublicKey)> = server_ids
        .iter()
        .map(|&sid| (sid, server_keys[sid.0 as usize].public()))
        .collect();
    let client_ids: Vec<ClientId> = (0..n as u32).map(ClientId).collect();

    // Active clients carry a distinct payload; cover clients contribute zero.
    // Mse: a ξ-element. Scheduled: a (rand,size) token, then the client's slot.
    let mse_payloads: Vec<Vec<i64>> = match codec {
        AppCodec::Mse => (0..active)
            .map(|i| (0..cfg.payload_symbols).map(|j| i as i64 + j as i64 + 1).collect())
            .collect(),
        AppCodec::Prony => {
            let xi = prony_params.as_ref().unwrap().payload_symbols;
            (0..active).map(|i| (0..xi).map(|j| i as i64 + j as i64 + 1).collect()).collect()
        }
        AppCodec::Scheduled { .. } => {
            (0..active).map(|i| vec![i as i64 + 1, slot_bytes as i64]).collect()
        }
    };
    let byte_payloads: Vec<Vec<u8>> =
        (0..active).map(|i| vec![(i as u8).wrapping_add(1); slot_bytes]).collect();
    let ranges: Vec<(usize, usize)> = (0..active).map(|i| (i * slot_bytes, slot_bytes)).collect();
    let client_polys: Vec<Vec<KahePoly>> = (0..n)
        .map(|i| {
            let mut polys = match (&prony_params, i < active) {
                (Some(pp), true) => {
                    let mut sk = PronySketch::new(pp.clone());
                    sk.insert(&mut rng, &mse_payloads[i]);
                    sk.pack()
                }
                (Some(pp), false) => PronySketch::cover(pp),
                (None, true) => {
                    let mut enc = MseEncoding::new(mse_params.clone());
                    enc.insert(&mut rng, &mse_payloads[i]);
                    enc.pack()
                }
                (None, false) => MseEncoding::cover(&mse_params),
            };
            polys.resize(sched_polys, KahePoly::default());
            if matches!(codec, AppCodec::Scheduled { .. }) && i < active {
                polys.extend(codec::encode_at(ranges[i].0, msg_vector_bytes, &byte_payloads[i]));
            }
            // Pad covers/short encodings to the joint plaintext width — NOT to
            // the μ·l chunk boundary; the final KAHE chunk may be partial.
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
        let round =
            run_client_round(&mut rng, &pp, &SESSION, cid, client_polys[i].clone(), &servers);
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
    let (enc_app, _) = match codec {
        AppCodec::Mse => measure(|| {
            let mut enc = MseEncoding::new(mse_params.clone());
            enc.insert(&mut rng, &mse_payloads[0]);
            enc.pack()
        }),
        AppCodec::Prony => measure(|| {
            let mut sk = PronySketch::new(prony_params.as_ref().unwrap().clone());
            sk.insert(&mut rng, &mse_payloads[0]);
            sk.pack()
        }),
        AppCodec::Scheduled { .. } => measure(|| {
            let mut enc = MseEncoding::new(mse_params.clone());
            enc.insert(&mut rng, &mse_payloads[0]);
            let mut p = enc.pack();
            p.resize(sched_polys, KahePoly::default());
            p.extend(codec::encode_at(ranges[0].0, msg_vector_bytes, &byte_payloads[0]));
            p
        }),
    };
    let key0 = kahe_keygen(&mut rng, &pp);
    let (kahe_enc, _ctxt0) = measure(|| kahe_encrypt(&mut rng, &pp, &key0, &client_polys[0]));
    let (share, shares0) = measure(|| shamir_share(&mut rng, &pp, &key0, s));
    let (cs, (_comm0, openings0)) = measure(|| cs_commit(&mut rng, &pp, &shares0));
    let (seal, _) = measure(|| seal_openings(&mut rng, &pp, &SESSION, client_ids[0], &openings0, &servers));
    let (unseal, _) = measure(|| unseal_openings(&server_keys[0], &SESSION, servers[0].0, &sealed_inbox0));
    let (server, _) = measure(|| run_server_round(&inboxes[0], &canonical).unwrap());

    // Verify: REPS full runs, field-wise medians over the returned timings.
    // Recovery is ATTEMPTED, not asserted: at irrecoverable parameter sets
    // aggregation decodes to garbage (or Errs). We still report every timing.
    let mut vts: Vec<VerifyTimings> = Vec::with_capacity(REPS);
    let mut recovered: Option<Vec<KahePoly>> = None;
    for _ in 0..REPS {
        if let Ok((rec, vt)) = aggregate_and_decrypt_timed(&pp, &canonical, &client_entries, &outputs)
        {
            vts.push(vt);
            recovered = Some(rec);
        }
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
        let mut p = mse_payloads.clone();
        p.sort();
        p
    };
    let check_decode = |rec: &[KahePoly]| -> bool {
        if let Some(pp) = &prony_params {
            return PronySketch::unpack(pp, &rec[..n_polys])
                .decode()
                .map(|d| d == expected_payloads)
                .unwrap_or(false);
        }
        let tokens = match codec {
            AppCodec::Scheduled { .. } => &rec[..sched_polys],
            _ => &rec[..n_polys],
        };
        let tokens_ok = MseEncoding::unpack(&mse_params, tokens)
            .decode()
            .map(|d| d == expected_payloads)
            .unwrap_or(false);
        match codec {
            AppCodec::Scheduled { .. } => {
                tokens_ok
                    && codec::decode_ranges(&rec[sched_polys..n_polys], &ranges)
                        .map(|d| d == byte_payloads)
                        .unwrap_or(false)
            }
            _ => tokens_ok,
        }
    };
    let recovered_ok = recovered.as_deref().map(check_decode).unwrap_or(false);

    let dec_app = match &recovered {
        None => Stat::default(),
        // May Err (PeelStalled) on unrecoverable params — we time the attempt.
        Some(rec) => match codec {
            AppCodec::Mse => {
                measure(|| {
                    let _ = MseEncoding::unpack(&mse_params, &rec[..n_polys]).decode();
                })
                .0
            }
            AppCodec::Prony => {
                let pp = prony_params.as_ref().unwrap();
                measure(|| {
                    let _ = PronySketch::unpack(pp, &rec[..n_polys]).decode();
                })
                .0
            }
            AppCodec::Scheduled { .. } => {
                measure(|| {
                    let _ = MseEncoding::unpack(&mse_params, &rec[..sched_polys]).decode();
                    let _ = codec::decode_ranges(&rec[sched_polys..n_polys], &ranges);
                })
                .0
            }
        },
    };

    // Aggregated flow at each depth in AGG_LAYERS, measured live. Group size is
    // balanced so the largest per-hop fan-in is ρ^{1/(layers+1)}: 1-layer ⌈√ρ⌉,
    // 2-layer ⌈∛ρ⌉. Recovery is re-checked end-to-end through the tree.
    let agg_plans: Vec<AggPlan> = AGG_LAYERS
        .iter()
        .map(|&layers| {
            let g = match layers {
                1 => (n as f64).sqrt(),
                _ => (n as f64).powf(1.0 / (layers as f64 + 1.0)),
            }
            .ceil() as usize;
            let g = g.max(2);

            let a1 = n.div_ceil(g);
            let l1_groups: Vec<Vec<_>> = (0..a1)
                .map(|grp| {
                    client_entries
                        .iter()
                        .filter(|(cid, _)| cid.0 as usize % a1 == grp)
                        .cloned()
                        .collect()
                })
                .collect();
            let (agg_cpu, _) = measure(|| run_aggregator_round(&l1_groups[0]));
            let l1: Vec<_> = l1_groups.iter().map(|g| run_aggregator_round(g)).collect();

            // Fold up the tree: each higher level sums g aggregates at a time.
            let mut layer_counts = vec![a1];
            let mut ctxts: Vec<Vec<KahePoly>> = l1.iter().map(|a| a.summed_ctxt.clone()).collect();
            let mut comms: Vec<Commitment> = l1.iter().map(|a| a.summed_comm.clone()).collect();
            for _ in 1..layers {
                let groups = ctxts.len().div_ceil(g);
                ctxts = (0..groups)
                    .map(|grp| {
                        let cs: Vec<_> = ctxts
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| i % groups == grp)
                            .map(|(_, c)| c.clone())
                            .collect();
                        Kahe::agg_ctxt(&cs)
                    })
                    .collect();
                comms = (0..groups)
                    .map(|grp| {
                        let ms: Vec<_> = comms
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| i % groups == grp)
                            .map(|(_, m)| m.clone())
                            .collect();
                        HidingMerkleCommitment::sum_commitments(&ms)
                    })
                    .collect();
                layer_counts.push(ctxts.len());
            }

            let (leader, res) = measure(|| {
                let total_ctxt = Kahe::agg_ctxt(&ctxts);
                let total_comm = HidingMerkleCommitment::sum_commitments(&comms);
                decrypt_aggregate(&pp, &total_ctxt, &total_comm, &outputs)
            });
            let recovered = res.map(|rec| check_decode(&rec)).unwrap_or(false);

            AggPlan {
                layers,
                group_size: g,
                layer_counts,
                agg_cpu,
                leader,
                recovered,
            }
        })
        .collect();

    // RS-sharded ingress, measured live: its own `pp` (μ = n_polys + 1, the
    // extra slot holding Enc(H(m))), its own client rounds, one node round per
    // lane, and the full verify. The ciphertext never reaches the bulletin, so
    // the comparison against the direct/aggregated flows is per-role wire, not
    // just CPU.
    let t_modulus = match codec {
        AppCodec::Prony => PRONY_PRIME,
        _ => T_MODULUS_DEFAULT,
    };
    let rs_plans: Vec<RsPlan> = rs_specs()
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
            let rs_keys: Vec<pke::PrivateKey> =
                (0..s).map(|_| pke::PrivateKey::generate(&mut rrng)).collect();
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

            let rs_outputs: Vec<_> = (0..s)
                .map(|j| {
                    let items = rounds
                        .iter()
                        .map(|r| {
                            let (sid, sealed) = &r.sealed_openings[j];
                            let op = unseal_opening(&rs_keys[j], &SESSION, r.client_id, *sid, sealed)
                                .unwrap();
                            (r.client_id, op)
                        })
                        .collect();
                    run_server_round(
                        &ServerInbox { server_id: server_ids[j], items },
                        &canonical,
                    )
                    .unwrap()
                })
                .collect();

            let node_inboxes: Vec<RsNodeInbox> = (0..n_nodes)
                .map(|j| RsNodeInbox {
                    node_id: NodeId(j as u32),
                    items: rounds
                        .iter()
                        .map(|r| (r.client_id, r.rs_shares[j].clone()))
                        .collect(),
                })
                .collect();
            let (node_sum, _) = measure(|| run_node_round(&node_inboxes[0], &canonical).unwrap());
            let node_outputs: Vec<RsNodeBulletinEntry> = node_inboxes
                .iter()
                .map(|inb| run_node_round(inb, &canonical).unwrap())
                .collect();

            // Embedding into the digest ring is n_polys forward NTTs; the Ajtai
            // hash is DIGEST_POLYS AVX2 dots over them; encoding is
            // (n−k)·n_polys·N modmuls, since systematic shares are the blocks.
            let dp = pp_rs.digest.as_ref().unwrap();
            let key0 = kahe_keygen(&mut rrng, &pp_rs);
            let ctxt0 = kahe_encrypt(&mut rrng, &pp_rs, &key0, &client_polys[0]);
            let (dgt_embed, embedded0) = measure(|| embed(&ctxt0));
            let (dgt_hash, _) = measure(|| digest(dp, &embedded0));
            let rs_params = pp_rs.rs.as_ref().unwrap();
            let (rs_enc, shares0) = measure(|| Rs::encode(rs_params, &embedded0));
            let share_b = shares0[0].len() * dgt_packed_len();

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

            // A single flipped coefficient in one lane's sum must not decode.
            // A +1 coefficient-domain bump: stays inside the norm bound, so it
            // reaches the algebraic checks rather than tripping the range check.
            let mut lying = node_outputs.clone();
            let mut unit = [0i64; POLY_N];
            unit[0] = 1;
            lying[0].share_sum[0] =
                lying[0].share_sum[0] + DgtNTTPoly::from_kahe(&KahePoly::from_coeffs(unit));
            let lane_lie_caught = match run_verify(&lying) {
                Err(VerifyError::LaneMismatch(_)) => "syndrome",
                Err(VerifyError::DigestMismatch) => "digest",
                Err(VerifyError::CiphertextOutOfRange) => "norm",
                Err(_) => "other",
                Ok(_) => "MISSED",
            };

            RsPlan {
                k,
                n_nodes,
                share_b,
                bulletin_b: RsClientBulletinEntry::packed_len(DIGEST_POLYS),
                dgt_embed,
                dgt_hash,
                rs_enc,
                node_sum,
                v_sig: rstat(|v| v.sig_verify_us),
                v_open: rstat(|v| v.opening_verify_us),
                v_interp: rstat(|v| v.interpolation_us),
                v_reconstruct: rstat(|v| v.reconstruct_us),
                v_reconstruct_lagrange: if lag.is_empty() {
                    Stat::default()
                } else {
                    stat(lag)
                },
                v_syndrome: rstat(|v| v.syndrome_us),
                v_kahe_dec: rstat(|v| v.kahe_dec_us),
                v_digest: rstat(|v| v.digest_us),
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
        // For `prony` the structural width is the Vandermonde column count,
        // reported in place of the IBLT's cell count.
        iblt_cells: match &prony_params {
            Some(pp) => pp.cols(),
            None => mse_params.total_cells(),
        },
        flow: match codec {
            AppCodec::Mse => "mse",
            AppCodec::Prony => "prony",
            AppCodec::Scheduled { .. } => "sched",
        },
        payload_client_b: match codec {
            AppCodec::Mse => cfg.payload_symbols * BITS_PER_SYMBOL / 8,
            AppCodec::Prony => {
                prony_params.as_ref().unwrap().payload_symbols * prony_bits() / 8
            }
            AppCodec::Scheduled { .. } => slot_bytes,
        },
        mu_kahe,
        delta: match &prony_params {
            Some(pp) => pp.cols(),
            None => delta,
        },
        xi: match codec {
            AppCodec::Mse => cfg.payload_symbols,
            AppCodec::Prony => prony_params.as_ref().unwrap().payload_symbols,
            AppCodec::Scheduled { .. } => SCHED_TOKEN_SYMBOLS,
        },
        l,
        n_polys,
        recovered_ok,
        enc_app,
        kahe_enc,
        share,
        cs,
        seal,
        unseal,
        server,
        v_agg_ctxt,
        v_sum_comm,
        v_open,
        v_interp,
        v_kahe_dec,
        dec_app,
        open_env_b,
        comm_client_b,
        ctxt_client_b,
        agg_open_b,
        agg_share_b,
        useful_b: match codec {
            AppCodec::Mse => (active * cfg.payload_symbols * BITS_PER_SYMBOL) as f64 / 8.0,
            AppCodec::Prony => {
                (active * prony_params.as_ref().unwrap().payload_symbols * prony_bits()) as f64
                    / 8.0
            }
            AppCodec::Scheduled { .. } => (active * slot_bytes) as f64,
        },
        wire_ctxt_b: n as f64 * ctxt_client_b as f64,
        wire_comm_b: n as f64 * poly_packed_len(HVC_MODULUS) as f64,
        wire_opening_b: (n * s) as f64 * open_env_b as f64,
        wire_server_b: s as f64 * (agg_open_b + agg_share_b) as f64,
        agg_plans,
        rs_plans,
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
    h.update(b"panetiere/bench/rs/v1");
    h.update(cell);
    h.update((k as u64).to_le_bytes());
    h.update((n_nodes as u64).to_le_bytes());
    h.finalize().into()
}

// ── [C] cost model over [M] medians ─────────────────────────────────────────

struct Model {
    fixed_us: f64,
    chunk_us: f64, // per single chunk
    wall_us: f64,
    wire_total_b: f64,
    efficiency: f64,
    post_b: f64,        // one client bulletin post = comm + all l ctxt chunks
    client_open_b: f64, // one client's S sealed openings
    agg_cpu_us: Vec<f64>, // per AggPlan
    rs_cpu_us: Vec<f64>,  // per RsPlan
}

fn model(r: &Row) -> Model {
    // fixed = paid once per round (key-related); chunk_total = scales with l.
    let fixed_us = r.share.med
        + r.cs.med
        + r.seal.med
        + r.unseal.med
        + r.server.med
        + r.v_open.med
        + r.v_interp.med
        + r.v_sum_comm.med;
    let chunk_total_us =
        r.enc_app.med + r.kahe_enc.med + r.v_agg_ctxt.med + r.v_kahe_dec.med + r.dec_app.med;
    let wall_us = fixed_us + chunk_total_us;
    let wire_total_b = r.wire_ctxt_b + r.wire_comm_b + r.wire_opening_b + r.wire_server_b;
    // Aggregated flow replaces the leader's direct ingest phases with the tree.
    let cpu_base_agg_us = (fixed_us - r.v_open.med - r.v_interp.med - r.v_sum_comm.med)
        + (chunk_total_us - r.v_agg_ctxt.med - r.v_kahe_dec.med);
    let agg_cpu_us = r.agg_plans.iter().map(|p| cpu_base_agg_us + p.leader.med).collect();
    // RS flow: client pays app-encode + encrypt + RS-encode + key work; one
    // node sums its lane; the verifier no longer sums ciphertexts at all.
    let rs_cpu_us = r
        .rs_plans
        .iter()
        .map(|p| {
            r.enc_app.med
                + r.kahe_enc.med
                + p.dgt_embed.med
                + p.dgt_hash.med
                + p.rs_enc.med
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
                + p.v_reconstruct.med
                + p.v_syndrome.med
                + p.v_kahe_dec.med
                + p.v_digest.med
                + r.dec_app.med
        })
        .collect();
    Model {
        fixed_us,
        // Mean cost of one actual chunk, so `fixed + l·chunk == wall` exactly.
        // The final chunk may be partial — read `polys` vs `μ` in the cells
        // table for how full it is.
        chunk_us: chunk_total_us / r.l as f64,
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
//   A  client upload — N parallel uplinks; each client streams its bulletin
//      post and its S openings as parallel flows. Gated by the slowest of:
//      a client→bulletin flow, a client→servers flow, a server downlink
//      (ingesting N openings), the bulletin ingest (N posts). Every phase is
//      gated by both ends of each flow.
//   B  server post — S entries onto the bulletin.
//   C  verifier read — the full bulletin over the verifier's downlink.
// No compute/transfer overlap is modelled, so e2e = CPU wall + net is the
// conservative end of pipelined reality. Deterministic jitter seed per row.

struct NetPoint {
    direct_net_us: f64,
    direct_e2e_us: f64,
    agg: Vec<(f64, f64)>, // (net_us, e2e_us) per AggPlan
    rs: Vec<(f64, f64)>,  // (net_us, e2e_us) per RsPlan
}

fn net_sim(r: &Row, m: &Model, prof: &NetProfile, nrng: &mut ChaCha20Rng) -> NetPoint {
    // Mbit/s → bytes/µs. Client links vs cloud (server/bulletin/verifier).
    let xfer_cl = |b: f64| b / (prof.client_mbps / 8.0);
    let xfer_srv = |b: f64| b / (prof.server_mbps / 8.0);
    let mut maxlat = |k: usize| -> f64 {
        (0..k)
            .map(|_| prof.lat_ms + nrng.gen::<f64>() * prof.jitter_ms)
            .fold(0.0, f64::max)
            * 1e3
    };

    let a = (maxlat(r.n) + xfer_cl(m.post_b)) // client→bulletin flow
        .max(maxlat(r.n) + xfer_cl(m.client_open_b)) // client→servers flow
        .max(maxlat(r.n) + xfer_srv(r.wire_opening_b / r.s as f64)) // server downlink
        .max(maxlat(r.n) + xfer_srv(r.n as f64 * m.post_b)); // bulletin ingest
    let b = maxlat(r.s) + xfer_srv(r.wire_server_b);
    let c = maxlat(1) + xfer_srv(r.wire_comm_b + r.wire_server_b + r.wire_ctxt_b);
    let direct_net_us = a + b + c;

    // Aggregated flow: ctxt+comm go to aggregators (not broadcast); each level
    // sums g and forwards one aggregate up the tree. Openings→servers and the
    // server post are unchanged. Agg phase = Σ over levels of (one aggregator's
    // CPU + one hop's transfer). Conservative (no overlap).
    let agg = r
        .agg_plans
        .iter()
        .zip(&m.agg_cpu_us)
        .map(|(p, &agg_cpu_us)| {
            let g = p.group_size as f64;
            let a = (maxlat(r.n) + xfer_cl(m.post_b)) // client→aggregator flow
                .max(maxlat(r.n) + xfer_cl(m.client_open_b)) // client→servers flow
                .max(maxlat(r.n) + xfer_srv(r.wire_opening_b / r.s as f64)) // server downlink
                .max(maxlat(p.layer_counts[0]) + xfer_srv(g * m.post_b)); // L1 ingest
            let mut counts = p.layer_counts.clone();
            counts.push(1); // leader
            let mut agg_us = 0.0;
            for h in 0..p.layers {
                let (senders, receivers) = (counts[h], counts[h + 1]);
                let fan_in = if receivers == 1 { senders as f64 } else { g };
                agg_us += p.agg_cpu.med
                    + (maxlat(senders) + xfer_srv(m.post_b)) // sender uplink
                        .max(maxlat(receivers) + xfer_srv(fan_in * m.post_b)); // receiver ingest
            }
            let b = maxlat(r.s) + xfer_srv(r.wire_server_b);
            let leader_in = *p.layer_counts.last().unwrap();
            let c = maxlat(1) + xfer_srv(r.wire_server_b + leader_in as f64 * m.post_b);
            let net_us = a + agg_us + b + c;
            (net_us, agg_cpu_us + net_us)
        })
        .collect();

    // RS flow: the client uplinks n coded shares (total (n/k)·C) plus its S
    // openings; the bulletin only ever sees the constant-size post. A node
    // ingests ρ shares of C/k. The verifier reads k lane sums, the server
    // entries, the node posts and every client's bulletin post (it must check
    // the signatures and sum the digests).
    let rs = r
        .rs_plans
        .iter()
        .zip(&m.rs_cpu_us)
        .map(|(p, &rs_cpu_us)| {
            let share_b = p.share_b as f64;
            let client_shares_b = p.n_nodes as f64 * share_b;
            let node_in_b = r.n as f64 * share_b;
            let a = (maxlat(r.n) + xfer_cl(client_shares_b + p.bulletin_b as f64)) // client uplink
                .max(maxlat(r.n) + xfer_cl(m.client_open_b)) // client→servers flow
                .max(maxlat(r.n) + xfer_srv(r.wire_opening_b / r.s as f64)) // server downlink
                .max(maxlat(r.n) + xfer_srv(node_in_b)) // one lane's ingest
                .max(maxlat(r.n) + xfer_srv(r.n as f64 * p.bulletin_b as f64)); // bulletin ingest
            let b = (maxlat(r.s) + xfer_srv(r.wire_server_b))
                .max(maxlat(p.n_nodes) + xfer_srv(p.n_nodes as f64 * share_b));
            // All n share-sums: k reconstruct, the rest are the syndrome.
            let c = maxlat(1)
                + xfer_srv(
                    p.n_nodes as f64 * share_b
                        + r.wire_server_b
                        + r.n as f64 * p.bulletin_b as f64,
                );
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

fn print_tables(rows: &[Row]) {
    let models: Vec<Model> = rows.iter().map(model).collect();

    println!("── cells ───────────────────────────────────────────────────────────────────");
    println!(
        "{:<4}{:>4}{:>6}{:>6}{:>6}  {:<6}{:>12}{:>6}{:>7}{:>4}{:>8}{:>7}{:>7}  {}",
        "id", "S", "ρ", "act", "cov", "flow", "payload/cl", "δ", "ξ", "l", "polys", "μ", "cells",
        "recovered"
    );
    for (i, r) in rows.iter().enumerate() {
        println!(
            "{:<4}{:>4}{:>6}{:>6}{:>6}  {:<6}{:>12}{:>6}{:>7}{:>4}{:>8}{:>7}{:>7}  {}",
            cell_id(i),
            r.s,
            r.n,
            r.active,
            r.cover,
            r.flow,
            fmt_bytes(r.payload_client_b as f64),
            r.delta,
            r.xi,
            r.l,
            r.n_polys,
            r.mu_kahe,
            r.iblt_cells,
            if r.recovered_ok { "yes" } else { "NO" },
        );
    }
    println!("    (polys < l·μ ⇒ the final KAHE chunk is partial)");
    println!();

    println!(
        "── [M] cpu — per-round totals for one party, median of {} (min/max in csv) ────",
        REPS
    );
    println!(
        "{:4}{:─^50} {:─^20} {:─^60}",
        "", " client ", " server ", " verifier "
    );
    println!(
        "{:<4}{:>10}{:>10}{:>10}{:>10}{:>10} {:>10}{:>10} {:>10}{:>10}{:>10}{:>10}{:>10}{:>10}",
        "id", "enc_app", "kahe_enc", "share", "cs", "seal", "unseal", "round", "agg_ctxt",
        "sum_comm", "open", "interp", "kahe_dec", "dec_app"
    );
    for (i, r) in rows.iter().enumerate() {
        println!(
            "{:<4}{:>10}{:>10}{:>10}{:>10}{:>10} {:>10}{:>10} {:>10}{:>10}{:>10}{:>10}{:>10}{:>10}",
            cell_id(i),
            fmt_us(r.enc_app.med),
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

    println!("── [M] cpu — aggregated flow (per level / leader) ───────────────────────────");
    println!(
        "{:<4}{:>3}{:>4}  {:<16}{:>11}{:>13}  {}",
        "id", "L", "g", "fan-in", "agg/level", "leader", "recovered"
    );
    for (i, r) in rows.iter().enumerate() {
        for p in &r.agg_plans {
            let tree: Vec<String> = std::iter::once(r.n)
                .chain(p.layer_counts.iter().copied())
                .chain(std::iter::once(1))
                .map(|c| c.to_string())
                .collect();
            println!(
                "{:<4}{:>3}{:>4}  {:<16}{:>11}{:>13}  {}",
                cell_id(i),
                p.layers,
                p.group_size,
                tree.join("→"),
                fmt_us(p.agg_cpu.med),
                fmt_us(p.leader.med),
                if p.recovered { "yes" } else { "NO" },
            );
        }
    }
    println!();

    if rows.iter().any(|r| !r.rs_plans.is_empty()) {
        println!("── [M] cpu — rs-sharded ingress (client rs_enc / node sum / verifier) ───────");
        println!(
            "{:<4}{:>8}  {:>8}{:>8}{:>9}{:>10} {:>9}{:>9}{:>11}{:>10}{:>9}{:>10}  {:<10}{}",
            "id", "k/n", "embed", "hash", "rs_enc", "node_sum", "sig", "open", "reconstruct",
            "syndrome", "digest", "kahe_dec", "recovered", "lane lie"
        );
        for (i, r) in rows.iter().enumerate() {
            for p in &r.rs_plans {
                println!(
                    "{:<4}{:>8}  {:>8}{:>8}{:>9}{:>10} {:>9}{:>9}{:>11}{:>10}{:>9}{:>10}  {:<10}{}",
                    cell_id(i),
                    format!("{}/{}", p.k, p.n_nodes),
                    fmt_us(p.dgt_embed.med),
                    fmt_us(p.dgt_hash.med),
                    fmt_us(p.rs_enc.med),
                    fmt_us(p.node_sum.med),
                    fmt_us(p.v_sig.med),
                    fmt_us(p.v_open.med),
                    fmt_us(p.v_reconstruct.med),
                    fmt_us(p.v_syndrome.med),
                    fmt_us(p.v_digest.med),
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

        println!("── [D] rs sizing — measured at the run k/n, priced across k at n = k+2 ──────");
        println!(
            "{:<4}{:>5}{:>14}{:>12}{:>16}{:>15}{:>14}  {}",
            "id", "k", "share = C/k", "bulletin", "client egress", "node ingress", "verifier in",
            "gate"
        );
        for (i, r) in rows.iter().enumerate() {
            if r.rs_plans.is_empty() {
                continue;
            }
            let bulletin_b = r.rs_plans[0].bulletin_b as f64;
            let open_b = (r.open_env_b * r.s) as f64;
            let ctxt_b = r.ctxt_client_b as f64;
            for &k in RS_SIZING_K {
                let n_nodes = (k + 2).max(r.s);
                let share_b = (r.n_polys.div_ceil(k) * dgt_packed_len()) as f64;
                let egress = n_nodes as f64 * share_b + open_b + bulletin_b;
                let node_in = r.n as f64 * share_b;
                // All n share-sums, not k: the verifier runs the spare-lane
                // syndrome, which needs every share the bulletin carries.
                let verifier_in =
                    n_nodes as f64 * share_b + r.n as f64 * bulletin_b + r.wire_server_b;
                // 100 Mbit client uplink against a 1 Gbit lane downlink.
                let gate = if egress / 12.5e6 > node_in / 125e6 {
                    "client NIC"
                } else {
                    "node ingest"
                };
                println!(
                    "{:<4}{:>5}{:>14}{:>12}{:>16}{:>15}{:>14}  {}",
                    if k == RS_SIZING_K[0] { cell_id(i) } else { String::new() },
                    k,
                    fmt_bytes(share_b),
                    fmt_bytes(bulletin_b),
                    fmt_bytes(egress),
                    fmt_bytes(node_in),
                    fmt_bytes(verifier_in),
                    gate,
                );
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
        "{:<4}{:>10}{:>15}   {:<30}{}",
        "id", "comm/cl", "ctxt/cl", "open_env →1srv (×S /cl)", "srv_entry = agg_open + agg_share"
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
        "{:<4}{:<44}{:<24}{:<24}{}",
        "id", "client → (comm + l·ctxt + S·open)", "server ← / →", "L1-agg ← / →", "leader ← direct / agg-1"
    );
    for (i, r) in rows.iter().enumerate() {
        let m = &models[i];
        let client_ctxt = r.wire_ctxt_b / r.n as f64;
        let client_out = r.comm_client_b as f64 + client_ctxt + m.client_open_b;
        let l1 = r.agg_plans.first();
        let l1_in = l1.map_or(0.0, |p| p.group_size as f64 * m.post_b);
        let leader_agg = l1.map_or(0.0, |p| {
            r.wire_server_b + *p.layer_counts.last().unwrap() as f64 * m.post_b
        });
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

    println!("── [C] model: wall = fixed + l·chunk (client → server → verifier, serial) ───");
    println!(
        "{:<4}{:>10}{:>10}{:>4}{:>10}{:>12}{:>13}{:>10}{:>12}",
        "id", "fixed", "chunk", "l", "wall", "useful", "wire/round", "eff", "agg1-wall"
    );
    for (i, r) in rows.iter().enumerate() {
        let m = &models[i];
        println!(
            "{:<4}{:>10}{:>10}{:>4}{:>10}{:>12}{:>13}{:>10.2e}{:>12}",
            cell_id(i),
            fmt_us(m.fixed_us),
            fmt_us(m.chunk_us),
            r.l,
            fmt_us(m.wall_us),
            fmt_bytes(r.useful_b),
            fmt_bytes(m.wire_total_b),
            m.efficiency,
            m.agg_cpu_us.first().map_or("-".into(), |&us| fmt_us(us)),
        );
    }
    println!();

    println!("── [P] network sim: e2e = [C] wall + synthetic net; MB/s = useful/e2e ───────");
    println!(
        "{:<4}{:<7}{:>12}{:>13}{:>12}{:>12}{}",
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
            let (agg_e2e, agg_mbps) = np.agg.first().map_or(("-".into(), "-".into()), |(_, e2e)| {
                (fmt_us(*e2e), format!("{:.3}", r.useful_b / (e2e / 1e6) / 1e6))
            });
            let rs = np.rs.first().map_or(String::new(), |(_, e2e)| {
                format!(
                    "{:>12}{:>12.3}",
                    fmt_us(*e2e),
                    r.useful_b / (e2e / 1e6) / 1e6
                )
            });
            println!(
                "{:<4}{:<7}{:>12}{:>13.3}{:>12}{:>12}{}",
                cell_id(i),
                prof.label,
                fmt_us(np.direct_e2e_us),
                r.useful_b / (np.direct_e2e_us / 1e6) / 1e6,
                agg_e2e,
                agg_mbps,
                rs,
            );
        }
    }
    println!(
        "    ({}+U[0,{})ms … per profile; client/cloud Mbps: {})",
        NETWORKS[0].lat_ms,
        NETWORKS[0].jitter_ms,
        NETWORKS
            .iter()
            .map(|p| format!("{} {}/{}", p.label, p.client_mbps, p.server_mbps))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

// ── csv ─────────────────────────────────────────────────────────────────────

const CSV_PATH: &str = "scaling_sweep.csv";

fn rs0(r: &Row) -> Option<&RsPlan> {
    r.rs_plans.first()
}

fn write_csv(rows: &[Row]) {
    type PhaseGet = fn(&Row) -> Stat;
    let phases: &[(&str, PhaseGet)] = &[
        ("enc_app", |r| r.enc_app),
        ("kahe_enc", |r| r.kahe_enc),
        ("share", |r| r.share),
        ("cs_commit", |r| r.cs),
        ("seal", |r| r.seal),
        ("unseal", |r| r.unseal),
        ("server_round", |r| r.server),
        ("verify_agg_ctxt", |r| r.v_agg_ctxt),
        ("verify_sum_comm", |r| r.v_sum_comm),
        ("verify_open", |r| r.v_open),
        ("verify_interp", |r| r.v_interp),
        ("verify_kahe_dec", |r| r.v_kahe_dec),
        ("dec_app", |r| r.dec_app),
        ("agg1_level", |r| r.agg_plans.first().map_or(Stat::default(), |p| p.agg_cpu)),
        ("agg1_leader", |r| r.agg_plans.first().map_or(Stat::default(), |p| p.leader)),
        ("rs_dgt_embed", |r| rs0(r).map_or(Stat::default(), |p| p.dgt_embed)),
        ("rs_dgt_hash", |r| rs0(r).map_or(Stat::default(), |p| p.dgt_hash)),
        ("rs_enc", |r| rs0(r).map_or(Stat::default(), |p| p.rs_enc)),
        ("rs_node_sum", |r| rs0(r).map_or(Stat::default(), |p| p.node_sum)),
        ("rs_verify_sig", |r| rs0(r).map_or(Stat::default(), |p| p.v_sig)),
        ("rs_verify_open", |r| rs0(r).map_or(Stat::default(), |p| p.v_open)),
        ("rs_verify_interp", |r| rs0(r).map_or(Stat::default(), |p| p.v_interp)),
        ("rs_reconstruct", |r| rs0(r).map_or(Stat::default(), |p| p.v_reconstruct)),
        ("rs_reconstruct_lagrange", |r| {
            rs0(r).map_or(Stat::default(), |p| p.v_reconstruct_lagrange)
        }),
        ("rs_verify_kahe_dec", |r| rs0(r).map_or(Stat::default(), |p| p.v_kahe_dec)),
        ("rs_syndrome", |r| rs0(r).map_or(Stat::default(), |p| p.v_syndrome)),
        ("rs_verify_digest", |r| rs0(r).map_or(Stat::default(), |p| p.v_digest)),
    ];

    let mut header = String::new();
    header.push_str("id,s,rho,active,cover,flow,payload_client_b,delta,xi,l,n_polys,mu_kahe,iblt_cells,recovered,agg1_recovered,rs_k,rs_n,rs_recovered,rs_lane_lie");
    for (name, _) in phases {
        write!(header, ",m_{0}_us_med,m_{0}_us_min,m_{0}_us_max", name).unwrap();
    }
    header.push_str(
        ",m_open_env_b\
         ,d_comm_client_b,d_ctxt_client_b,d_agg_open_b,d_agg_share_b\
         ,d_wire_ctxt_b,d_wire_comm_b,d_wire_opening_b,d_wire_server_b,d_useful_b\
         ,d_rs_share_b,d_rs_bulletin_b,d_rs_client_egress_b,d_rs_node_ingress_b\
         ,c_fixed_us,c_chunk_us,c_wall_us,c_efficiency,c_agg1_wall_us,c_rs_wall_us",
    );
    for prof in NETWORKS {
        write!(
            header,
            ",p_{0}_direct_net_us,p_{0}_direct_e2e_us,p_{0}_direct_mbps,p_{0}_agg1_net_us,p_{0}_agg1_e2e_us,p_{0}_agg1_mbps,p_{0}_rs_net_us,p_{0}_rs_e2e_us,p_{0}_rs_mbps",
            prof.label
        )
        .unwrap();
    }

    // Param columns s..iblt_cells (skipping the run-local `id`) identify a
    // cell for cross-run dedup in append mode.
    let param_key = |line: &str| -> String {
        line.split(',').skip(1).take(12).collect::<Vec<_>>().join(",")
    };

    let mut out = String::new();
    for (i, r) in rows.iter().enumerate() {
        let m = model(r);
        write!(
            out,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            cell_id(i),
            r.s,
            r.n,
            r.active,
            r.cover,
            r.flow,
            r.payload_client_b,
            r.delta,
            r.xi,
            r.l,
            r.n_polys,
            r.mu_kahe,
            r.iblt_cells,
            r.recovered_ok,
            r.agg_plans.first().map_or(false, |p| p.recovered),
            rs0(r).map_or(0, |p| p.k),
            rs0(r).map_or(0, |p| p.n_nodes),
            rs0(r).map_or(false, |p| p.recovered),
            rs0(r).map_or("-", |p| p.lane_lie_caught),
        )
        .unwrap();
        for (_, get) in phases {
            let st = get(r);
            write!(out, ",{:.3},{:.3},{:.3}", st.med, st.min, st.max).unwrap();
        }
        write!(
            out,
            ",{},{},{},{},{},{:.0},{:.0},{:.0},{:.0},{:.0}",
            r.open_env_b,
            r.comm_client_b,
            r.ctxt_client_b,
            r.agg_open_b,
            r.agg_share_b,
            r.wire_ctxt_b,
            r.wire_comm_b,
            r.wire_opening_b,
            r.wire_server_b,
            r.useful_b,
        )
        .unwrap();
        let (rs_share_b, rs_bulletin_b, rs_egress_b, rs_node_in_b) = match rs0(r) {
            None => (0.0, 0.0, 0.0, 0.0),
            Some(p) => (
                p.share_b as f64,
                p.bulletin_b as f64,
                p.n_nodes as f64 * p.share_b as f64
                    + (r.open_env_b * r.s) as f64
                    + p.bulletin_b as f64,
                r.n as f64 * p.share_b as f64,
            ),
        };
        write!(
            out,
            ",{:.0},{:.0},{:.0},{:.0}",
            rs_share_b, rs_bulletin_b, rs_egress_b, rs_node_in_b
        )
        .unwrap();
        write!(
            out,
            ",{:.3},{:.3},{:.3},{:.6e},{:.3},{:.3}",
            m.fixed_us,
            m.chunk_us,
            m.wall_us,
            m.efficiency,
            m.agg_cpu_us.first().copied().unwrap_or(0.0),
            m.rs_cpu_us.first().copied().unwrap_or(0.0),
        )
        .unwrap();
        let mbps = |e2e_us: f64| r.useful_b / (e2e_us / 1e6) / 1e6;
        for np in net_all(r, &m) {
            let (agg_net, agg_e2e) = np.agg.first().copied().unwrap_or((0.0, 0.0));
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
    let append = std::env::var("SWEEP_APPEND").map_or(false, |v| v == "1");
    let mut lines: Vec<String> = Vec::new();
    if append {
        if let Ok(old) = std::fs::read_to_string(CSV_PATH) {
            let mut it = old.lines();
            if it.next() == Some(header.as_str()) {
                let new_keys: Vec<String> = out.lines().map(param_key).collect();
                lines.extend(
                    it.filter(|l| !new_keys.contains(&param_key(l))).map(String::from),
                );
            } else {
                eprintln!("csv schema changed; overwriting {CSV_PATH}");
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
    std::fs::write(CSV_PATH, merged).unwrap();
}

pub fn run() {
    let budget = Duration::from_secs(
        std::env::var("BENCH_BUDGET_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300),
    );
    let start = Instant::now();

    println!("Panetière scaling bench  (budget: {}s, {} reps/phase)", budget.as_secs(), REPS);
    println!(
        "ring: HVC {} bits/coef ({} B/poly) | KAHE {} bits/coef ({} B/poly) | t bits/symbol {}",
        poly_packed_len(HVC_MODULUS) * 8 / POLY_N,
        poly_packed_len(HVC_MODULUS),
        poly_packed_len64(KAHE_MODULUS) * 8 / POLY_N,
        poly_packed_len64(KAHE_MODULUS),
        BITS_PER_SYMBOL,
    );
    println!("provenance: [M] measured   — median of {} one-party reps (spread in csv)", REPS);
    println!("            [D] derived    — exact byte arithmetic from packing formulas");
    println!("            [C] composed   — model wall = fixed + l·chunk over [M] medians");
    println!("            [P] projected  — extrapolation / synthetic network sim");
    println!("useful = active · ξ · log₂(t) bits/round (only active clients carry payload)");
    println!();

    let servers = env_list::<usize>("SWEEP_SERVERS").unwrap_or_else(|| SERVERS.to_vec());
    let clients = sweep_clients();
    let configs = sweep_configs();
    let sched_bytes =
        env_list::<usize>("SWEEP_SCHED_BYTES").unwrap_or_else(|| SCHED_MESSAGE_BYTES.to_vec());

    // The protocol sweep, flattened once. Scheduled ignores ξ (its token IBLT
    // is fixed at 2 symbols), so it varies over sched_bytes with configs[0].
    let variants: Vec<(&Config, AppCodec, String)> = configs
        .iter()
        .flat_map(|cfg| {
            [
                (cfg, AppCodec::Mse, format!("mse {}", cfg.label)),
                (cfg, AppCodec::Prony, format!("prony {}", cfg.label)),
            ]
        })
        .chain(sched_bytes.iter().map(|&mb| {
            (
                &configs[0],
                AppCodec::Scheduled { message_bytes: mb },
                format!("sched msg={}", fmt_bytes(mb as f64)),
            )
        }))
        .collect();

    let mut rows: Vec<Row> = Vec::new();
    let mut skipped = false;

    // Nesting is clients → servers → protocol, protocol innermost, so a budget
    // cut-off leaves *complete* protocol comparisons at every (client, server)
    // point it reached rather than one protocol across all of them.
    'sweep: for &(clients_total, clients_active) in &clients {
        for &s in &servers {
            for (cfg, codec, desc) in &variants {
                if start.elapsed() >= budget {
                    skipped = true;
                    break 'sweep;
                }
                eprintln!(
                    "running cell {}: clients={clients_total}x{clients_active} S={s} {desc}",
                    cell_id(rows.len())
                );
                rows.push(run_cell(s, clients_total, clients_active, cfg, codec));
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
    println!("csv: {} ({} cells)", CSV_PATH, rows.len());
    println!("done in {:.1}s", start.elapsed().as_secs_f64());
}
