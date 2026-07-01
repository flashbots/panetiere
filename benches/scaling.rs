//! End-to-end scaling bench: sweeps `(S, N)` × MSE/KAHE configs.
//!
//! Headline metric: **useful multiset throughput** — bits/sec of recovered
//! application payload through the full pipeline. MSE is the coding-rate
//! knob, so the bench leads with bandwidth (wire bytes / useful bits /
//! efficiency). Per-phase CPU times follow for bottleneck attribution:
//!   client.mse, client.proto, server.proto, verify.proto, verify.mse
//!
//! Each cell runs ONE round with `μ_kahe = MseEncoding::n_polys(mse_params)`;
//! every client encodes a distinct (element, r). Recovery is asserted
//! before timing.
//!
//! ──────────────────────────────────────────────────────────────────────
//! INPUTS (knobs — sweep by editing CELLS / CONFIGS, or via env)
//! ──────────────────────────────────────────────────────────────────────
//!  CELLS:   (S, ρ)        servers, anonymity-set size (= total clients)
//!  ACTIVE_CLIENTS:        clients carrying real msgs; ρ−active send cover.
//!                         IBLT is sized to active, not ρ.
//!  CONFIGS: γ             MSE rows (practical sweet spot = 4)
//!           δ_factor      δ = δ_factor · active (paper sweet spot = 2)
//!           ξ             payload symbols/client (useful bits = ξ·log₂ t)
//!           κ_kahe        KAHE key components (coupled to μ_cs in ProtocolParams)
//!  NETWORKS: per-profile one-way latency + uniform jitter, link Mbit/s
//!           (wire-time sim; reported separately from CPU, then combined)
//!  env:     BENCH_BUDGET_SECS  default 300
//!           RAYON_NUM_THREADS  recommend 8
//!
//! ──────────────────────────────────────────────────────────────────────
//! DERIVED FROM INPUTS (computed per cell — not knobs)
//! ──────────────────────────────────────────────────────────────────────
//!  μ_kahe         = MseEncoding::n_polys(mse_params)
//!                 = ⌈γ · δ · (1 + K_LIMBS + ξ) / 512⌉   (KAHE message width)
//!  total_cells    = γ · δ
//!  total_scalars  = (1 + K_LIMBS + ξ) · total_cells    (Z_t coefs per MSE bucket)
//!  opening_polys  = κ_cs + μ_cs + (2·block_size−2)·HVC_WIDTH
//!                 + stored_path_len·2·HVC_WIDTH   (read off live Opening)
//!  Shamir threshold, κ_cs, β, β_agg, block_size — chosen by
//!  `ProtocolParams::setup_with_kahe_dims_full(s, μ_kahe, κ_kahe, …)`
//!
//! ──────────────────────────────────────────────────────────────────────
//! CONSTANTS (pinned upstream — edit the source, not this file)
//! ──────────────────────────────────────────────────────────────────────
//!  chipmunk/src/param.rs:
//!    q_cs       = HVC_MODULUS    = 25_601           (~15 bits) // good to keep it < 2^15 so that
//!    it fits into 16bit AVX. paper suggests 40k.
//!    q_kahe     = KAHE_MODULUS   = 1_073_738_753    (~30 bits, fits i32 NTT). bigger q_kahe gives
//!    a lot more useful plaintext bits, but at a big performance cost since we need a slower algo
//!    N (ring degree)             = 512 (hardcoded in chipmunk, could change it if wanted)
//!    HVC_WIDTH                   = 3
//!
//!  src/kahe.rs:
//!    t          = T_MODULUS_DEFAULT = 262_144 (= 2^18) // must be within noise budget
//!    σ_s        = SIGMA_S_DEFAULT   = 4.5 // noise
//!    σ_e        = SIGMA_E_DEFAULT   = √2 · σ_s
//!    noise budget: t·8σ_e·√ρ + ρ·t/2 < q_kahe/2  (holds for ρ ≤ ~300)
//!
//!  src/mse.rs:
//!    BITS_PER_SYMBOL = log₂ t          = 18
//!    K_LIMBS                           = 2   (r ∈ Z_{t^2} fits in u64)
//!
//!  bench-internal (this file):
//!    MSE PRF key      = [0xAA; 32]            (deterministic)
//!    ChaCha20Rng seed = f(S, N, κ_kahe, γ, ξ, δ_factor)  (deterministic)
//!    per-client payload = small distinct values in [−t/2, t/2)
//!
//!  protocol coupling (asserted at runtime):
//!    μ_cs = κ_kahe          (src/protocol/client.rs:37). Doesnt have to be the case, but it makes
//!    things simpler
//!
//! Run with:
//!   RAYON_NUM_THREADS=8 cargo bench -j 8 --bench scaling
//!   BENCH_BUDGET_SECS=600 cargo bench --bench scaling

use std::time::{Duration, Instant};

use chipmunk_code::{KahePoly, CS_MODULUS, HVC_MODULUS, KAHE_MODULUS, N as POLY_N, ZETA};
use panetiere::cs::{poly_packed_len, Commitment, Cs, HidingMerkleCommitment};
use panetiere::kahe::{Kahe, KaheScheme, SIGMA_E_DEFAULT, SIGMA_S_DEFAULT, T_MODULUS_DEFAULT};
use panetiere::mse::{MseEncoding, MseParams, BITS_PER_SYMBOL, K_LIMBS};
use panetiere::protocol::aggregator::run_aggregator_round;
use panetiere::protocol::client::{
    cs_commit, kahe_encrypt, kahe_keygen, run_client_round, shamir_share,
};
use panetiere::protocol::server::{run_server_round, ServerInbox};
use panetiere::protocol::verify::{aggregate_and_decrypt_timed, decrypt_aggregate, VerifyTimings};
use panetiere::protocol::ProtocolParams;
use panetiere::protocol::{ClientId, ServerId};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

// (S, ρ): servers, anonymity-set size (= total clients, noise-budget-bounded).
const CELLS: &[(usize, usize)] = &[(128, 100)];

// Two points only: no cover (active = ρ) and the 100-100 active/cover split.
// IBLT is sized to active, so the second row is "half the IBLT, same anonymity
// set". Both at ρ = 200 (CELLS), within the noise budget.
const ACTIVE_CLIENTS: &[usize] = &[100];

struct Config {
    label: &'static str,
    gamma: usize,
    payload_symbols: usize, // ξ: per-client element size = ξ·log₂t bits (512 ⇒ 1 KB)
    kappa_kahe: usize,
}

// MU_KAHE — the KAHE encryption chunk size, in polys. A fixed number, picked so
// one chunk holds the encoding of a 1 KB-element IBLT for 100 clients:
// ⌈BUCKETS · (1 + K_LIMBS + 512) / N⌉ = ⌈300·515/2048⌉ = 76. A larger message's
// IBLT encoding spans l = ⌈n_polys/MU_KAHE⌉ chunks, each under its own matrix
// A_i but the same key sk, so the Shamir+CS pass amortizes over all l chunks.
const MU_KAHE: usize = 76;

// Aggregation depths to evaluate. Group size per depth is balanced so the
// largest per-hop fan-in is ρ^{1/(layers+1)} (see `AggPlan`): 1-layer → ⌈√ρ⌉,
// 2-layer → ⌈∛ρ⌉.
const AGG_LAYERS: &[usize] = &[1];

// Max total KAHE width the scheme supports (noise budget / IBLT capacity): the
// encoding is at most MU_FULL polys ⇒ at most ⌈MU_FULL/MU_KAHE⌉ ≈ 189 chunks.
// Each row extrapolates throughput to this full-utilization point.
const MU_FULL: f64 = 14401.0;

// Per-client element size = ξ symbols · 16 bits: 2048 ⇒ 4 KiB, 8192 ⇒ 16 KiB.
const CONFIGS: &[Config] = &[
    //    Config { label: "4KB", gamma: 4, payload_symbols: 2048, kappa_kahe: 1 },
    Config {
        label: "1KB",
        gamma: 4,
        payload_symbols: 512,
        kappa_kahe: 1,
    },
];

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
        label: "LAN  ",
        lat_ms: 0.5,
        jitter_ms: 0.2,
        client_mbps: 1000.0,
        server_mbps: 1000.0,
    },
    NetProfile {
        label: "fiber",
        lat_ms: 25.0,
        jitter_ms: 10.0,
        client_mbps: 100.0,
        server_mbps: 1000.0,
    },
    NetProfile {
        label: "big  ",
        lat_ms: 25.0,
        jitter_ms: 10.0,
        client_mbps: 100.0,
        server_mbps: 4000.0,
    },
];

fn time_us<F: FnOnce() -> R, R>(f: F) -> (f64, R) {
    let t = Instant::now();
    let r = f();
    (t.elapsed().as_secs_f64() * 1e6, r)
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

// `layer_counts`: aggregator count per level bottom-up; leader ingests the last.
#[derive(Clone)]
struct AggPlan {
    layers: usize,
    group_size: usize,
    layer_counts: Vec<usize>,
    agg_cpu_us: f64,
    leader_us: f64,
    recovered: bool,
}

struct Row {
    s: usize,
    n: usize, // ρ — total clients = anonymity set
    active: usize,
    cover: usize,
    iblt_cells: usize,
    label: &'static str,
    mu_kahe: usize,
    kappa: usize,
    gamma: usize,
    delta: usize,
    xi: usize,
    l: usize,
    recovered_ok: bool,
    // CPU (one round, no averaging). Fixed-per-round vs per-chunk split:
    //   fixed:    client_share + client_cs + server + verify_opening + verify_interp + verify_sum_comm
    //   per-chunk (×l): mse_c + client_kahe + verify_agg_ctxt + verify_kahe_dec + mse_v
    mse_c_us: f64, // total over l inserts
    client_kahe_us: f64,
    client_share_us: f64,
    client_cs_us: f64,
    server_us: f64,
    verify_agg_ctxt_us: f64,
    verify_sum_comm_us: f64,
    verify_opening_us: f64,
    verify_interp_us: f64,
    verify_kahe_us: f64,
    mse_v_us: f64,
    // Bandwidth (bytes per round, totalled across all parties)
    useful_b: f64,
    wire_ctxt_b: f64,
    wire_comm_b: f64,
    wire_opening_b: f64,
    wire_server_b: f64,
    // Aggregated flow, one entry per AGG_LAYERS depth.
    agg_plans: Vec<AggPlan>,
}

fn run_cell(s: usize, n: usize, active: usize, cfg: &Config) -> Row {
    let cover = n - active;
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&[
        s as u8,
        n as u8,
        (n >> 8) as u8,
        active as u8,
        cfg.kappa_kahe as u8,
        cfg.gamma as u8,
        cfg.payload_symbols as u8,
        (cfg.payload_symbols >> 8) as u8,
    ]);
    let mut rng = ChaCha20Rng::from_seed(seed);

    // IBLT sized to the active count only (cover clients add nothing): ~3
    // buckets per insertion so peeling recovers all `active` messages.
    let delta = (3 * active).div_ceil(cfg.gamma);
    let mse_params = MseParams::new(cfg.gamma, delta, cfg.payload_symbols, [0xAA; 32]);
    let n_polys = MseEncoding::n_polys(&mse_params);
    // KAHE encrypts the encoding in l chunks of the fixed size MU_KAHE.
    let mu_kahe = MU_KAHE;
    let l = n_polys.div_ceil(mu_kahe);

    let pp = ProtocolParams::setup_with_kahe_dims_full(
        &mut rng,
        s,
        mu_kahe,
        cfg.kappa_kahe,
        l,
        SIGMA_S_DEFAULT,
        SIGMA_E_DEFAULT,
        T_MODULUS_DEFAULT,
    );
    let server_ids: Vec<ServerId> = (0..s as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n as u32).map(ClientId).collect();

    // One ξ-symbol element per active client. Small distinct values, ≪ t/2,
    // so summing across clients doesn't wrap.
    let payloads: Vec<Vec<i32>> = (0..active)
        .map(|i| {
            (0..cfg.payload_symbols)
                .map(|j| i as i32 + j as i32 + 1)
                .collect()
        })
        .collect();
    // Active clients encode their element; the rest send cover traffic. Both
    // zero-pad to mu_kahe·l polys for KAHE encryption.
    let client_polys: Vec<Vec<KahePoly>> = (0..n)
        .map(|i| {
            let mut polys = if i < active {
                let mut enc = MseEncoding::new(mse_params.clone());
                enc.insert(&mut rng, &payloads[i]);
                enc.pack()
            } else {
                MseEncoding::cover(&mse_params)
            };
            polys.resize(mu_kahe * l, KahePoly::default());
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
    for (i, &cid) in client_ids.iter().enumerate() {
        let round = run_client_round(&mut rng, &pp, cid, client_polys[i].clone(), &server_ids);
        client_entries.push((round.client_id, round.encrypted_message));
        for (idx, (_sid, ops)) in round.encrypted_openings.into_iter().enumerate() {
            inboxes[idx].items.push((cid, ops));
        }
    }
    let canonical = client_ids.clone();
    let outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).unwrap())
        .collect();

    // Recovery is ATTEMPTED, not asserted: at irrecoverable parameter sets
    // (e.g. small q_kahe collapsing the noise budget) aggregation decodes to
    // garbage. We still time every phase and report whether the multiset
    // actually came back, so the bench yields CPU/bandwidth numbers regardless.
    let recovered_ok = match aggregate_and_decrypt_timed(&pp, &canonical, &client_entries, &outputs)
    {
        Ok((recovered, _t0)) => MseEncoding::unpack(&mse_params, &recovered[..n_polys])
            .decode()
            .map(|d| d.len() == active)
            .unwrap_or(false),
        Err(_) => false,
    };

    // Wire ledger. Ciphertext + commitment (root) are uniform-mod-q → tight at
    // ⌈log₂ q⌉. Openings are decomposed digits + small r/s, so we measure the
    // REAL packed size via `Opening::pack` (per-region bit widths).
    let o = &inboxes[0].items[0].1;
    // Client→server opening is FRESH: r ≤ β_cs, tree digits ≤ ζ, s ≤ q_cs/2.
    let cs_half = (CS_MODULUS as u32) / 2;
    let fresh_open_b = o.pack(pp.cs.beta_cs, cs_half, ZETA).body_len();
    // Server-posted aggregate: r ≤ β_agg, tree digits ≤ ρ·ζ (ρ = canonical count);
    // plus agg_share = κ_kahe CS-ring polys at ⌈log₂ q_cs⌉ bits.
    let rho = canonical.len() as u32;
    let agg_open_b = outputs[0]
        .agg_open
        .pack(pp.cs.r_bound, cs_half, rho * ZETA)
        .body_len();
    let agg_share_b = cfg.kappa_kahe * poly_packed_len(CS_MODULUS);

    // Per-phase timings (one round each).
    let (mse_c_us, _) = time_us(|| {
        let mut enc = MseEncoding::new(mse_params.clone());
        enc.insert(&mut rng, &payloads[0]);
        enc.pack()
    });
    let key0 = kahe_keygen(&mut rng, &pp);
    let (client_kahe_us, _ctxt0) = time_us(|| kahe_encrypt(&mut rng, &pp, &key0, &client_polys[0]));
    let (client_share_us, shares0) = time_us(|| shamir_share(&mut rng, &pp, &key0, s));
    let (client_cs_us, _) = time_us(|| cs_commit(&mut rng, &pp, &shares0));
    let (server_us, _) = time_us(|| run_server_round(&inboxes[0], &canonical).unwrap());
    let (verify_us_total, (recovered2, vt)) = time_us(|| {
        aggregate_and_decrypt_timed(&pp, &canonical, &client_entries, &outputs).unwrap()
    });
    let _ = verify_us_total;
    let VerifyTimings {
        agg_ctxt_us,
        sum_comm_us,
        opening_verify_us,
        interpolation_us,
        kahe_dec_us,
    } = vt;
    let (mse_v_us, _) = time_us(|| {
        // May Err (PeelStalled) on unrecoverable params — we time the attempt.
        let _ = MseEncoding::unpack(&mse_params, &recovered2[..n_polys]).decode();
    });

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
            let l1_entries = |grp: usize| -> Vec<_> {
                client_entries
                    .iter()
                    .filter(|(cid, _)| cid.0 as usize % a1 == grp)
                    .cloned()
                    .collect::<Vec<_>>()
            };
            let (agg_cpu_us, _) = time_us(|| run_aggregator_round(&l1_entries(0)));
            let l1: Vec<_> = (0..a1)
                .map(|grp| run_aggregator_round(&l1_entries(grp)))
                .collect();

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

            let (leader_us, recovered) = {
                let (us, res) = time_us(|| {
                    let total_ctxt = Kahe::agg_ctxt(&ctxts);
                    let total_comm = HidingMerkleCommitment::sum_commitments(&comms);
                    decrypt_aggregate(&pp, &total_ctxt, &total_comm, &outputs)
                });
                let ok = match res {
                    Ok(rec) => MseEncoding::unpack(&mse_params, &rec[..n_polys])
                        .decode()
                        .map(|d| d.len() == active)
                        .unwrap_or(false),
                    Err(_) => false,
                };
                (us, ok)
            };

            AggPlan {
                layers,
                group_size: g,
                layer_counts,
                agg_cpu_us,
                leader_us,
                recovered,
            }
        })
        .collect();

    Row {
        s,
        n,
        active,
        cover,
        iblt_cells: mse_params.total_cells(),
        label: cfg.label,
        mu_kahe,
        kappa: cfg.kappa_kahe,
        gamma: cfg.gamma,
        delta,
        xi: cfg.payload_symbols,
        l,
        recovered_ok,
        mse_c_us,
        client_kahe_us,
        client_share_us,
        client_cs_us,
        server_us,
        verify_agg_ctxt_us: agg_ctxt_us,
        verify_sum_comm_us: sum_comm_us,
        verify_opening_us: opening_verify_us,
        verify_interp_us: interpolation_us,
        verify_kahe_us: kahe_dec_us,
        mse_v_us,
        useful_b: (active * cfg.payload_symbols * BITS_PER_SYMBOL) as f64 / 8.0,
        wire_ctxt_b: n as f64 * (mu_kahe * l) as f64 * poly_packed_len(KAHE_MODULUS) as f64,
        wire_comm_b: n as f64 * poly_packed_len(HVC_MODULUS) as f64,
        wire_opening_b: (n * s) as f64 * fresh_open_b as f64,
        wire_server_b: s as f64 * (agg_open_b + agg_share_b) as f64,
        agg_plans,
    }
}

fn print_row(r: &Row) {
    // Fixed-per-round = Shamir share + CS commit + server + opening verify
    //                 + Shamir interpolation + commitment summation.
    // Per-chunk (×l)   = MSE encode + KAHE enc + KAHE agg_ctxt + KAHE dec + MSE decode.
    let fixed_us = r.client_share_us
        + r.client_cs_us
        + r.server_us
        + r.verify_opening_us
        + r.verify_interp_us
        + r.verify_sum_comm_us;
    let per_chunk_total_us =
        r.mse_c_us + r.client_kahe_us + r.verify_agg_ctxt_us + r.verify_kahe_us + r.mse_v_us;
    let per_chunk_us = per_chunk_total_us / r.l as f64;
    let per_round_us = fixed_us + per_chunk_total_us;
    let wire_total = r.wire_ctxt_b + r.wire_comm_b + r.wire_opening_b + r.wire_server_b;
    let useful_mb_s = r.useful_b / (per_round_us / 1e6) / 1e6;
    let wire_mb_s = wire_total / (per_round_us / 1e6) / 1e6;
    let efficiency = r.useful_b / wire_total;

    println!(
        "S={:>2} ρ={:>4} active={:>4} cover={:>4} {:>5} (cells={}, μ_kahe={}, κ_kahe={}, γ={}, δ={}, K={}, ξ={}, l={})  recovered: {}",
        r.s,
        r.n,
        r.active,
        r.cover,
        r.label,
        r.iblt_cells,
        r.mu_kahe,
        r.kappa,
        r.gamma,
        r.delta,
        K_LIMBS,
        r.xi,
        r.l,
        if r.recovered_ok {
            "YES"
        } else {
            "NO (irrecoverable params — timings still valid)"
        },
    );
    println!(
        "  fixed/round: {:>10}   [share {} + cs {} + server {} + verify_open {} + verify_interp {} + sum_comm {}]",
        fmt_us(fixed_us),
        fmt_us(r.client_share_us),
        fmt_us(r.client_cs_us),
        fmt_us(r.server_us),
        fmt_us(r.verify_opening_us),
        fmt_us(r.verify_interp_us),
        fmt_us(r.verify_sum_comm_us),
    );
    println!(
        "  per-chunk:   {:>10}   [mse_c {} + kahe_enc {} + verify_agg_ctxt {} + verify_kahe_dec {} + mse_v {}]   (×l={})",
        fmt_us(per_chunk_us),
        fmt_us(r.mse_c_us / r.l as f64),
        fmt_us(r.client_kahe_us / r.l as f64),
        fmt_us(r.verify_agg_ctxt_us / r.l as f64),
        fmt_us(r.verify_kahe_us / r.l as f64),
        fmt_us(r.mse_v_us / r.l as f64),
        r.l,
    );
    println!(
        "  total wall:  {:>10}  = {} + {} · {}",
        fmt_us(per_round_us),
        fmt_us(fixed_us),
        r.l,
        fmt_us(per_chunk_us),
    );
    println!(
        "  useful: {} / round  ⇒  {:.3} MB/s  (wall per 1 MiB useful: {})",
        fmt_bytes(r.useful_b),
        useful_mb_s,
        fmt_us(1024.0 * 1024.0 / r.useful_b * per_round_us),
    );
    let leader_direct_b = r.wire_ctxt_b + r.wire_comm_b;
    let one_pub_b = leader_direct_b / r.n as f64; // one client post = one aggregate
    let direct_verify_us = r.verify_agg_ctxt_us
        + r.verify_sum_comm_us
        + r.verify_opening_us
        + r.verify_interp_us
        + r.verify_kahe_us;
    println!(
        "  aggregated (direct leader: decode {}, ingest {} ctxt+comm):",
        fmt_us(direct_verify_us),
        fmt_bytes(leader_direct_b),
    );
    for p in &r.agg_plans {
        let leader_in = *p.layer_counts.last().unwrap();
        let tree: Vec<String> = std::iter::once(r.n)
            .chain(p.layer_counts.iter().copied())
            .map(|c| c.to_string())
            .collect();
        println!(
            "    {}-layer (g={}, fan-in {}→1): agg/level {}, leader decode {}, ingest {} ({} aggregates){}",
            p.layers,
            p.group_size,
            tree.join("→"),
            fmt_us(p.agg_cpu_us),
            fmt_us(p.leader_us),
            fmt_bytes(leader_in as f64 * one_pub_b),
            leader_in,
            if p.recovered { "" } else { "  [UNRECOVERED]" },
        );
    }

    // Extrapolate to full utilization: total width MU_FULL ⇒ l_full chunks of
    // MU_KAHE. The per-chunk cost is ~constant, the fixed (Shamir+CS+tree) cost
    // is paid once, and useful payload ∝ chunk count.
    let l_full = MU_FULL / r.mu_kahe as f64;
    let per_chunk = per_chunk_total_us / r.l as f64;
    let wall_full_us = fixed_us + per_chunk * l_full;
    let useful_full_b = r.useful_b * l_full / r.l as f64;
    let thr_full = useful_full_b / (wall_full_us / 1e6) / 1e6;
    println!(
        "  ⇒ extrapolated @ full util (μ_total={:.0}, l≈{:.0}): {} useful, wall {}, {:.3} MB/s",
        MU_FULL,
        l_full,
        fmt_bytes(useful_full_b),
        fmt_us(wall_full_us),
        thr_full,
    );
    println!(
        "  wire:   {} / round  ({:.2} MB/s)   →   efficiency {:.3e}",
        fmt_bytes(wire_total),
        wire_mb_s,
        efficiency,
    );
    println!(
        "          ctxt={} (l·μ) comm={} opening={} (fixed) server_pub={}",
        fmt_bytes(r.wire_ctxt_b),
        fmt_bytes(r.wire_comm_b),
        fmt_bytes(r.wire_opening_b),
        fmt_bytes(r.wire_server_b),
    );

    // ── Per-role segmentation ────────────────────────────────────────────
    // Re-grouping the same wire bytes by who emits them, to expose the true
    // per-party bottleneck instead of the all-parties total above.
    //
    //  single client out = its bulletin post (ctxt+comm, broadcast once)
    //                    + its S point-to-point openings (one per server)
    //  single server out = one ServerBulletinEntry (agg_open + agg_share)
    //  public bulletin   = persistent public state = all client posts
    //                      (ctxt+comm) + all server entries.  Point-to-point
    //                      openings are NOT on the bulletin.
    // Only the KAHE ciphertext scales with the chunk count l; commitments,
    // openings, and server entries are key-related and paid once per round.
    let l = r.l as f64;
    let client_ctxt = r.wire_ctxt_b / r.n as f64; // μ·l polys per client
    let client_comm = r.wire_comm_b / r.n as f64;
    let client_open = r.wire_opening_b / r.n as f64; // → all S servers
    let client_fixed = client_comm + client_open;
    let server_fixed = r.wire_server_b / r.s as f64; // one ServerBulletinEntry
    let bulletin_fixed = r.wire_comm_b + r.wire_server_b;

    let post_b = client_comm + client_ctxt;
    println!("  per-role wire (out = emitted / in = ingested):");
    println!(
        "    1 client out = {}  [comm {} + {} openings→{}S  +  {}·{} ctxt]",
        fmt_bytes(client_fixed + client_ctxt),
        fmt_bytes(client_comm),
        fmt_bytes(client_open),
        r.s,
        r.l,
        fmt_bytes(client_ctxt / l),
    );
    println!(
        "    1 server: out {} (agg_open+agg_share)   in {} ({} clients' openings)",
        fmt_bytes(server_fixed),
        fmt_bytes(r.wire_opening_b / r.s as f64),
        r.n,
    );
    println!(
        "    1 L1 aggregator: in {} (g posts)   out {} (1 aggregate)",
        fmt_bytes(r.agg_plans[0].group_size as f64 * post_b),
        fmt_bytes(post_b),
    );
    println!(
        "    leader in: direct {} (comm {} + server entries {} + ctxt {})",
        fmt_bytes(bulletin_fixed + r.wire_ctxt_b),
        fmt_bytes(r.wire_comm_b),
        fmt_bytes(r.wire_server_b),
        fmt_bytes(r.wire_ctxt_b),
    );
    for p in &r.agg_plans {
        let leader_in = *p.layer_counts.last().unwrap();
        println!(
            "      {}-layer aggregated: {} (server entries {} + {} aggregates {})",
            p.layers,
            fmt_bytes(r.wire_server_b + leader_in as f64 * post_b),
            fmt_bytes(r.wire_server_b),
            leader_in,
            fmt_bytes(leader_in as f64 * post_b),
        );
    }

    // ── network sim ─────────────────────────────────────────────────────
    // Three sequential wire phases per round, kept separate from CPU wall:
    //   A  client upload — N parallel uplinks; each client streams its
    //      bulletin post and its S openings as parallel flows (each at the
    //      full client rate). Gated by the slowest of: a client→bulletin
    //      flow, a client→servers flow, a server downlink (ingesting N
    //      openings), the bulletin ingest (N posts).
    //      Every phase is gated by both ends of each flow — sender uplink
    //      and receiver downlink each serialize their own total bytes.
    //   B  server post — S entries onto the bulletin (receiver ingest of
    //      all S entries dominates each server's single-entry uplink).
    //   C  verifier read — the full bulletin over the verifier's downlink
    //      (bulletin egress is the same bytes at the same class rate).
    // No compute/transfer overlap is modelled, so e2e = CPU wall + net is
    // the conservative end of pipelined reality. Deterministic jitter seed.
    let mut nrng = ChaCha20Rng::from_seed([0x5E; 32]);
    println!(
        "  network sim (net = A client upload + B server post + C verifier read; e2e = cpu + net):"
    );
    for p in NETWORKS {
        // Mbit/s → bytes/µs. Client links vs cloud (server/bulletin/verifier).
        let xfer_cl = |b: f64| b / (p.client_mbps / 8.0);
        let xfer_srv = |b: f64| b / (p.server_mbps / 8.0);
        let mut maxlat = |k: usize| -> f64 {
            (0..k)
                .map(|_| p.lat_ms + nrng.gen::<f64>() * p.jitter_ms)
                .fold(0.0, f64::max)
                * 1e3
        };
        println!(
            "    {} ({}+U[0,{})ms, client {} / cloud {} Mbps):",
            p.label, p.lat_ms, p.jitter_ms, p.client_mbps, p.server_mbps,
        );
        // Direct flow: every client broadcasts its post (ctxt+comm) to the
        // bulletin; phase A is gated by the bulletin's ingest of all N posts.
        let a = (maxlat(r.n) + xfer_cl(post_b)) // client→bulletin flow
            .max(maxlat(r.n) + xfer_cl(client_open)) // client→servers flow
            .max(maxlat(r.n) + xfer_srv(r.wire_opening_b / r.s as f64)) // server downlink
            .max(maxlat(r.n) + xfer_srv(r.n as f64 * post_b)); // bulletin ingest
        let b = maxlat(r.s) + xfer_srv(r.wire_server_b);
        let c = maxlat(1) + xfer_srv(bulletin_fixed + r.wire_ctxt_b);
        let net = a + b + c;
        let e2e = per_round_us + net;
        println!(
            "      direct: net {:>10} [A {} + B {} + C {}]   e2e {:>10}  →  {:.3} MB/s (cpu-only {:.3})",
            fmt_us(net),
            fmt_us(a),
            fmt_us(b),
            fmt_us(c),
            fmt_us(e2e),
            r.useful_b / (e2e / 1e6) / 1e6,
            r.useful_b / (per_round_us / 1e6) / 1e6,
        );

        // Aggregated flow: ctxt+comm go to aggregators (not broadcast); each
        // level sums g and forwards one aggregate up the tree. Openings→servers
        // and the server post are unchanged. Agg phase = Σ over levels of (one
        // aggregator's CPU + one hop's transfer). Conservative (no overlap).
        let cpu_base_agg =
            (fixed_us - r.verify_opening_us - r.verify_interp_us - r.verify_sum_comm_us)
                + (per_chunk_total_us - r.verify_agg_ctxt_us - r.verify_kahe_us);
        for p in &r.agg_plans {
            let g = p.group_size as f64;
            let a = (maxlat(r.n) + xfer_cl(post_b)) // client→aggregator flow
                .max(maxlat(r.n) + xfer_cl(client_open)) // client→servers flow
                .max(maxlat(r.n) + xfer_srv(r.wire_opening_b / r.s as f64)) // server downlink
                .max(maxlat(p.layer_counts[0]) + xfer_srv(g * post_b)); // L1 ingest of g posts
            let mut counts = p.layer_counts.clone();
            counts.push(1); // leader
            let mut agg = 0.0;
            for h in 0..p.layers {
                let (senders, receivers) = (counts[h], counts[h + 1]);
                let fan_in = if receivers == 1 { senders as f64 } else { g };
                agg += p.agg_cpu_us
                    + (maxlat(senders) + xfer_srv(post_b)) // each sender uploads one aggregate
                        .max(maxlat(receivers) + xfer_srv(fan_in * post_b)); // receiver ingest
            }
            let b = maxlat(r.s) + xfer_srv(r.wire_server_b);
            let leader_in = *p.layer_counts.last().unwrap();
            let c = maxlat(1) + xfer_srv(r.wire_server_b + leader_in as f64 * post_b);
            let net = a + agg + b + c;
            let cpu_agg = cpu_base_agg + p.leader_us;
            let e2e = cpu_agg + net;
            println!(
                "      agg L={} (g={:>2}): net {:>10} [A {} + Agg {} + B {} + C {}]   e2e {:>10}  →  {:.3} MB/s (cpu-only {:.3})",
                p.layers,
                p.group_size,
                fmt_us(net),
                fmt_us(a),
                fmt_us(agg),
                fmt_us(b),
                fmt_us(c),
                fmt_us(e2e),
                r.useful_b / (e2e / 1e6) / 1e6,
                r.useful_b / (cpu_agg / 1e6) / 1e6,
            );
        }
    }
}

fn main() {
    let budget = Duration::from_secs(
        std::env::var("BENCH_BUDGET_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300),
    );
    let start = Instant::now();

    println!("Panetière scaling bench  (budget: {}s)", budget.as_secs());
    println!(
        "ring: HVC {} bits/coef ({} B/poly) | KAHE {} bits/coef ({} B/poly) | t bits/symbol {}",
        poly_packed_len(HVC_MODULUS) * 8 / POLY_N,
        poly_packed_len(HVC_MODULUS),
        poly_packed_len(KAHE_MODULUS) * 8 / POLY_N,
        poly_packed_len(KAHE_MODULUS),
        BITS_PER_SYMBOL,
    );
    println!("per-round wall = fixed + l · per-chunk");
    println!(
        "  fixed   = shamir_share + cs_commit + server + verify_open + verify_interp + sum_comm"
    );
    println!("  chunk   = mse_c + kahe_enc + kahe_agg_ctxt + kahe_dec + mse_v");
    println!(
        "useful = l · active · ξ · log₂(t) bits per round  (only active clients carry payload)"
    );
    println!();

    'outer: for &(s, n) in CELLS {
        for &active in ACTIVE_CLIENTS {
            if active > n {
                continue;
            }
            for cfg in CONFIGS {
                if start.elapsed() >= budget {
                    println!("(budget exhausted; remaining cells skipped)");
                    break 'outer;
                }
                print_row(&run_cell(s, n, active, cfg));
            }
            println!();
        }
    }
    println!("done in {:.1}s", start.elapsed().as_secs_f64());
}
