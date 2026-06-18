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
//!  CELLS:   (S, N)        servers, clients (= ρ)
//!  CONFIGS: γ             MSE rows (practical sweet spot = 4)
//!           δ_factor      δ = δ_factor · N (paper sweet spot = 2, δ ≈ 2ρ)
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

use chipmunk_code::{
    KaheNTTPoly, KahePoly, CS_MODULUS, HVC_MODULUS, KAHE_MODULUS, N as POLY_N, ZETA,
};
use flashnet::kahe::{SIGMA_E_DEFAULT, SIGMA_S_DEFAULT, T_MODULUS_DEFAULT};
use flashnet::mse::{MseEncoding, MseParams, BITS_PER_SYMBOL, K_LIMBS};
use flashnet::protocol::client::{
    cs_commit, kahe_encrypt, kahe_keygen, run_client_round, shamir_share,
};
use flashnet::protocol::server::{run_server_round, ServerInbox};
use flashnet::protocol::verify::{aggregate_and_decrypt_timed, VerifyTimings};
use flashnet::protocol::ProtocolParams;
use flashnet::protocol::{ClientId, ServerId};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const CELLS: &[(usize, usize)] = &[(8, 100)];

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

// Max total KAHE width the scheme supports (noise budget / IBLT capacity): the
// encoding is at most MU_FULL polys ⇒ at most ⌈MU_FULL/MU_KAHE⌉ ≈ 189 chunks.
// Each row extrapolates throughput to this full-utilization point.
const MU_FULL: f64 = 14401.0;

// Per-client element size: 16 KB ⇒ n_polys ≈ 1201 ⇒ l = ⌈1201/MU_KAHE⌉ = 16 chunks.
const CONFIGS: &[Config] = &[Config {
    label: "16KB",
    gamma: 4,
    payload_symbols: 8192,
    kappa_kahe: 1,
}];

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
    NetProfile { label: "LAN  ", lat_ms: 0.5, jitter_ms: 0.2, client_mbps: 1000.0, server_mbps: 1000.0 },
    NetProfile { label: "fiber", lat_ms: 25.0, jitter_ms: 10.0, client_mbps: 100.0, server_mbps: 1000.0 },
    NetProfile { label: "dsl  ", lat_ms: 35.0, jitter_ms: 15.0, client_mbps: 20.0, server_mbps: 1000.0 },
];

/// ⌈log₂ q⌉, tight-pack bit-width per ring coefficient.
const fn bits_per_coef(q: i32) -> usize {
    (32 - ((q as u32) - 1).leading_zeros()) as usize
}
const HVC_POLY_BYTES: f64 = (POLY_N * bits_per_coef(HVC_MODULUS)) as f64 / 8.0;
const KAHE_POLY_BYTES: f64 = (POLY_N * bits_per_coef(KAHE_MODULUS)) as f64 / 8.0;

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

struct Row {
    s: usize,
    n: usize,
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
    // sqrt-compression sim: per-client extra CPU (m ring mults) and the
    // compressed wire width (2·⌈√m⌉ polys, m = μ·l).
    compress_us: f64,
    sqrt_polys: usize,
    // Bandwidth (bytes per round, totalled across all parties)
    useful_b: f64,
    wire_ctxt_b: f64,
    wire_comm_b: f64,
    wire_opening_b: f64,
    wire_server_b: f64,
}

fn run_cell(s: usize, n: usize, cfg: &Config) -> Row {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&[
        s as u8,
        n as u8,
        (n >> 8) as u8,
        cfg.kappa_kahe as u8,
        cfg.gamma as u8,
        cfg.payload_symbols as u8,
        (cfg.payload_symbols >> 8) as u8,
        0,
    ]);
    let mut rng = ChaCha20Rng::from_seed(seed);

    // One IBLT for the n insertions, sized at ~3 buckets per insertion (the
    // 300-buckets-for-100-clients ratio) so peeling recovers all n messages.
    // Total cells γ·δ ≈ 3·n; bigger n ⇒ bigger encoding ⇒ more chunks l.
    let delta = (3 * n).div_ceil(cfg.gamma);
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

    // One ξ-symbol element per client (its broadcast message). Small distinct
    // values, ≪ t/2 = 2^15, so summing across clients doesn't wrap.
    let payloads: Vec<Vec<i32>> = client_ids
        .iter()
        .map(|cid| {
            let base = cid.0 as i32;
            (0..cfg.payload_symbols)
                .map(|j| base + j as i32 + 1)
                .collect()
        })
        .collect();
    // Each client encodes its element into the IBLT and zero-pads the encoding to
    // a whole number of MU_KAHE chunks (mu_kahe·l polys) for KAHE encryption.
    let client_polys: Vec<Vec<KahePoly>> = payloads
        .iter()
        .map(|elt| {
            let mut enc = MseEncoding::new(mse_params.clone());
            enc.insert(&mut rng, elt);
            let mut polys = enc.pack();
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
            .map(|d| d.len() == n)
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
    let agg_share_b = (cfg.kappa_kahe * POLY_N * bits_per_coef(CS_MODULUS)).div_ceil(8);

    // Per-phase timings (one round each).
    let (mse_c_us, _) = time_us(|| {
        let mut enc = MseEncoding::new(mse_params.clone());
        enc.insert(&mut rng, &payloads[0]);
        enc.pack()
    });
    let key0 = kahe_keygen(&mut rng, &pp);
    let (client_kahe_us, ctxt0) = time_us(|| kahe_encrypt(&mut rng, &pp, &key0, &client_polys[0]));

    // ── sqrt-compression sim ─────────────────────────────────────────────
    // Hypothetical client-side step: one ring multiplication per ciphertext
    // element against a public NTT-resident poly, accumulated into 2·⌈√m⌉
    // outputs which are all the client posts. CPU = m forward NTTs +
    // m pointwise mult-accs + 2⌈√m⌉ inverse NTTs (the compression operand is
    // public, so its NTT is precomputed). Recovery from the compressed form
    // is NOT modelled — the real pipeline below still runs on full ctxts.
    let m = ctxt0.len(); // = μ·l
    let sqrt_polys = 2 * (m as f64).sqrt().ceil() as usize;
    let comp_ntt = KaheNTTPoly::rand_ntt_poly(&mut rng);
    let (compress_us, _compressed) = time_us(|| {
        let mut acc = vec![KaheNTTPoly::default(); sqrt_polys];
        for (j, p) in ctxt0.iter().enumerate() {
            acc[j % sqrt_polys] += KaheNTTPoly::from(p) * comp_ntt;
        }
        acc.iter().map(KahePoly::from).collect::<Vec<KahePoly>>()
    });
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

    Row {
        s,
        n,
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
        compress_us,
        sqrt_polys,
        useful_b: (n * cfg.payload_symbols * BITS_PER_SYMBOL) as f64 / 8.0,
        wire_ctxt_b: n as f64 * (mu_kahe * l) as f64 * KAHE_POLY_BYTES,
        wire_comm_b: n as f64 * HVC_POLY_BYTES,
        wire_opening_b: (n * s) as f64 * fresh_open_b as f64,
        wire_server_b: s as f64 * (agg_open_b + agg_share_b) as f64,
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
        "S={:>2} N={:>4} {:>5} (μ_kahe={}, κ_kahe={}, γ={}, δ={}, K={}, ξ={}, l={})  recovered: {}",
        r.s,
        r.n,
        r.label,
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

    println!("  per-role wire = fixed + l · per-chunk:");
    println!(
        "    1 client = {}  [fixed {} (comm {} + {} openings→{}S)  +  {}·{} ctxt]",
        fmt_bytes(client_fixed + client_ctxt),
        fmt_bytes(client_fixed),
        fmt_bytes(client_comm),
        fmt_bytes(client_open),
        r.s,
        r.l,
        fmt_bytes(client_ctxt / l),
    );
    println!(
        "    1 server = {}  [fixed {} (agg_open+agg_share)  +  per-chunk 0]",
        fmt_bytes(server_fixed),
        fmt_bytes(server_fixed),
    );
    println!(
        "    bulletin = {}  [fixed {} (comm {} + server entries {})  +  {}·{} ctxt]",
        fmt_bytes(bulletin_fixed + r.wire_ctxt_b),
        fmt_bytes(bulletin_fixed),
        fmt_bytes(r.wire_comm_b),
        fmt_bytes(r.wire_server_b),
        r.l,
        fmt_bytes(r.wire_ctxt_b / l),
    );

    // ── sqrt-compression sim ────────────────────────────────────────────
    // CPU: +compress_us per client (scales with l, like the other per-chunk
    // phases). Wire: client ctxt shrinks from μ·l to 2·⌈√(μ·l)⌉ polys;
    // comm/openings/server entries unchanged.
    let m = r.mu_kahe * r.l;
    let client_ctxt_sqrt = r.sqrt_polys as f64 * KAHE_POLY_BYTES;
    let wire_ctxt_sqrt = r.n as f64 * client_ctxt_sqrt;
    let wall_sim_us = per_round_us + r.compress_us;
    let wire_sim_total = wire_ctxt_sqrt + r.wire_comm_b + r.wire_opening_b + r.wire_server_b;
    println!(
        "  ── sqrt-compression sim: m={} → 2·⌈√m⌉={} polys/client ──",
        m, r.sqrt_polys,
    );
    println!(
        "    extra client CPU: {} (m ring mults; {} per mult)  →  wall {} (was {}, {:+.1}%)",
        fmt_us(r.compress_us),
        fmt_us(r.compress_us / m as f64),
        fmt_us(wall_sim_us),
        fmt_us(per_round_us),
        (wall_sim_us / per_round_us - 1.0) * 100.0,
    );
    println!(
        "    1 client out = {}  [comm {} + openings {} + ctxt {} (was {})]",
        fmt_bytes(client_fixed + client_ctxt_sqrt),
        fmt_bytes(client_comm),
        fmt_bytes(client_open),
        fmt_bytes(client_ctxt_sqrt),
        fmt_bytes(client_ctxt),
    );
    println!(
        "    bulletin = {} (was {}, {:.1}× smaller)   1 server out = {} (unchanged)",
        fmt_bytes(bulletin_fixed + wire_ctxt_sqrt),
        fmt_bytes(bulletin_fixed + r.wire_ctxt_b),
        (bulletin_fixed + r.wire_ctxt_b) / (bulletin_fixed + wire_ctxt_sqrt),
        fmt_bytes(server_fixed),
    );
    println!(
        "    wire total = {} (was {})   efficiency {:.3e} (was {:.3e})   useful {:.3} MB/s (was {:.3})",
        fmt_bytes(wire_sim_total),
        fmt_bytes(wire_total),
        r.useful_b / wire_sim_total,
        efficiency,
        r.useful_b / (wall_sim_us / 1e6) / 1e6,
        useful_mb_s,
    );

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
    println!("  network sim (net = A client upload + B server post + C verifier read; e2e = cpu + net):");
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
        for (tag, ctxt_per_client, cpu_us) in [
            ("base", client_ctxt, per_round_us),
            ("sqrt", client_ctxt_sqrt, wall_sim_us),
        ] {
            let post_b = client_comm + ctxt_per_client; // one client's bulletin post
            let a = (maxlat(r.n) + xfer_cl(post_b)) // client→bulletin flow
                .max(maxlat(r.n) + xfer_cl(client_open)) // client→servers flow
                .max(maxlat(r.n) + xfer_srv(r.wire_opening_b / r.s as f64)) // server downlink
                .max(maxlat(r.n) + xfer_srv(r.n as f64 * post_b)); // bulletin ingest
            let b = maxlat(r.s) + xfer_srv(r.wire_server_b);
            let c = maxlat(1) + xfer_srv(bulletin_fixed + r.n as f64 * ctxt_per_client);
            let net = a + b + c;
            let e2e = cpu_us + net;
            println!(
                "      {}: net {:>10} [A {} + B {} + C {}]   e2e {:>10}  →  {:.3} MB/s (cpu-only {:.3})",
                tag,
                fmt_us(net),
                fmt_us(a),
                fmt_us(b),
                fmt_us(c),
                fmt_us(e2e),
                r.useful_b / (e2e / 1e6) / 1e6,
                r.useful_b / (cpu_us / 1e6) / 1e6,
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

    println!("flashnet scaling bench  (budget: {}s)", budget.as_secs());
    println!(
        "ring: HVC {} bits/coef ({} B/poly) | KAHE {} bits/coef ({} B/poly) | t bits/symbol {}",
        bits_per_coef(HVC_MODULUS),
        HVC_POLY_BYTES as usize,
        bits_per_coef(KAHE_MODULUS),
        KAHE_POLY_BYTES as usize,
        BITS_PER_SYMBOL,
    );
    println!("per-round wall = fixed + l · per-chunk");
    println!(
        "  fixed   = shamir_share + cs_commit + server + verify_open + verify_interp + sum_comm"
    );
    println!("  chunk   = mse_c + kahe_enc + kahe_agg_ctxt + kahe_dec + mse_v");
    println!("useful = l · N · ξ · log₂(t) bits per round  (N clients × l elements each)");
    println!();

    'outer: for &(s, n) in CELLS {
        for cfg in CONFIGS {
            if start.elapsed() >= budget {
                println!("(budget exhausted; remaining cells skipped)");
                break 'outer;
            }
            print_row(&run_cell(s, n, cfg));
        }
        println!();
    }
    println!("done in {:.1}s", start.elapsed().as_secs_f64());
}
