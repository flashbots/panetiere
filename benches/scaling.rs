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

use chipmunk_code::{KahePoly, HVC_MODULUS, HVC_WIDTH, KAHE_MODULUS, N as POLY_N};
use flashnet::kahe::{SIGMA_E_DEFAULT, SIGMA_S_DEFAULT, T_MODULUS_DEFAULT};
use flashnet::mse::{MseEncoding, MseParams, BITS_PER_SYMBOL, K_LIMBS};
use flashnet::protocol::client::{cs_commit, kahe_encrypt, kahe_keygen, run_client_round, shamir_share};
use flashnet::protocol::{ClientId, ServerId};
use flashnet::protocol::server::{run_server_round, ServerInbox};
use flashnet::protocol::verify::{aggregate_and_decrypt_timed, VerifyTimings};
use flashnet::protocol::ProtocolParams;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const CELLS: &[(usize, usize)] = &[(4, 100), (8, 100), (16, 100)];

struct Config {
    label: &'static str,
    gamma: usize,
    delta_factor: usize, // δ = delta_factor · N · l
    payload_symbols: usize,
    kappa_kahe: usize,
    /// Number of MSE elements packed per client, per round. Equivalently the
    /// number of KAHE ciphertext chunks per round — one Shamir+CS commit
    /// amortizes over all `l` chunks. `l = 1` is the original (no-amortize)
    /// regime; `l > 1` widens MSE's `δ` so per-row load stays the same.
    l: usize,
}

// ξ = 29 ⇒ 29·18 = 522 useful bits per element (≥ 512-bit target).
const CONFIGS: &[Config] = &[
    Config { label: " L=1",  gamma: 4, delta_factor: 2, payload_symbols: 29, kappa_kahe: 31, l: 1 },
    Config { label: " L=20", gamma: 4, delta_factor: 2, payload_symbols: 29, kappa_kahe: 31, l: 20 },
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
    // CPU (one round, no averaging). Fixed-per-round vs per-chunk split:
    //   fixed:    client_share + client_cs + server + verify_opening + verify_interp + verify_sum_comm
    //   per-chunk (×l): mse_c + client_kahe + verify_agg_ctxt + verify_kahe_dec + mse_v
    mse_c_us: f64,           // total over l inserts
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
        cfg.delta_factor as u8,
        cfg.l as u8,
    ]);
    let mut rng = ChaCha20Rng::from_seed(seed);

    // Widen MSE so total elements scale with l while per-row peeling load
    // (ρ_total / δ) stays the same as the l=1 baseline.
    let delta = cfg.delta_factor * n * cfg.l;
    let mse_params = MseParams::new(cfg.gamma, delta, cfg.payload_symbols, [0xAA; 32]);
    let n_polys = MseEncoding::n_polys(&mse_params);
    assert!(
        n_polys % cfg.l == 0,
        "n_polys={n_polys} must divide evenly by l={}",
        cfg.l
    );
    let mu_kahe = n_polys / cfg.l;

    let pp = ProtocolParams::setup_with_kahe_dims_full(
        &mut rng,
        s,
        mu_kahe,
        cfg.kappa_kahe,
        cfg.l,
        SIGMA_S_DEFAULT,
        SIGMA_E_DEFAULT,
        T_MODULUS_DEFAULT,
    );
    let server_ids: Vec<ServerId> = (0..s as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n as u32).map(ClientId).collect();

    // Each client inserts l distinct elements. Small distinct values; max
    // magnitude ≈ (n·l + ξ) ≪ t/2 = 2^17, so no wrap when summed across n.
    let payloads: Vec<Vec<Vec<i32>>> = client_ids
        .iter()
        .map(|cid| {
            (0..cfg.l)
                .map(|elt_idx| {
                    let base = cid.0 as i32 * cfg.l as i32 + elt_idx as i32;
                    (0..cfg.payload_symbols)
                        .map(|j| base + j as i32 + 1)
                        .collect()
                })
                .collect()
        })
        .collect();
    let client_polys: Vec<Vec<KahePoly>> = payloads
        .iter()
        .map(|elts| {
            let mut enc = MseEncoding::new(mse_params.clone());
            for e in elts {
                enc.insert(&mut rng, e);
            }
            enc.pack()
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

    // Recovery must succeed before timing — otherwise we're benching broken params.
    let (recovered, _t0) =
        aggregate_and_decrypt_timed(&pp, &canonical, &client_entries, &outputs).unwrap();
    let decoded = MseEncoding::unpack(&mse_params, &recovered)
        .decode()
        .expect("MSE decode");
    assert_eq!(
        decoded.len(),
        n * cfg.l,
        "MSE recovered wrong multiset size"
    );

    // Wire ledger (structural — tight ⌈log₂ q⌉ packing).
    let o = &inboxes[0].items[0].1;
    let opening_polys = o.kappa_cs()
        + o.mu_cs()
        + (2 * o.block_size() - 2) * HVC_WIDTH
        + o.stored_path_len() * 2 * HVC_WIDTH;

    // Per-phase timings (one round each).
    let (mse_c_us, _) = time_us(|| {
        let mut enc = MseEncoding::new(mse_params.clone());
        for e in &payloads[0] {
            enc.insert(&mut rng, e);
        }
        enc.pack()
    });
    let key0 = kahe_keygen(&mut rng, &pp);
    let (client_kahe_us, _ctxt0) =
        time_us(|| kahe_encrypt(&mut rng, &pp, &key0, &client_polys[0]));
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
        MseEncoding::unpack(&mse_params, &recovered2)
            .decode()
            .expect("decode")
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
        l: cfg.l,
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
        useful_b: (n * cfg.l * cfg.payload_symbols * BITS_PER_SYMBOL) as f64 / 8.0,
        wire_ctxt_b: n as f64 * (mu_kahe * cfg.l) as f64 * KAHE_POLY_BYTES,
        wire_comm_b: n as f64 * HVC_POLY_BYTES,
        wire_opening_b: (n * s) as f64 * opening_polys as f64 * HVC_POLY_BYTES,
        wire_server_b: s as f64 * (opening_polys + cfg.kappa_kahe) as f64 * HVC_POLY_BYTES,
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
        "S={:>2} N={:>4} {:>5} (μ_kahe={}, κ_kahe={}, γ={}, δ={}, K={}, ξ={}, l={})",
        r.s, r.n, r.label, r.mu_kahe, r.kappa, r.gamma, r.delta, K_LIMBS, r.xi, r.l,
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
    println!("  fixed   = shamir_share + cs_commit + server + verify_open + verify_interp + sum_comm");
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
