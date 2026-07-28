//! Prony sketch vs the MSE IBLT: client plaintext size, ciphertext size, and
//! measured encode/decode cost at the swept operating points.
//!
//! ```sh
//! RAYON_NUM_THREADS=8 cargo run -j 8 --release --example prony_sizing
//! ```
//!
//! Env knobs for the timed cell: `PRONY_RHO`, `PRONY_ELEM_BYTES`.

use std::time::Instant;

use chipmunk_code::{KahePoly, KAHE_MODULUS, N as POLY_N};
use panetiere::cs::poly_packed_len64;
use panetiere::kahe::{
    Kahe, KaheScheme, SIGMA_E_DEFAULT, SIGMA_S_DEFAULT, T_MODULUS_DEFAULT,
};
use panetiere::mse::{MseEncoding, MseParams, BITS_PER_SYMBOL};
use panetiere::prony::{PronyParams, PronySketch, PRONY_PRIME};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const GAMMA: usize = 4;

fn mib(b: usize) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let ct_poly = poly_packed_len64(KAHE_MODULUS);
    let prony_bits = PronyParams::new(2, 1).bits_per_symbol();

    println!(
        "ciphertext {} B/poly ({} coeffs); MSE {} bits/symbol, Prony {} bits/symbol\n",
        ct_poly, POLY_N, BITS_PER_SYMBOL, prony_bits,
    );
    println!(
        "{:>5} {:>8} | {:>10} {:>7} {:>9} | {:>10} {:>7} {:>9} | {:>6}",
        "ρ", "element", "mse cells", "polys", "C", "prony ξ", "polys", "C", "ratio",
    );

    for &elem_bytes in &[4096usize, 20480] {
        for &rho in &[100usize, 300] {
            let bits = elem_bytes * 8;
            let xi_mse = bits.div_ceil(BITS_PER_SYMBOL);
            let mse = MseParams::new(GAMMA, (3 * rho).div_ceil(GAMMA), xi_mse, [0xAA; 32]);
            let mse_polys = MseEncoding::n_polys(&mse);

            let xi_prony = bits.div_ceil(prony_bits);
            let prony = PronyParams::new(rho, xi_prony);
            let prony_polys = PronySketch::n_polys(&prony);

            println!(
                "{:>5} {:>7}K | {:>10} {:>7} {:>8.2}M | {:>10} {:>7} {:>8.2}M | {:>5.2}x",
                rho,
                elem_bytes / 1024,
                mse.total_cells(),
                mse_polys,
                mib(mse_polys * ct_poly),
                xi_prony,
                prony_polys,
                mib(prony_polys * ct_poly),
                mse_polys as f64 / prony_polys as f64,
            );
        }
    }

    // Timed cell: one full round's worth of inserts summed into one sketch
    // (by linearity that *is* Σ of the per-client sketches), then decoded.
    let rho = env_usize("PRONY_RHO", 300);
    let elem_bytes = env_usize("PRONY_ELEM_BYTES", 20480);
    let xi = (elem_bytes * 8).div_ceil(prony_bits);
    let params = PronyParams::new(rho, xi);
    println!(
        "\ntimed: ρ={} element={} KiB ξ={} cols={} scalars={} polys={}",
        rho,
        elem_bytes / 1024,
        xi,
        params.cols(),
        params.total_scalars(),
        PronySketch::n_polys(&params),
    );

    let mut rng = ChaCha20Rng::from_seed([0x11u8; 32]);
    let payloads: Vec<Vec<i64>> = (0..rho)
        .map(|i| (0..xi).map(|s| ((i * 7 + s * 13) % 1000) as i64).collect())
        .collect();

    let mut agg = PronySketch::new(params.clone());
    let t0 = Instant::now();
    for m in &payloads {
        agg.insert(&mut rng, m);
    }
    let enc = t0.elapsed();

    let t1 = Instant::now();
    let got = agg.decode().expect("decode");
    let dec = t1.elapsed();

    let mut want = payloads;
    want.sort();
    assert_eq!(got, want, "decoded multiset mismatch");

    // Locator + root-finding alone, isolated by dropping the payload block.
    let mut ids_only = PronySketch::new(PronyParams::new(rho, 1));
    let mut rng2 = ChaCha20Rng::from_seed([0x11u8; 32]);
    for i in 0..rho {
        ids_only.insert(&mut rng2, &[i as i64]);
    }
    let t2 = Instant::now();
    ids_only.decode().expect("decode ids");
    let roots = t2.elapsed();

    println!(
        "  prony  encode {:>7.2} ms/client   decode {:>8.2} ms  (locator+roots {:.0} ms, \
         payload solve {:.0} ms), {} recovered",
        enc.as_secs_f64() * 1e3 / rho as f64,
        dec.as_secs_f64() * 1e3,
        roots.as_secs_f64() * 1e3,
        (dec.as_secs_f64() - roots.as_secs_f64()).max(0.0) * 1e3,
        got.len(),
    );

    // Same cell through the IBLT. Inserting all ρ elements into one encoding is
    // the coefficient-wise sum of the per-client encodings.
    let xi_mse = (elem_bytes * 8).div_ceil(BITS_PER_SYMBOL);
    let mse = MseParams::new(GAMMA, (3 * rho).div_ceil(GAMMA), xi_mse, [0xAA; 32]);
    let mse_payloads: Vec<Vec<i64>> = (0..rho)
        .map(|i| (0..xi_mse).map(|s| ((i * 7 + s * 13) % 1000) as i64).collect())
        .collect();

    let mut menc = MseEncoding::new(mse.clone());
    let t3 = Instant::now();
    for m in &mse_payloads {
        menc.insert(&mut rng, m);
    }
    let m_enc = t3.elapsed();

    let t4 = Instant::now();
    let m_got = menc.decode().expect("mse decode");
    let m_dec = t4.elapsed();

    let mut m_want = mse_payloads;
    m_want.sort();
    assert_eq!(m_got, m_want, "mse decoded multiset mismatch");

    println!(
        "  mse    encode {:>7.2} ms/client   decode {:>8.2} ms  (ξ={} cells={} scalars={} polys={}), \
         {} recovered",
        m_enc.as_secs_f64() * 1e3 / rho as f64,
        m_dec.as_secs_f64() * 1e3,
        xi_mse,
        mse.total_cells(),
        mse.total_scalars(),
        MseEncoding::n_polys(&mse),
        m_got.len(),
    );
    println!(
        "  prony/mse app-layer: encode {:.2}x, decode {:.2}x",
        enc.as_secs_f64() / m_enc.as_secs_f64(),
        dec.as_secs_f64() / m_dec.as_secs_f64(),
    );

    // The app layer is not the whole client/verifier cost: KAHE runs over the
    // packed plaintext, so prony's narrower message makes enc/dec cheaper.
    let (p_kenc, p_kdec) = kahe_cost(PronySketch::n_polys(&params), PRONY_PRIME);
    let (m_kenc, m_kdec) = kahe_cost(MseEncoding::n_polys(&mse), T_MODULUS_DEFAULT);
    println!(
        "  kahe   prony {} polys: enc {:.1} ms dec {:.1} ms | mse {} polys: enc {:.1} ms dec {:.1} ms",
        PronySketch::n_polys(&params),
        p_kenc * 1e3,
        p_kdec * 1e3,
        MseEncoding::n_polys(&mse),
        m_kenc * 1e3,
        m_kdec * 1e3,
    );
    println!(
        "  total  client (app enc + kahe enc): prony {:.1} ms vs mse {:.1} ms | \
         verifier (kahe dec + app dec): prony {:.0} ms vs mse {:.0} ms",
        enc.as_secs_f64() * 1e3 / rho as f64 + p_kenc * 1e3,
        m_enc.as_secs_f64() * 1e3 / rho as f64 + m_kenc * 1e3,
        dec.as_secs_f64() * 1e3 + p_kdec * 1e3,
        m_dec.as_secs_f64() * 1e3 + m_kdec * 1e3,
    );
}

/// `(enc_secs, dec_secs)` for one client's `n_polys`-wide message under one key.
fn kahe_cost(n_polys: usize, t_modulus: u64) -> (f64, f64) {
    const MU_KAHE: usize = 2002;
    let mut rng = ChaCha20Rng::from_seed([0x77u8; 32]);
    let l = n_polys.div_ceil(MU_KAHE);
    let pp =
        Kahe::setup_with_dims(&mut rng, MU_KAHE, l, SIGMA_S_DEFAULT, SIGMA_E_DEFAULT, t_modulus);
    let key = Kahe::gen(&mut rng, &pp);
    let m: Vec<KahePoly> = vec![KahePoly::default(); n_polys];
    let t = Instant::now();
    let ct = Kahe::enc(&mut rng, &pp, &key, &m);
    let enc = t.elapsed().as_secs_f64();
    let agg = Kahe::agg_key(std::slice::from_ref(&key));
    let t = Instant::now();
    let _ = Kahe::dec(&pp, &ct, &agg);
    (enc, t.elapsed().as_secs_f64())
}
