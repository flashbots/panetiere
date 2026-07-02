//! Profiling driver. Runs one fixed (S, N) cell of the protocol many times
//! so callgrind / perf has a steady-state hot path to collect on.
//!
//! Build with debug info:
//!   cargo build --profile profiling --example profile
//!
//! Run under callgrind:
//!   valgrind --tool=callgrind --callgrind-out-file=callgrind.out \
//!            ./target/profiling/examples/profile
//!
//! Then open `callgrind.out` in kcachegrind.
//!
//! Workload knobs via env vars:
//!   PROFILE_S=8         server count (default 8)
//!   PROFILE_N=16        canonical client count (default 16)
//!   PROFILE_ITERS=10    iterations of the timed inner loop (default 10)
//!
//! Tip: keep ITERS low under callgrind (it adds ~50× overhead). For perf
//! sampling, push ITERS to 100+ to dilute startup noise.

use chipmunk_code::KahePoly;
use rand::Rng;
use panetiere::pke;
use panetiere::protocol::client::run_client_round;
use panetiere::protocol::{ClientId, ServerId};
use panetiere::protocol::server::{run_server_round, unseal_opening, ServerInbox};
use panetiere::protocol::verify::aggregate_and_decrypt;
use panetiere::protocol::ProtocolParams;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn rand_message_poly<R: Rng>(rng: &mut R, t: u32) -> KahePoly {
    let half = t as i32 / 2;
    let mut coeffs = [0i32; chipmunk_code::N];
    for c in coeffs.iter_mut() {
        *c = (rng.gen_range(0..t) as i32) - half;
    }
    KahePoly::from_coeffs(coeffs)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let s = env_usize("PROFILE_S", 8);
    let n = env_usize("PROFILE_N", 16);
    let iters = env_usize("PROFILE_ITERS", 10);
    let mu = env_usize("PROFILE_MU", 0);
    let kappa = env_usize("PROFILE_KAPPA", 0);
    let t_modulus = env_usize("PROFILE_T", 0) as u32;

    let mut rng = ChaCha20Rng::from_seed([(s as u8).wrapping_mul(7) ^ n as u8; 32]);
    let pp = if mu > 0 && kappa > 0 && t_modulus > 0 {
        ProtocolParams::setup_with_kahe_dims_full(
            &mut rng, s, mu, kappa,
            1,
            panetiere::kahe::SIGMA_S_DEFAULT,
            panetiere::kahe::SIGMA_E_DEFAULT,
            t_modulus,
        )
    } else if mu > 0 && kappa > 0 {
        ProtocolParams::setup_with_kahe_dims(&mut rng, s, mu, kappa)
    } else {
        ProtocolParams::setup(&mut rng, s)
    };

    eprintln!(
        "profile: S={} N={} iters={} (μ_kahe={}, κ_kahe={}, μ_cs={}, σ_s={:.3}, σ_e={:.3}, t={}, threshold={})",
        s, n, iters,
        pp.kahe.mu_kahe, pp.kahe.kappa_kahe, pp.cs.mu_cs,
        pp.kahe.sigma_s, pp.kahe.sigma_e, pp.kahe.t_modulus, pp.shamir.t
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
    let canonical = client_ids.clone();

    for iter in 0..iters {
        let mut client_entries = Vec::with_capacity(n);
        let mut inboxes: Vec<ServerInbox> = server_ids
            .iter()
            .map(|&sid| ServerInbox {
                server_id: sid,
                items: vec![],
            })
            .collect();
        for &cid in &client_ids {
            let m: Vec<KahePoly> = (0..pp.kahe.mu_kahe)
                .map(|_| rand_message_poly(&mut rng, pp.kahe.t_modulus))
                .collect();
            let round = run_client_round(&mut rng, &pp, cid, m, &servers);
            client_entries.push((round.client_id, round.encrypted_message));
            for (idx, (_, sealed)) in round.sealed_openings.into_iter().enumerate() {
                let opening = unseal_opening(&server_keys[idx], &sealed).expect("unseal");
                inboxes[idx].items.push((cid, opening));
            }
        }
        let outputs: Vec<_> = inboxes
            .iter()
            .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
            .collect();
        let _recovered =
            aggregate_and_decrypt(&pp, &canonical, &client_entries, &outputs).expect("verify");
        if iter == 0 {
            eprintln!("profile: first iteration completed");
        }
    }
    eprintln!("profile: done {} iterations", iters);
}
