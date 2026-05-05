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

use chipmunk_code::{HVCPoly, Polynomial};
use flashnet::protocol::client::run_client_round;
use flashnet::protocol::message::{ClientId, ServerId};
use flashnet::protocol::server::{run_server_round, ServerInbox};
use flashnet::protocol::verify::aggregate_and_decrypt;
use flashnet::protocol::ProtocolParams;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

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
    let beta = env_usize("PROFILE_BETA", 0) as u32;

    let mut rng = ChaCha20Rng::from_seed([(s as u8).wrapping_mul(7) ^ n as u8; 32]);
    let pp = if mu > 0 && kappa > 0 && beta > 0 {
        ProtocolParams::setup_with_kahe_dims_beta(&mut rng, s, mu, kappa, beta)
    } else {
        ProtocolParams::setup(&mut rng, s)
    };

    eprintln!(
        "profile: S={} N={} iters={} (μ_kahe={}, κ_kahe={}, μ_cs={}, β={}, t={})",
        s, n, iters,
        pp.kahe.mu_kahe, pp.kahe.kappa_kahe, pp.cs.mu_cs, pp.kahe.sk_bound, pp.shamir.t
    );
    let server_ids: Vec<ServerId> = (0..s as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n as u32).map(ClientId).collect();
    let canonical = client_ids.clone();

    for iter in 0..iters {
        let mut publics = Vec::with_capacity(n);
        let mut inboxes: Vec<ServerInbox> = server_ids
            .iter()
            .map(|&sid| ServerInbox {
                server_id: sid,
                items: vec![],
            })
            .collect();
        for &cid in &client_ids {
            let m: Vec<HVCPoly> = (0..pp.kahe.mu_kahe)
                .map(|_| HVCPoly::rand_poly(&mut rng))
                .collect();
            let round = run_client_round(&mut rng, &pp, cid, m, &server_ids);
            publics.push((round.client_id, round.public));
            for (idx, (_, op)) in round.private.into_iter().enumerate() {
                inboxes[idx].items.push((cid, op));
            }
        }
        let outputs: Vec<_> = inboxes
            .iter()
            .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
            .collect();
        let _recovered =
            aggregate_and_decrypt(&pp, &canonical, &publics, &outputs).expect("verify");
        if iter == 0 {
            eprintln!("profile: first iteration completed");
        }
    }
    eprintln!("profile: done {} iterations", iters);
}
