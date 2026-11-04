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
//!   PROFILE_MU=0        KAHE width in polys (0 = the default parameter set)
//!   PROFILE_T=0         plaintext modulus, only with PROFILE_MU (0 = default)
//!   PROFILE_ROLE=all    which party's work the timed loop repeats:
//!                       `all` runs the whole round per iteration;
//!                       `client`, `server` or `recipient` builds that role's
//!                       inputs once and then repeats only its own work, which
//!                       is what one process in a deployment actually runs.
//!
//! Each role is parallelised the way a deployment is: client rounds run
//! concurrently across clients, every server unseals and sums its inbox in
//! parallel with the others, and the recipient runs alone. Messages are drawn
//! once, outside the loop, so the profile is the protocol and not the RNG.
//!
//! Tip: keep ITERS low under callgrind (it adds ~50× overhead). For perf
//! sampling, push ITERS to 100+ to dilute startup noise.

use panetiere::bulletin::{ClientBulletinEntry, ServerBulletinEntry};
use panetiere::cs::Commitment;
use panetiere::pke;
use panetiere::protocol::client::ClientRound;
use panetiere::protocol::server::{run_server_round, unseal_openings, ServerInbox};
use panetiere::protocol::verify::aggregate_and_decrypt_unverified;
use panetiere::protocol::ProtocolParams;
use panetiere::protocol::{ClientId, ServerId, SessionId};
use panetiere::KahePoly;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

fn rand_message_poly<R: Rng>(rng: &mut R, t: u64) -> KahePoly {
    let half = t as i64 / 2;
    let mut coeffs = [0i64; panetiere::N];
    for c in coeffs.iter_mut() {
        *c = rng.gen_range(0..t) as i64 - half;
    }
    KahePoly::from_coeffs(coeffs)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[derive(Clone, Copy, PartialEq)]
enum Role {
    All,
    Client,
    Server,
    Recipient,
}

impl Role {
    fn from_env() -> Self {
        match std::env::var("PROFILE_ROLE").as_deref() {
            Err(_) | Ok("all") => Role::All,
            Ok("client") => Role::Client,
            Ok("server") => Role::Server,
            Ok("recipient") => Role::Recipient,
            Ok(other) => panic!("PROFILE_ROLE={other:?}: want all|client|server|recipient"),
        }
    }
}

fn session(iter: usize) -> SessionId {
    let mut sid_bytes = [0u8; 32];
    sid_bytes[..8].copy_from_slice(&(iter as u64).to_le_bytes());
    SessionId(sid_bytes)
}

struct Cell {
    pp: ProtocolParams,
    servers: Vec<(ServerId, pke::PublicKey)>,
    server_keys: Vec<pke::PrivateKey>,
    client_ids: Vec<ClientId>,
    messages: Vec<Vec<KahePoly>>,
}

/// What the clients publish and what each server receives, sealed.
struct ClientPhase {
    entries: Vec<(ClientId, ClientBulletinEntry)>,
    sealed_per_server: Vec<Vec<(ClientId, Vec<u8>)>>,
}

impl Cell {
    /// Every client's round, concurrently, each on its own forked seed.
    fn client_phase<R: Rng>(&self, rng: &mut R, sid: &SessionId) -> ClientPhase {
        let seeds: Vec<[u8; 32]> = (0..self.client_ids.len()).map(|_| rng.gen()).collect();
        let rounds: Vec<_> = self
            .client_ids
            .par_iter()
            .zip(self.messages.par_iter())
            .zip(seeds.par_iter())
            .map(|((&cid, message), seed)| {
                let mut client_rng = ChaCha20Rng::from_seed(*seed);
                ClientRound::new(&mut client_rng, &self.pp, sid, cid, &self.servers)
                    .finalize(message)
            })
            .collect();

        let mut entries = Vec::with_capacity(rounds.len());
        let mut sealed_per_server: Vec<Vec<(ClientId, Vec<u8>)>> = (0..self.servers.len())
            .map(|_| Vec::with_capacity(rounds.len()))
            .collect();
        for round in rounds {
            entries.push((round.client_id, round.encrypted_message));
            for (idx, (_, sealed)) in round.sealed_openings.into_iter().enumerate() {
                sealed_per_server[idx].push((round.client_id, sealed));
            }
        }
        ClientPhase {
            entries,
            sealed_per_server,
        }
    }

    /// Every server's round, concurrently: unseal the inbox, then verify and sum.
    fn server_phase(
        &self,
        sid: &SessionId,
        sealed_per_server: &[Vec<(ClientId, Vec<u8>)>],
        commitments: &[(ClientId, Commitment)],
    ) -> Vec<ServerBulletinEntry> {
        sealed_per_server
            .par_iter()
            .enumerate()
            .map(|(idx, sealed)| {
                let server_id = ServerId(idx as u32);
                let items = unseal_openings(&self.server_keys[idx], sid, server_id, sealed)
                    .into_iter()
                    .zip(sealed)
                    .map(|(opening, (cid, _))| {
                        (
                            *cid,
                            opening.expect("envelope unseals under its own session"),
                        )
                    })
                    .collect();
                let inbox = ServerInbox { server_id, items };
                run_server_round(&self.pp.cs, &inbox, &self.client_ids, commitments)
                    .expect("every canonical client posted")
            })
            .collect()
    }

    fn recipient_phase(
        &self,
        entries: &[(ClientId, ClientBulletinEntry)],
        outputs: &[ServerBulletinEntry],
    ) -> Vec<KahePoly> {
        aggregate_and_decrypt_unverified(&self.pp, &self.client_ids, entries, outputs)
            .expect("honest round verifies")
    }
}

fn commitments_of(entries: &[(ClientId, ClientBulletinEntry)]) -> Vec<(ClientId, Commitment)> {
    entries
        .iter()
        .map(|(cid, e)| (*cid, e.comm.clone()))
        .collect()
}

fn main() {
    let s = env_usize("PROFILE_S", 8);
    let n = env_usize("PROFILE_N", 16);
    let iters = env_usize("PROFILE_ITERS", 10);
    let mu = env_usize("PROFILE_MU", 0);
    let t_modulus = env_usize("PROFILE_T", 0) as u64;
    let role = Role::from_env();

    let mut rng = ChaCha20Rng::from_seed([(s as u8).wrapping_mul(7) ^ n as u8; 32]);
    let pp = if mu > 0 && t_modulus > 0 {
        ProtocolParams::setup_with_kahe_dims_full(
            &mut rng,
            s,
            mu,
            panetiere::kahe::SIGMA_S_DEFAULT,
            panetiere::kahe::SIGMA_E_DEFAULT,
            t_modulus,
        )
    } else if mu > 0 {
        ProtocolParams::setup_with_kahe_dims(&mut rng, s, mu)
    } else {
        ProtocolParams::setup(&mut rng, s)
    };

    eprintln!(
        "profile: S={} N={} iters={} role={} threads={} (μ_kahe={}, μ_cs={}, σ_s={:.3}, σ_e={:.3}, t={}, threshold={})",
        s,
        n,
        iters,
        std::env::var("PROFILE_ROLE").unwrap_or_else(|_| "all".into()),
        rayon::current_num_threads(),
        pp.kahe.mu_kahe,
        pp.cs.mu_cs,
        pp.kahe.sigma_s,
        pp.kahe.sigma_e,
        pp.kahe.t_modulus,
        pp.shamir.t
    );
    let server_keys: Vec<pke::PrivateKey> = (0..s)
        .map(|_| pke::PrivateKey::generate(&mut rng))
        .collect();
    let servers: Vec<(ServerId, pke::PublicKey)> = server_keys
        .iter()
        .enumerate()
        .map(|(idx, key)| (ServerId(idx as u32), key.public()))
        .collect();
    let client_ids: Vec<ClientId> = (0..n as u32).map(ClientId).collect();
    let messages: Vec<Vec<KahePoly>> = (0..n)
        .map(|_| {
            (0..pp.kahe.mu_kahe)
                .map(|_| rand_message_poly(&mut rng, pp.kahe.t_modulus))
                .collect()
        })
        .collect();
    let cell = Cell {
        pp,
        servers,
        server_keys,
        client_ids,
        messages,
    };

    match role {
        Role::All => {
            for iter in 0..iters {
                // one session per iteration, as a real deployment would do per round
                let sid = session(iter);
                let clients = cell.client_phase(&mut rng, &sid);
                let commitments = commitments_of(&clients.entries);
                let outputs = cell.server_phase(&sid, &clients.sealed_per_server, &commitments);
                let _recovered = cell.recipient_phase(&clients.entries, &outputs);
                if iter == 0 {
                    eprintln!("profile: first iteration completed");
                }
            }
        }
        Role::Client => {
            for iter in 0..iters {
                let sid = session(iter);
                let _clients = cell.client_phase(&mut rng, &sid);
                if iter == 0 {
                    eprintln!("profile: first iteration completed");
                }
            }
        }
        Role::Server => {
            let sid = session(0);
            let clients = cell.client_phase(&mut rng, &sid);
            let commitments = commitments_of(&clients.entries);
            eprintln!("profile: server inputs ready");
            for iter in 0..iters {
                let _outputs = cell.server_phase(&sid, &clients.sealed_per_server, &commitments);
                if iter == 0 {
                    eprintln!("profile: first iteration completed");
                }
            }
        }
        Role::Recipient => {
            let sid = session(0);
            let clients = cell.client_phase(&mut rng, &sid);
            let commitments = commitments_of(&clients.entries);
            let outputs = cell.server_phase(&sid, &clients.sealed_per_server, &commitments);
            eprintln!("profile: recipient inputs ready");
            for iter in 0..iters {
                let _recovered = cell.recipient_phase(&clients.entries, &outputs);
                if iter == 0 {
                    eprintln!("profile: first iteration completed");
                }
            }
        }
    }
    eprintln!("profile: done {} iterations", iters);
}
