//! Prony sketch (`src/prony.rs`) carried through the Panetière protocol,
//! mirroring `tests/mse_e2e.rs`.
//!
//! The only protocol-level change against the MSE flow is the plaintext
//! modulus: `t = PRONY_PRIME = 2^36 − 5` instead of `T_MODULUS_DEFAULT = 2^36`,
//! passed through `ProtocolParams::setup_with_kahe_dims_full`. Everything else
//! — KAHE, Shamir, CS, the verifier's per-poly sum — is untouched, which is the
//! point: the sketch is just another additively homomorphic plaintext.

use panetiere::kahe::{SIGMA_E_DEFAULT, SIGMA_S_DEFAULT};
use panetiere::pke;
use panetiere::prony::{PronyParams, PronySketch, PRONY_PRIME};
use panetiere::protocol::client::run_client_round;
use panetiere::protocol::server::{run_server_round, unseal_opening, ServerInbox};
use panetiere::protocol::verify::aggregate_and_decrypt;
use panetiere::protocol::ProtocolParams;
use panetiere::protocol::{ClientId, ServerId, SessionId};
use panetiere::KahePoly;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Run `client_polys` (one `Vec<KahePoly>` per client, all equal length)
/// through a full round and return the verifier's recovered per-poly sums.
fn round_trip(
    rng: &mut ChaCha20Rng,
    n_servers: usize,
    client_polys: &[Vec<KahePoly>],
) -> Vec<KahePoly> {
    let n_polys = client_polys[0].len();
    assert!(client_polys.iter().all(|p| p.len() == n_polys));

    let pp = ProtocolParams::setup_with_kahe_dims_full(
        rng,
        n_servers,
        1,
        SIGMA_S_DEFAULT,
        SIGMA_E_DEFAULT,
        PRONY_PRIME,
    );
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let server_keys: Vec<pke::PrivateKey> = (0..n_servers)
        .map(|_| pke::PrivateKey::generate(rng))
        .collect();
    let servers: Vec<(ServerId, pke::PublicKey)> = server_ids
        .iter()
        .map(|&sid| (sid, server_keys[sid.0 as usize].public()))
        .collect();
    let client_ids: Vec<ClientId> = (0..client_polys.len() as u32).map(ClientId).collect();
    let canonical = client_ids.clone();

    (0..n_polys)
        .map(|k| {
            // Each poly index is its own protocol execution, so it gets its own sid.
            let mut sid_bytes = [0u8; 32];
            sid_bytes[..8].copy_from_slice(&(k as u64).to_le_bytes());
            let session = SessionId(sid_bytes);
            let mut client_entries = Vec::new();
            let mut inboxes: Vec<ServerInbox> = server_ids
                .iter()
                .map(|&sid| ServerInbox {
                    server_id: sid,
                    items: vec![],
                })
                .collect();
            for (i, &cid) in client_ids.iter().enumerate() {
                let round =
                    run_client_round(rng, &pp, &session, cid, vec![client_polys[i][k]], &servers);
                client_entries.push((round.client_id, round.encrypted_message));
                for (idx, (sid, sealed)) in round.sealed_openings.into_iter().enumerate() {
                    let opening = unseal_opening(&server_keys[idx], &session, cid, sid, &sealed)
                        .expect("unseal");
                    inboxes[idx].items.push((cid, opening));
                }
            }
            let outputs: Vec<_> = inboxes
                .iter()
                .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
                .collect();
            let recovered = aggregate_and_decrypt(&pp, &canonical, &client_entries, &outputs)
                .unwrap_or_else(|e| panic!("verify failed at poly {}: {:?}", k, e));
            assert_eq!(recovered.len(), 1);
            recovered[0]
        })
        .collect()
}

#[test]
fn prony_recovers_through_panetiere() {
    let mut rng = ChaCha20Rng::from_seed([0xFEu8; 32]);
    let n_clients = 6;
    let params = PronyParams::new(n_clients, 2);

    let messages: Vec<Vec<i64>> = (0..n_clients)
        .map(|i| vec![1000 + i as i64, (i as i64 + 1) << 30])
        .collect();
    let client_polys: Vec<Vec<KahePoly>> = messages
        .iter()
        .map(|m| {
            let mut sk = PronySketch::new(params.clone());
            sk.insert(&mut rng, m);
            sk.pack()
        })
        .collect();

    let recovered = round_trip(&mut rng, 3, &client_polys);
    let mut got = PronySketch::unpack(&params, &recovered)
        .decode()
        .expect("prony decode");
    let mut want = messages;
    got.sort();
    want.sort();
    assert_eq!(got, want);
}

/// Cover clients run the full protocol — canonical set, anonymity set, sealed
/// openings — but send zero polys, so they consume no sketch capacity. The
/// sketch is sized to the active count only.
#[test]
fn cover_clients_do_not_consume_capacity() {
    let mut rng = ChaCha20Rng::from_seed([0x5Au8; 32]);
    let (n_active, n_cover) = (4, 8);
    let params = PronyParams::new(n_active, 1);

    let elements: Vec<i64> = (0..n_active).map(|i| 7 * i as i64 + 3).collect();
    let mut client_polys: Vec<Vec<KahePoly>> = elements
        .iter()
        .map(|&x| {
            let mut sk = PronySketch::new(params.clone());
            sk.insert(&mut rng, &[x]);
            sk.pack()
        })
        .collect();
    client_polys.extend((0..n_cover).map(|_| PronySketch::cover(&params)));

    let recovered = round_trip(&mut rng, 3, &client_polys);
    let mut got: Vec<i64> = PronySketch::unpack(&params, &recovered)
        .decode()
        .expect("prony decode")
        .into_iter()
        .map(|t| t[0])
        .collect();
    let mut want = elements;
    got.sort();
    want.sort();
    assert_eq!(got, want);
}
