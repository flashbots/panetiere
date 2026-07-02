//! Paper-aligned MSE (`src/mse.rs`) carried through the Panetière protocol.
//!
//! Each client builds a 1-element MSE encoding, packs it into KahePolys, and
//! contributes them to the protocol. The verifier's recovered per-poly sums
//! unpack into the multiset union, which decodes to all clients' elements.

use chipmunk_code::{KahePoly, Polynomial};
use panetiere::mse::{MseEncoding, MseParams};
use panetiere::pke;
use panetiere::protocol::client::run_client_round;
use panetiere::protocol::{ClientId, ServerId};
use panetiere::protocol::server::{run_server_round, unseal_opening, ServerInbox};
use panetiere::protocol::verify::aggregate_and_decrypt;
use panetiere::protocol::ProtocolParams;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// MSE arithmetic now lives in `Z_t` (KAHE plaintext modulus, default
/// `t = 262_144`), large enough to carry random `r` values and small element
/// values without wrap. The protocol's recovered sum decodes back to the
/// multiset union.
#[test]
fn mse_recovers_through_panetiere() {
    let mut rng = ChaCha20Rng::from_seed([0xFEu8; 32]);
    let n_servers = 3;
    let n_clients = 6;

    // Pick γ = 4 (paper Theorem 3 sweet spot), δ = 2·ρ, ξ = 1.
    let mse_params = MseParams::new(4, 2 * n_clients, 1, [0xAA; 32]);

    // Each client picks a unique element x ∈ Z_q with small magnitude.
    let elements: Vec<i32> = (0..n_clients).map(|i| 1000 + i as i32).collect();
    let client_polys: Vec<Vec<KahePoly>> = elements
        .iter()
        .map(|&x| {
            let mut enc = MseEncoding::new(mse_params.clone());
            // Per-client random `r ∈ [0, t^K_LIMBS)` drawn from the protocol RNG.
            let r: u64 = rng.gen_range(0..mse_params.r_space());
            enc.insert_with_r(&[x], r);
            enc.pack()
        })
        .collect();
    let n_polys = client_polys[0].len();
    assert_eq!(n_polys, MseEncoding::n_polys(&mse_params));

    let pp = ProtocolParams::setup(&mut rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let server_keys: Vec<pke::PrivateKey> = (0..n_servers)
        .map(|_| pke::PrivateKey::generate(&mut rng))
        .collect();
    let servers: Vec<(ServerId, pke::PublicKey)> = server_ids
        .iter()
        .map(|&sid| (sid, server_keys[sid.0 as usize].public()))
        .collect();
    let client_ids: Vec<ClientId> = (0..n_clients as u32).map(ClientId).collect();
    let canonical = client_ids.clone();

    let mut recovered_polys = Vec::with_capacity(n_polys);
    for k in 0..n_polys {
        let mut client_entries = Vec::new();
        let mut inboxes: Vec<ServerInbox> = server_ids
            .iter()
            .map(|&sid| ServerInbox {
                server_id: sid,
                items: vec![],
            })
            .collect();
        for (i, &cid) in client_ids.iter().enumerate() {
            let m = vec![client_polys[i][k]];
            let round = run_client_round(&mut rng, &pp, cid, m, &servers);
            client_entries.push((round.client_id, round.encrypted_message));
            for (idx, (_sid, sealed)) in round.sealed_openings.into_iter().enumerate() {
                let opening = unseal_opening(&server_keys[idx], &sealed).expect("unseal");
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
        recovered_polys.push(recovered[0]);
    }

    let union = MseEncoding::unpack(&mse_params, &recovered_polys);
    let mut got: Vec<i32> = union
        .decode()
        .expect("MSE decode")
        .into_iter()
        .map(|t| t[0])
        .collect();
    let mut expected = elements.clone();
    got.sort();
    expected.sort();
    assert_eq!(got, expected);
}

/// Cover traffic: `n_active` clients carry real elements, `n_cover` send
/// `MseEncoding::cover` (zero polys). All `n_active + n_cover` clients run the
/// full protocol — they're in the canonical set and the anonymity set — but the
/// IBLT is sized only to `n_active` and the recovered multiset is exactly the
/// active elements. The cover clients add nothing to peeling.
#[test]
fn cover_clients_do_not_inflate_iblt() {
    let mut rng = ChaCha20Rng::from_seed([0x5Au8; 32]);
    let n_servers = 3;
    let n_active = 6;
    let n_cover = 14;
    let n_total = n_active + n_cover;

    // IBLT sized to the active count, not the anonymity set.
    let mse_params = MseParams::new(4, 2 * n_active, 1, [0xAA; 32]);

    let elements: Vec<i32> = (0..n_active).map(|i| 1000 + i as i32).collect();
    let client_polys: Vec<Vec<KahePoly>> = (0..n_total)
        .map(|i| {
            if i < n_active {
                let mut enc = MseEncoding::new(mse_params.clone());
                let r: u64 = rng.gen_range(0..mse_params.r_space());
                enc.insert_with_r(&[elements[i]], r);
                enc.pack()
            } else {
                MseEncoding::cover(&mse_params)
            }
        })
        .collect();
    let n_polys = MseEncoding::n_polys(&mse_params);
    assert!(client_polys.iter().all(|p| p.len() == n_polys));

    let pp = ProtocolParams::setup(&mut rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let server_keys: Vec<pke::PrivateKey> = (0..n_servers)
        .map(|_| pke::PrivateKey::generate(&mut rng))
        .collect();
    let servers: Vec<(ServerId, pke::PublicKey)> = server_ids
        .iter()
        .map(|&sid| (sid, server_keys[sid.0 as usize].public()))
        .collect();
    let client_ids: Vec<ClientId> = (0..n_total as u32).map(ClientId).collect();
    let canonical = client_ids.clone();

    let mut recovered_polys = Vec::with_capacity(n_polys);
    for k in 0..n_polys {
        let mut client_entries = Vec::new();
        let mut inboxes: Vec<ServerInbox> = server_ids
            .iter()
            .map(|&sid| ServerInbox {
                server_id: sid,
                items: vec![],
            })
            .collect();
        for (i, &cid) in client_ids.iter().enumerate() {
            let m = vec![client_polys[i][k]];
            let round = run_client_round(&mut rng, &pp, cid, m, &servers);
            client_entries.push((round.client_id, round.encrypted_message));
            for (idx, (_sid, sealed)) in round.sealed_openings.into_iter().enumerate() {
                let opening = unseal_opening(&server_keys[idx], &sealed).expect("unseal");
                inboxes[idx].items.push((cid, opening));
            }
        }
        let outputs: Vec<_> = inboxes
            .iter()
            .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
            .collect();
        let recovered = aggregate_and_decrypt(&pp, &canonical, &client_entries, &outputs)
            .unwrap_or_else(|e| panic!("verify failed at poly {}: {:?}", k, e));
        recovered_polys.push(recovered[0]);
    }

    let mut got: Vec<i32> = MseEncoding::unpack(&mse_params, &recovered_polys)
        .decode()
        .expect("MSE decode")
        .into_iter()
        .map(|t| t[0])
        .collect();
    let mut expected = elements.clone();
    got.sort();
    expected.sort();
    assert_eq!(got, expected);
}

/// Guards the cover/insert distinction: `cover` adds nothing, but a
/// zero-*payload* `insert(&[0])` occupies cells and peels back out as a real
/// element. MSE-level, no protocol needed.
#[test]
fn zero_payload_insert_is_not_cover() {
    let mut rng = ChaCha20Rng::from_seed([0x11u8; 32]);
    let mse_params = MseParams::new(4, 16, 1, [0xAA; 32]);

    let mut enc = MseEncoding::new(mse_params.clone());
    enc.insert(&mut rng, &[7]);
    enc.insert(&mut rng, &[0]); // zero payload — still a real insert
    let recovered = enc.decode().expect("decode");
    let mut got: Vec<i32> = recovered.into_iter().map(|t| t[0]).collect();
    got.sort();
    assert_eq!(got, vec![0, 7]);

    // cover() leaves the matrices empty: nothing to peel.
    assert!(MseEncoding::cover(&mse_params)
        .iter()
        .all(|p| { let mut q = *p; q.normalize(); q.coeffs().iter().all(|&c| c == 0) }));
}
