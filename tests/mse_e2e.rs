//! Paper-aligned MSE (`src/mse.rs`) carried through the flashnet protocol.
//!
//! Each client builds a 1-element MSE encoding, packs it into KahePolys, and
//! contributes them to the protocol. The verifier's recovered per-poly sums
//! unpack into the multiset union, which decodes to all clients' elements.

use chipmunk_code::KahePoly;
use flashnet::mse::{MseEncoding, MseParams};
use flashnet::protocol::client::run_client_round;
use flashnet::protocol::message::{ClientId, ServerId};
use flashnet::protocol::server::{run_server_round, ServerInbox};
use flashnet::protocol::verify::aggregate_and_decrypt;
use flashnet::protocol::ProtocolParams;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// MSE arithmetic now lives in `Z_t` (KAHE plaintext modulus, default
/// `t = 262_144`), large enough to carry random `r` values and small element
/// values without wrap. The protocol's recovered sum decodes back to the
/// multiset union.
#[test]
fn mse_recovers_through_flashnet() {
    let mut rng = ChaCha20Rng::from_seed([0xFEu8; 32]);
    let n_servers = 3;
    let n_clients = 6;

    // Pick γ = 4 (paper Theorem 3 sweet spot), δ = 2·ρ.
    let mse_params = MseParams::new(4, 2 * n_clients, [0xAA; 32]);

    // Each client picks a unique element x ∈ Z_q with small magnitude.
    let elements: Vec<i32> = (0..n_clients).map(|i| 1000 + i as i32).collect();
    let client_polys: Vec<Vec<KahePoly>> = elements
        .iter()
        .map(|&x| {
            let mut enc = MseEncoding::new(mse_params.clone());
            // Use a per-client random `r` drawn from the protocol RNG so each
            // client's encoding has independent randomness. Keep `r` well
            // inside `[-t/2, t/2)` (t = 262_144 by default).
            let r: i32 = rng.gen_range(-100_000..100_000);
            enc.insert_with_r(x, r);
            enc.pack()
        })
        .collect();
    let n_polys = client_polys[0].len();
    assert_eq!(n_polys, MseEncoding::n_polys(&mse_params));

    let pp = ProtocolParams::setup(&mut rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n_clients as u32).map(ClientId).collect();
    let canonical = client_ids.clone();

    let mut recovered_polys = Vec::with_capacity(n_polys);
    for k in 0..n_polys {
        let mut publics = Vec::new();
        let mut inboxes: Vec<ServerInbox> = server_ids
            .iter()
            .map(|&sid| ServerInbox {
                server_id: sid,
                items: vec![],
            })
            .collect();
        for (i, &cid) in client_ids.iter().enumerate() {
            let m = vec![client_polys[i][k]];
            let round = run_client_round(&mut rng, &pp, cid, m, &server_ids);
            publics.push((round.client_id, round.public));
            for (idx, (sid, ops)) in round.private.into_iter().enumerate() {
                let _ = sid;
                inboxes[idx].items.push((cid, ops));
            }
        }
        let outputs: Vec<_> = inboxes
            .iter()
            .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
            .collect();
        let recovered = aggregate_and_decrypt(&pp, &canonical, &publics, &outputs)
            .unwrap_or_else(|e| panic!("verify failed at poly {}: {:?}", k, e));
        assert_eq!(recovered.len(), 1);
        recovered_polys.push(recovered[0]);
    }

    let union = MseEncoding::unpack(&mse_params, &recovered_polys);
    let mut got = union.decode().expect("MSE decode");
    let mut expected = elements.clone();
    got.sort();
    expected.sort();
    assert_eq!(got, expected);
}
