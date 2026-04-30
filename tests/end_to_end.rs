use chipmunk_code::{HVCPoly, Polynomial};
use flashnet::codec;
use flashnet::cs::{Cs, HidingMerkleCommitment};
use flashnet::protocol::client::run_client_round;
use flashnet::protocol::message::{ClientId, ServerId};
use flashnet::protocol::server::{run_server_round, ServerInbox};
use flashnet::protocol::verify::aggregate_and_decrypt;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn run<R: rand::Rng>(rng: &mut R, n_servers: usize, n_clients: usize) -> (HVCPoly, HVCPoly) {
    let pp = HidingMerkleCommitment::setup(rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n_clients as u32).map(ClientId).collect();

    let mut messages = Vec::with_capacity(n_clients);
    let mut publics = Vec::with_capacity(n_clients);
    let mut inboxes: Vec<ServerInbox> = server_ids
        .iter()
        .map(|&sid| ServerInbox {
            server_id: sid,
            items: vec![],
        })
        .collect();

    for &cid in &client_ids {
        let m = HVCPoly::rand_poly(rng);
        messages.push(m);
        let round = run_client_round(rng, &pp, cid, m, &server_ids);
        publics.push((round.client_id, round.public));
        for (idx, (sid, op, sh)) in round.private.into_iter().enumerate() {
            assert_eq!(sid, server_ids[idx]);
            inboxes[idx].items.push((cid, op, sh));
        }
    }

    let canonical = client_ids.clone();
    let server_outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
        .collect();

    let recovered = aggregate_and_decrypt(&pp, &canonical, &publics, &server_outputs)
        .expect("verify failed");

    let expected = messages
        .iter()
        .copied()
        .fold(HVCPoly::default(), |a, x| a + x);
    (expected, recovered)
}

/// Several clients write distinct byte payloads into disjoint coefficient slots
/// of a single `HVCPoly`. The protocol's recovered sum decodes back to a
/// 1024-byte buffer with each client's bytes intact in its own slot — i.e.
/// anonymous broadcast in slot mode.
#[test]
fn slot_mode_disjoint_clients_recover_each_payload() {
    let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
    let n_servers = 3;

    let messages: &[&[u8]] = &[
        b"hello from client 0",
        b"client 1 says hi",
        b"another from #2",
        b"#3: anonymous broadcast",
        b"client 4 here",
        b"final note from 5",
    ];
    let n_clients = messages.len();
    const SLOT_SIZE: usize = 128;
    const TOTAL_BYTES: usize = 1024; // exactly one HVCPoly worth (no length header)
    assert!(n_clients * SLOT_SIZE <= TOTAL_BYTES);

    let pp = HidingMerkleCommitment::setup(&mut rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n_clients as u32).map(ClientId).collect();

    let mut publics = Vec::new();
    let mut inboxes: Vec<ServerInbox> = server_ids
        .iter()
        .map(|&sid| ServerInbox {
            server_id: sid,
            items: vec![],
        })
        .collect();

    for (slot, &cid) in client_ids.iter().enumerate() {
        let payload = messages[slot];
        let mut buf = vec![0u8; TOTAL_BYTES];
        let start = slot * SLOT_SIZE;
        buf[start..start + payload.len()].copy_from_slice(payload);

        let polys = codec::encode_raw(&buf);
        assert_eq!(polys.len(), 1, "slot buffer sized to one HVCPoly");
        let m = polys[0];

        let round = run_client_round(&mut rng, &pp, cid, m, &server_ids);
        publics.push((round.client_id, round.public));
        for (idx, (sid, op, sh)) in round.private.into_iter().enumerate() {
            assert_eq!(sid, server_ids[idx]);
            inboxes[idx].items.push((cid, op, sh));
        }
    }

    let canonical = client_ids;
    let outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
        .collect();

    let recovered_poly = aggregate_and_decrypt(&pp, &canonical, &publics, &outputs)
        .expect("verify failed");
    let recovered = codec::decode_raw(&[recovered_poly]).expect("decode");
    assert_eq!(recovered.len(), TOTAL_BYTES);

    for slot in 0..n_clients {
        let start = slot * SLOT_SIZE;
        let payload = messages[slot];
        assert_eq!(
            &recovered[start..start + payload.len()],
            payload,
            "slot {} should hold its client's payload",
            slot
        );
        // Bytes outside the payload but inside the slot are zero (no other
        // client wrote there).
        assert!(
            recovered[start + payload.len()..start + SLOT_SIZE]
                .iter()
                .all(|&b| b == 0),
            "non-payload bytes in slot {} should be zero",
            slot
        );
    }
}

/// 8 KB broadcast: total buffer is 8 polys (8192 bytes). Each of 4 clients
/// writes a 2 KB payload into its own slot; we run the full protocol once per
/// poly index of the encoded buffer (8 sub-rounds), recover the per-poly sums,
/// concatenate, and assert each client's slot decodes byte-for-byte.
#[test]
fn slot_mode_8kb_message_multi_poly() {
    let mut rng = ChaCha20Rng::from_seed([13u8; 32]);
    let n_servers = 3;
    let n_clients = 4;
    const SLOT_SIZE: usize = 2048;
    const TOTAL_BYTES: usize = 8 * 1024; // 8 polys × 1024 bytes/poly
    const N_POLYS: usize = TOTAL_BYTES / 1024;
    assert_eq!(n_clients * SLOT_SIZE, TOTAL_BYTES);

    // Build per-client 8 KB buffers: pseudo-random payload in each client's slot,
    // zeros elsewhere.
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(n_clients);
    let mut buffers: Vec<Vec<u8>> = Vec::with_capacity(n_clients);
    for slot in 0..n_clients {
        let payload: Vec<u8> = (0..SLOT_SIZE)
            .map(|j| ((slot * 131 + j * 17) & 0xFF) as u8)
            .collect();
        let mut buf = vec![0u8; TOTAL_BYTES];
        buf[slot * SLOT_SIZE..(slot + 1) * SLOT_SIZE].copy_from_slice(&payload);
        payloads.push(payload);
        buffers.push(buf);
    }

    // Each buffer encodes to N_POLYS polys.
    let encoded: Vec<Vec<HVCPoly>> = buffers
        .iter()
        .map(|b| {
            let polys = flashnet::codec::encode_raw(b);
            assert_eq!(polys.len(), N_POLYS);
            polys
        })
        .collect();

    let pp = HidingMerkleCommitment::setup(&mut rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n_clients as u32).map(ClientId).collect();
    let canonical = client_ids.clone();

    // Run the protocol once per poly index. Concatenate recovered polys.
    let mut recovered_polys = Vec::with_capacity(N_POLYS);
    for poly_idx in 0..N_POLYS {
        let mut publics = Vec::new();
        let mut inboxes: Vec<ServerInbox> = server_ids
            .iter()
            .map(|&sid| ServerInbox {
                server_id: sid,
                items: vec![],
            })
            .collect();
        for (i, &cid) in client_ids.iter().enumerate() {
            let m = encoded[i][poly_idx];
            let round = run_client_round(&mut rng, &pp, cid, m, &server_ids);
            publics.push((round.client_id, round.public));
            for (idx, (sid, op, sh)) in round.private.into_iter().enumerate() {
                assert_eq!(sid, server_ids[idx]);
                inboxes[idx].items.push((cid, op, sh));
            }
        }
        let outputs: Vec<_> = inboxes
            .iter()
            .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
            .collect();
        let recovered = aggregate_and_decrypt(&pp, &canonical, &publics, &outputs)
            .unwrap_or_else(|e| panic!("verify failed at poly {}: {:?}", poly_idx, e));
        recovered_polys.push(recovered);
    }

    let recovered_bytes = flashnet::codec::decode_raw(&recovered_polys).expect("decode");
    assert_eq!(recovered_bytes.len(), TOTAL_BYTES);

    for slot in 0..n_clients {
        let start = slot * SLOT_SIZE;
        assert_eq!(
            &recovered_bytes[start..start + SLOT_SIZE],
            payloads[slot].as_slice(),
            "slot {} mismatch",
            slot
        );
    }
}

#[test]
fn end_to_end_recovers_sum() {
    let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
    let (expected, recovered) = run(&mut rng, 4, 8);
    assert_eq!(expected, recovered);
}

#[test]
fn tampered_agg_share_rejected() {
    let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
    let n_servers = 4;
    let n_clients = 4;
    let pp = HidingMerkleCommitment::setup(&mut rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n_clients as u32).map(ClientId).collect();

    let mut publics = Vec::new();
    let mut inboxes: Vec<ServerInbox> = server_ids
        .iter()
        .map(|&sid| ServerInbox {
            server_id: sid,
            items: vec![],
        })
        .collect();
    for &cid in &client_ids {
        let m = HVCPoly::rand_poly(&mut rng);
        let round = run_client_round(&mut rng, &pp, cid, m, &server_ids);
        publics.push((round.client_id, round.public));
        for (idx, (sid, op, sh)) in round.private.into_iter().enumerate() {
            assert_eq!(sid, server_ids[idx]);
            inboxes[idx].items.push((cid, op, sh));
        }
    }
    let canonical = client_ids;
    let mut server_outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
        .collect();

    // Tamper: swap the agg_share of two servers (changes individual shares
    // but not their sum), so decryption silently produces the wrong message
    // unless the commitment opening check rejects.
    server_outputs[0].agg_share = server_outputs[0].agg_share + HVCPoly::rand_poly(&mut rng);
    let result = aggregate_and_decrypt(&pp, &canonical, &publics, &server_outputs);
    assert!(matches!(
        result,
        Err(flashnet::protocol::verify::VerifyError::ShareOpeningMismatch(_))
    ));
}

#[test]
fn high_norm_r_rejected_in_protocol() {
    let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
    let n_servers = 4;
    let n_clients = 4;
    let pp = HidingMerkleCommitment::setup(&mut rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n_clients as u32).map(ClientId).collect();

    let mut publics = Vec::new();
    let mut inboxes: Vec<ServerInbox> = server_ids
        .iter()
        .map(|&sid| ServerInbox {
            server_id: sid,
            items: vec![],
        })
        .collect();
    for &cid in &client_ids {
        let m = HVCPoly::rand_poly(&mut rng);
        let round = run_client_round(&mut rng, &pp, cid, m, &server_ids);
        publics.push((round.client_id, round.public));
        for (idx, (sid, op, sh)) in round.private.into_iter().enumerate() {
            assert_eq!(sid, server_ids[idx]);
            inboxes[idx].items.push((cid, op, sh));
        }
    }
    // Corrupt one private opening before aggregation: replace one entry of r
    // with a uniform polynomial — ||r||_∞ vastly exceeds r_bound.
    inboxes[0].items[0].1.r[0] = HVCPoly::rand_poly(&mut rng);

    let canonical = client_ids;
    let server_outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
        .collect();

    let result = aggregate_and_decrypt(&pp, &canonical, &publics, &server_outputs);
    assert!(matches!(
        result,
        Err(flashnet::protocol::verify::VerifyError::InvalidServerOpening(_))
    ));
}

#[test]
fn end_to_end_small() {
    let mut rng = ChaCha20Rng::from_seed([99u8; 32]);
    for (s, c) in [(1usize, 1usize), (2, 1), (2, 3), (3, 5), (4, 1)] {
        let (expected, recovered) = run(&mut rng, s, c);
        assert_eq!(expected, recovered, "s={} c={}", s, c);
    }
}
