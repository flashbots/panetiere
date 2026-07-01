use chipmunk_code::{CsPoly, KahePoly, Polynomial, N};
use panetiere::codec;
use panetiere::protocol::client::run_client_round;
use panetiere::protocol::{ClientId, ServerId};
use panetiere::protocol::server::{run_server_round, ServerInbox};
use panetiere::protocol::verify::aggregate_and_decrypt;
use panetiere::protocol::ProtocolParams;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// Sample a polynomial with coefficients centered in `[-t/2, t/2)` —
/// matches the KAHE plaintext space `R_t`.
fn rand_message_poly<R: Rng>(rng: &mut R, t: u32) -> KahePoly {
    let half = t as i32 / 2;
    let mut coeffs = [0i32; N];
    for c in coeffs.iter_mut() {
        *c = (rng.gen_range(0..t) as i32) - half;
    }
    KahePoly::from_coeffs(coeffs)
}

/// Reduce each coefficient of `poly` mod `t` into centered range — used to
/// compute the expected aggregate decryption (`Σm mod t`). Uses `normalize`
/// (centered `[-q/2, q/2]`) so the mod-t residue isn't skewed by `q mod t`.
fn reduce_centered_mod_t(poly: KahePoly, t: u32) -> KahePoly {
    let mut p = poly;
    p.normalize();
    let t_i = t as i32;
    let half = t_i / 2;
    let mut coeffs = [0i32; N];
    for (out, &c) in coeffs.iter_mut().zip(p.coeffs().iter()) {
        let r = c.rem_euclid(t_i);
        *out = if r >= half { r - t_i } else { r };
    }
    KahePoly::from_coeffs(coeffs)
}

fn run<R: rand::Rng>(rng: &mut R, n_servers: usize, n_clients: usize) -> (KahePoly, KahePoly) {
    let pp = ProtocolParams::setup(rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n_clients as u32).map(ClientId).collect();

    let mut messages = Vec::with_capacity(n_clients);
    let mut client_entries = Vec::with_capacity(n_clients);
    let mut inboxes: Vec<ServerInbox> = server_ids
        .iter()
        .map(|&sid| ServerInbox {
            server_id: sid,
            items: vec![],
        })
        .collect();

    for &cid in &client_ids {
        let m = rand_message_poly(rng, pp.kahe.t_modulus);
        messages.push(m);
        let round = run_client_round(rng, &pp, cid, vec![m], &server_ids);
        client_entries.push((round.client_id, round.encrypted_message));
        for (idx, (sid, ops)) in round.encrypted_openings.into_iter().enumerate() {
            assert_eq!(sid, server_ids[idx]);
            inboxes[idx].items.push((cid, ops));
        }
    }

    let canonical = client_ids.clone();
    let server_outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
        .collect();

    let recovered = aggregate_and_decrypt(&pp, &canonical, &client_entries, &server_outputs)
        .expect("verify failed");
    assert_eq!(recovered.len(), pp.kahe.mu_kahe);

    let expected = reduce_centered_mod_t(
        messages
            .iter()
            .copied()
            .fold(KahePoly::default(), |a, x| a + x),
        pp.kahe.t_modulus,
    );
    (expected, recovered[0])
}

/// Several clients write distinct byte payloads into disjoint coefficient slots
/// of a single `HVCPoly`. The protocol's recovered sum decodes back to a
/// 1024-byte buffer with each client's bytes intact in its own slot — i.e.
/// anonymous broadcast in slot mode.
///
/// After the q_kahe decoupling, `t = T_MODULUS_DEFAULT = 262_144` (≥ 2^16),
/// so the codec's 16-bit-per-coefficient layout fits inside the plaintext
/// modulus without further reworking.
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
    // One KahePoly = N·2 bytes = 4096 at N=2048.
    const TOTAL_BYTES: usize = 4096;
    assert!(n_clients * SLOT_SIZE <= TOTAL_BYTES);

    let pp = ProtocolParams::setup(&mut rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n_clients as u32).map(ClientId).collect();

    let mut client_entries = Vec::new();
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
        assert_eq!(polys.len(), 1, "slot buffer sized to one KahePoly");

        let round = run_client_round(&mut rng, &pp, cid, polys, &server_ids);
        client_entries.push((round.client_id, round.encrypted_message));
        for (idx, (sid, ops)) in round.encrypted_openings.into_iter().enumerate() {
            assert_eq!(sid, server_ids[idx]);
            inboxes[idx].items.push((cid, ops));
        }
    }

    let canonical = client_ids;
    let outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
        .collect();

    let recovered = aggregate_and_decrypt(&pp, &canonical, &client_entries, &outputs)
        .expect("verify failed");
    let recovered_bytes = codec::decode_raw(&recovered).expect("decode");
    assert_eq!(recovered_bytes.len(), TOTAL_BYTES);

    for slot in 0..n_clients {
        let start = slot * SLOT_SIZE;
        let payload = messages[slot];
        assert_eq!(
            &recovered_bytes[start..start + payload.len()],
            payload,
            "slot {} should hold its client's payload",
            slot
        );
        assert!(
            recovered_bytes[start + payload.len()..start + SLOT_SIZE]
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
///
#[test]
fn slot_mode_8kb_message_multi_poly() {
    let mut rng = ChaCha20Rng::from_seed([13u8; 32]);
    let n_servers = 3;
    let n_clients = 4;
    const SLOT_SIZE: usize = 2048;
    const TOTAL_BYTES: usize = 8 * 1024;
    // One KahePoly = N·2 bytes = 4096 at N=2048 → 8 KiB spans 2 polys.
    const N_POLYS: usize = TOTAL_BYTES / 4096;
    assert_eq!(n_clients * SLOT_SIZE, TOTAL_BYTES);

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

    let encoded: Vec<Vec<KahePoly>> = buffers
        .iter()
        .map(|b| {
            let polys = panetiere::codec::encode_raw(b);
            assert_eq!(polys.len(), N_POLYS);
            polys
        })
        .collect();

    let pp = ProtocolParams::setup(&mut rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n_clients as u32).map(ClientId).collect();
    let canonical = client_ids.clone();

    let mut recovered_polys = Vec::with_capacity(N_POLYS);
    for poly_idx in 0..N_POLYS {
        let mut client_entries = Vec::new();
        let mut inboxes: Vec<ServerInbox> = server_ids
            .iter()
            .map(|&sid| ServerInbox {
                server_id: sid,
                items: vec![],
            })
            .collect();
        for (i, &cid) in client_ids.iter().enumerate() {
            let m = encoded[i][poly_idx];
            let round = run_client_round(&mut rng, &pp, cid, vec![m], &server_ids);
            client_entries.push((round.client_id, round.encrypted_message));
            for (idx, (sid, ops)) in round.encrypted_openings.into_iter().enumerate() {
                assert_eq!(sid, server_ids[idx]);
                inboxes[idx].items.push((cid, ops));
            }
        }
        let outputs: Vec<_> = inboxes
            .iter()
            .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
            .collect();
        let recovered = aggregate_and_decrypt(&pp, &canonical, &client_entries, &outputs)
            .unwrap_or_else(|e| panic!("verify failed at poly {}: {:?}", poly_idx, e));
        assert_eq!(recovered.len(), 1);
        recovered_polys.push(recovered[0]);
    }

    let recovered_bytes = panetiere::codec::decode_raw(&recovered_polys).expect("decode");
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

/// Aggregated flow recovers the same sum as the direct flow: clients' public
/// ciphertexts/commitments are summed per group, re-summed across groups, and
/// fed to `decrypt_aggregate` — openings still go to every server as usual.
#[test]
fn aggregated_recovers_same_sum() {
    use panetiere::cs::{Cs, HidingMerkleCommitment};
    use panetiere::kahe::{Kahe, KaheScheme};
    use panetiere::protocol::aggregator::run_aggregator_round;
    use panetiere::protocol::verify::decrypt_aggregate;

    let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
    let n_servers = 4;
    let n_clients = 45; // > 40, spans 2 groups of ≤ 40
    let n_groups = 2;

    let pp = ProtocolParams::setup(&mut rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n_clients as u32).map(ClientId).collect();

    let mut client_entries = Vec::with_capacity(n_clients);
    let mut inboxes: Vec<ServerInbox> = server_ids
        .iter()
        .map(|&sid| ServerInbox { server_id: sid, items: vec![] })
        .collect();

    for &cid in &client_ids {
        let m = rand_message_poly(&mut rng, pp.kahe.t_modulus);
        let round = run_client_round(&mut rng, &pp, cid, vec![m], &server_ids);
        client_entries.push((round.client_id, round.encrypted_message));
        for (idx, (sid, ops)) in round.encrypted_openings.into_iter().enumerate() {
            assert_eq!(sid, server_ids[idx]);
            inboxes[idx].items.push((cid, ops));
        }
    }

    let canonical = client_ids.clone();
    let server_outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
        .collect();

    let direct = aggregate_and_decrypt(&pp, &canonical, &client_entries, &server_outputs)
        .expect("direct verify failed");

    // Partition clients into groups; each aggregator sums its group's publics.
    let groups: Vec<Vec<(ClientId, _)>> = (0..n_groups)
        .map(|g| {
            client_entries
                .iter()
                .filter(|(cid, _)| cid.0 as usize % n_groups == g)
                .cloned()
                .collect()
        })
        .collect();
    let aggregates: Vec<_> = groups.iter().map(|g| run_aggregator_round(g)).collect();

    // Leader re-sums per-group aggregates across groups.
    let group_ctxts: Vec<Vec<KahePoly>> =
        aggregates.iter().map(|a| a.summed_ctxt.clone()).collect();
    let group_comms: Vec<_> = aggregates.iter().map(|a| a.summed_comm.clone()).collect();
    let total_ctxt = Kahe::agg_ctxt(&group_ctxts);
    let total_comm = HidingMerkleCommitment::sum_commitments(&group_comms);

    let aggregated = decrypt_aggregate(&pp, &total_ctxt, &total_comm, &server_outputs)
        .expect("aggregated verify failed");

    assert_eq!(direct, aggregated);
}

#[test]
fn tampered_agg_share_rejected() {
    let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
    let n_servers = 4;
    let n_clients = 4;
    let pp = ProtocolParams::setup(&mut rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n_clients as u32).map(ClientId).collect();

    let mut client_entries = Vec::new();
    let mut inboxes: Vec<ServerInbox> = server_ids
        .iter()
        .map(|&sid| ServerInbox {
            server_id: sid,
            items: vec![],
        })
        .collect();
    for &cid in &client_ids {
        let m = rand_message_poly(&mut rng, pp.kahe.t_modulus);
        let round = run_client_round(&mut rng, &pp, cid, vec![m], &server_ids);
        client_entries.push((round.client_id, round.encrypted_message));
        for (idx, (sid, ops)) in round.encrypted_openings.into_iter().enumerate() {
            assert_eq!(sid, server_ids[idx]);
            inboxes[idx].items.push((cid, ops));
        }
    }
    let canonical = client_ids;
    let mut server_outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
        .collect();

    server_outputs[0].agg_share[0] =
        server_outputs[0].agg_share[0] + CsPoly::rand_poly(&mut rng);
    let result = aggregate_and_decrypt(&pp, &canonical, &client_entries, &server_outputs);
    assert!(matches!(
        result,
        Err(panetiere::protocol::verify::VerifyError::ShareOpeningMismatch(_))
    ));
}

#[test]
fn high_norm_r_rejected_in_protocol() {
    let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
    let n_servers = 4;
    let n_clients = 4;
    let pp = ProtocolParams::setup(&mut rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n_clients as u32).map(ClientId).collect();

    let mut client_entries = Vec::new();
    let mut inboxes: Vec<ServerInbox> = server_ids
        .iter()
        .map(|&sid| ServerInbox {
            server_id: sid,
            items: vec![],
        })
        .collect();
    for &cid in &client_ids {
        let m = rand_message_poly(&mut rng, pp.kahe.t_modulus);
        let round = run_client_round(&mut rng, &pp, cid, vec![m], &server_ids);
        client_entries.push((round.client_id, round.encrypted_message));
        for (idx, (sid, ops)) in round.encrypted_openings.into_iter().enumerate() {
            assert_eq!(sid, server_ids[idx]);
            inboxes[idx].items.push((cid, ops));
        }
    }
    // Corrupt one private opening before aggregation.
    inboxes[0].items[0].1.r_mut()[0] = CsPoly::rand_poly(&mut rng);

    let canonical = client_ids;
    let server_outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
        .collect();

    let result = aggregate_and_decrypt(&pp, &canonical, &client_entries, &server_outputs);
    assert!(matches!(
        result,
        Err(panetiere::protocol::verify::VerifyError::InvalidServerOpening(_))
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

/// Threshold recovery: with t = ⌊γ/2⌋+1, decryption succeeds with any t of γ
/// server outputs in any order. Verifier picks the first `t` after sorting.
#[test]
fn recovers_from_t_of_n_servers() {
    let mut rng = ChaCha20Rng::from_seed([101u8; 32]);
    let n_servers = 5;
    let n_clients = 4;
    let pp = ProtocolParams::setup(&mut rng, n_servers);
    assert_eq!(pp.shamir.t, 3);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n_clients as u32).map(ClientId).collect();

    let mut messages = Vec::with_capacity(n_clients);
    let mut client_entries = Vec::new();
    let mut inboxes: Vec<ServerInbox> = server_ids
        .iter()
        .map(|&sid| ServerInbox {
            server_id: sid,
            items: vec![],
        })
        .collect();
    for &cid in &client_ids {
        let m = rand_message_poly(&mut rng, pp.kahe.t_modulus);
        messages.push(m);
        let round = run_client_round(&mut rng, &pp, cid, vec![m], &server_ids);
        client_entries.push((round.client_id, round.encrypted_message));
        for (idx, (sid, ops)) in round.encrypted_openings.into_iter().enumerate() {
            assert_eq!(sid, server_ids[idx]);
            inboxes[idx].items.push((cid, ops));
        }
    }
    let canonical = client_ids.clone();
    let outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
        .collect();

    let expected = reduce_centered_mod_t(
        messages
            .iter()
            .copied()
            .fold(KahePoly::default(), |a, x| a + x),
        pp.kahe.t_modulus,
    );

    // Pick any t outputs and verify decryption succeeds.
    for &chosen in &[
        [0usize, 1, 2].as_slice(),
        &[0, 2, 4],
        &[1, 3, 4],
        &[2, 3, 4],
    ] {
        let subset: Vec<_> = chosen.iter().map(|&i| outputs[i].clone()).collect();
        let recovered = aggregate_and_decrypt(&pp, &canonical, &client_entries, &subset)
            .expect("verify failed");
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0], expected, "subset {:?}", chosen);
    }
}
