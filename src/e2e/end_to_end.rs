#![allow(clippy::needless_range_loop, clippy::type_complexity)]

use crate::bulletin::ClientBulletinEntry;
use crate::cs::Commitment;
use crate::kahe::poly_mod_t;
use crate::pke;
use crate::protocol::client::{run_client_round, ClientRound};
use crate::protocol::server::{run_server_round, unseal_opening, ServerInbox, ServerRoundError};
use crate::protocol::verify::aggregate_and_decrypt_unverified;
use crate::protocol::ProtocolParams;
use crate::protocol::{ClientId, ServerId, SessionId};
use crate::{CsPoly, KahePoly, N};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// These tests each run a single protocol execution, so one session id suffices.
/// `session_binding_rejects_replay` uses its own distinct ids.
const SESSION: SessionId = SessionId([0x5D; 32]);

/// Full round from a fixed seed: per-client ctxt + commitment bytes + sealed
/// envelopes, plus the recovered message. Used to check that panetiere's
/// rayon-parallelized phases are thread-count-invariant.
fn run_full_round_bytes(
    seed: [u8; 32],
    n_servers: usize,
    n_clients: usize,
) -> (
    Vec<Vec<KahePoly>>,
    Vec<Vec<u8>>,
    Vec<Vec<Vec<u8>>>,
    Vec<KahePoly>,
) {
    let mut rng = ChaCha20Rng::from_seed(seed);
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

    let mut ctxts = Vec::with_capacity(n_clients);
    let mut comm_bytes = Vec::with_capacity(n_clients);
    let mut sealed_all = Vec::with_capacity(n_clients);
    let mut client_entries = Vec::with_capacity(n_clients);
    let mut inboxes: Vec<ServerInbox> = server_ids
        .iter()
        .map(|&sid| ServerInbox {
            server_id: sid,
            items: vec![],
        })
        .collect();

    for &cid in &client_ids {
        let m = rand_message_poly(&mut rng, pp.kahe.t_modulus);
        let round = run_client_round(&mut rng, &pp, &SESSION, cid, vec![m], &servers);
        ctxts.push(round.encrypted_message.ctxt.clone());
        comm_bytes.push(round.encrypted_message.comm.to_bytes());
        let mut sealed_for_client = Vec::with_capacity(n_servers);
        for (idx, (sid, sealed)) in round.sealed_openings.into_iter().enumerate() {
            assert_eq!(sid, server_ids[idx]);
            let opening =
                unseal_opening(&server_keys[idx], &SESSION, cid, sid, &sealed).expect("unseal");
            sealed_for_client.push(sealed);
            inboxes[idx].items.push((cid, opening));
        }
        sealed_all.push(sealed_for_client);
        client_entries.push((round.client_id, round.encrypted_message));
    }

    let canonical = client_ids.clone();
    let commitments: Vec<(ClientId, Commitment)> = client_entries
        .iter()
        .map(|(cid, e)| (*cid, e.comm.clone()))
        .collect();
    let server_outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(&pp, inb, &canonical, &commitments).expect("missing client"))
        .collect();
    let recovered =
        aggregate_and_decrypt_unverified(&pp, &canonical, &client_entries, &server_outputs)
            .expect("verify failed");

    (ctxts, comm_bytes, sealed_all, recovered)
}

/// The rayon-parallelized phases (kahe enc/dec/agg_ctxt, cs commit, seal,
/// verify) fork RNG streams per work item so the transcript is bit-identical
/// regardless of how many threads actually run it.
#[test]
fn full_round_is_thread_count_invariant() {
    let seed = [77u8; 32];
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| run_full_round_bytes(seed, 5, 6));
    let many = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap()
        .install(|| run_full_round_bytes(seed, 5, 6));
    assert_eq!(one, many);
}

/// Sample a polynomial with centered coefficients in `[−(t−1)/2, (t−1)/2]`,
/// the KAHE plaintext space `R_t`.
fn rand_message_poly<R: Rng>(rng: &mut R, t: u64) -> KahePoly {
    let half = t as i64 / 2;
    let mut coeffs = [0i64; N];
    for c in coeffs.iter_mut() {
        *c = rng.gen_range(0..t) as i64 - half;
    }
    KahePoly::from_coeffs(coeffs)
}

#[test]
fn client_round_accepts_message_after_offline_work() {
    let mut rng = ChaCha20Rng::from_seed([0x39; 32]);
    let pp = ProtocolParams::setup(&mut rng, 4);
    let server_keys: Vec<_> = (0..4)
        .map(|_| pke::PrivateKey::generate(&mut rng))
        .collect();
    let servers: Vec<_> = server_keys
        .iter()
        .enumerate()
        .map(|(i, key)| (ServerId(i as u32), key.public()))
        .collect();
    let message = rand_message_poly(&mut rng, pp.kahe.t_modulus);
    let mut direct_rng = rng.clone();
    let direct = run_client_round(
        &mut direct_rng,
        &pp,
        &SESSION,
        ClientId(0),
        vec![message],
        &servers,
    );
    let round =
        ClientRound::new(&mut rng, &pp, &SESSION, ClientId(0), &servers).finalize(&[message]);
    assert_eq!(round.encrypted_message.ctxt, direct.encrypted_message.ctxt);
    assert_eq!(
        round.encrypted_message.comm.to_bytes(),
        direct.encrypted_message.comm.to_bytes(),
    );
    assert_eq!(round.sealed_openings, direct.sealed_openings);
    let mut outputs = Vec::new();
    for (i, (server_id, sealed)) in round.sealed_openings.iter().enumerate() {
        let opening =
            unseal_opening(&server_keys[i], &SESSION, ClientId(0), *server_id, sealed).unwrap();
        outputs.push(
            run_server_round(
                &pp,
                &ServerInbox {
                    server_id: *server_id,
                    items: vec![(ClientId(0), opening)],
                },
                &[ClientId(0)],
                &[(ClientId(0), round.encrypted_message.comm.clone())],
            )
            .unwrap(),
        );
    }
    let recovered = aggregate_and_decrypt_unverified(
        &pp,
        &[ClientId(0)],
        &[(ClientId(0), round.encrypted_message)],
        &outputs,
    )
    .unwrap();
    assert_eq!(recovered, vec![poly_mod_t(&message, pp.kahe.t_modulus)]);
}

fn run<R: rand::Rng + rand::CryptoRng>(
    rng: &mut R,
    n_servers: usize,
    n_clients: usize,
) -> (KahePoly, KahePoly) {
    let mut rng = rng;
    let pp = ProtocolParams::setup(rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let server_keys: Vec<pke::PrivateKey> = (0..n_servers)
        .map(|_| pke::PrivateKey::generate(&mut rng))
        .collect();
    let servers: Vec<(ServerId, pke::PublicKey)> = server_ids
        .iter()
        .map(|&sid| (sid, server_keys[sid.0 as usize].public()))
        .collect();
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
        let round = run_client_round(rng, &pp, &SESSION, cid, vec![m], &servers);
        client_entries.push((round.client_id, round.encrypted_message));
        for (idx, (sid, sealed)) in round.sealed_openings.into_iter().enumerate() {
            assert_eq!(sid, server_ids[idx]);
            let opening =
                unseal_opening(&server_keys[idx], &SESSION, cid, sid, &sealed).expect("unseal");
            inboxes[idx].items.push((cid, opening));
        }
    }

    let canonical = client_ids.clone();
    let commitments: Vec<(ClientId, Commitment)> = client_entries
        .iter()
        .map(|(cid, e)| (*cid, e.comm.clone()))
        .collect();
    let server_outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(&pp, inb, &canonical, &commitments).expect("missing client"))
        .collect();

    let recovered =
        aggregate_and_decrypt_unverified(&pp, &canonical, &client_entries, &server_outputs)
            .expect("verify failed");
    assert_eq!(recovered.len(), pp.kahe.mu_kahe);

    let expected = poly_mod_t(
        &messages
            .iter()
            .copied()
            .fold(KahePoly::default(), |a, x| a + x),
        pp.kahe.t_modulus,
    );
    (expected, recovered[0])
}

#[test]
fn end_to_end_recovers_sum() {
    let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
    let (expected, recovered) = run(&mut rng, 4, 8);
    assert_eq!(expected, recovered);
}

/// An envelope cannot be replayed into another session, re-attributed to another
/// client, or misfiled against another server's slot.
#[test]
fn session_binding_rejects_replay() {
    let mut rng = ChaCha20Rng::from_seed([0x9Cu8; 32]);
    let n_servers = 4;
    let pp = ProtocolParams::setup(&mut rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let server_keys: Vec<pke::PrivateKey> = (0..n_servers)
        .map(|_| pke::PrivateKey::generate(&mut rng))
        .collect();
    let servers: Vec<(ServerId, pke::PublicKey)> = server_ids
        .iter()
        .map(|&sid| (sid, server_keys[sid.0 as usize].public()))
        .collect();

    let round_a = SessionId([0xA1; 32]);
    let round_b = SessionId([0xB2; 32]);
    let cid = ClientId(7);
    let m = rand_message_poly(&mut rng, pp.kahe.t_modulus);
    let round = run_client_round(&mut rng, &pp, &round_a, cid, vec![m], &servers);
    let (target, sealed) = &round.sealed_openings[0];
    assert_eq!(*target, server_ids[0]);

    assert!(
        unseal_opening(&server_keys[0], &round_a, cid, *target, sealed).is_some(),
        "the honest context must open"
    );
    assert!(
        unseal_opening(&server_keys[0], &round_b, cid, *target, sealed).is_none(),
        "an envelope from another session must not open"
    );
    assert!(
        unseal_opening(&server_keys[0], &round_a, ClientId(8), *target, sealed).is_none(),
        "an envelope must not be re-attributed to another client"
    );
    assert!(
        unseal_opening(&server_keys[0], &round_a, cid, server_ids[1], sealed).is_none(),
        "an envelope must not be misfiled against another server's slot"
    );
}

#[test]
fn tampered_server_opening_rejected() {
    let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
    let n_servers = 4;
    let n_clients = 4;
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
        let round = run_client_round(&mut rng, &pp, &SESSION, cid, vec![m], &servers);
        client_entries.push((round.client_id, round.encrypted_message));
        for (idx, (sid, sealed)) in round.sealed_openings.into_iter().enumerate() {
            assert_eq!(sid, server_ids[idx]);
            let opening =
                unseal_opening(&server_keys[idx], &SESSION, cid, sid, &sealed).expect("unseal");
            inboxes[idx].items.push((cid, opening));
        }
    }
    let canonical = client_ids;
    let commitments: Vec<(ClientId, Commitment)> = client_entries
        .iter()
        .map(|(cid, e)| (*cid, e.comm.clone()))
        .collect();
    let mut server_outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(&pp, inb, &canonical, &commitments).expect("missing client"))
        .collect();

    // swapping two polynomials of r keeps the opening within its bounds, but it
    // doesn't open the summed commitment, so server 0's share can't be recovered
    server_outputs[0].agg_open.r_mut().swap(0, 1);
    let result =
        aggregate_and_decrypt_unverified(&pp, &canonical, &client_entries, &server_outputs);
    assert_eq!(
        result,
        Err(crate::protocol::verify::VerifyError::InvalidServerOpening(
            0
        ))
    );
}

#[test]
fn below_floor_anonymity_set_rejected() {
    use crate::protocol::verify::VerifyError;

    let mut rng = ChaCha20Rng::from_seed([21u8; 32]);
    let n_servers = 4;
    let n_clients = 4;
    let mut pp = ProtocolParams::setup(&mut rng, n_servers);
    pp.min_clients = n_clients + 1;
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let server_keys: Vec<pke::PrivateKey> = (0..n_servers)
        .map(|_| pke::PrivateKey::generate(&mut rng))
        .collect();
    let servers: Vec<(ServerId, pke::PublicKey)> = server_ids
        .iter()
        .map(|&sid| (sid, server_keys[sid.0 as usize].public()))
        .collect();
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
        let round = run_client_round(&mut rng, &pp, &SESSION, cid, vec![m], &servers);
        client_entries.push((round.client_id, round.encrypted_message));
        for (idx, (sid, sealed)) in round.sealed_openings.into_iter().enumerate() {
            assert_eq!(sid, server_ids[idx]);
            let opening =
                unseal_opening(&server_keys[idx], &SESSION, cid, sid, &sealed).expect("unseal");
            inboxes[idx].items.push((cid, opening));
        }
    }
    let canonical = client_ids;
    let commitments: Vec<(ClientId, Commitment)> = client_entries
        .iter()
        .map(|(cid, e)| (*cid, e.comm.clone()))
        .collect();
    let server_outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(&pp, inb, &canonical, &commitments).expect("missing client"))
        .collect();

    let result =
        aggregate_and_decrypt_unverified(&pp, &canonical, &client_entries, &server_outputs);
    assert_eq!(
        result,
        Err(VerifyError::AnonymitySetTooSmall {
            got: n_clients,
            min: n_clients + 1
        })
    );

    pp.min_clients = n_clients;
    aggregate_and_decrypt_unverified(&pp, &canonical, &client_entries, &server_outputs)
        .expect("at-floor set must verify");
}

/// Clients, servers and per-server inboxes for one round, plus the public
/// posts and commitments the servers and recipient read off the bulletin.
struct Round {
    pp: ProtocolParams,
    canonical: Vec<ClientId>,
    client_entries: Vec<(ClientId, ClientBulletinEntry)>,
    commitments: Vec<(ClientId, Commitment)>,
    inboxes: Vec<ServerInbox>,
}

fn setup_round(rng: &mut ChaCha20Rng, n_servers: usize, n_clients: usize) -> Round {
    let pp = ProtocolParams::setup(rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let server_keys: Vec<pke::PrivateKey> = (0..n_servers)
        .map(|_| pke::PrivateKey::generate(rng))
        .collect();
    let servers: Vec<(ServerId, pke::PublicKey)> = server_ids
        .iter()
        .map(|&sid| (sid, server_keys[sid.0 as usize].public()))
        .collect();
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
        let m = rand_message_poly(rng, pp.kahe.t_modulus);
        let round = run_client_round(rng, &pp, &SESSION, cid, vec![m], &servers);
        client_entries.push((round.client_id, round.encrypted_message));
        for (idx, (sid, sealed)) in round.sealed_openings.into_iter().enumerate() {
            assert_eq!(sid, server_ids[idx]);
            let opening =
                unseal_opening(&server_keys[idx], &SESSION, cid, sid, &sealed).expect("unseal");
            inboxes[idx].items.push((cid, opening));
        }
    }
    let commitments = client_entries
        .iter()
        .map(|(cid, e)| (*cid, e.comm.clone()))
        .collect();
    Round {
        pp,
        canonical: client_ids,
        client_entries,
        commitments,
        inboxes,
    }
}

/// A client whose opening does not open its commitment is named by the server
/// it targeted, which posts nothing. The other servers are unaffected and still
/// reach the threshold, so the round completes without blaming an honest server.
#[test]
fn bad_opening_is_named_by_server_not_blamed_on_it() {
    let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
    let mut round = setup_round(&mut rng, 4, 4);
    assert_eq!(round.pp.shamir.t, 3);
    round.inboxes[0].items[1].1.r_mut()[0] = CsPoly::rand_poly(&mut rng);

    let results: Vec<_> = round
        .inboxes
        .iter()
        .map(|inb| run_server_round(&round.pp, inb, &round.canonical, &round.commitments))
        .collect();
    assert!(matches!(
        results[0],
        Err(ServerRoundError::BadOpening(ClientId(1)))
    ));
    let server_outputs: Vec<_> = results.into_iter().skip(1).map(|r| r.unwrap()).collect();
    aggregate_and_decrypt_unverified(
        &round.pp,
        &round.canonical,
        &round.client_entries,
        &server_outputs,
    )
    .expect("remaining servers reach the threshold");
}

/// A commitment on the bulletin that the client's openings do not open is
/// caught by every server, each naming that client.
#[test]
fn opening_against_wrong_commitment_rejected() {
    let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
    let mut round = setup_round(&mut rng, 4, 4);
    round.commitments[2].1 = round.commitments[3].1.clone();

    for inb in &round.inboxes {
        let result = run_server_round(&round.pp, inb, &round.canonical, &round.commitments);
        assert!(matches!(
            result,
            Err(ServerRoundError::BadOpening(ClientId(2)))
        ));
    }
}

/// A valid opening for another server's position is rejected, since summing it
/// at this position would corrupt the aggregate.
#[test]
fn opening_for_other_position_rejected() {
    let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
    let mut round = setup_round(&mut rng, 4, 4);
    let from_server_1 = round.inboxes[1].items[0].1.clone();
    round.inboxes[0].items[0].1 = from_server_1;

    let result = run_server_round(
        &round.pp,
        &round.inboxes[0],
        &round.canonical,
        &round.commitments,
    );
    assert!(matches!(
        result,
        Err(ServerRoundError::BadOpening(ClientId(0)))
    ));
}

/// A canonical client with no commitment on the bulletin is a missing client,
/// the same as one with no opening in the inbox.
#[test]
fn missing_commitment_is_missing_client() {
    let mut rng = ChaCha20Rng::from_seed([5u8; 32]);
    let mut round = setup_round(&mut rng, 4, 4);
    round.commitments.pop();

    let result = run_server_round(
        &round.pp,
        &round.inboxes[0],
        &round.canonical,
        &round.commitments,
    );
    assert_eq!(
        result.err(),
        Some(ServerRoundError::MissingClient(ClientId(3)))
    );
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
    let server_keys: Vec<pke::PrivateKey> = (0..n_servers)
        .map(|_| pke::PrivateKey::generate(&mut rng))
        .collect();
    let servers: Vec<(ServerId, pke::PublicKey)> = server_ids
        .iter()
        .map(|&sid| (sid, server_keys[sid.0 as usize].public()))
        .collect();
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
        let round = run_client_round(&mut rng, &pp, &SESSION, cid, vec![m], &servers);
        client_entries.push((round.client_id, round.encrypted_message));
        for (idx, (sid, sealed)) in round.sealed_openings.into_iter().enumerate() {
            assert_eq!(sid, server_ids[idx]);
            let opening =
                unseal_opening(&server_keys[idx], &SESSION, cid, sid, &sealed).expect("unseal");
            inboxes[idx].items.push((cid, opening));
        }
    }
    let canonical = client_ids.clone();
    let commitments: Vec<(ClientId, Commitment)> = client_entries
        .iter()
        .map(|(cid, e)| (*cid, e.comm.clone()))
        .collect();
    let outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(&pp, inb, &canonical, &commitments).expect("missing client"))
        .collect();

    let expected = poly_mod_t(
        &messages
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
        let recovered = aggregate_and_decrypt_unverified(&pp, &canonical, &client_entries, &subset)
            .expect("verify failed");
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0], expected, "subset {:?}", chosen);
    }
}
