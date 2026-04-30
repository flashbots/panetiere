//! IBLT carried through the flashnet protocol end-to-end.
//!
//! Each client builds an IBLT containing a unique chunk, packs it into HVCPolys,
//! and contributes those polys to the protocol. The verifier recovers the
//! per-poly summed message; concatenated, those polys unpack into the union
//! IBLT, which peels to recover all clients' chunks.

use chipmunk_code::HVCPoly;
use flashnet::cs::{Cs, HidingMerkleCommitment};
use flashnet::iblt::{IbltParams, IbltVector, IBLT_CHUNK_BYTES};
use flashnet::protocol::client::run_client_round;
use flashnet::protocol::message::{ClientId, ServerId};
use flashnet::protocol::server::{run_server_round, ServerInbox};
use flashnet::protocol::verify::{aggregate_and_decrypt, VerifyError};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn rand_chunk<R: Rng>(rng: &mut R) -> [u8; IBLT_CHUNK_BYTES] {
    let mut c = [0u8; IBLT_CHUNK_BYTES];
    rng.fill(&mut c[..]);
    c
}

#[test]
fn iblt_recovers_through_flashnet() {
    let mut rng = ChaCha20Rng::from_seed([0xABu8; 32]);
    let n_servers = 3;
    let n_clients = 4;

    let params = IbltParams {
        message_slots: 16,
        base_bits: 12,
    };
    assert!(params.max_clients() as usize >= n_clients);

    // Each client builds their own IBLT with one unique chunk.
    let chunks: Vec<[u8; IBLT_CHUNK_BYTES]> =
        (0..n_clients).map(|_| rand_chunk(&mut rng)).collect();
    let client_polys: Vec<Vec<HVCPoly>> = chunks
        .iter()
        .map(|c| {
            let mut iblt = IbltVector::new(params.clone());
            iblt.insert_chunk(*c);
            iblt.pack()
        })
        .collect();
    let n_polys = client_polys[0].len();
    assert_eq!(n_polys, IbltVector::n_polys(&params));

    let pp = HidingMerkleCommitment::setup(&mut rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n_clients as u32).map(ClientId).collect();
    let canonical = client_ids.clone();

    // Run flashnet once per poly index of the packed IBLT.
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
            let m = client_polys[i][k];
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
            .unwrap_or_else(|e| panic!("verify failed at poly {}: {:?}", k, e));
        recovered_polys.push(recovered);
    }

    let union = IbltVector::unpack(&params, &recovered_polys);
    let mut got = union.recover().expect("IBLT recover");
    got.sort();
    let mut expected = chunks.clone();
    expected.sort();
    assert_eq!(got, expected);
}

#[test]
fn iblt_protocol_rejects_tampered_share() {
    let mut rng = ChaCha20Rng::from_seed([0xCDu8; 32]);
    let n_servers = 3;
    let n_clients = 3;
    let params = IbltParams {
        message_slots: 8,
        base_bits: 12,
    };

    let chunks: Vec<[u8; IBLT_CHUNK_BYTES]> =
        (0..n_clients).map(|_| rand_chunk(&mut rng)).collect();
    let client_polys: Vec<Vec<HVCPoly>> = chunks
        .iter()
        .map(|c| {
            let mut iblt = IbltVector::new(params.clone());
            iblt.insert_chunk(*c);
            iblt.pack()
        })
        .collect();

    let pp = HidingMerkleCommitment::setup(&mut rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n_clients as u32).map(ClientId).collect();
    let canonical = client_ids.clone();

    // Run only the first poly; tamper one server's agg_share.
    let k = 0;
    let mut publics = Vec::new();
    let mut inboxes: Vec<ServerInbox> = server_ids
        .iter()
        .map(|&sid| ServerInbox {
            server_id: sid,
            items: vec![],
        })
        .collect();
    for (i, &cid) in client_ids.iter().enumerate() {
        let m = client_polys[i][k];
        let round = run_client_round(&mut rng, &pp, cid, m, &server_ids);
        publics.push((round.client_id, round.public));
        for (idx, (sid, op, sh)) in round.private.into_iter().enumerate() {
            assert_eq!(sid, server_ids[idx]);
            inboxes[idx].items.push((cid, op, sh));
        }
    }
    let mut outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).unwrap())
        .collect();
    use chipmunk_code::Polynomial;
    outputs[0].agg_share = outputs[0].agg_share + HVCPoly::rand_poly(&mut rng);
    let result = aggregate_and_decrypt(&pp, &canonical, &publics, &outputs);
    assert!(matches!(result, Err(VerifyError::ShareOpeningMismatch(_))));
}
