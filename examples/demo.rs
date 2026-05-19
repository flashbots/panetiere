//! End-to-end demo with byte messages.
//!
//! Each client encodes its own short byte string into a single `KahePoly` slot.
//! Slots are non-overlapping (each client gets a unique window of coefficients
//! within the 1024-byte poly), so the protocol's recovered sum decodes back to
//! a buffer where each client's bytes occupy their assigned slot. This
//! demonstrates how the codec composes with the protocol when the application
//! picks a slot layout that doesn't overflow.

use flashnet::codec;
use flashnet::protocol::client::run_client_round;
use flashnet::protocol::message::{ClientId, ServerId};
use flashnet::protocol::server::{run_server_round, ServerInbox};
use flashnet::protocol::verify::aggregate_and_decrypt;
use flashnet::protocol::ProtocolParams;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn pack_slot(payload: &[u8], slot: usize, slot_size: usize, total_bytes: usize) -> Vec<u8> {
    assert!(payload.len() <= slot_size);
    assert!(slot * slot_size + slot_size <= total_bytes);
    let mut buf = vec![0u8; total_bytes];
    let start = slot * slot_size;
    buf[start..start + payload.len()].copy_from_slice(payload);
    buf
}

fn main() {
    let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
    let n_servers = 4;

    let messages: &[&[u8]] = &[
        b"hello from client 0",
        b"client 1 says hi",
        b"another from #2",
        b"#3: anonymous broadcast works",
        b"client 4 here",
        b"final note from 5",
    ];
    let n_clients = messages.len();

    const SLOT_SIZE: usize = 128;
    const TOTAL_BYTES: usize = 1024;
    assert!(n_clients * SLOT_SIZE <= TOTAL_BYTES);

    println!(
        "flashnet demo: {} clients × {}-byte slots, {} servers (t = ⌊γ/2⌋+1)",
        n_clients, SLOT_SIZE, n_servers
    );

    let pp = ProtocolParams::setup(&mut rng, n_servers);
    let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
    let client_ids: Vec<ClientId> = (0..n_clients as u32).map(ClientId).collect();

    let mut publics = vec![];
    let mut inboxes: Vec<ServerInbox> = server_ids
        .iter()
        .map(|&sid| ServerInbox {
            server_id: sid,
            items: vec![],
        })
        .collect();

    for (slot, &cid) in client_ids.iter().enumerate() {
        let buf = pack_slot(messages[slot], slot, SLOT_SIZE, TOTAL_BYTES);
        let polys = codec::encode_raw(&buf);
        assert_eq!(polys.len(), 1, "slot layout sized to fit one KahePoly");

        let round = run_client_round(&mut rng, &pp, cid, polys, &server_ids);
        publics.push((round.client_id, round.public));
        for (idx, (sid, ops)) in round.private.into_iter().enumerate() {
            assert_eq!(sid, server_ids[idx]);
            inboxes[idx].items.push((cid, ops));
        }
    }

    let canonical = client_ids;
    let outputs: Vec<_> = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).expect("missing client"))
        .collect();

    let recovered =
        aggregate_and_decrypt(&pp, &canonical, &publics, &outputs).expect("verify failed");
    let recovered_bytes =
        codec::decode_raw(&recovered).expect("decode of summed message failed");

    println!("\nrecovered slots:");
    for slot in 0..n_clients {
        let start = slot * SLOT_SIZE;
        let raw = &recovered_bytes[start..start + SLOT_SIZE];
        let len = raw.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
        let s = std::str::from_utf8(&raw[..len]).unwrap_or("<non-utf8>");
        println!("  client {}: {:?}", slot, s);
        assert_eq!(raw[..messages[slot].len()], *messages[slot]);
    }
    println!("\nok: all client messages recovered from summed ciphertext");
}
