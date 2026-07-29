//! End-to-end demo over the path the README recommends: a `channel` payload
//! layer, one client round each, the servers' aggregate, and the recipient's
//! canonical-set policy.
//!
//! Eight clients take part; six carry a message and two send cover traffic, so
//! six elements come back out of an anonymity set of eight. Recovery order
//! follows the encoding, not the client order — that is the point.

use std::collections::HashMap;

use panetiere::bulletin::{ClientBulletinEntry, InMemoryBulletin};
use panetiere::channel::{self, ChannelParams};
use panetiere::pke;
use panetiere::protocol::client::run_client_round;
use panetiere::protocol::recipient::{recover_direct, SetPolicy};
use panetiere::protocol::server::{run_server_round, unseal_opening, ServerInbox};
use panetiere::protocol::{ClientId, ServerId, SessionId};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const N_SERVERS: usize = 4;
const N_COVER: usize = 2;
/// Anonymity floor: never decrypt over fewer clients than this.
const MIN_CLIENTS: usize = 4;
const MESSAGE_BYTES: usize = 32;

/// Trim the zero padding `channel` leaves in place — framing is the caller's.
fn unpad(payload: &[u8]) -> &[u8] {
    let len = payload.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
    &payload[..len]
}

fn main() {
    let mut rng = ChaCha20Rng::from_seed([0u8; 32]);

    let messages: &[&[u8]] = &[
        b"hello from client 0",
        b"client 1 says hi",
        b"another from #2",
        b"#3: anonymous broadcast",
        b"client 4 here",
        b"final note from 5",
    ];
    let n_clients = messages.len() + N_COVER;

    // The payload layer comes first: it sizes the multiset encoding for "this
    // many messages of this many bytes", and that fixes the KAHE width every
    // client must contribute.
    let ch = ChannelParams::for_messages(messages.len() as u32, MESSAGE_BYTES, [0x5C; 32]);
    let mut pp = ch.protocol_params(&mut rng, N_SERVERS);
    pp.min_clients = MIN_CLIENTS;

    println!(
        "Panetière demo: {} clients ({} messages + {} cover), {} servers, t = {}",
        n_clients,
        messages.len(),
        N_COVER,
        N_SERVERS,
        pp.shamir.t,
    );
    println!(
        "  payload width {} KahePoly, up to {} bytes per message",
        ch.n_polys(),
        ch.max_payload_bytes(),
    );

    // Long-lived server keys, plus a session id unique to this execution.
    let server_keys: Vec<pke::PrivateKey> = (0..N_SERVERS)
        .map(|_| pke::PrivateKey::generate(&mut rng))
        .collect();
    let servers: Vec<(ServerId, pke::PublicKey)> = server_keys
        .iter()
        .enumerate()
        .map(|(i, k)| (ServerId(i as u32), k.public()))
        .collect();
    let session = SessionId([0xD1; 32]);

    let bulletin = InMemoryBulletin::new();
    let mut inboxes: Vec<ServerInbox> = servers
        .iter()
        .map(|(sid, _)| ServerInbox {
            server_id: *sid,
            items: vec![],
        })
        .collect();

    for i in 0..n_clients {
        let cid = ClientId(i as u32);
        let contribution = match messages.get(i) {
            Some(m) => channel::encode_message(&mut rng, &ch, m).expect("payload fits"),
            None => channel::cover(&ch),
        };
        let round = run_client_round(&mut rng, &pp, &session, cid, contribution, &servers);

        // The public half goes on the bulletin as bytes — a transport carries
        // the wire form, not the in-memory type.
        let wire = round.encrypted_message.to_bytes();
        let post = ClientBulletinEntry::from_bytes(&wire).expect("wire round-trip");
        bulletin.publish_client(cid, post);

        // The private half is one sealed envelope per server, each holding that
        // server's Shamir share of this client's KAHE key.
        for (idx, (sid, sealed)) in round.sealed_openings.iter().enumerate() {
            let share = unseal_opening(&server_keys[idx], &session, cid, *sid, sealed)
                .expect("unseal");
            inboxes[idx].items.push((cid, share));
        }
    }

    // Whoever fixes the canonical set announces it; here everyone took part.
    let canonical: Vec<ClientId> = (0..n_clients as u32).map(ClientId).collect();
    bulletin.publish_canonical(canonical.clone());

    for inbox in &inboxes {
        let entry = run_server_round(inbox, &canonical).expect("canonical client missing");
        bulletin.publish_server(entry);
    }

    // The recipient holds nothing secret: it decides the set, drops servers
    // whose shares fail their opening, then decrypts.
    let posts: HashMap<ClientId, ClientBulletinEntry> = bulletin.clients().into_iter().collect();
    let policy = SetPolicy::anchored(&canonical, MIN_CLIENTS, n_clients);
    let recovered = recover_direct(
        &pp,
        &policy,
        |cid| posts.get(&cid).cloned(),
        &bulletin.servers(),
    )
    .expect("recover");

    let payloads = channel::decode_messages(&ch, &recovered.plaintext, Some(messages.len()))
        .expect("peel");

    println!(
        "\nanonymity set {}, servers excluded {:?}",
        recovered.canonical.len(),
        recovered.culprits,
    );
    println!(
        "recovered {} messages, in encoding order — it says nothing about who sent what:",
        payloads.len(),
    );
    for p in &payloads {
        println!("  {:?}", std::str::from_utf8(unpad(p)).unwrap_or("<non-utf8>"));
    }

    let mut got: Vec<&[u8]> = payloads.iter().map(|p| unpad(p)).collect();
    let mut want: Vec<&[u8]> = messages.to_vec();
    got.sort();
    want.sort();
    assert_eq!(got, want);
    println!("\nok: every message recovered from the summed ciphertext");
}
