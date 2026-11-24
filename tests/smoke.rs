//! One round through the public API only, with everything that crosses the
//! wire sent as bytes. A missing public item fails to compile here, and a
//! round that compiles but does not recover fails the assertions.

use std::collections::HashMap;

use panetiere::bulletin::{ClientBulletinEntry, ServerBulletinEntry};
use panetiere::channel::{self, ChannelParams};
use panetiere::pke;
use panetiere::protocol::client::run_client_round;
use panetiere::protocol::recipient::{recover_unverified_direct, SetPolicy};
use panetiere::protocol::server::{run_server_round, unseal_opening, ServerInbox};
use panetiere::protocol::{round_wire_sizes, ClientId, Commitment, ServerId, SessionId};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const N_SERVERS: usize = 4;
const N_COVER: usize = 1;
const MESSAGE_BYTES: usize = 16;

#[test]
fn round_over_the_wire() {
    let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
    let messages: &[&[u8]] = &[b"first", b"second", b"third"];
    let n_clients = messages.len() + N_COVER;

    let ch = ChannelParams::for_messages(&mut rng, messages.len() as u32, MESSAGE_BYTES);
    let pp = ch
        .protocol_params(&mut rng, N_SERVERS)
        .with_min_clients(n_clients);
    let sizes = round_wire_sizes(pp.n_servers(), pp.message_polys(), n_clients as u32);
    let session = SessionId::new([0x5A; 32]);

    // servers publish their keys as bytes and keep the private half as bytes.
    let server_keys: Vec<Vec<u8>> = (0..N_SERVERS)
        .map(|_| pke::PrivateKey::generate(&mut rng).to_bytes())
        .collect();
    let servers: Vec<(ServerId, pke::PublicKey)> = server_keys
        .iter()
        .enumerate()
        .map(|(i, k)| {
            let public = pke::PrivateKey::try_from_bytes(k)
                .expect("private key round-trips")
                .public()
                .to_bytes();
            let public = pke::PublicKey::try_from_bytes(&public).expect("public key round-trips");
            (ServerId::new(i as u32), public)
        })
        .collect();

    let mut posts: HashMap<ClientId, Vec<u8>> = HashMap::new();
    let mut envelopes: Vec<Vec<(ClientId, Vec<u8>)>> = vec![Vec::new(); N_SERVERS];
    for i in 0..n_clients {
        let cid = ClientId::new(i as u32);
        let contribution = match messages.get(i) {
            Some(m) => channel::encode_message(&mut rng, &ch, &session, m).expect("message fits"),
            None => channel::cover(&ch),
        };
        let (post, sealed) =
            run_client_round(&mut rng, &pp, &session, cid, contribution, &servers).into_parts();
        let post = post.to_bytes();
        assert_eq!(post.len(), sizes.client_post());
        posts.insert(cid, post);
        for (sid, envelope) in sealed {
            envelopes[sid.get() as usize].push((cid, envelope));
        }
    }

    let canonical: Vec<ClientId> = (0..n_clients as u32).map(ClientId::new).collect();
    let decode_post =
        |bytes: &[u8]| ClientBulletinEntry::try_from_bytes(bytes).expect("client post parses");
    let commitments: Vec<(ClientId, Commitment)> = posts
        .iter()
        .map(|(cid, bytes)| (*cid, decode_post(bytes).commitment().clone()))
        .collect();

    let mut server_entries = Vec::new();
    for (idx, (sid, _)) in servers.iter().enumerate() {
        let key = pke::PrivateKey::try_from_bytes(&server_keys[idx]).expect("key parses");
        let items = envelopes[idx]
            .iter()
            .map(|(cid, envelope)| {
                let opening =
                    unseal_opening(&key, &session, *cid, *sid, envelope).expect("envelope opens");
                (*cid, opening)
            })
            .collect();
        let entry = run_server_round(
            &pp,
            &ServerInbox::new(*sid, items),
            &canonical,
            &commitments,
        )
        .expect("every canonical client posted");
        let bytes = entry.to_bytes(&pp);
        assert_eq!(bytes.len(), 8 + 4 * n_clients + sizes.server_entry());
        server_entries.push(bytes);
    }

    let server_entries: Vec<ServerBulletinEntry> = server_entries
        .iter()
        .map(|b| ServerBulletinEntry::try_from_bytes(b).expect("server entry parses"))
        .collect();
    let policy = SetPolicy::anchored(&canonical, n_clients, n_clients);
    let recovered = recover_unverified_direct(
        &pp,
        &policy,
        |cid| posts.get(&cid).map(|b| decode_post(b)),
        &server_entries,
    )
    .expect("round recovers");
    assert_eq!(recovered.canonical(), canonical.as_slice());
    assert!(recovered.culprits().is_empty());

    let payloads =
        channel::decode_messages(&ch, &session, recovered.plaintext(), Some(messages.len()))
            .expect("messages decode");
    let mut want: Vec<&[u8]> = messages.to_vec();
    want.sort();
    let got: Vec<&[u8]> = payloads.iter().map(Vec::as_slice).collect();
    assert_eq!(got, want);
}
