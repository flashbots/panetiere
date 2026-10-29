//! The paper's lift from relaxed to full anonymous broadcast: two sequential
//! executions of the one-round protocol, each with fresh parameters and fresh
//! server keys.
//!
//! One execution delivers the *relaxed* guarantee: a corrupt client can see the
//! honest messages before choosing its own. Running it twice closes that gap.
//! In the first execution every client broadcasts a commitment to its message;
//! in the second it broadcasts the opening. A corrupt client is bound to its
//! choice after the first execution, when all it has seen is a list of
//! commitments, so its message can no longer depend on the honest ones.
//!
//! Six clients commit. Four open honestly, one goes silent in the second
//! execution, and one opens to a message it never committed to. The recipient
//! keeps exactly the four honestly opened messages. Two more clients send cover
//! traffic in both executions.

use std::collections::HashMap;

use panetiere::bulletin::{ClientBulletinEntry, InMemoryBulletin};
use panetiere::channel::{self, ChannelParams};
use panetiere::pke;
use panetiere::protocol::client::run_client_round;
use panetiere::protocol::recipient::{recover_unverified_direct, SetPolicy};
use panetiere::protocol::server::{run_server_round, unseal_opening, ServerInbox};
use panetiere::protocol::{ClientId, ServerId, SessionId};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

const N_SERVERS: usize = 4;
const N_COVER: usize = 2;
/// Anonymity floor: never decrypt over fewer clients than this.
const MIN_CLIENTS: usize = 4;
const MESSAGE_BYTES: usize = 32;
const NONCE_BYTES: usize = 16;
const COMMIT_BYTES: usize = 32;
/// An opening payload is `nonce ‖ one length byte ‖ message`.
const OPENING_BYTES: usize = NONCE_BYTES + 1 + MESSAGE_BYTES;

/// A hash commitment: binding under collision resistance, and both extractable
/// and equivocable in the random oracle model, which is what the paper's
/// composition argument asks of the commitment.
fn commit(nonce: &[u8; NONCE_BYTES], message: &[u8]) -> [u8; COMMIT_BYTES] {
    let mut h = Sha256::new();
    h.update(b"panetiere/two-run/commit/v1");
    h.update(nonce);
    h.update(message);
    h.finalize().into()
}

fn opening_payload(nonce: &[u8; NONCE_BYTES], message: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(OPENING_BYTES);
    out.extend_from_slice(nonce);
    out.push(message.len() as u8);
    out.extend_from_slice(message);
    out
}

fn parse_opening(payload: &[u8]) -> Option<([u8; NONCE_BYTES], &[u8])> {
    let nonce: [u8; NONCE_BYTES] = payload.get(..NONCE_BYTES)?.try_into().ok()?;
    let len = *payload.get(NONCE_BYTES)? as usize;
    let message = payload.get(NONCE_BYTES + 1..NONCE_BYTES + 1 + len)?;
    Some((nonce, message))
}

/// One execution of the relaxed protocol, with its own parameters, server keys
/// and session id. Each participating client carries a payload or cover.
/// Returns the decoded payloads, padded to the channel's symbol width.
fn execute(
    rng: &mut ChaCha20Rng,
    session: SessionId,
    message_bytes: usize,
    clients: &[(ClientId, Option<Vec<u8>>)],
) -> Vec<Vec<u8>> {
    let n_senders = clients.iter().filter(|(_, p)| p.is_some()).count();
    let ch = ChannelParams::for_messages(n_senders as u32, message_bytes, rng.gen());
    let mut pp = ch.protocol_params(rng, N_SERVERS);
    pp.min_clients = MIN_CLIENTS;

    // fresh server keys per execution: the paper's composition argument needs
    // the two executions to run under separate instances of the pki
    let server_keys: Vec<pke::PrivateKey> = (0..N_SERVERS)
        .map(|_| pke::PrivateKey::generate(rng))
        .collect();
    let servers: Vec<(ServerId, pke::PublicKey)> = server_keys
        .iter()
        .enumerate()
        .map(|(i, k)| (ServerId(i as u32), k.public()))
        .collect();

    let bulletin = InMemoryBulletin::new();
    let mut inboxes: Vec<ServerInbox> = servers
        .iter()
        .map(|(sid, _)| ServerInbox {
            server_id: *sid,
            items: vec![],
        })
        .collect();

    for (cid, payload) in clients {
        let contribution = match payload {
            Some(p) => channel::encode_message(rng, &ch, p).expect("payload fits"),
            None => channel::cover(&ch),
        };
        let round = run_client_round(rng, &pp, &session, *cid, contribution, &servers);
        let wire = round.encrypted_message.to_bytes();
        let post = ClientBulletinEntry::from_bytes(&wire).expect("wire round-trip");
        bulletin.publish_client(*cid, post);
        for (idx, (sid, sealed)) in round.sealed_openings.iter().enumerate() {
            let share =
                unseal_opening(&server_keys[idx], &session, *cid, *sid, sealed).expect("unseal");
            inboxes[idx].items.push((*cid, share));
        }
    }

    let canonical: Vec<ClientId> = clients.iter().map(|(cid, _)| *cid).collect();
    bulletin.publish_canonical(canonical.clone());
    for inbox in &inboxes {
        let entry = run_server_round(inbox, &canonical).expect("canonical client missing");
        bulletin.publish_server(entry);
    }

    let posts: HashMap<ClientId, ClientBulletinEntry> = bulletin.clients().into_iter().collect();
    let policy = SetPolicy::anchored(&canonical, MIN_CLIENTS, clients.len());
    let recovered = recover_unverified_direct(
        &pp,
        &policy,
        |cid| posts.get(&cid).cloned(),
        &bulletin.servers(),
    )
    .expect("recover");
    channel::decode_messages(&ch, &recovered.plaintext, Some(n_senders)).expect("peel")
}

fn main() {
    let mut rng = ChaCha20Rng::from_seed([0u8; 32]);

    let messages: &[&[u8]] = &[
        b"hello from client 0",
        b"client 1 says hi",
        b"another from #2",
        b"#3: anonymous broadcast",
        b"client 4 commits then vanishes",
        b"client 5 commits to this",
    ];
    let nonces: Vec<[u8; NONCE_BYTES]> = messages.iter().map(|_| rng.gen()).collect();
    let n_clients = messages.len() + N_COVER;

    println!(
        "Panetière two-run demo: {} clients ({} committing + {} cover), {} servers",
        n_clients,
        messages.len(),
        N_COVER,
        N_SERVERS,
    );

    // execution one: every client broadcasts a commitment, cover clients send cover
    let first: Vec<(ClientId, Option<Vec<u8>>)> = (0..n_clients)
        .map(|i| {
            let payload = messages.get(i).map(|m| commit(&nonces[i], m).to_vec());
            (ClientId(i as u32), payload)
        })
        .collect();
    let commitments = execute(&mut rng, SessionId([0xA1; 32]), COMMIT_BYTES, &first);
    println!("\nexecution 1: {} commitments broadcast", commitments.len());

    // execution two: openings. client 4 never shows up, and client 5 opens to a
    // message it did not commit to.
    let second: Vec<(ClientId, Option<Vec<u8>>)> = (0..n_clients)
        .filter(|&i| i != 4)
        .map(|i| {
            let payload = match i {
                5 => Some(opening_payload(
                    &nonces[5],
                    b"client 5 opens to this instead",
                )),
                _ => messages.get(i).map(|m| opening_payload(&nonces[i], m)),
            };
            (ClientId(i as u32), payload)
        })
        .collect();
    let openings = execute(&mut rng, SessionId([0xA2; 32]), OPENING_BYTES, &second);
    println!("execution 2: {} openings broadcast", openings.len());

    // the recipient keeps an opening only if it opens one of the commitments
    // from execution one, each commitment at most once
    let mut unopened: HashMap<[u8; COMMIT_BYTES], usize> = HashMap::new();
    for c in &commitments {
        let c: [u8; COMMIT_BYTES] = c[..COMMIT_BYTES].try_into().expect("commitment width");
        *unopened.entry(c).or_default() += 1;
    }
    let mut output: Vec<Vec<u8>> = Vec::new();
    let mut dropped = 0usize;
    for payload in &openings {
        let kept = parse_opening(payload).and_then(|(nonce, message)| {
            let left = unopened.get_mut(&commit(&nonce, message))?;
            *left = left.checked_sub(1)?;
            Some(message.to_vec())
        });
        match kept {
            Some(m) => output.push(m),
            None => dropped += 1,
        }
    }
    let never_opened: usize = unopened.values().sum();

    println!(
        "\nrecipient: {} messages, {} openings without a commitment, {} commitments never opened",
        output.len(),
        dropped,
        never_opened,
    );
    for m in &output {
        println!("  {:?}", std::str::from_utf8(m).unwrap_or("<non-utf8>"));
    }

    let mut got = output.clone();
    let mut want: Vec<Vec<u8>> = messages[..4].iter().map(|m| m.to_vec()).collect();
    got.sort();
    want.sort();
    assert_eq!(got, want);
    assert_eq!(dropped, 1);
    assert_eq!(never_opened, 2);
    println!("\nok: only messages committed before anyone saw a message came through");
}
