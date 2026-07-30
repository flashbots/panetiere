//! The recipient over a real round: anchor admission, culprit exclusion and the
//! rejection reasons it reports.

use chipmunk_code::{CsPoly, KahePoly, Polynomial, N};
use panetiere::bulletin::{ClientBulletinEntry, ServerBulletinEntry};
use panetiere::pke;
use panetiere::protocol::client::run_client_round;
use panetiere::protocol::recipient::{
    recover_direct, recover_once, RecipientError, RejectReason, SetPolicy,
};
use panetiere::protocol::server::{run_server_round, unseal_opening, ServerInbox};
use panetiere::protocol::verify::{aggregate_and_decrypt, VerifyError};
use panetiere::protocol::{ClientId, ProtocolParams, ServerId, SessionId};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const SESSION: SessionId = SessionId([0x71; 32]);
/// Above every set size the tests build, so `max_clients` never binds unless a
/// test sets it deliberately.
const NO_CAP: usize = 1024;

fn rand_message_poly<R: Rng>(rng: &mut R, t: u64) -> KahePoly {
    let half = t as i64 / 2;
    let mut coeffs = [0i64; N];
    for c in coeffs.iter_mut() {
        *c = rng.gen_range(0..t) as i64 - half;
    }
    KahePoly::from_coeffs(coeffs)
}

struct Round {
    pp: ProtocolParams,
    canonical: Vec<ClientId>,
    publics: Vec<(ClientId, ClientBulletinEntry)>,
    outputs: Vec<ServerBulletinEntry>,
}

impl Round {
    fn lookup(&self) -> impl Fn(ClientId) -> Option<ClientBulletinEntry> + '_ {
        |cid| {
            self.publics
                .iter()
                .find(|(c, _)| *c == cid)
                .map(|(_, e)| e.clone())
        }
    }
}

/// One honest round: `n_servers` servers all sharing over the same client set.
fn round(seed: u8, n_servers: usize, n_clients: usize) -> Round {
    let mut rng = ChaCha20Rng::from_seed([seed; 32]);
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

    let mut publics = Vec::new();
    let mut inboxes: Vec<ServerInbox> = server_ids
        .iter()
        .map(|&sid| ServerInbox {
            server_id: sid,
            items: vec![],
        })
        .collect();
    for &cid in &client_ids {
        let m = rand_message_poly(&mut rng, pp.kahe.t_modulus);
        let r = run_client_round(&mut rng, &pp, &SESSION, cid, vec![m], &servers);
        publics.push((r.client_id, r.encrypted_message));
        for (idx, (sid, sealed)) in r.sealed_openings.into_iter().enumerate() {
            let opening =
                unseal_opening(&server_keys[idx], &SESSION, cid, sid, &sealed).expect("unseal");
            inboxes[idx].items.push((cid, opening));
        }
    }
    let canonical = client_ids;
    let outputs = inboxes
        .iter()
        .map(|inb| run_server_round(inb, &canonical).expect("server round"))
        .collect();
    Round {
        pp,
        canonical,
        publics,
        outputs,
    }
}

#[test]
fn recover_once_matches_aggregate_and_decrypt() {
    let r = round(1, 4, 6);
    let want = aggregate_and_decrypt(&r.pp, &r.canonical, &r.publics, &r.outputs).unwrap();
    let got = recover_once(&r.pp, &r.canonical, &r.publics, &r.outputs).unwrap();
    assert_eq!(got, want);
}

/// The paper-faithful entry point names the faulty server and stops; it does not
/// retry without it.
#[test]
fn recover_once_names_the_faulty_server() {
    let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
    let mut r = round(2, 5, 4);
    r.outputs[1].agg_share = r.outputs[1].agg_share + CsPoly::rand_poly(&mut rng);
    assert_eq!(
        recover_once(&r.pp, &r.canonical, &r.publics, &r.outputs),
        Err(VerifyError::ShareOpeningMismatch(1))
    );
}

/// The wrapper drops exactly the corrupted servers and reports them, because
/// t-of-n decryption does not need them.
#[test]
fn recover_direct_excludes_culprits() {
    let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
    let mut r = round(3, 5, 4);
    assert_eq!(r.pp.shamir.t, 3);
    for i in [1usize, 3] {
        r.outputs[i].agg_share = r.outputs[i].agg_share + CsPoly::rand_poly(&mut rng);
    }
    let honest = recover_once(&r.pp, &r.canonical, &r.publics, &{
        let mut v = r.outputs.clone();
        v.retain(|sp| sp.server_id.0 != 1 && sp.server_id.0 != 3);
        v
    })
    .unwrap();

    let policy = SetPolicy::anchored(&r.canonical, 0, NO_CAP);
    let got = recover_direct(&r.pp, &policy, r.lookup(), &r.outputs).unwrap();
    assert_eq!(got.plaintext, honest);
    let mut culprits: Vec<u32> = got.culprits.iter().map(|s| s.0).collect();
    culprits.sort();
    assert_eq!(culprits, vec![1, 3]);
    assert_eq!(got.canonical, r.canonical);
}

#[test]
fn recover_direct_reports_exhausted_shares() {
    let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
    let mut r = round(4, 4, 4);
    // t = 3 of 4, so two bad shares leave too few.
    assert_eq!(r.pp.shamir.t, 3);
    for i in [0usize, 2] {
        r.outputs[i].agg_share = r.outputs[i].agg_share + CsPoly::rand_poly(&mut rng);
    }
    let policy = SetPolicy::anchored(&r.canonical, 0, NO_CAP);
    match recover_direct(&r.pp, &policy, r.lookup(), &r.outputs) {
        Err(RecipientError::Rejected(reason)) => assert!(
            matches!(reason, RejectReason::ExhaustedShares { need: 3, .. }),
            "got {reason:?}"
        ),
        other => panic!("expected Rejected, got {other:?}"),
    }
}

#[test]
fn below_share_threshold_is_its_own_error() {
    let r = round(5, 5, 4);
    let policy = SetPolicy::anchored(&r.canonical, 0, NO_CAP);
    match recover_direct(&r.pp, &policy, r.lookup(), &r.outputs[..2]) {
        Err(e) => assert_eq!(e, RecipientError::BelowShareThreshold { have: 2, need: 3 }),
        Ok(other) => panic!("expected BelowShareThreshold, got {other:?}"),
    }
}

/// Regression for the three-way canonical-bound disagreement: one number, and an
/// oversized announced set is refused rather than half-served.
#[test]
fn max_clients_rejects_an_oversized_anchor() {
    let r = round(6, 4, 6);
    let policy = SetPolicy::anchored(&r.canonical, 0, 4);
    assert_eq!(
        recover_direct(&r.pp, &policy, r.lookup(), &r.outputs).unwrap_err(),
        RecipientError::Rejected(RejectReason::AboveMaxClients { got: 6, max: 4 })
    );
}

#[test]
fn missing_client_public_is_rejected() {
    let mut r = round(7, 4, 5);
    r.publics.remove(2);
    let policy = SetPolicy::anchored(&r.canonical, 0, NO_CAP);
    assert_eq!(
        recover_direct(&r.pp, &policy, r.lookup(), &r.outputs).unwrap_err(),
        RecipientError::Rejected(RejectReason::MissingClientPublics { have: 4, need: 5 })
    );
}

/// Servers that shared over a different set than the anchor do not count towards
/// the threshold, however many of them there are.
#[test]
fn servers_disagreeing_with_the_anchor_do_not_count() {
    let r = round(9, 5, 5);
    assert_eq!(r.pp.shamir.t, 3);
    let subset: Vec<ClientId> = r.canonical[..4].to_vec();

    // Only servers 0,1 keep the anchored set; 2,3,4 claim the subset.
    let mut outputs = r.outputs.clone();
    for sp in outputs.iter_mut().filter(|sp| sp.server_id.0 >= 2) {
        sp.clients = subset.clone();
    }
    let policy = SetPolicy::anchored(&r.canonical, 0, NO_CAP);
    assert_eq!(
        recover_direct(&r.pp, &policy, r.lookup(), &outputs).unwrap_err(),
        RecipientError::Rejected(RejectReason::TooFewAgreeingServers {
            agreeing: 2,
            need: 3
        })
    );
}

#[test]
fn anonymity_floor_rejects_a_small_set() {
    let mut r = round(10, 4, 4);
    r.pp.min_clients = 5;
    let policy = SetPolicy::anchored(&r.canonical, 0, NO_CAP);
    assert_eq!(
        recover_direct(&r.pp, &policy, r.lookup(), &r.outputs).unwrap_err(),
        RecipientError::Rejected(RejectReason::BelowAnonymityFloor { got: 4, min: 5 })
    );
}
