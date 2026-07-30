//! End-to-end RS-sharded ingress mode: no client ciphertext on the bulletin,
//! `k`-of-`n` lane share-sums reconstruct `Σ ct`, and the homomorphic digest
//! gates the result.

use chipmunk_code::{DgtNTTPoly, KahePoly};
use panetiere::bulletin::{RsClientBulletinEntry, RsNodeBulletinEntry, ServerBulletinEntry};
use panetiere::mse::{MseEncoding, MseParams};
use panetiere::prony::{PronyParams, PronySketch, PRONY_PRIME};
use panetiere::protocol::client::{run_client_round_rs, RsClientRound};
use panetiere::protocol::server::{
    run_rs_node_round, run_server_round, unseal_opening, RsNodeInbox, ServerInbox,
};
use panetiere::protocol::verify::{aggregate_and_decrypt_rs, VerifyError};
use panetiere::protocol::{ClientId, NodeId, ProtocolParams, ServerId, SessionId};
use panetiere::digest::DIGEST_POLYS;
use panetiere::rs::Share;
use panetiere::sig::SigningKey;
use panetiere::{kahe::T_MODULUS_DEFAULT, pke};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const SESSION: SessionId = SessionId([0x77; 32]);
const S: usize = 16;
const K: usize = 14;
const N_NODES: usize = 16;
/// Aggregation ceiling the digest ring is sized for.
const RHO_MAX: usize = 300;

/// One live round: every party, wired up.
struct Round {
    pp: ProtocolParams,
    canonical: Vec<ClientId>,
    entries: Vec<(ClientId, RsClientBulletinEntry)>,
    rounds: Vec<RsClientRound>,
    server_keys: Vec<pke::PrivateKey>,
}

impl Round {
    fn build(rho: usize, payloads: Vec<Vec<KahePoly>>, payload_polys: usize, t: u64) -> Self {
        let mut rng = ChaCha20Rng::from_seed([0x21; 32]);
        let pp = ProtocolParams::setup_rs_mode(
            &mut rng,
            S,
            payload_polys,
            K,
            N_NODES,
            t,
            RHO_MAX,
            [0x42; 32],
        );
        let server_keys: Vec<pke::PrivateKey> =
            (0..S).map(|_| pke::PrivateKey::generate(&mut rng)).collect();
        let servers: Vec<(ServerId, pke::PublicKey)> = (0..S)
            .map(|i| (ServerId(i as u32), server_keys[i].public()))
            .collect();

        let mut rounds = Vec::with_capacity(rho);
        let mut entries = Vec::with_capacity(rho);
        for (i, m) in payloads.into_iter().enumerate() {
            let cid = ClientId(i as u32);
            let sk = SigningKey::generate(&mut rng);
            let r = run_client_round_rs(&mut rng, &pp, &SESSION, cid, m, &servers, &sk);
            entries.push((cid, r.bulletin.clone()));
            rounds.push(r);
        }
        let canonical: Vec<ClientId> = (0..rho as u32).map(ClientId).collect();
        Round { pp, canonical, entries, rounds, server_keys }
    }

    /// Servers and lanes over an arbitrary subset, exercising the all-or-nothing
    /// canonical-set path.
    fn outputs(&self, subset: &[ClientId]) -> (Vec<ServerBulletinEntry>, Vec<RsNodeBulletinEntry>) {
        let servers = (0..S)
            .map(|j| {
                let items = self
                    .rounds
                    .iter()
                    .map(|r| {
                        let (_, sealed) = &r.sealed_openings[j];
                        let op = unseal_opening(
                            &self.server_keys[j],
                            &SESSION,
                            r.client_id,
                            ServerId(j as u32),
                            sealed,
                        )
                        .unwrap();
                        (r.client_id, op)
                    })
                    .collect();
                let inbox = ServerInbox { server_id: ServerId(j as u32), items };
                run_server_round(&inbox, subset).unwrap()
            })
            .collect();

        let nodes = (0..N_NODES)
            .map(|j| {
                let items: Vec<(ClientId, Share)> = self
                    .rounds
                    .iter()
                    .map(|r| (r.client_id, r.rs_shares[j].clone()))
                    .collect();
                let inbox = RsNodeInbox { node_id: NodeId(j as u32), items };
                run_rs_node_round(&inbox, subset).unwrap()
            })
            .collect();

        (servers, nodes)
    }
}

fn mse_round(rho: usize, xi: usize) -> (Round, MseParams, Vec<Vec<i64>>) {
    let mut rng = ChaCha20Rng::from_seed([0x31; 32]);
    let params = MseParams::new(4, (3 * rho).div_ceil(4), xi, [0xAA; 32]);
    let payload_polys = MseEncoding::n_polys(&params);
    let elements: Vec<Vec<i64>> = (0..rho)
        .map(|i| (0..xi).map(|j| (i + j + 1) as i64).collect())
        .collect();
    let payloads: Vec<Vec<KahePoly>> = elements
        .iter()
        .map(|e| {
            let mut enc = MseEncoding::new(params.clone());
            enc.insert(&mut rng, e);
            let mut p = enc.pack();
            p.resize(payload_polys, KahePoly::default());
            p
        })
        .collect();
    (
        Round::build(rho, payloads, payload_polys, T_MODULUS_DEFAULT),
        params,
        elements,
    )
}

/// ξ chosen so the payload spans more polys than `K` — otherwise every block
/// past the first is zero padding and the coding is not exercised at all.
const MULTI_BLOCK_XI: usize = 640;

#[test]
fn mse_round_trip_through_rs_lanes() {
    let (r, params, elements) = mse_round(24, MULTI_BLOCK_XI);
    assert!(r.pp.kahe.mu_kahe > K);
    let (servers, nodes) = r.outputs(&r.canonical);

    let (plain, tt) =
        aggregate_and_decrypt_rs(&r.pp, &SESSION, &r.canonical, &r.entries, &servers, &nodes)
            .expect("rs verify");
    assert!(tt.reconstruct_us > 0.0);

    let mut got = MseEncoding::unpack(&params, &plain).decode().expect("peel");
    got.sort();
    let mut want = elements.clone();
    want.sort();
    assert_eq!(got, want);
}

#[test]
fn prony_round_trip_through_rs_lanes() {
    let rho = 20;
    let xi = 1400;
    let mut rng = ChaCha20Rng::from_seed([0x32; 32]);
    let params = PronyParams::new(rho, xi);
    let payload_polys = PronySketch::n_polys(&params);
    assert!(payload_polys > K);
    let elements: Vec<Vec<i64>> = (0..rho)
        .map(|i| (0..xi).map(|j| (i * 7 + j + 1) as i64).collect())
        .collect();
    let payloads: Vec<Vec<KahePoly>> = elements
        .iter()
        .map(|e| {
            let mut sk = PronySketch::new(params.clone());
            sk.insert(&mut rng, e);
            let mut p = sk.pack();
            p.resize(payload_polys, KahePoly::default());
            p
        })
        .collect();

    let r = Round::build(rho, payloads, payload_polys, PRONY_PRIME);
    let (servers, nodes) = r.outputs(&r.canonical);
    let (plain, _) =
        aggregate_and_decrypt_rs(&r.pp, &SESSION, &r.canonical, &r.entries, &servers, &nodes)
            .expect("rs verify");

    let mut got = PronySketch::unpack(&params, &plain).decode().expect("prony decode");
    got.sort();
    let mut want = elements.clone();
    want.sort();
    assert_eq!(got, want);
}

/// Any `k` of the `n` lanes suffice, including an all-parity selection.
#[test]
fn recovers_from_any_k_lanes() {
    let (r, params, elements) = mse_round(16, MULTI_BLOCK_XI);
    let (servers, nodes) = r.outputs(&r.canonical);

    for start in 0..=(N_NODES - K) {
        let subset: Vec<RsNodeBulletinEntry> = nodes[start..start + K].to_vec();
        let (plain, _) =
            aggregate_and_decrypt_rs(&r.pp, &SESSION, &r.canonical, &r.entries, &servers, &subset)
                .unwrap_or_else(|e| panic!("start={start}: {e:?}"));
        let mut got = MseEncoding::unpack(&params, &plain).decode().expect("peel");
        got.sort();
        let mut want = elements.clone();
        want.sort();
        assert_eq!(got, want, "start={start}");
    }
}

/// The bulletin post does not grow with the message — the whole point.
#[test]
fn bulletin_post_is_constant_size() {
    let (small, _, _) = mse_round(16, 8);
    let (big, _, _) = mse_round(16, MULTI_BLOCK_XI);
    assert!(big.pp.kahe.mu_kahe > small.pp.kahe.mu_kahe);
    assert_eq!(
        small.entries[0].1.to_bytes().len(),
        big.entries[0].1.to_bytes().len()
    );
    assert_eq!(
        small.entries[0].1.to_bytes().len(),
        RsClientBulletinEntry::packed_len(DIGEST_POLYS)
    );
    let back = RsClientBulletinEntry::from_bytes(&big.entries[0].1.to_bytes(), DIGEST_POLYS).unwrap();
    assert_eq!(back.digest, big.entries[0].1.digest);
    assert_eq!(back.sig, big.entries[0].1.sig);
}

/// A lying lane is caught three ways, and which one fires says something.
#[test]
fn a_lying_lane_is_caught() {
    let (r, _, _) = mse_round(16, MULTI_BLOCK_XI);
    assert!(r.pp.kahe.mu_kahe > K);

    // (a) Perturbing an NTT slot spreads densely over the coefficient domain, so
    // the reconstruction leaves the honest ρ·q_kahe/2 range and the norm check
    // rejects it. Shown at exactly k lanes, because with spares present the
    // syndrome is cheaper and fires first.
    {
        let (servers, mut nodes) = r.outputs(&r.canonical);
        let mut c = *nodes[3].share_sum[0].coeffs();
        c[5] = (c[5] + 1) % chipmunk_code::DGT_MODULUS;
        nodes[3].share_sum[0] = DgtNTTPoly::from_raw(&c);
        let exactly_k: Vec<RsNodeBulletinEntry> = nodes[..K].to_vec();
        assert_eq!(
            aggregate_and_decrypt_rs(
                &r.pp,
                &SESSION,
                &r.canonical,
                &r.entries,
                &servers,
                &exactly_k
            )
            .err(),
            Some(VerifyError::CiphertextOutOfRange)
        );
    }

    // (b) A single coefficient off by one stays inside the range, so it reaches
    // the algebraic checks. Adding the embedding of a unit KahePoly is exactly a
    // +1 coefficient-domain bump, since embedding is coefficient-wise and the
    // NTT is linear.
    let mut unit = [0i64; chipmunk_code::N];
    unit[5] = 1;
    let bump = DgtNTTPoly::from_kahe(&KahePoly::from_coeffs(unit));

    // With the n−k spares present the syndrome fires first. It names the shares
    // that contradict the reconstruction — the spares, not the liar; identifying
    // the liar among the k used would need Berlekamp–Welch.
    {
        let (servers, mut nodes) = r.outputs(&r.canonical);
        nodes[3].share_sum[0] = nodes[3].share_sum[0] + bump;
        match aggregate_and_decrypt_rs(
            &r.pp, &SESSION, &r.canonical, &r.entries, &servers, &nodes,
        ) {
            Err(VerifyError::LaneMismatch(bad)) => assert!(!bad.is_empty()),
            other => panic!("expected LaneMismatch, got {other:?}"),
        }
    }

    // (c) With exactly k lanes there is no syndrome left, and the Ajtai digest
    // catches it — collision-resistantly, not probabilistically.
    {
        let (servers, mut nodes) = r.outputs(&r.canonical);
        nodes[3].share_sum[0] = nodes[3].share_sum[0] + bump;
        let exactly_k: Vec<RsNodeBulletinEntry> = nodes[..K].to_vec();
        assert_eq!(
            aggregate_and_decrypt_rs(
                &r.pp,
                &SESSION,
                &r.canonical,
                &r.entries,
                &servers,
                &exactly_k
            )
            .err(),
            Some(VerifyError::DigestMismatch)
        );
    }

    // Dropping the liar recovers cleanly — the n−k redundancy doing its job.
    {
        let (servers, mut nodes) = r.outputs(&r.canonical);
        nodes[3].share_sum[0] = nodes[3].share_sum[0] + bump;
        let clean: Vec<RsNodeBulletinEntry> = nodes
            .iter()
            .filter(|n| n.node_id != NodeId(3))
            .cloned()
            .collect();
        assert!(aggregate_and_decrypt_rs(
            &r.pp,
            &SESSION,
            &r.canonical,
            &r.entries,
            &servers,
            &clean
        )
        .is_ok());
    }
}

#[test]
fn a_tampered_bulletin_post_is_rejected() {
    let (r, _, _) = mse_round(16, 8);
    let (servers, nodes) = r.outputs(&r.canonical);

    let mut entries = r.entries.clone();
    let mut c = *entries[2].1.digest[0].coeffs();
    c[0] = (c[0] + 1) % chipmunk_code::DGT_MODULUS;
    entries[2].1.digest[0] = DgtNTTPoly::from_raw(&c);

    assert_eq!(
        aggregate_and_decrypt_rs(&r.pp, &SESSION, &r.canonical, &entries, &servers, &nodes).err(),
        Some(VerifyError::BadSignature(ClientId(2)))
    );
}

#[test]
fn roster_mismatch_between_lanes_and_servers_is_rejected() {
    let (r, _, _) = mse_round(16, 8);
    let (servers, _) = r.outputs(&r.canonical);
    let short = &r.canonical[..15];
    let (_, nodes_short) = r.outputs(short);

    assert_eq!(
        aggregate_and_decrypt_rs(
            &r.pp,
            &SESSION,
            &r.canonical,
            &r.entries,
            &servers,
            &nodes_short
        )
        .err(),
        Some(VerifyError::InconsistentNodeCanonical(NodeId(0)))
    );
}

/// A digest that is not the hash of its ciphertext but *is* correctly signed
/// clears the signature check, so only the aggregate digest check can catch it.
/// The counterpart to `a_tampered_bulletin_post_is_rejected`, which stops at
/// `BadSignature` and never reaches the digest.
#[test]
fn a_signed_but_inconsistent_digest_is_rejected() {
    let (mut r, _, _) = mse_round(16, 8);

    let culprit = 11usize;
    {
        let mut rng = ChaCha20Rng::from_seed([0x34; 32]);
        let e = &mut r.entries[culprit].1;
        let mut c = *e.digest[0].coeffs();
        c[3] = (c[3] + 1) % chipmunk_code::DGT_MODULUS;
        e.digest[0] = DgtNTTPoly::from_raw(&c);
        let sk = SigningKey::generate(&mut rng);
        e.sig = sk.sign(&RsClientBulletinEntry::signing_bytes(
            &SESSION,
            ClientId(culprit as u32),
            &e.comm,
            &e.digest,
        ));
        e.pubkey = sk.verifying_key().to_sec1_bytes();
    }

    let (servers, nodes) = r.outputs(&r.canonical);
    assert_eq!(
        aggregate_and_decrypt_rs(&r.pp, &SESSION, &r.canonical, &r.entries, &servers, &nodes).err(),
        Some(VerifyError::DigestMismatch)
    );
}
