//! End-to-end RS-sharded ingress mode: no client ciphertext on the bulletin,
//! `k`-of-`n` lane sums reconstruct `Σ ct`, and every lane's post opens the
//! summed client-signed roots — so a lying lane is named, not merely detected.

use chipmunk_code::{DgtNTTPoly, HVCPoly, KahePoly};
use panetiere::bulletin::{RsClientBulletinEntry, RsNodeBulletinEntry, ServerBulletinEntry};
use panetiere::mse::{MseEncoding, MseParams};
use panetiere::prony::{PronyParams, PronySketch, PRONY_PRIME};
use panetiere::protocol::client::{run_client_round_rs, RsClientRound};
use panetiere::protocol::server::{
    run_rs_node_round, run_server_round, unseal_opening, RsNodeInbox, ServerInbox, ServerRoundError,
};
use panetiere::protocol::verify::{aggregate_and_decrypt_rs, VerifyError};
use panetiere::protocol::{ClientId, NodeId, ProtocolParams, ServerId, SessionId};
use panetiere::rs::Share;
use panetiere::share_commitment::SharePath;
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
        let server_keys: Vec<pke::PrivateKey> = (0..S)
            .map(|_| pke::PrivateKey::generate(&mut rng))
            .collect();
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
        Round {
            pp,
            canonical,
            entries,
            rounds,
            server_keys,
        }
    }

    /// The signed roots the lanes verify shares against.
    fn roots(&self) -> Vec<(ClientId, HVCPoly)> {
        self.entries
            .iter()
            .map(|(cid, e)| (*cid, e.share_root))
            .collect()
    }

    /// One lane's round, so a test can inspect the per-lane ingest verdict.
    fn node_round(
        &self,
        j: usize,
        subset: &[ClientId],
        roots: &[(ClientId, HVCPoly)],
    ) -> Result<RsNodeBulletinEntry, ServerRoundError> {
        let items: Vec<(ClientId, Share, SharePath)> = self
            .rounds
            .iter()
            .map(|r| {
                (
                    r.client_id,
                    r.rs_shares[j].clone(),
                    r.share_paths[j].clone(),
                )
            })
            .collect();
        let inbox = RsNodeInbox {
            node_id: NodeId(j as u32),
            items,
        };
        run_rs_node_round(self.pp.share_comm.as_ref().unwrap(), &inbox, subset, roots)
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
                let inbox = ServerInbox {
                    server_id: ServerId(j as u32),
                    items,
                };
                run_server_round(&inbox, subset).unwrap()
            })
            .collect();

        let roots = self.roots();
        let nodes = (0..N_NODES)
            .map(|j| self.node_round(j, subset, &roots).unwrap())
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

    let mut got = PronySketch::unpack(&params, &plain)
        .decode()
        .expect("prony decode");
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
        RsClientBulletinEntry::packed_len()
    );
    let back = RsClientBulletinEntry::from_bytes(&big.entries[0].1.to_bytes()).unwrap();
    assert_eq!(back.share_root, big.entries[0].1.share_root);
    assert_eq!(back.sig, big.entries[0].1.sig);
}

/// A lying lane is *named*, with or without spares — the property the flat
/// digest could not give: it detected a bad sum but attributed nothing.
#[test]
fn a_lying_lane_is_caught() {
    let (r, _, _) = mse_round(16, MULTI_BLOCK_XI);
    assert!(r.pp.kahe.mu_kahe > K);

    // A +1 digit bump stays inside β_agg, so it reaches the algebraic check
    // rather than tripping the norm gate.
    let bump = |n: &mut RsNodeBulletinEntry| {
        let d = &mut n.agg_open.data_mut()[0];
        let mut c = *d.coeffs();
        c[5] += 1;
        *d = HVCPoly::from_coeffs(c);
    };

    // With every lane reporting, only the liar fails to open the summed roots.
    {
        let (servers, mut nodes) = r.outputs(&r.canonical);
        bump(&mut nodes[3]);
        assert_eq!(
            aggregate_and_decrypt_rs(&r.pp, &SESSION, &r.canonical, &r.entries, &servers, &nodes,)
                .err(),
            Some(VerifyError::LaneOpeningFailed(vec![NodeId(3)]))
        );
    }

    // And at exactly k lanes, where no syndrome exists at all: still named.
    {
        let (servers, mut nodes) = r.outputs(&r.canonical);
        bump(&mut nodes[3]);
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
            Some(VerifyError::LaneOpeningFailed(vec![NodeId(3)]))
        );
    }

    // An out-of-range digit is caught by the norm gate, which is what stops a
    // lane from smuggling a different integer past an identical hash.
    {
        let (servers, mut nodes) = r.outputs(&r.canonical);
        let d = &mut nodes[3].agg_open.data_mut()[0];
        let mut c = *d.coeffs();
        c[5] -= chipmunk_code::HVC_MODULUS;
        *d = HVCPoly::from_coeffs(c);
        assert_eq!(
            aggregate_and_decrypt_rs(&r.pp, &SESSION, &r.canonical, &r.entries, &servers, &nodes,)
                .err(),
            Some(VerifyError::LaneOpeningFailed(vec![NodeId(3)]))
        );
    }

    // Dropping the liar recovers cleanly — the n−k redundancy doing its job.
    {
        let (servers, mut nodes) = r.outputs(&r.canonical);
        bump(&mut nodes[3]);
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
    let mut c = *entries[2].1.share_root.coeffs();
    c[0] += 1;
    entries[2].1.share_root = HVCPoly::from_coeffs(c);

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

/// A root that is not the commitment to the client's shares but *is* correctly
/// signed clears the signature check — and is then caught at ingest, per lane,
/// naming the client. The counterpart to `a_tampered_bulletin_post_is_rejected`,
/// which stops at `BadSignature` and never reaches a lane.
#[test]
fn a_signed_root_binds_the_shares() {
    let (mut r, _, _) = mse_round(16, 8);

    let culprit = 11usize;
    {
        let mut rng = ChaCha20Rng::from_seed([0x34; 32]);
        let e = &mut r.entries[culprit].1;
        let mut c = *e.share_root.coeffs();
        c[3] += 1;
        e.share_root = HVCPoly::from_coeffs(c);
        let sk = SigningKey::generate(&mut rng);
        e.sig = sk.sign(&RsClientBulletinEntry::signing_bytes(
            &SESSION,
            ClientId(culprit as u32),
            &e.comm,
            &e.share_root,
        ));
        e.pubkey = sk.verifying_key().to_sec1_bytes();
    }

    let roots = r.roots();
    for j in 0..N_NODES {
        assert_eq!(
            r.node_round(j, &r.canonical, &roots).err(),
            Some(ServerRoundError::BadShare(ClientId(culprit as u32))),
            "lane {j}"
        );
    }
}

/// The lane proves its aggregate with one hash and only re-checks per client
/// when that fails, so the two paths must agree on who is at fault — and an
/// over-capacity roster must not be blamed on a client.
#[test]
fn the_lane_pays_for_attribution_only_on_failure() {
    let (r, _, _) = mse_round(16, 8);
    let roots = r.roots();

    // Honest input: the aggregate opens, so no per-client hash is needed and
    // every lane posts.
    for j in 0..N_NODES {
        assert!(r.node_round(j, &r.canonical, &roots).is_ok(), "lane {j}");
    }

    // A root that no share opens: the fallback runs and names that client
    // rather than reporting a capacity fault.
    let mut roots_bad = roots.clone();
    let mut c = *roots_bad[4].1.coeffs();
    c[0] += 1;
    roots_bad[4].1 = HVCPoly::from_coeffs(c);
    assert_eq!(
        r.node_round(0, &r.canonical, &roots_bad).err(),
        Some(ServerRoundError::BadShare(ClientId(4)))
    );

    // ρ beyond what the digit bound admits: every client's share opens its own
    // root, so the lane reports capacity rather than inventing a culprit.
    let mut rng = ChaCha20Rng::from_seed([0x51; 32]);
    let tight = ProtocolParams::setup_rs_mode(
        &mut rng,
        S,
        20,
        K,
        N_NODES,
        T_MODULUS_DEFAULT,
        1,
        [0x42; 32],
    );
    let keys: Vec<pke::PrivateKey> = (0..S)
        .map(|_| pke::PrivateKey::generate(&mut rng))
        .collect();
    let servers: Vec<(ServerId, pke::PublicKey)> = (0..S)
        .map(|i| (ServerId(i as u32), keys[i].public()))
        .collect();
    let rounds: Vec<RsClientRound> = (0..4u32)
        .map(|i| {
            let sk = SigningKey::generate(&mut rng);
            let msg = vec![KahePoly::default(); tight.kahe.mu_kahe];
            run_client_round_rs(&mut rng, &tight, &SESSION, ClientId(i), msg, &servers, &sk)
        })
        .collect();
    let scp = tight.share_comm.as_ref().unwrap();
    let tight_roots: Vec<(ClientId, HVCPoly)> = rounds
        .iter()
        .map(|x| (x.client_id, x.bulletin.share_root))
        .collect();
    let canonical: Vec<ClientId> = rounds.iter().map(|x| x.client_id).collect();

    // One client alone is inside ρ_max = 1 and posts fine.
    let one = RsNodeInbox {
        node_id: NodeId(0),
        items: vec![(
            rounds[0].client_id,
            rounds[0].rs_shares[0].clone(),
            rounds[0].share_paths[0].clone(),
        )],
    };
    assert!(run_rs_node_round(scp, &one, &canonical[..1], &tight_roots).is_ok());

    let all = RsNodeInbox {
        node_id: NodeId(0),
        items: rounds
            .iter()
            .map(|x| {
                (
                    x.client_id,
                    x.rs_shares[0].clone(),
                    x.share_paths[0].clone(),
                )
            })
            .collect(),
    };
    assert_eq!(
        run_rs_node_round(scp, &all, &canonical, &tight_roots).err(),
        Some(ServerRoundError::AggregateOverCapacity)
    );
}

/// Ingest names the client, not the round: one bad share (or one bad path)
/// stops only the lane it was sent to.
#[test]
fn ingest_rejects_a_bad_share_naming_the_client() {
    // Multi-block, so every systematic lane holds real data rather than the
    // zero padding a payload shorter than `k` would leave it.
    let (mut r, _, _) = mse_round(16, MULTI_BLOCK_XI);
    let roots = r.roots();
    let (culprit, lane) = (7usize, 5usize);

    let good = r.rounds[culprit].rs_shares[lane].clone();
    let p = r.rounds[culprit].rs_shares[lane][0];
    assert_ne!(p, DgtNTTPoly::default(), "tamper must change the share");
    r.rounds[culprit].rs_shares[lane][0] = p + p;
    for j in 0..N_NODES {
        let got = r.node_round(j, &r.canonical, &roots);
        if j == lane {
            assert_eq!(
                got.err(),
                Some(ServerRoundError::BadShare(ClientId(culprit as u32)))
            );
        } else {
            assert!(got.is_ok(), "lane {j} should be unaffected");
        }
    }
    r.rounds[culprit].rs_shares[lane] = good;

    let mut nodes = r.rounds[culprit].share_paths[lane].nodes.to_vec();
    let mut c = *nodes[0].coeffs();
    c[0] += 1;
    nodes[0] = HVCPoly::from_coeffs(c);
    r.rounds[culprit].share_paths[lane].nodes = nodes.into_boxed_slice();
    assert_eq!(
        r.node_round(lane, &r.canonical, &roots).err(),
        Some(ServerRoundError::BadShare(ClientId(culprit as u32)))
    );
}
