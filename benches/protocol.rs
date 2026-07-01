use chipmunk_code::{KahePoly, Polynomial};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use panetiere::cs::{Cs, HidingMerkleCommitment};
use panetiere::kahe::{Kahe, KaheScheme};
use panetiere::protocol::aggregator::run_aggregator_round;
use panetiere::protocol::client::run_client_round;
use panetiere::protocol::{ClientId, ServerId};
use panetiere::protocol::server::{run_server_round, ServerInbox};
use panetiere::protocol::verify::{aggregate_and_decrypt, decrypt_aggregate};
use panetiere::protocol::ProtocolParams;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn bench_client_round(c: &mut Criterion) {
    let mut g = c.benchmark_group("client_round");
    for &n_servers in &[2usize, 4, 8, 16] {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        let pp = ProtocolParams::setup(&mut rng, n_servers);
        let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
        let m = vec![KahePoly::rand_poly(&mut rng); pp.kahe.mu_kahe];
        g.bench_with_input(BenchmarkId::from_parameter(n_servers), &n_servers, |b, _| {
            b.iter(|| run_client_round(&mut rng, &pp, ClientId(0), m.clone(), &server_ids))
        });
    }
    g.finish();
}

fn bench_server_round(c: &mut Criterion) {
    let mut g = c.benchmark_group("server_round");
    for &(n_servers, n_clients) in &[(4usize, 4usize), (4, 16), (4, 64)] {
        let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
        let pp = ProtocolParams::setup(&mut rng, n_servers);
        let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
        let mut inbox = ServerInbox {
            server_id: server_ids[0],
            items: vec![],
        };
        let mut canonical = vec![];
        for ci in 0..n_clients {
            let cid = ClientId(ci as u32);
            canonical.push(cid);
            let m = vec![KahePoly::rand_poly(&mut rng); pp.kahe.mu_kahe];
            let round = run_client_round(&mut rng, &pp, cid, m, &server_ids);
            let (sid, ops) = round.encrypted_openings.into_iter().next().unwrap();
            assert_eq!(sid, server_ids[0]);
            inbox.items.push((cid, ops));
        }
        g.bench_with_input(
            BenchmarkId::from_parameter(format!("S{}_C{}", n_servers, n_clients)),
            &(n_servers, n_clients),
            |b, _| b.iter(|| run_server_round(&inbox, &canonical).unwrap()),
        );
    }
    g.finish();
}

fn bench_verify(c: &mut Criterion) {
    let mut g = c.benchmark_group("verify");
    for &(n_servers, n_clients) in &[(4usize, 4usize), (4, 16), (8, 16)] {
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        let pp = ProtocolParams::setup(&mut rng, n_servers);
        let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
        let mut inboxes: Vec<ServerInbox> = server_ids
            .iter()
            .map(|&sid| ServerInbox {
                server_id: sid,
                items: vec![],
            })
            .collect();
        let mut client_entries: Vec<_> = vec![];
        let mut canonical = vec![];
        for ci in 0..n_clients {
            let cid = ClientId(ci as u32);
            canonical.push(cid);
            let m = vec![KahePoly::rand_poly(&mut rng); pp.kahe.mu_kahe];
            let round = run_client_round(&mut rng, &pp, cid, m, &server_ids);
            client_entries.push((round.client_id, round.encrypted_message));
            for (idx, (sid, ops)) in round.encrypted_openings.into_iter().enumerate() {
                assert_eq!(sid, server_ids[idx]);
                inboxes[idx].items.push((cid, ops));
            }
        }
        let outputs: Vec<_> = inboxes
            .iter()
            .map(|inb| run_server_round(inb, &canonical).unwrap())
            .collect();
        g.bench_with_input(
            BenchmarkId::from_parameter(format!("S{}_C{}", n_servers, n_clients)),
            &(n_servers, n_clients),
            |b, _| b.iter(|| aggregate_and_decrypt(&pp, &canonical, &client_entries, &outputs).unwrap()),
        );
    }
    g.finish();
}

/// One aggregator summing its group's public ciphertexts + commitments.
fn bench_aggregator_round(c: &mut Criterion) {
    let mut g = c.benchmark_group("aggregator_round");
    let n_servers = 4;
    for &group_size in &[20usize, 40] {
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let pp = ProtocolParams::setup(&mut rng, n_servers);
        let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
        let mut entries = Vec::with_capacity(group_size);
        for ci in 0..group_size {
            let cid = ClientId(ci as u32);
            let m = vec![KahePoly::rand_poly(&mut rng); pp.kahe.mu_kahe];
            let round = run_client_round(&mut rng, &pp, cid, m, &server_ids);
            entries.push((round.client_id, round.encrypted_message));
        }
        g.bench_with_input(
            BenchmarkId::from_parameter(group_size),
            &group_size,
            |b, _| b.iter(|| run_aggregator_round(&entries)),
        );
    }
    g.finish();
}

/// Leader decode over re-summed group aggregates (aggregated flow), vs the
/// direct `bench_verify` which re-sums every individual `ClientPublic`.
fn bench_verify_aggregated(c: &mut Criterion) {
    let mut g = c.benchmark_group("verify_aggregated");
    let group_size = 40;
    for &(n_servers, n_clients) in &[(4usize, 40usize), (4, 80), (8, 80)] {
        let mut rng = ChaCha20Rng::from_seed([5u8; 32]);
        let pp = ProtocolParams::setup(&mut rng, n_servers);
        let server_ids: Vec<ServerId> = (0..n_servers as u32).map(ServerId).collect();
        let mut inboxes: Vec<ServerInbox> = server_ids
            .iter()
            .map(|&sid| ServerInbox { server_id: sid, items: vec![] })
            .collect();
        let mut client_entries: Vec<_> = vec![];
        let mut canonical = vec![];
        for ci in 0..n_clients {
            let cid = ClientId(ci as u32);
            canonical.push(cid);
            let m = vec![KahePoly::rand_poly(&mut rng); pp.kahe.mu_kahe];
            let round = run_client_round(&mut rng, &pp, cid, m, &server_ids);
            client_entries.push((round.client_id, round.encrypted_message));
            for (idx, (sid, ops)) in round.encrypted_openings.into_iter().enumerate() {
                assert_eq!(sid, server_ids[idx]);
                inboxes[idx].items.push((cid, ops));
            }
        }
        let outputs: Vec<_> = inboxes
            .iter()
            .map(|inb| run_server_round(inb, &canonical).unwrap())
            .collect();

        let n_groups = n_clients.div_ceil(group_size);
        let aggregates: Vec<_> = (0..n_groups)
            .map(|grp| {
                let entries: Vec<_> = client_entries
                    .iter()
                    .filter(|(cid, _)| cid.0 as usize % n_groups == grp)
                    .cloned()
                    .collect();
                run_aggregator_round(&entries)
            })
            .collect();
        let group_ctxts: Vec<Vec<KahePoly>> =
            aggregates.iter().map(|a| a.summed_ctxt.clone()).collect();
        let group_comms: Vec<_> = aggregates.iter().map(|a| a.summed_comm.clone()).collect();
        let total_ctxt = Kahe::agg_ctxt(&group_ctxts);
        let total_comm = HidingMerkleCommitment::sum_commitments(&group_comms);

        let entry_bytes = client_entries[0].1.to_bytes().len();
        eprintln!(
            "verify_aggregated S{n_servers}_C{n_clients}: leader ingests {n_groups} group aggregates (~{} B) vs {n_clients} ClientPublics (~{} B)",
            n_groups * entry_bytes,
            n_clients * entry_bytes,
        );

        g.bench_with_input(
            BenchmarkId::from_parameter(format!("S{}_C{}", n_servers, n_clients)),
            &(n_servers, n_clients),
            |b, _| b.iter(|| decrypt_aggregate(&pp, &total_ctxt, &total_comm, &outputs).unwrap()),
        );
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_client_round,
    bench_server_round,
    bench_verify,
    bench_aggregator_round,
    bench_verify_aggregated
);
criterion_main!(benches);
