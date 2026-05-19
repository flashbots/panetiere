use chipmunk_code::{KahePoly, Polynomial};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use flashnet::protocol::client::run_client_round;
use flashnet::protocol::message::{ClientId, ServerId};
use flashnet::protocol::server::{run_server_round, ServerInbox};
use flashnet::protocol::verify::aggregate_and_decrypt;
use flashnet::protocol::ProtocolParams;
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
            let (sid, ops) = round.private.into_iter().next().unwrap();
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
        let mut publics = vec![];
        let mut canonical = vec![];
        for ci in 0..n_clients {
            let cid = ClientId(ci as u32);
            canonical.push(cid);
            let m = vec![KahePoly::rand_poly(&mut rng); pp.kahe.mu_kahe];
            let round = run_client_round(&mut rng, &pp, cid, m, &server_ids);
            publics.push((round.client_id, round.public));
            for (idx, (sid, ops)) in round.private.into_iter().enumerate() {
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
            |b, _| b.iter(|| aggregate_and_decrypt(&pp, &canonical, &publics, &outputs).unwrap()),
        );
    }
    g.finish();
}

criterion_group!(benches, bench_client_round, bench_server_round, bench_verify);
criterion_main!(benches);
