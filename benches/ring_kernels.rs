use std::time::Duration;

use chipmunk_code::{pointwise_dot as pointwise_dot_hvc, HVCNTTPoly, HVCPoly};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use panetiere::{hvc_stream::StreamingHvcDot, hvc_sum::sum_hvc_polys, N};
use rand::{rngs::StdRng, SeedableRng};

fn hvc(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0x4856_43);
    let polys: Vec<_> = (0..63).map(|_| HVCPoly::rand_poly(&mut rng)).collect();
    let ntt: Vec<_> = polys.iter().map(HVCNTTPoly::from).collect();
    let streaming: Vec<_> = [6, 24, 30, 63]
        .into_iter()
        .map(|width| StreamingHvcDot::init(&mut rng, width))
        .collect();
    let sum_polys: Vec<_> = (0..100).map(|_| HVCPoly::rand_poly(&mut rng)).collect();
    let sum_refs: Vec<_> = sum_polys.iter().collect();
    let mut group = c.benchmark_group("ring_kernels/hvc");
    group.throughput(Throughput::Elements(N as u64));

    group.bench_function("coefficient_to_ntt", |b| {
        b.iter(|| HVCNTTPoly::from(black_box(&polys[0])))
    });
    group.bench_function("ntt_to_coefficient", |b| {
        b.iter(|| HVCPoly::from(black_box(&ntt[0])))
    });
    group.bench_function("pointwise_sum_100", |b| {
        b.iter(|| sum_hvc_polys(black_box(&sum_refs)))
    });
    group.bench_function("pointwise_mac_6", |b| {
        b.iter(|| pointwise_dot_hvc(black_box(&ntt[..6]), black_box(&ntt[..6])))
    });
    group.bench_function("pointwise_mac_24", |b| {
        b.iter(|| pointwise_dot_hvc(black_box(&ntt[..24]), black_box(&ntt[..24])))
    });
    group.bench_function("pointwise_mac_30", |b| {
        b.iter(|| pointwise_dot_hvc(black_box(&ntt[..30]), black_box(&ntt[..30])))
    });
    group.bench_function("pointwise_mac_63", |b| {
        b.iter(|| pointwise_dot_hvc(black_box(&ntt), black_box(&ntt)))
    });
    for (dot, width) in streaming.iter().zip([6, 24, 30, 63]) {
        group.bench_function(format!("streaming_hash_{width}"), |b| {
            b.iter(|| dot.hash(black_box(&polys[..width])))
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = hvc
}
criterion_main!(benches);
