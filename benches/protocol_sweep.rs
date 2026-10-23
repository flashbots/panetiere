#[global_allocator]
static ALLOCATOR: panetiere::scaling_bench::CountingAlloc =
    panetiere::scaling_bench::CountingAlloc;

fn main() {
    panetiere::scaling_bench::run();
}
