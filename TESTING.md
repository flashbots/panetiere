# Tests and benchmarks

How to test and benchmark Panetière.

## Tests and benches

Add all the target architectures:

```sh
rustup target add x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu riscv64gc-unknown-linux-gnu
```

```sh
RAYON_NUM_THREADS=8 cargo test -j 8
```

KAHE runs on a single 64-bit NTT over its prime modulus; the CS auxiliary
domain uses a two-prime RNS parameter set from `src/rings.rs`, whose 32-bit NTT
channels dispatch to scalar, AVX2, or NEON. CS and NG retain their own
field arithmetic. The discrete Gaussian sampler (`src/dgauss.rs`) has an AVX2
and a scalar kernel; its tests compare both with a plain comparison at every
threshold of its table. Whether the compiled kernels are free of branches on
the sample depends on the compiler and the target, so after changing the sampler
or the toolchain, run `python3 scripts/dgauss_branches.py`. It checks the
assembly for x86-64 and aarch64 and needs both Rust targets installed.

The end-to-end tests in `src/e2e/` are the executable spec: `end_to_end.rs` (canonical round,
*t*-of-*n*, replay/tamper/norm rejection, anonymity floor, thread
invariance), `iblt_e2e.rs` and `ng_e2e.rs` (each encoding carried end-to-end,
with cover traffic), `recipient_e2e.rs` (anchor admission, culprit exclusion, every `RejectReason`).
`tests/smoke.rs` runs one round through the public API only, with every
wire item sent as bytes.

### Lint

CI also checks formatting and runs clippy:

```sh
rustup component add rustfmt clippy
cargo fmt --all --check
RUSTFLAGS="" cargo clippy --target x86_64-unknown-linux-gnu --all-targets -- -D warnings
RUSTFLAGS="" cargo clippy --target aarch64-unknown-linux-gnu --all-targets -- -D warnings
```

`RUSTFLAGS=""` overrides a local `target-cpu=native`.

### Other architectures

Emulate other architectures with qemu. On Ubuntu 26.04:

```sh
sudo apt install qemu-user
```

For each architecture other than your own, install its linker and C library,
edit the linker and runner in `~/.cargo/config.toml` and run the tests.

To emulate x86-64:

```sh
sudo apt install gcc-x86-64-linux-gnu libc6-dev-amd64-cross
```

```toml
[target.x86_64-unknown-linux-gnu]
linker = "x86_64-linux-gnu-gcc"
runner = "qemu-x86_64 -L /usr/x86_64-linux-gnu"
```

```sh
RUSTFLAGS="" cargo test --target x86_64-unknown-linux-gnu
```

To emulate aarch64:

```sh
sudo apt install gcc-aarch64-linux-gnu libc6-dev-arm64-cross
```

```toml
[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"
runner = "qemu-aarch64 -L /usr/aarch64-linux-gnu"
```

```sh
RUSTFLAGS="" cargo test --target aarch64-unknown-linux-gnu
```

To emulate riscv64:

```sh
sudo apt install gcc-riscv64-linux-gnu libc6-dev-riscv64-cross
```

```toml
[target.riscv64gc-unknown-linux-gnu]
linker = "riscv64-linux-gnu-gcc"
runner = "qemu-riscv64 -L /usr/riscv64-linux-gnu"
```

```sh
RUSTFLAGS="" cargo test --target riscv64gc-unknown-linux-gnu
```

## Paper benchmark

`bench/` is a separate package in the workspace, `paper_bench`. It measures the compute time of each role at the paper's parameters, through the public API only. For each combination of encoding, number of clients, number of servers and message size, it times one client, one server and the recipient, from the bytes each receives to the bytes it sends, and reports the exact length of every message.

```sh
cargo build --release -p paper_bench
RAYON_NUM_THREADS=8 taskset -c 0-7 target/release/paper_bench
```

It prints one table per setting and rewrites the CSV file after each one, so a run that is stopped early keeps what it measured. Both start with a line that records the revision, compiler, CPU, the CPUs the process may use, threads, repetitions and seed. Pinning with `taskset` is optional; it keeps runs comparable.

The settings are environment variables, all optional:

| Variable | Default | Meaning |
|---|---|---|
| `PAPER_ENCODINGS` | `ng,iblt` | multiset encodings |
| `PAPER_CLIENTS` | `100,300,300x100` | client counts: `ρ`, or `ρxC` for ρ clients of which C send a message and ρ − C send cover traffic |
| `PAPER_SERVERS` | `8,16,32` | server counts γ |
| `PAPER_MSG_BYTES` | `4096,16384` | message sizes in bytes |
| `PAPER_REPS` | `5` | timed repetitions per role; odd, so the median is a measured value |
| `PAPER_SEED` | `0` | seed, hashed into every setting's inputs |
| `PAPER_CSV` | `paper_bench.csv` | output file |
| `RAYON_NUM_THREADS` | all CPUs | threads |

The default grid is the paper's, 36 settings, and takes about 5 minutes on a laptop. For a single setting:

```sh
PAPER_ENCODINGS=ng PAPER_CLIENTS=100 PAPER_SERVERS=8 PAPER_MSG_BYTES=4096 target/release/paper_bench
```
