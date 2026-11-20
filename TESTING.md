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
channels dispatch to scalar, AVX2, or NEON. CS and Prony retain their own
field arithmetic.

The end-to-end tests in `src/e2e/` are the executable spec: `end_to_end.rs` (canonical round,
*t*-of-*n*, replay/tamper/norm rejection, anonymity floor, thread
invariance), `mse_e2e.rs` and `prony_e2e.rs` (each encoding carried end-to-end,
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
