#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RAYON_NUM_THREADS=8 cargo run --release -j 8 --example demo
