#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
RAYON_NUM_THREADS=8 cargo bench -j 8 "$@"
