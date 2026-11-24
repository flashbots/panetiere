#!/usr/bin/env bash
# runs sim/run.sh once per value of one setting and collects every round into
# target/sim/sweep-<setting>.jsonl. other settings pass through as usual.
#
#   sim/sweep.sh CLIENT_UP 5mbit 20mbit 100mbit
#   SIM_ROUNDS=3 sim/sweep.sh SIM_CLIENTS 8 16 32
#
# then: sim/analyze.py target/sim/sweep-CLIENT_UP.jsonl --csv
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 2 ]; then
    echo "usage: $0 <setting> <value>..." >&2
    exit 1
fi
setting=$1
shift

mkdir -p target/sim
combined="target/sim/sweep-$setting.jsonl"
: >"$combined"
for value in "$@"; do
    echo "== $setting=$value" >&2
    env "$setting=$value" SIM_LABEL="$setting=$value" \
        SIM_RESULTS="target/sim/sweep-$setting-$value.jsonl" sim/run.sh
    cat "target/sim/sweep-$setting-$value.jsonl" >>"$combined"
done
echo "sweep written to $combined" >&2
