#!/usr/bin/env bash
# runs sim/run.sh once for every combination of the given settings and
# collects every round into target/sim/sweep.jsonl. other settings pass
# through as usual.
#
#   sim/sweep.sh CLIENT_UP=5mbit,20mbit,100mbit
#   SIM_ROUNDS=2 sim/sweep.sh SIM_CLIENTS=8,16,32 SIM_SERVERS=4,8 SIM_MESSAGE_BYTES=1024,16384
#
# then: sim/analyze.py target/sim/sweep.jsonl --csv
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
    echo "usage: $0 <SETTING=v1,v2,...>..." >&2
    exit 1
fi

# every combination, one line each, as space separated SETTING=value pairs
combos=("")
for arg in "$@"; do
    setting=${arg%%=*}
    IFS=, read -ra values <<<"${arg#*=}"
    next=()
    for combo in "${combos[@]}"; do
        for value in "${values[@]}"; do
            next+=("${combo:+$combo }$setting=$value")
        done
    done
    combos=("${next[@]}")
done

mkdir -p target/sim
combined=target/sim/sweep.jsonl
: >"$combined"
failed=0
for combo in "${combos[@]}"; do
    echo "== $combo" >&2
    read -ra pairs <<<"$combo"
    part="target/sim/sweep-${combo// /_}.jsonl"
    if env "${pairs[@]}" SIM_LABEL="$combo" SIM_RESULTS="$part" sim/run.sh; then
        cat "$part" >>"$combined"
    else
        echo "!! $combo failed, continuing" >&2
        failed=$((failed + 1))
    fi
done
echo "sweep written to $combined ($failed of ${#combos[@]} runs failed)" >&2
