#!/usr/bin/env bash
# runs one simulated deployment in docker: a bulletin, the servers, the
# clients and a recipient, each in its own container with its own shaped link,
# then a coordinator that drives the rounds. one json line per round is written
# to $SIM_RESULTS (target/sim/results.jsonl by default); read it with
# sim/analyze.py.
#
# every setting is an environment variable, for example:
#   SIM_SERVERS=5 SIM_CLIENTS=20 SIM_MESSAGES=10 CLIENT_UP=10mbit sim/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${SIM_SEED:=0}"
: "${SIM_SERVERS:=4}"
: "${SIM_CLIENTS:=8}"
: "${SIM_MESSAGES:=6}"
: "${SIM_MESSAGE_BYTES:=32}"
: "${SIM_MIN_CLIENTS:=1}"
: "${SIM_ROUNDS:=1}"
: "${SIM_LABEL:=}"

# link profiles: one-way delay, loss, upload and download rate per role.
# an empty value leaves that part of the link unshaped.
: "${BULLETIN_DELAY=5ms}" "${BULLETIN_LOSS=}" "${BULLETIN_UP=1gbit}" "${BULLETIN_DOWN=1gbit}"
: "${SERVER_DELAY=10ms}" "${SERVER_LOSS=}" "${SERVER_UP=1gbit}" "${SERVER_DOWN=1gbit}"
: "${CLIENT_DELAY=40ms}" "${CLIENT_LOSS=}" "${CLIENT_UP=20mbit}" "${CLIENT_DOWN=100mbit}"
: "${RECIPIENT_DELAY=40ms}" "${RECIPIENT_LOSS=}" "${RECIPIENT_UP=20mbit}" "${RECIPIENT_DOWN=100mbit}"

out=target/sim
compose="$out/compose.yml"
: "${SIM_RESULTS:=$out/results.jsonl}"
mkdir -p "$out"

cargo build --release -p panetiere-sim
docker build -q -t panetiere-sim sim/docker >/dev/null

url() { echo "http://$1:8080"; }
join() { local IFS=,; echo "$*"; }

server_urls=()
for ((i = 0; i < SIM_SERVERS; i++)); do server_urls+=("$(url "server-$i")"); done
client_urls=()
for ((i = 0; i < SIM_CLIENTS; i++)); do client_urls+=("$(url "client-$i")"); done

# writes one service. $1 name, $2 link profile prefix, $3 role, rest are extra env lines
service() {
    local name=$1 profile=$2 role=$3
    shift 3
    local delay="${profile}_DELAY" loss="${profile}_LOSS" up="${profile}_UP" down="${profile}_DOWN"
    cat <<YAML
  $name:
    <<: *node
    command: ["$role"]
    environment:
      <<: *setup
      NET_DELAY: "${!delay}"
      NET_LOSS: "${!loss}"
      NET_UP: "${!up}"
      NET_DOWN: "${!down}"
YAML
    for line in "$@"; do echo "      $line"; done
}

{
    cat <<YAML
name: panetiere-sim
x-setup: &setup
  SIM_SEED: "$SIM_SEED"
  SIM_SERVERS: "$SIM_SERVERS"
  SIM_CLIENTS: "$SIM_CLIENTS"
  SIM_MESSAGES: "$SIM_MESSAGES"
  SIM_MESSAGE_BYTES: "$SIM_MESSAGE_BYTES"
  SIM_MIN_CLIENTS: "$SIM_MIN_CLIENTS"
  RUST_LOG: "\${RUST_LOG:-info}"
x-node: &node
  image: panetiere-sim
  cap_add: [NET_ADMIN]
  volumes:
    - $PWD/target/release/panetiere-sim:/usr/local/bin/panetiere-sim:ro
services:
YAML
    service bulletin BULLETIN bulletin
    for ((i = 0; i < SIM_SERVERS; i++)); do
        service "server-$i" SERVER server "SIM_INDEX: \"$i\"" "SIM_BULLETIN: $(url bulletin)"
    done
    for ((i = 0; i < SIM_CLIENTS; i++)); do
        service "client-$i" CLIENT client "SIM_INDEX: \"$i\"" "SIM_BULLETIN: $(url bulletin)" \
            "SIM_SERVER_URLS: $(join "${server_urls[@]}")"
    done
    service recipient RECIPIENT recipient "SIM_BULLETIN: $(url bulletin)"
    cat <<YAML
  coordinator:
    <<: *node
    profiles: [coordinator]
    command: ["coordinator"]
    environment:
      <<: *setup
      SIM_ROUNDS: "$SIM_ROUNDS"
      SIM_LABEL: "$SIM_LABEL"
      SIM_BULLETIN: $(url bulletin)
      SIM_SERVER_URLS: $(join "${server_urls[@]}")
      SIM_CLIENT_URLS: $(join "${client_urls[@]}")
      SIM_RECIPIENT: $(url recipient)
YAML
} >"$compose"

dc() { docker compose -f "$compose" "$@"; }
trap 'dc --profile coordinator down --remove-orphans >/dev/null 2>&1' EXIT

dc up -d --quiet-pull
dc run --rm -T coordinator >"$SIM_RESULTS"
echo "results written to $SIM_RESULTS" >&2
