# panetiere-sim

Runs a Panetière round over a network instead of in one process. Each role is
its own node with a small HTTP API, and `run.sh` puts every node in its own
Docker container with its own shaped link (delay, loss, upload and download
rate).

```bash
sim/run.sh                                        # 4 servers, 8 clients, 8 messages
SIM_CLIENTS=50 SIM_MESSAGES=40 CLIENT_UP=10mbit SIM_ROUNDS=3 sim/run.sh
```

Each round writes one JSON line to `target/sim/results.jsonl`. It holds the
coordinator's phase times and a list of timed steps ("spans") from every node.

## Metrics

Each span has a start and end time (unix microseconds), and where it applies
the bytes moved and the CPU time used. Containers on one host share a clock, so
spans from different nodes line up on one timeline.

| Node | Spans |
|---|---|
| client | `download_keys`, `client_round` (CPU), `upload_post`, `upload_opening_<server>` |
| server | `download_canonical`, `download_commitments`, `receive_openings` (first to last arrival), `unseal` (CPU), `server_round` (CPU), `upload_entry` |
| recipient | `download_canonical`, `download_entry_<server>`, `download_post_<client>`, `parse`, `recover`, `decode` (all CPU) |
| bulletin | `receive_posts`, `receive_entries` (first to last arrival) |

CPU spans record the whole process's CPU time, so CPU time divided by wall
time is the number of cores the step kept busy.

```bash
sim/analyze.py target/sim/results.jsonl                          # table per round
sim/analyze.py target/sim/results.jsonl --html target/sim/timeline.html
sim/sweep.sh SIM_CLIENTS=8,16,32 SIM_MESSAGE_BYTES=1024,16384     # every combination
sim/analyze.py target/sim/sweep.jsonl --csv                      # one row per round
```

The table gives, for each step: count, median and slowest time, the slowest
node, bytes, throughput and cores used, and for each role the share of the
round it spent busy. The HTML page draws every node's spans on one timeline,
so idle time and the critical path are visible at a glance.

Needs Docker with Compose. The containers need `NET_ADMIN`, and the host needs
the `ifb` and `sch_netem` kernel modules for download shaping.

## Roles

| Role | Does |
|---|---|
| `bulletin` | Stores client posts, server entries and the canonical set per session. Serves commitments on their own, so servers do not download ciphertexts. |
| `server` | Serves its public key, collects sealed openings, and on `/round` fetches the canonical set and commitments, runs the server round and posts its entry. |
| `client` | On `/round`, encodes its message (or cover), runs the client round, and uploads its post and every opening at the same time. |
| `recipient` | On `/round`, downloads every server entry and client post, recovers and decodes. |
| `coordinator` | Waits for every node, then drives each round phase by phase and checks the recovered messages. |

Every node derives the public parameters from the same `SIM_*` settings and
seed, so nothing has to be sent around for setup.

## Settings

All settings are environment variables.

| Variable | Default | |
|---|---|---|
| `SIM_SERVERS` / `SIM_CLIENTS` | 4 / 8 | |
| `SIM_MESSAGES` | `SIM_CLIENTS` | clients `0..SIM_MESSAGES` send a message, the rest send cover. Every post has room for this many messages, so the recipient downloads about `clients × messages × message bytes × 7` |
| `SIM_MESSAGE_BYTES` | 32 | |
| `SIM_ROUNDS` | 1 | the first round includes connection setup, so use 2 or more and compare |
| `SIM_LABEL` | none | copied into every report; `sweep.sh` sets it |
| `SIM_RESULTS` | `target/sim/results.jsonl` | |
| `SIM_SEED` | 0 | seed for the shared parameters |
| `<ROLE>_DELAY` | see `run.sh` | one-way delay on outgoing traffic |
| `<ROLE>_LOSS` | none | outgoing packet loss, e.g. `0.1%` |
| `<ROLE>_UP` / `<ROLE>_DOWN` | see `run.sh` | rate limits, e.g. `20mbit` |
| `<ROLE>_CPUS` | client 1, server 4, recipient 8 | CPU budget per node, as if it were its own machine; empty for no limit |
| `SIM_READY_TIMEOUT_SECS` | 120 | how long the coordinator waits for every node to start |

`<ROLE>` is `BULLETIN`, `SERVER`, `CLIENT` or `RECIPIENT`. Set a value to the
empty string to leave that part of the link unshaped.

## How big a run fits

- **CPU.** All clients compute at once. Compute times are only realistic while
  `SIM_CLIENTS × CLIENT_CPUS` stays under the host's CPU count; `run.sh` warns
  past that. Network times stay valid either way. To go bigger, give clients a
  fraction of a CPU (`CLIENT_CPUS=0.5`), which models slower clients.
- **ARP table.** Every pair of containers that talk needs an entry in one
  table shared by the whole host, about `2 × clients × (servers + 3)` entries.
  Linux allows 1024 by default, which is about 70 clients with 4 servers.
  `run.sh` stops with the `sysctl` command to raise it.
- **Memory.** About 55 MB per client at 8 messages of 16 kB. The bulletin keeps
  every post in memory.
- **Docker.** A bridge network takes at most about 1000 containers.

## Limits

- Every client is honest and every server answers, so the canonical set is
  always all clients.
- All containers share the host's CPUs, so compute times for clients that run
  at the same time are slower than on separate machines.
- The roles start from a coordinator request rather than running freely, so
  the phases never overlap.
- Spans across nodes only line up when the nodes share a clock. On separate
  machines, clock skew shifts each node's spans.
