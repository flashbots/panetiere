# panetiere-sim

Runs a Panetière round over a network instead of in one process. Each role is
its own node with a small HTTP API, and `run.sh` puts every node in its own
Docker container with its own shaped link (delay, loss, upload and download
rate).

```bash
sim/run.sh                                        # 4 servers, 8 clients, 6 messages
SIM_CLIENTS=50 SIM_MESSAGES=40 CLIENT_UP=10mbit SIM_ROUNDS=3 sim/run.sh
```

Each round prints one JSON line, also written to `target/sim/results.jsonl`,
with the wall time of each phase and every node's own report: compute time,
transfer time and bytes sent or received.

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
| `SIM_SERVERS` / `SIM_CLIENTS` / `SIM_MESSAGES` | 4 / 8 / 6 | clients `0..SIM_MESSAGES` send a message, the rest send cover |
| `SIM_MESSAGE_BYTES` | 32 | |
| `SIM_ROUNDS` | 1 | |
| `SIM_SEED` | 0 | seed for the shared parameters |
| `<ROLE>_DELAY` | see `run.sh` | one-way delay on outgoing traffic |
| `<ROLE>_LOSS` | none | outgoing packet loss, e.g. `0.1%` |
| `<ROLE>_UP` / `<ROLE>_DOWN` | see `run.sh` | rate limits, e.g. `20mbit` |

`<ROLE>` is `BULLETIN`, `SERVER`, `CLIENT` or `RECIPIENT`. Set a value to the
empty string to leave that part of the link unshaped.

## Limits

- Every client is honest and every server answers, so the canonical set is
  always all clients.
- All containers share the host's CPUs, so compute times for clients that run
  at the same time are slower than on separate machines.
- The roles start from a coordinator request rather than running freely, so
  the phases never overlap.
