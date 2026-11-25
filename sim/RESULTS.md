# Sim results

First runs of the network sim, 2026-11-25, on one 32-core host.

Every client sends a message (`SIM_MESSAGES = SIM_CLIENTS`). Links are the
`run.sh` defaults: clients and recipient 40 ms delay, 20 Mbit/s up and
100 Mbit/s down; servers 10 ms at 1 Gbit/s; bulletin 5 ms at 1 Gbit/s. Servers
get 2 CPUs, clients 1 CPU (0.5 at 64 clients), the recipient 8. Every number is
from the second round, so connection setup is not included. Times are seconds.

## Grid

| clients | servers | message | round | client phase | server phase | recipient phase | post | one opening | recipient download | client CPU | recipient CPU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 4 | 1 kB | 0.75 | 0.32 | 0.11 | 0.31 | 146 kB | 47 kB | 1.4 MB | 6 ms | 4 ms |
| 8 | 8 | 1 kB | 0.86 | 0.41 | 0.11 | 0.34 | 146 kB | 57 kB | 1.8 MB | 10 ms | 5 ms |
| 8 | 16 | 1 kB | 1.18 | 0.68 | 0.10 | 0.39 | 146 kB | 67 kB | 2.6 MB | 19 ms | 8 ms |
| 16 | 4 | 1 kB | 0.90 | 0.32 | 0.11 | 0.46 | 182 kB | 47 kB | 3.2 MB | 6 ms | 6 ms |
| 16 | 8 | 1 kB | 1.03 | 0.43 | 0.11 | 0.49 | 182 kB | 57 kB | 3.6 MB | 11 ms | 7 ms |
| 16 | 16 | 1 kB | 1.42 | 0.70 | 0.19 | 0.52 | 182 kB | 67 kB | 4.5 MB | 20 ms | 10 ms |
| 32 | 4 | 1 kB | 1.34 | 0.37 | 0.11 | 0.85 | 252 kB | 47 kB | 8.4 MB | 10 ms | 13 ms |
| 32 | 8 | 1 kB | 1.50 | 0.46 | 0.12 | 0.92 | 252 kB | 57 kB | 8.8 MB | 16 ms | 13 ms |
| 32 | 16 | 1 kB | 1.91 | 0.73 | 0.21 | 0.97 | 252 kB | 67 kB | 9.7 MB | 28 ms | 16 ms |
| 64 | 4 | 1 kB | 2.77 | 0.46 | 0.12 | 2.18 | 370 kB | 47 kB | 24.0 MB | 23 ms | 33 ms |
| 64 | 8 | 1 kB | 2.90 | 0.55 | 0.13 | 2.21 | 370 kB | 57 kB | 24.4 MB | 32 ms | 33 ms |
| 8 | 4 | 16 kB | 3.05 | 1.15 | 0.09 | 1.80 | 2.05 MB | 47 kB | 16.7 MB | 29 ms | 27 ms |
| 8 | 8 | 16 kB | 3.20 | 1.23 | 0.11 | 1.85 | 2.05 MB | 57 kB | 17.0 MB | 32 ms | 29 ms |
| 8 | 16 | 16 kB | 3.33 | 1.50 | 0.11 | 1.71 | 2.05 MB | 67 kB | 17.9 MB | 41 ms | 31 ms |
| 16 | 4 | 16 kB | 5.45 | 1.42 | 0.11 | 3.91 | 2.66 MB | 47 kB | 42.7 MB | 37 ms | 55 ms |
| 16 | 8 | 16 kB | 5.62 | 1.50 | 0.11 | 4.00 | 2.66 MB | 57 kB | 43.1 MB | 44 ms | 56 ms |
| 16 | 16 | 16 kB | 5.87 | 1.78 | 0.21 | 3.88 | 2.66 MB | 67 kB | 44.0 MB | 52 ms | 57 ms |
| 32 | 4 | 16 kB | 12.15 | 1.90 | 0.11 | 10.13 | 3.68 MB | 47 kB | 118 MB | 74 ms | 123 ms |
| 32 | 8 | 16 kB | 12.27 | 1.98 | 0.12 | 10.17 | 3.68 MB | 57 kB | 118 MB | 81 ms | 122 ms |
| 32 | 16 | 16 kB | 12.74 | 2.27 | 0.21 | 10.25 | 3.68 MB | 67 kB | 119 MB | 92 ms | 123 ms |
| 64 | 4 | 16 kB | 34.14 | 3.48 | 0.11 | 30.54 | 5.61 MB | 47 kB | 359 MB | 214 ms | 313 ms |
| 64 | 8 | 16 kB | 34.26 | 3.57 | 0.13 | 30.56 | 5.61 MB | 57 kB | 360 MB | 219 ms | 315 ms |

Server CPU stayed between 5 and 20 ms in every case.

What it shows:

- **The recipient's download decides the round.** At 16 kB messages it is
  83–89% of the round, at about 95 Mbit/s on a 100 Mbit/s link.
- **Recipient download grows close to the square of the client count.** The
  recipient downloads one post per client, and each post has room for every
  message. At 16 kB a post is about 1.6 MB plus about 62 kB per message slot.
  Capping message slots below the client count would make growth linear.
- **Compute is about 1% of the round.** Overlapping phases or summing while
  downloading would save little at these link speeds.
- **Adding servers mostly costs clients.** Each opening grows by about 10 kB
  every time the server count doubles, and clients upload one per server.

## Recipient at 1 Gbit/s

Same as the grid, 32 clients and 4 servers, with `RECIPIENT_DOWN=1gbit`.

| message | round | client phase | recipient phase | download rate | recipient CPU |
|---|---|---|---|---|---|
| 1 kB | 0.88 (was 1.34) | 0.36 | 0.40 | 226 Mbit/s | 14 ms |
| 16 kB | 3.82 (was 12.15) | 1.90 | 1.80 | 595 Mbit/s | 126 ms |

With a fast recipient, the client phase is as long as the recipient phase.
The download reaches only about 60% of the link; the cause is not traced yet.

## Openings and client upload

8 clients, 1 kB messages, by client upload rate. Client phase, with the round
in brackets.

| servers | one opening | client upload | 5 Mbit/s | 20 Mbit/s | 100 Mbit/s |
|---|---|---|---|---|---|
| 4 | 47 kB | 334 kB | 0.71 (1.14) | 0.32 (0.75) | 0.28 (0.72) |
| 8 | 57 kB | 604 kB | 1.17 (1.60) | 0.41 (0.86) | 0.29 (0.72) |
| 16 | 67 kB | 1,224 kB | 2.22 (2.73) | 0.68 (1.18) | 0.30 (0.82) |

The client phase is about 0.2 s plus upload bytes divided by the link rate.
At 16 servers openings are 88% of what a client uploads. If each opening
were cut to the part that does not grow with servers (about 27 kB), the 16
server round would drop by about 38% at 5 Mbit/s, 22% at 20 Mbit/s and 6% at
100 Mbit/s (estimated from the formula above, not measured).

## Stability

One 64 client run failed once before the host's ARP table limit was raised
(it was near the default 1024 entries). After raising it, the same case ran
10 rounds over two runs with no failure, and round times varied by less than
1%.

## Reproduce

```bash
SIM_ROUNDS=2 SERVER_CPUS=2 sim/sweep.sh SIM_CLIENTS=8,16,32 SIM_SERVERS=4,8,16 SIM_MESSAGE_BYTES=1024,16384
SIM_ROUNDS=2 SERVER_CPUS=2 CLIENT_CPUS=0.5 sim/sweep.sh SIM_CLIENTS=64 SIM_SERVERS=4,8 SIM_MESSAGE_BYTES=1024,16384
SIM_ROUNDS=2 SERVER_CPUS=2 RECIPIENT_DOWN=1gbit sim/sweep.sh SIM_CLIENTS=32 SIM_SERVERS=4 SIM_MESSAGE_BYTES=1024,16384
SIM_ROUNDS=2 SERVER_CPUS=2 SIM_CLIENTS=8 SIM_MESSAGE_BYTES=1024 sim/sweep.sh SIM_SERVERS=4,8,16 CLIENT_UP=5mbit,20mbit,100mbit
```
