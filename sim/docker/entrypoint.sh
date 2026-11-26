#!/bin/sh
# shapes this container's link from the NET_* variables, then runs the node.
# delay and loss apply to outgoing traffic only, so a round trip between two
# nodes pays both of their delays.
set -eu

egress=""
if [ -n "${NET_DELAY:-}" ]; then egress="$egress delay $NET_DELAY"; fi
if [ -n "${NET_LOSS:-}" ]; then egress="$egress loss $NET_LOSS"; fi
if [ -n "${NET_UP:-}" ]; then egress="$egress rate $NET_UP"; fi
if [ -n "$egress" ]; then
    tc qdisc add dev eth0 root netem limit 100000 $egress
fi

# incoming traffic is redirected through an ifb device so it can be rate limited too
if [ -n "${NET_DOWN:-}" ]; then
    ip link add ifb0 type ifb
    ip link set ifb0 up
    tc qdisc add dev eth0 handle ffff: ingress
    tc filter add dev eth0 parent ffff: matchall action mirred egress redirect dev ifb0
    tc qdisc add dev ifb0 root netem limit 100000 rate "$NET_DOWN"
fi

exec panetiere-sim "$@"
