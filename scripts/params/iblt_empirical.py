#!/usr/bin/env python3
"""Empirical search for IBLT shapes (theta rows x delta cells) by simulation.

The target failure probability is 2^-eps (as eps_mse in generate_params.py).
For every theta, the script searches for the smallest delta such that the
decoding failure rate at full load (rho elements) is at most the target with
the given confidence: a shape is accepted if the one-sided Clopper-Pearson
upper bound on the failure rate, computed from the observed failures in
trials-factor * 2^eps trials, is at most the target.  Key collisions are not
included (see iblt_common.py).

The search per theta starts from the asymptotic peeling threshold as the lower
end and from the provable delta (or the size limit given by the best shape so
far, if smaller) as the upper end; if the upper end fails, it grows by 1.25x up
to the limit.  It then bisects until the interval is below --resolution
(default 1%) of delta, since the pass/fail outcome close to the threshold is
dominated by noise anyway.  The reported delta always passed the test.  The
range of thetas is determined automatically and thetas are excluded by rigorous
lower bounds where possible (see iblt_common.py).

For comparison, the script also reports the exact union bound at the empirical
delta and the smallest delta for which this provable bound is at most the
target.  The empirical search needs the true failure rate to be clearly below
the target to pass the confidence test, so for small theta (where the union
bound is nearly tight) the provable delta can even be smaller.

The number of trials per tested shape is trials-factor * 2^eps, so this is only
feasible up to eps of about 25-30.  For larger eps, use --no-sim for the
provable shapes only.

As a library, empirical_shape() returns the best empirical shape (used by
generate_params.py --empirical).

Usage examples:
    python3 iblt_empirical.py --rho 300 --eps 20
    python3 iblt_empirical.py --rho 300 --eps 20 --check 3 225   # evaluate one shape
    python3 iblt_empirical.py --rho 300 --eps 32 --no-sim        # provable shapes only
    python3 iblt_empirical.py --rho 300 --eps 14 --thetas 3-8 --workers 8   # restrict thetas
"""

import argparse
import math
import os
import sys
import time

from iblt_common import (Simulator, _load_lib, common_arguments, cp_upper, make_args,
                         max_failures, peeling_factor, search_thetas, union_bound_log2)


def evaluate(sim, rho, theta, delta, trials, target, alpha):
    """Simulates one shape.  It passes if after all trials the Clopper-Pearson
    upper bound is at most target; the simulation stops early once too many
    failures occurred to pass.  Returns (ok, failures, trials, upper bound, seconds)."""
    kmax = max_failures(trials, target, alpha)
    t0 = time.time()
    f, n, _ = sim.run(rho, theta, delta, trials, maxfail=kmax)
    ub = cp_upper(f, n, alpha)
    ok = n >= trials and f <= kmax
    return ok, f, n, ub, time.time() - t0


def search_theta(sim, rho, theta, target, alpha, trials, verbose, max_cells=None, guess=None,
                 resolution=0.01):
    """Smallest delta for this theta that passes the test, up to the relative
    resolution (the returned delta passes; the true smallest one is at most
    resolution * delta below it).

    Lower end: the peeling threshold (shapes below it fail).  Upper end: the
    smallest shape that is still interesting (max_cells, from pruning) and the
    guess (the provable delta), whichever is smaller; if that fails, grow by
    1.25x (2x without a guess) up to the cap.  Then binary search.  If the lower
    end itself passes (possible for small rho), the search continues below it.
    Returns None if no shape with theta*delta <= max_cells passes."""
    def test(delta):
        ok, f, n, ub, dt = evaluate(sim, rho, theta, delta, trials, target, alpha)
        if verbose:
            print(f"    theta={theta:2d} delta={delta:6d}: {f} failures / {n} trials, "
                  f"upper bound 2^{math.log2(ub):.2f}  {'pass' if ok else 'fail'}  ({dt:.1f}s)",
                  flush=True)
        return ok, (delta, f, n, ub)

    lo = max(2, math.floor(peeling_factor(theta) * rho / theta))
    cap = (1 << 22) if max_cells is None else max_cells // theta
    if cap < 2:
        return None
    lo = min(lo, cap)
    hi = min(cap, guess) if guess is not None else lo
    hi = max(hi, lo)
    known = False  # whether lo - 1 is known to fail
    # find a passing upper end
    while True:
        ok, res = test(hi)
        if ok:
            break
        if hi >= cap:
            return None
        lo, known = hi + 1, True
        hi = min(max(hi + 1, math.ceil(hi * (1.25 if guess is not None else 2))), cap)
    best = res
    while True:
        # binary search on [lo, hi]; hi passes
        while hi - lo >= max(1, resolution * hi):
            mid = (lo + hi) // 2
            ok, res = test(mid)
            if ok:
                hi, best = mid, res
            else:
                lo, known = mid + 1, True
        if known or lo <= 2:
            return best
        # the untested lower end passed: continue below it
        if verbose:
            print(f"    theta={theta:2d}: passes at the peeling lower end, searching below", flush=True)
        lo = max(2, lo // 2)


def empirical_shape(rho, target, alpha=0.05, trials_factor=10.0, resolution=0.01, workers=None,
                    seed=1, thetas=None, verbose=False):
    """Library interface (used by generate_params.py --empirical): the smallest shape
    (fewest cells) whose simulated failure rate passes the confidence test for the
    failure probability target, searched as in main().  Returns None if no shape is
    found, and otherwise a dict with theta, delta, cells, the failures and trials of
    the accepted test, the log2 of its Clopper-Pearson upper bound, the settings, and
    the best provable shape of the same search (float union bound, for reference)."""
    if not _load_lib():
        print("warning: no C compiler found, using the slow pure-Python simulator", file=sys.stderr)
    trials = math.ceil(trials_factor / target)
    sim = Simulator(workers or os.cpu_count() or 1, seed)
    try:
        def solve(theta, pdelta, max_cells):
            res = search_theta(sim, rho, theta, target, alpha, trials, verbose, max_cells,
                               guess=pdelta, resolution=resolution)
            if res is None:
                return None
            delta, f, n, ub = res
            return delta, {"failures": f, "trials": n, "upper_log2": math.log2(ub)}

        res = search_thetas(make_args(rho, target, thetas, quiet=not verbose), solve, report=False)
    finally:
        sim.close()
    if res["best"] is None:
        return None
    theta, delta, info = res["best"]
    return dict(theta=theta, delta=delta, cells=theta * delta, alpha=alpha,
                trials_per_shape=trials, resolution=resolution, seed=seed,
                provable_float=res["provable"], **info)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_arguments(ap)
    ap.add_argument("--trials-factor", type=float, default=10.0,
                    help="trials per tested shape = factor * 2^eps (default 10)")
    ap.add_argument("--resolution", type=float, default=0.01,
                    help="stop the binary search once the interval is below this fraction of delta "
                         "(default 0.01; 0 = exact)")
    ap.add_argument("--check", nargs=2, type=int, metavar=("THETA", "DELTA"),
                    help="only simulate one shape (trials-factor * 2^eps trials) and report its "
                         "failure rate and union bound")
    ap.add_argument("--no-sim", action="store_true",
                    help="no simulation, only compute the provable shapes")
    args = ap.parse_args()
    args.target = 2.0 ** -args.eps

    if not args.no_sim and not _load_lib():
        print("warning: no C compiler found, using the slow pure-Python simulator", file=sys.stderr)
    trials = math.ceil(args.trials_factor / args.target)
    if trials > 1e9 and not args.no_sim:
        print(f"warning: {trials:.2e} trials per tested shape; consider --no-sim or a "
              f"smaller eps", file=sys.stderr)
    sim = Simulator(args.workers, args.seed)
    try:
        if args.check:
            theta, delta = args.check
            t0 = time.time()
            f, n, _ = sim.run(args.rho, theta, delta, trials)
            print(f"rho={args.rho} theta={theta} delta={delta} ({theta * delta} cells): "
                  f"{f} failures / {n} trials, rate {f / n:.3e}, "
                  f"{100 * (1 - args.alpha):.0f}% upper bound 2^{math.log2(cp_upper(f, n, args.alpha)):.2f} "
                  f"({time.time() - t0:.1f}s)")
            if not args.no_bound:
                print(f"exact union bound: 2^{union_bound_log2(args.rho, theta, delta):.2f}")
            return

        print(f"rho={args.rho}, target failure probability 2^-{args.eps:g} = {args.target:.3g}"
              f" (key collisions not included)", flush=True)
        if not args.no_sim:
            print(f"empirical: {100 * (1 - args.alpha):.0f}% confidence, {trials} trials per tested "
                  f"shape, {args.workers} workers", flush=True)
        print(flush=True)

        def solve(theta, pdelta, max_cells):
            res = search_theta(sim, args.rho, theta, args.target, args.alpha, trials, not args.quiet,
                               max_cells, guess=pdelta, resolution=args.resolution)
            if res is None:
                return None
            delta, f, n, ub = res
            bound = "-" if args.no_bound else f"2^{union_bound_log2(args.rho, theta, delta):.2f}"
            return delta, [f"{f}/{n}", f"2^{math.log2(ub):.2f}", bound]

        search_thetas(args, solve, ["failures/trials", "upper bound", "union bound"], "empirical",
                      simulate=not args.no_sim)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
