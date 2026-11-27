#!/usr/bin/env python3
# Table for the discrete Gaussian sampler of KAHE keys and errors (src/dgauss.rs).
# Written by Claude (Anthropic).
#
# Usage:
#   python3 dgauss_cdt.py            # write src/dgauss_table.rs
#   python3 dgauss_cdt.py --check    # fail unless src/dgauss_table.rs is up to date
#
# Plain Python 3 with mpmath (see requirements.txt).
#
# The sampler draws x from the discrete Gaussian D_sigma over Z, with probabilities
# proportional to exp(-x^2 / (2 sigma^2)) (the paper's convention, prelim.tex), by
# inversion: it draws |x| from a cumulative distribution table (CDT) and a uniform sign.
# For k = 1, ..., K, the table holds the threshold
#     T_k = floor(2^B * P[|X| < k]),
# and for a uniform u in [0, 2^B) the sampler outputs |x| = #{k : u >= T_k}.  So
# |x| = j < K has probability (T_{j+1} - T_j) / 2^B (with T_0 = 0), and |x| = K takes
# the rest, including the tail beyond K.  A uniform sign is applied to |x| > 0.
#
# The statistical distance between the sampler's output and D_sigma equals the one
# between the two laws of |x|, since both are symmetric and the sign is uniform.  It is
# computed exactly from the table (up to the working precision, see PREC).  Over one
# round at the reference parameters (rho clients, each sampling a key and mu error
# polynomials of n coefficients, so rho (mu + 1) n samples), it adds up to at most
# (number of samples) * (distance per sample).
#
# The budget: the proof of addition privacy loses 2(rho - 1)(mu + 1) eps to the
# smoothing parameter eta_eps, and generate_params.py chooses sigma so that this is at
# most 2^-lambda.  Since sigma is rounded up to a multiple of 1/100, the loss at the
# actual sigma is smaller, and the sampler's distance per round has to fit into what is
# left: smoothing loss + sampler distance <= 2^-lambda.  The script checks that.
#
# K is the smallest value with tail mass P[|X| > K] < 2^-tail_bits, with tail_bits
# chosen so that the tail costs at most 2^-(lambda + 2) per round, and B = 192 makes
# the rounding of the thresholds negligible next to it.
#
# The kernel in src/dgauss.rs compares the B = 192-bit numbers as three 64-bit words
# (hi, mid, lo).  It relies on the structure checked in tiers() below: the first
# entries have distinct high words below 2^64 - 1, the next ones have high word
# 2^64 - 1 and distinct middle words below 2^64 - 1, and the rest have both equal to
# 2^64 - 1.

import argparse
import json
import math
import os
import sys

from mpmath import mp, mpf, exp, floor, log, pi

B = 192                 # bits per threshold
PREC = 1000             # working precision in bits; errors are far below 2^-(PREC - B - 64)
SUM_RANGE = 2000        # sum the Gaussian weights over |x| <= SUM_RANGE (the rest is below 2^-28000 for sigma < 10)
MAX = 2 ** 64 - 1

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "reference.json")
OUT = os.path.join(HERE, "..", "..", "src", "dgauss_table.rs")


def read_reference(path):
  with open(path) as f:
    r = json.load(f)
  sigma = r["kahe"]["sigma"]
  # sigma is chosen in steps of 1/100 (generate_params.py: kahe_sigma_100)
  sigma_100 = round(sigma * 100)
  assert sigma_100 / 100 == sigma, f"sigma = {sigma} is not a multiple of 1/100"
  return {
    "sigma_100": sigma_100,
    "rho": r["setting"]["rho"],
    "n": r["setting"]["n"],
    "mu": r["kahe"]["mu"],
    "secpar": r["setting"]["secpar"],
  }


def law_of_abs(sigma):
  """P[|X| = k] for k = 0, 1, ..., SUM_RANGE under D_sigma."""
  w = [exp(-mpf(k) ** 2 / (2 * sigma * sigma)) for k in range(SUM_RANGE + 1)]
  total = w[0] + 2 * sum(w[1:])
  return [w[0] / total] + [2 * x / total for x in w[1:]]


def smoothing_loss(sigma, ref):
  """The privacy proof's loss 2(rho - 1)(mu + 1) eps for the smallest eps that sigma
  allows, by the bound generate_params.py uses (kahe_sigma_100):
  sigma >= (2/sqrt(pi)) eta_eps(Z^n) with eta_eps(Z^n) <= sqrt(ln(2n(1 + 1/eps))/pi),
  so 1/eps <= exp(pi^2 sigma^2 / 4) / (2n) - 1."""
  inv_eps = exp(pi ** 2 * sigma ** 2 / 4) / (2 * ref["n"]) - 1
  return 2 * (ref["rho"] - 1) * (ref["mu"] + 1) / inv_eps


def build(ref):
  mp.prec = PREC
  sigma = mpf(ref["sigma_100"]) / 100
  samples = ref["rho"] * (ref["mu"] + 1) * ref["n"]
  log_samples = math.log2(samples)
  tail_bits = math.ceil(ref["secpar"] + log_samples) + 2
  p = law_of_abs(sigma)

  thresholds = []
  cum = p[0]                       # P[|X| < 1]
  k = 1
  while 1 - cum >= mpf(2) ** -tail_bits:
    x = cum * mpf(2) ** B
    t = int(floor(x))
    # floor() must be exact: x may not be within 2^-64 of an integer
    frac = x - t
    assert mpf(2) ** -64 < frac < 1 - mpf(2) ** -64, f"threshold {k} too close to an integer"
    thresholds.append(t)
    cum += p[k]                    # P[|X| < k + 1]
    k += 1
  K = len(thresholds)

  # the sampler's law of |x|
  bounds = [0] + thresholds + [2 ** B]
  q = [mpf(bounds[i + 1] - bounds[i]) / mpf(2) ** B for i in range(K + 1)]
  sd = (sum(abs(q[i] - p[i]) for i in range(K + 1)) + sum(p[K + 1:])) / 2
  sd_round = sd * samples
  smoothing = smoothing_loss(sigma, ref)
  total = smoothing + sd_round
  assert total <= mpf(2) ** -ref["secpar"], (
    f"smoothing loss 2^{float(log(smoothing, 2)):.2f} + sampler 2^{float(log(sd_round, 2)):.2f} "
    f"exceeds 2^-{ref['secpar']}")
  return {
    "sigma_100": ref["sigma_100"],
    "thresholds": thresholds,
    "tail_bits": tail_bits,
    "log2_sd": float(log(sd, 2)),
    "log2_samples": log_samples,
    "log2_sd_round": float(log(sd_round, 2)),
    "log2_smoothing": float(log(smoothing, 2)),
    "log2_total": float(log(total, 2)),
    "secpar": ref["secpar"],
  }


def tiers(thresholds):
  """The table as words, and (K0, K1) with the structure the kernel relies on (see
  the header)."""
  w = [((t >> 128) & MAX, (t >> 64) & MAX, t & MAX) for t in thresholds]
  assert all(a < b for a, b in zip(thresholds, thresholds[1:])), "not strictly increasing"
  k0 = next((i for i, x in enumerate(w) if x[0] == MAX), None)
  k1 = next((i for i, x in enumerate(w) if x[0] == MAX and x[1] == MAX), None)
  assert k0 is not None and k1 is not None, (
    "the table has no entries with high (and middle) word 2^64 - 1; the kernels need all three tiers")
  assert all(x[0] == MAX for x in w[k0:]) and all(x[1] == MAX for x in w[k1:])
  assert all(x[0] < MAX for x in w[:k0]) and all(x[1] < MAX for x in w[k0:k1])
  assert len({x[0] for x in w[:k0]}) == k0, "high words of tier 1 not distinct"
  assert len({x[1] for x in w[k0:k1]}) == k1 - k0, "middle words of tier 2 not distinct"
  return w, k0, k1


def render(t, w, k0, k1):
  k = len(w)
  sigma = t["sigma_100"] / 100
  lines = [
    "// Generated by scripts/params/dgauss_cdt.py from scripts/params/reference.json.",
    "// Do not edit; run the script instead.",
    "//",
    "// Cumulative distribution table of |x| for the discrete Gaussian D_sigma with",
    f"// sigma = {sigma:.2f}, probabilities proportional to exp(-x^2 / (2 sigma^2)).",
    f"// Entry k - 1 is floor(2^{B} * P[|X| < k]) for k = 1, ..., {k}, as words [hi, mid, lo].",
    f"// The cut tail P[|X| > {k}] is below 2^-{t['tail_bits']}. The statistical distance from",
    f"// D_sigma is 2^{t['log2_sd']:.1f} per sample and 2^{t['log2_sd_round']:.1f} for the 2^{t['log2_samples']:.1f} samples",
    "// of one round at the reference parameters. Together with the smoothing loss of the",
    f"// privacy proof at this sigma, 2^{t['log2_smoothing']:.2f}, that is 2^{t['log2_total']:.2f}, within the budget",
    f"// of 2^-{t['secpar']}.",
    "",
    "/// The parameter `σ` of the discrete Gaussian the table is generated for, with",
    "/// probabilities proportional to `exp(−x²/(2σ²))`.",
    f"pub(crate) const SIGMA: f64 = {sigma:.2f};",
    "/// The number of entries whose high word is below `u64::MAX`. Their high words",
    "/// are distinct.",
    f"pub(crate) const TIER1: usize = {k0};",
    "/// The index of the first entry whose high and middle words are both `u64::MAX`.",
    "/// The entries from `TIER1` up to here have high word `u64::MAX` and distinct",
    "/// middle words below it.",
    f"pub(crate) const TIER2: usize = {k1};",
    "",
    f"pub(crate) const TABLE: [[u64; 3]; {k}] = [",
  ]
  for x in w:
    lines.append(f"    [{x[0]:#018x}, {x[1]:#018x}, {x[2]:#018x}],")
  lines.append("];")
  return "\n".join(lines) + "\n"


def main():
  ap = argparse.ArgumentParser(description="Generate the discrete Gaussian sampler's table.")
  ap.add_argument("--reference", default=REFERENCE, help="parameter set to read sigma and the round size from")
  ap.add_argument("--out", default=OUT, help="Rust file to write")
  ap.add_argument("--check", action="store_true", help="only check that --out is up to date")
  args = ap.parse_args()

  t = build(read_reference(args.reference))
  w, k0, k1 = tiers(t["thresholds"])
  text = render(t, w, k0, k1)
  summary = (f"sigma = {t['sigma_100'] / 100:.2f}: {len(w)} entries (tiers {k0}, {k1 - k0}, {len(w) - k1}), "
             f"SD 2^{t['log2_sd']:.1f} per sample, 2^{t['log2_sd_round']:.1f} per round; "
             f"with the smoothing loss 2^{t['log2_smoothing']:.2f}: 2^{t['log2_total']:.2f} <= 2^-{t['secpar']}")
  if args.check:
    with open(args.out) as f:
      if f.read() != text:
        sys.exit(f"{args.out} is out of date; run {os.path.basename(__file__)}")
    print("up to date: " + summary)
  else:
    with open(args.out, "w") as f:
      f.write(text)
    print("wrote " + os.path.normpath(args.out) + ": " + summary)


if __name__ == "__main__":
  main()
