# Parameter generation for Panetiere, following the bounds stated in the paper.
# Written by Claude (Anthropic).
#
# Usage (all options have defaults, see --help):
#   python3 generate_params.py --rho 300 --gamma 8 --n 2048 --msg-bytes 20480 \
#        --json params.json
#   python3 generate_params.py --check-only params.json      # only check a parameter set
#
# Plain Python 3 with mpmath, sympy and tabulate (pip install mpmath sympy tabulate).
# The IBLT shape search imports iblt_common.py (and, for --empirical, iblt_empirical.py)
# from the same directory; numpy makes it faster (optional).
# The search only uses the built-in hardness estimates.  Sage is only needed for
# --confirm-with-estimator, which re-checks the final parameters with the lattice
# estimator: run as  sage -python generate_params.py --confirm-with-estimator ...
# with the lattice estimator (https://github.com/malb/lattice-estimator) on PYTHONPATH.
#
# Tags such as thm:broadcast are labels in the paper's LaTeX source; README.md maps
# them to the theorem and section numbers of the current draft.
#
# Every constraint enforced below is tagged with the statement it comes from:
#   [KAHE]   Theorem thm:kahe_secure (Lemmas lem:kahe_add_correct, lem:kahe_add_private;
#            subset-sum bound T_sigma from def:subsetsumbound, lem:subsetsumtail)
#   [MSE]    Theorem thm:multisetencoding (additive multiset encoding from IBLTs, --mse IBLT)
#   [NG]     Theorem thm:mse_newton (multiset encoding from Newton-Girard identities, --mse NG)
#   [LEAF]   Theorem thm:leafcom (Lemmas lem:leafcomcorrect, lem:leafcombinding, lem:ajtaihiding)
#   [AVC]    Chipmunk vector commitment theorem (lem:chipmunkcorrect, lem:avcposbind)
#   [HIDE]   Theorem on the hiding vector commitment (lem:hidevcom_addition_hiding)
#   [ABC]    Theorem thm:broadcast (anonymous broadcast with guaranteed output delivery
#            in the random oracle model).  Every MSE element is (s, m XOR RO(s)) with a
#            nonce s in N = {0,1}^l and the masked message in M = {0,1}^|m|, both chunked
#            like messages (b bits per element of Z_t, omega*b bits per element of F for
#            NG), so RO only outputs bits.  The theorem requires
#            rho (Q + rho) / |N| <= 2^-lambda, where Q bounds the adversary's RO queries.
#
# Notation follows the paper: theta is the number of IBLT rows (hash functions), gamma
# the number of servers, and omega the length of the IBLT keys (elements of Z_t^omega).
# Both multiset encodings work over the field Z_t: the plaintext modulus t is the
# smallest prime above 2^b with t = 1 mod 2^k (k = --t-ntt-log, default 20; 0 drops the
# condition), with t > C (MSE) and q not dividing t (KAHE).  The congruence is not
# needed by the paper; it lets an implementation multiply polynomials over Z_t with an
# NTT (root finding for NG) and build F_{t^omega} as Z_t[Y]/(Y^omega - c) for omega a
# power of two up to 2^k.  With simple chunking every coordinate in Z_t carries b bits
# (losing log2(t/2^b) bits, about 0.01 for b = 31 and k = 20, < 1e-8 without the
# congruence); for NG, an element of F = F_{t^omega} carries omega*b bits.
#
# Structure:
#   * The search (find_*) may use floating point, but every bound involving ln or
#     sqrt is evaluated with rigorous interval arithmetic (mpmath.iv), so the chosen
#     values provably satisfy it.
#   * check_all() re-derives every condition of the paper directly from the final
#     parameters, independently of the search code, using exact integer comparisons
#     (interval arithmetic only to enclose ln, sqrt, pi and the series defining K_sigma).
#     The only exception is the minimiser tau of the subset-sum bound T_sigma
#     (def:subsetsumbound), which check_all() searches for again; it is only a
#     certificate, since f(tau) >= T_sigma for every tau > 0, so the check does not
#     depend on the search being accurate.
#
# IBLT shape: by default the shape with the fewest cells satisfying the union bound of
# [MSE] (provable).  With --empirical, the shape is instead chosen by simulation with
# iblt_empirical.py (next to this script), and used only if it has fewer cells than the
# provable one.  An empirical shape is not covered by the paper's bound: check_all() then
# reports the stopping-set condition of [MSE] as EMPIRICAL (by choice); all other
# conditions are checked as usual.
#
# --check-only FILE does not search: it reads a parameter set from a JSON file (as written
# by --json and described by params.schema.json next to this script), runs all checks on
# it (with --empirical, an IBLT shape that misses the provable bound is checked by
# simulation instead; with --confirm-with-estimator, the hardness assumptions are also
# confirmed with the lattice estimator), and exits with status 1 if any check fails, or
# with status 2 if the file does not match the schema.  It needs the jsonschema package
# (pip install jsonschema); writing JSON does not.
#
# Cover traffic (sec:cover, --capacity C): rho is the number of clients and C <= rho the
# capacity of the MSE, i.e. the maximum number of senders.  All aggregation bounds (KAHE,
# leaf commitment, Chipmunk, recipient download) are with respect to rho; the encoding
# (its size, the IBLT shape, omega, t > C) only depends on C.  Guaranteed output delivery
# then only holds against at most C - h corrupted clients, h the number of honest senders.
# Without --capacity, C = rho.  The nonce length stays sized for rho (conservative).
#
# All logarithms are natural logarithms unless written log2.

import argparse
import contextlib
import dataclasses
import datetime
import json
import functools
import math
import operator
import os
import secrets
import sys
import time
from dataclasses import dataclass
from typing import Optional

# The estimator (if present) must be imported before mpmath: it loads Sage, and mpmath
# picks its integer backend at import time.  Imported the other way round, mpmath uses
# gmpy2 directly and Sage cannot convert the resulting gmpy2.mpz values.
try:                      # only needed for --confirm-with-estimator
  from estimator import LWE, SIS, ND
  HAVE_ESTIMATOR = True
except Exception:         # ImportError, or the estimator failing to import without Sage
  HAVE_ESTIMATOR = False

import mpmath
from mpmath import iv
from sympy import isprime as is_prime, nextprime as next_prime
from tabulate import tabulate

# The IBLT shape search (and the simulation for --empirical) lives in the iblt_*.py
# scripts next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iblt_common  # noqa: E402


iv.dps = 60
LN2 = math.log(2)

# Progress output.  With --log-search every hardness estimate made during the search
# is printed together with its result and duration.
LOG_SEARCH = False
_T0 = time.time()


def progress(msg):
  print("[%7.1fs] %s" % (time.time() - _T0, msg), flush=True)


# ANSI colours, only when writing to a terminal (and NO_COLOR is not set)
_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def red(text):
  return "\033[31m%s\033[0m" % text if _COLOR else text


def green(text):
  return "\033[32m%s\033[0m" % text if _COLOR else text


def yellow(text):
  return "\033[33m%s\033[0m" % text if _COLOR else text


def color_status(status):
  """'ok' in green, 'EMPIRICAL' (an IBLT shape chosen empirically on purpose) in yellow,
  any other status in red (tabulate ignores the escape codes when computing column
  widths)."""
  if status == "ok":
    return green(status)
  return yellow(status) if status == "EMPIRICAL" else red(status)


def log2(x):
  # Sage integers are not Python ints, and math.log converts them to float first,
  # which overflows to inf for huge values.  Convert integers to Python ints.
  try:
    x = int(operator.index(x))
  except TypeError:
    pass
  return math.log(x) / LN2


def bits_mod(q):
  """Bits needed for an element of Z_q, i.e. ceil(log2 q)."""
  return int(q - 1).bit_length()


def bits_range(bound):
  """Bits needed for an integer in [-bound, bound], i.e. ceil(log2(2 bound + 1))."""
  return int(2 * bound).bit_length()


# ---------------------------------------------------------------------------
# Rigorous real arithmetic (interval enclosures of ln, sqrt, pi)
# ---------------------------------------------------------------------------

def ivi(x):
  """Exact interval for an integer (Sage integers are converted to Python ints).
  (Not called I, which Sage predefines as the imaginary unit.)"""
  return iv.mpf(int(x))


def _upper_floor_ceil(x):
  """(floor, ceil) of the upper end of the interval x, computed exactly from its binary
  representation.  (mpmath.floor/ceil and int-vs-mpf comparisons go through the global
  53-bit working precision, which is wrong for numbers above 2^53.)"""
  sign, man, exp, _ = x._mpi_[1]          # raw mpf: (-1)^sign * man * 2^exp
  man, exp = int(man), int(exp)
  if sign:
    man = -man
  if exp >= 0:
    z = man << exp
    return z, z
  fl = man >> -exp                         # Python's >> floors, also for negative man
  return fl, fl + (0 if fl << -exp == man else 1)


def iv_ceil(x):
  """Smallest integer >= every point of the interval x."""
  return _upper_floor_ceil(x)[1]


def iv_floor_upper(x):
  """floor of the upper end of x; an integer z with z + 1 > every point of x."""
  return _upper_floor_ceil(x)[0]


def int_gt(z, x):
  """Integer z is strictly larger than every point of the interval x."""
  return int(z) > _upper_floor_ceil(x)[0]


def int_ge(z, x):
  """Integer z is at least every point of the interval x."""
  return int(z) >= _upper_floor_ceil(x)[1]


def iv_ge(x, y):
  """Every point of the interval x is at least every point of the interval y (exact)."""
  return mpmath.libmp.mpf_cmp(x._mpi_[0], y._mpi_[1]) >= 0


# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Setting:
  rho: int            # number of clients
  gamma: int          # number of servers
  n: int              # ring dimension
  secpar: int         # lambda
  msg_bytes: int      # message size
  eps_kahe: int       # KAHE decryption failure 2^-eps_kahe
  eps_mse: int        # MSE decoding failure 2^-eps_mse (+ PRF advantage for IBLT)
  max_q_bits: int = 64  # upper limit on all moduli (search only)
  t_ntt_log: int = 20   # t = 1 mod 2^t_ntt_log (search only; 0: smallest prime above 2^b)
  log_queries: int = 64  # [ABC] log2 Q, bound on the adversary's random-oracle queries
  mse: str = "NG"     # multiset encoding: "IBLT" or "NG"
  # IBLT shape: False = provable (default), True = by simulation (--empirical)
  empirical: bool = False
  sim_alpha: float = 0.05             # confidence 1 - alpha of the simulation tests
  sim_trials_factor: float = 10.0     # trials per tested shape = factor * 2^eps
  sim_resolution: float = 0.01        # relative resolution of the delta search
  sim_workers: int = 0                # simulation processes (0: number of CPUs)
  seed: int = 1                       # simulation seed (random unless --seed is set)
  # capacity C <= rho of the MSE (sec:cover); None: C = rho.  With cover traffic, the
  # aggregation bounds stay with respect to all rho clients, the encoding only has to
  # hold the elements of at most C senders.
  capacity: Optional[int] = None

  @property
  def cap(self):
    """Capacity C of the MSE (sec:cover); rho unless set."""
    return self.rho if self.capacity is None else self.capacity

  @property
  def msg_bits(self):
    return 8 * self.msg_bytes

  @property
  def tau(self):
    """Depth of the Chipmunk tree: smallest tau with 2^tau >= gamma."""
    return int(self.gamma - 1).bit_length()

  @property
  def secpar_hide(self):
    """[HIDE] the hiding VC loses a factor 2^tau >= gamma over the leaf commitment."""
    return self.secpar + self.tau


@dataclass(frozen=True)
class MseParams:
  """IBLT multiset encoding (thm:multisetencoding)."""
  theta: int        # number of hash functions (rows)
  delta: int          # cells per row
  omega: int          # key length (elements of Z_t)
  xi: int             # element length (elements of Z_t): xi_nonce nonce + rest message chunks
  chunk_bits: int     # message bits per element of Z_t
  empirical: object = None  # with --empirical: dict describing the empirical shape search
  xi_nonce: int = 0   # [ABC] nonce chunks (elements of Z_t), N = {0,1}^(xi_nonce * chunk_bits)

  @property
  def cells(self):
    return self.theta * self.delta

  @property
  def elems(self):
    """Elements of Z_t in an encoding: counter, key and value per cell."""
    return self.cells * (1 + self.omega + self.xi)

  kind = "IBLT"


@dataclass(frozen=True)
class NgMseParams:
  """Newton-Girard multiset encoding over F = F_{t^omega} (thm:mse_newton)."""
  rho: int            # capacity C of the encoding (sec:cover; = setting.rho by default)
  omega: int          # extension degree; keys and chunks are elements of F
  xi: int             # element length (elements of F): xi_nonce nonce + rest message chunks
  chunk_bits: int     # message bits per coordinate in Z_t (simple chunking)
  xi_nonce: int = 0   # [ABC] nonce chunks (elements of F), N = {0,1}^(xi_nonce * omega * chunk_bits)

  @property
  def elems(self):
    """Elements of Z_t in an encoding: C key power sums and C xi chunk sums, each an
    element of F = Z_t^omega."""
    return self.omega * self.rho * (1 + self.xi)

  kind = "NG"


@dataclass(frozen=True)
class KaheParams:
  t: int
  q: int
  mu: int
  sigma_100: int      # sigma = sigma_100 / 100
  nu: int
  mse: object         # MseParams or NgMseParams

  @property
  def sigma(self):
    return int(self.sigma_100) / float(100)

  @property
  def b(self):
    return int(self.t).bit_length() - 1


@dataclass(frozen=True)
class LeafParams:
  p: int              # modulus q_leaf = p
  beta: int
  beta_agg: int
  kappa: int
  mu: int


@dataclass(frozen=True)
class ChipmunkParams:
  q: int
  eta: int
  kappa: int          # limbs per element of R_q
  kappa_prime: int    # limbs per element of R_p
  beta_agg: int
  tau: int
  xi: int             # leaf length = leaf.mu + 1


# ---------------------------------------------------------------------------
# Sizes (in bits).  Every size used anywhere in the script is computed here.
# ---------------------------------------------------------------------------

def kahe_ct_bits(n, mu, q):
  """KAHE ciphertext: mu elements of R_q."""
  return mu * n * bits_mod(q)


def leaf_opening_bits(n, kappa, bound):
  """Leaf commitment randomness: kappa ring elements with coefficients in [-bound, bound]."""
  return kappa * n * bits_range(bound)


def chipmunk_opening_elems(tau, kappa, kappa_prime, xi):
  """Ring elements in a Chipmunk opening.  For each of the tau decomposed path elements
  one limb can be dropped, since it is determined by the others and the hash of the
  children: tau (2 kappa - 1) + xi kappa'."""
  return tau * (2 * kappa - 1) + xi * kappa_prime


def chipmunk_opening_bits(n, tau, kappa, kappa_prime, xi, bound):
  return chipmunk_opening_elems(tau, kappa, kappa_prime, xi) * n * bits_range(bound)


def vc_com_bits(n, q):
  """Chipmunk commitment: one element of R_q."""
  return n * bits_mod(q)


@dataclass(frozen=True)
class Sizes:
  ct: int             # KAHE ciphertext
  com: int            # vector commitment
  opening: int        # one client's opening for one server (Chipmunk + leaf)
  agg_opening: int    # one server's aggregated opening
  client_up: int      # ct + com + gamma openings (PKE overhead excluded)
  server_up: int
  recipient_down: int # rho (ct + com) + gamma aggregated openings


def compute_sizes(s, kahe, leaf, avc):
  ct = kahe_ct_bits(s.n, kahe.mu, kahe.q)
  com = vc_com_bits(s.n, avc.q)
  op = (chipmunk_opening_bits(s.n, avc.tau, avc.kappa, avc.kappa_prime, avc.xi, avc.eta)
        + leaf_opening_bits(s.n, leaf.kappa, leaf.beta))
  agg = (chipmunk_opening_bits(s.n, avc.tau, avc.kappa, avc.kappa_prime, avc.xi, avc.beta_agg)
         + leaf_opening_bits(s.n, leaf.kappa, leaf.beta_agg))
  return Sizes(ct=ct, com=com, opening=op, agg_opening=agg,
               client_up=ct + com + s.gamma * op, server_up=agg,
               recipient_down=s.rho * (ct + com) + s.gamma * agg)


def fmt_bytes(bits):
  """Human-readable size (decimal units) of a bit count."""
  b = int(bits) / 8.0
  for unit, scale in [("GB", 1e9), ("MB", 1e6), ("KB", 1e3)]:
    if b >= scale:
      return "%.2f %s" % (b / scale, unit)
  return "%d B" % int(math.ceil(b))


# ---------------------------------------------------------------------------
# Primes
# ---------------------------------------------------------------------------

def find_minimally_splitting_prime(min_q):
  """Smallest prime q > min_q with q = 5 (mod 8).  [LEAF] requires this so that
  X^n+1 splits into two irreducible factors mod q and short elements are units."""
  q = min_q + 1
  q += (5 - q) % 8          # first candidate > min_q with q = 5 (mod 8)
  while not is_prime(q):
    q += 8
  return q


def find_ntt_friendly_prime(n, min_q):
  """Smallest prime q > min_q with q = 1 (mod 2n), so that X^n + 1 splits
  completely mod q and negacyclic NTT-based multiplication is available.  The
  paper only needs q prime; NTT-friendliness is an implementation choice."""
  m = 2 * n
  q = min_q + 1
  q += (1 - q) % m          # first candidate > min_q with q = 1 (mod 2n)
  while not is_prime(q):
    q += m
  return q


# ---------------------------------------------------------------------------
# Hardness estimates (lattice estimator)
# ---------------------------------------------------------------------------

def _min_rop(results):
  return min((math.log2(float(r["rop"])) for r in results.values()), default=None)


def _estimator_failure_reason(estimate, params):
  try:
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
      estimate(params, catch_exceptions=False)
    return "unknown"
  except Exception as e:
    return "%s: %s" % (type(e).__name__, e)


# Ring-LWE: core-SVP estimate of the primal (uSVP) attack, a port of the lattice
# estimator's PrimalUSVP.cost_gsa as used by LWE.estimate.rough (ADPS16 cost model,
# 2^(0.292 b), unlimited samples, GSA).  Secret and error have the same width, so no
# rescaling is needed.  Matches the estimator's rough estimate on all instances we
# ran it on (n = 2048 and 4096, log q from 37 to 64, stddev 3.35) to 0.1 bits; the
# estimator's dual-hybrid attack was never cheaper there.

def _usvp_dim(n, logq, stddev, m, beta):
  """Smallest embedding dimension d in [n, m] satisfying the uSVP condition, else m
  (estimator: PrimalUSVP._solve_for_d with tau = stddev, xi = 1)."""
  dl = math.log(_delta_bkz(beta))
  a = -dl
  C = math.log(stddev ** 2 * (beta - 1) + stddev ** 2) / 2
  b = dl * (2 * beta - 1) + logq - C
  c = math.log(stddev) - (n + 1) * logq
  if a * n * n + b * n + c >= 0:
    return n
  disc = b * b - 4 * a * c
  if disc < 0:
    return m
  d1 = (-b + math.sqrt(disc)) / (2 * a)
  d2 = (-b - math.sqrt(disc)) / (2 * a)
  if a > 0:
    return min(m, math.ceil(d2))
  if n <= d1:
    return min(m, math.ceil(d1))
  if n > d2:
    return m
  return min(m, math.ceil(d2))


def _usvp_succeeds(n, logq, stddev, beta):
  dl = math.log(_delta_bkz(beta))
  m = 2 * math.ceil(math.sqrt(n * logq / dl))
  d = _usvp_dim(n, logq, stddev, m, beta)
  if d == beta and d < m:
    d += 1
  lhs = math.log(math.sqrt(stddev ** 2 * (beta - 1) + stddev ** 2))
  rhs = dl * (2 * beta - d - 1) + (math.log(stddev) + logq * (d - n - 1)) / d
  return lhs <= rhs


def lwe_usvp_core_svp(n, q, stddev):
  """(log2 cost, beta) of the primal uSVP attack: smallest BKZ block size that works."""
  n, logq, stddev = int(n), math.log(int(q)), float(stddev)
  lo, hi = 40, 2 * n
  if not _usvp_succeeds(n, logq, stddev, hi):
    return float('inf'), None
  while lo < hi:                         # the condition is monotone in beta
    mid = (lo + hi) // 2
    if _usvp_succeeds(n, logq, stddev, mid):
      hi = mid
    else:
      lo = mid + 1
  return _C_SVP * lo, lo


def rlweIsHard(n, q, stddev, secpar):
  """Ring-LWE in dimension n, modulus q, secret and error with the given standard
  deviation, is at least 2^secpar hard (built-in primal estimate).  The number of
  samples is left unbounded, which is conservative."""
  t0 = time.time()
  cost, beta = lwe_usvp_core_svp(n, q, stddev)
  if LOG_SEARCH:
    progress("  RLWE n=%d q=%d stddev=%.2f: 2^%.1f (core-SVP primal, b=%s)  (%.2fs)"
             % (n, q, stddev, cost, beta, time.time() - t0))
  return secpar < cost


def _sis_l2_cost(beta, q, n, m):
  """Cost of Euclidean Ring-SIS with bound ceil(beta sqrt(m n)), or None."""
  sq = int(beta) ** 2 * int(m) * int(n)
  beta2 = math.isqrt(sq)
  if beta2 * beta2 < sq:
    beta2 += 1
  if 2 * beta2 >= q - 1:
    return beta2, None
  params = SIS.Parameters(n=n, m=m * n, q=q, length_bound=beta2, norm=2, tag="RingSIS-l2")
  with contextlib.redirect_stdout(open(os.devnull, 'w')):
    return beta2, _min_rop(SIS.estimate.rough(params))


# Infinity-norm SIS: core-SVP estimate following the MSIS analysis of Dilithium
# (Ducas et al., CHES 2018, Sec. 6.2, and its published security scripts).  BKZ-b on a
# w-column sub-lattice of the q-ary SIS lattice (h = n rows) produces a vector whose
# nonzero coordinates are modelled as Gaussian over the region where the GSA profile is
# above 1; the attack succeeds if all of them lie in [-beta, beta], repeated as often
# as needed using the 2^(0.2075 b) vectors a sieve outputs.  Cost 2^(0.292 b) per BKZ
# call (classical core-SVP), minimised over b and w.  This reproduces the lattice
# estimator's infinity-norm estimate (identical to 0.1 bits on our instances) in a
# fraction of a second and, unlike the estimator, does not overflow for large m n.

# Python floats throughout (Sage would otherwise turn the literals into slow MPFR numbers)
_C_SVP = float('0.292')      # log2 cost of sieving in dimension b, per b (classical core-SVP)
_C_NVEC = float('0.2075')    # log2 number of short vectors a sieve outputs, per b


def _delta_bkz(b):
  """Root Hermite factor of BKZ-b."""
  b = float(b)
  return ((math.pi * b) ** (1 / b) * b / (2 * math.pi * math.e)) ** (1 / (2 * b - 2))


def _gsa_top(q, h, w, b):
  """Closed form of the randomized BKZ-b profile of a q-ary lattice with h q-vectors in
  dimension w: (log of the first Gram-Schmidt norm, length of the region above 1)."""
  s = 2 * math.log(_delta_bkz(b))
  glv = float(h) * math.log(float(q))
  k = int(math.floor((-1 + math.sqrt(1 + 8 * glv / s)) / 2))   # max k: s k(k+1)/2 <= glv
  k = max(1, min(k, w))
  return k * s + (glv - s * k * (k + 1) / 2) / k, k


def _sis_linf_attack_cost(q, w, h, beta, b):
  l0, d = _gsa_top(q, h, w, b)
  sigma = math.exp(l0) / math.sqrt(float(d))
  p_coord = math.erf(float(beta) / sigma / math.sqrt(float(2)))
  if p_coord <= 0:
    return float('inf')
  log2_eps = float(d) * math.log2(p_coord)
  return _C_SVP * b + max(float(0), -log2_eps - _C_NVEC * b)


def sis_linf_core_svp(q, n, m, beta, step_b=5, n_w=400):
  """(log2 cost, w, b) of the cheapest attack on SIS_{q, n x (m n), beta} (infinity norm)."""
  q, n, m, beta = int(q), int(n), int(m), int(beta)
  h, max_w = n, m * n
  best, arg = float('inf'), None
  step_w = max(5, (max_w - h) // n_w)
  for b in range(50, max_w + 1, step_b):
    if _C_SVP * b > best:
      break
    for w in range(max_w, h, -step_w):
      if b <= w:
        c = _sis_linf_attack_cost(q, w, h, beta, b)
        if c < best:
          best, arg = c, (w, b)
  if arg is not None:            # refine around the coarse optimum
    w0, b0 = arg
    for b in range(max(50, b0 - step_b), b0 + step_b + 1):
      for w in range(min(max_w, w0 + step_w), max(h, w0 - step_w), -1):
        if b <= w:
          c = _sis_linf_attack_cost(q, w, h, beta, b)
          if c < best:
            best, arg = c, (w, b)
  return (best,) + (arg if arg else (None, None))


def sis_trivially_broken(beta, q, n, secpar):
  """Attacks the estimator does not model: beta >= q/2, or the trivial attack (choose
  all but n coordinates, i.e. one ring element, freely and solve for the rest; its n
  coefficients are uniform in Z_q and all lie in [-beta, beta] with probability
  ((2 beta + 1)/q)^n)."""
  return 2 * beta >= q - 1 or n * log2(q / (2 * beta + 1)) < secpar


def rsisIsHard(beta, q, n, m, secpar):
  """Ring-SIS_{R,q,m,beta} (infinity norm, m ring elements) is at least 2^secpar hard
  (trivial attacks, then the built-in core-SVP estimate above)."""
  if sis_trivially_broken(beta, q, n, secpar):
    return False
  t0 = time.time()
  cost, w, b = sis_linf_core_svp(q, n, m, beta)
  if LOG_SEARCH:
    progress("  SIS  n=%d m=%d q=%d beta=%d: 2^%.1f (core-SVP, w=%s, b=%s)  (%.2fs)"
             % (n, m, q, beta, cost, w, b, time.time() - t0))
  return secpar < cost


# ---------------------------------------------------------------------------
# Confirmation with the lattice estimator (--confirm-with-estimator)
# ---------------------------------------------------------------------------

def hardness_instances(s, kahe, leaf, avc):
  """Every lattice problem check_all requires to be 2^secpar hard."""
  return [
    ("KAHE", "RLWE_{R,q,D_{sigma/2}}", "lwe", (s.n, kahe.q, kahe.sigma / 2)),
    ("LEAF", "SIS_{R,p,kappa,2 beta_agg}", "sis", (2 * leaf.beta_agg, leaf.p, s.n, leaf.kappa)),
    ("AVC", "SIS_{R,q,2 kappa,2 beta_agg}", "sis", (2 * avc.beta_agg, avc.q, s.n, 2 * avc.kappa)),
    ("AVC", "SIS_{R,q,xi kappa',2 beta_agg}", "sis",
     (2 * avc.beta_agg, avc.q, s.n, avc.xi * avc.kappa_prime)),
  ]


def estimator_lwe(n, q, stddev):
  """(log2 cost, note) from LWE.estimate.rough; cost None if the estimator fails."""
  params = LWE.Parameters(n=n, q=q, Xs=ND.DiscreteGaussian(stddev),
                          Xe=ND.DiscreteGaussian(stddev), tag="KAHE-RLWE")
  with contextlib.redirect_stdout(open(os.devnull, 'w')):
    cost = _min_rop(LWE.estimate.rough(params))
  if cost is None:
    return None, "estimator failed (%s)" % _estimator_failure_reason(LWE.estimate.rough, params)
  return cost, "rough"


def estimator_sis(beta, q, n, m, secpar):
  """(log2 cost, note): the Euclidean bound beta sqrt(m n) if that alone is conclusive
  (sufficient), else the estimator's infinity-norm estimate; cost None on failure."""
  beta2, cost2 = _sis_l2_cost(beta, q, n, m)
  if cost2 is not None and secpar < cost2:
    return cost2, "Euclidean bound %d" % beta2
  from sage.all import oo
  params = SIS.Parameters(n=n, m=m * n, q=q, length_bound=beta, norm=oo, tag="RingSIS")
  with contextlib.redirect_stdout(open(os.devnull, 'w')):
    cost = _min_rop(SIS.estimate.rough(params))
  if cost is None:
    return None, "estimator failed (%s)" % _estimator_failure_reason(SIS.estimate.rough, params)
  return cost, "infinity norm"


def confirm_with_estimator(s, kahe, leaf, avc):
  """Re-check every hardness assumption of the final parameters with the lattice
  estimator.  Returns (rows, all_ok)."""
  rows = []
  for src, what, kind, args in hardness_instances(s, kahe, leaf, avc):
    t0 = time.time()
    if kind == "lwe":
      builtin = lwe_usvp_core_svp(*args)[0]
      cost, note = estimator_lwe(*args)
    else:
      beta, q, n, m = args
      builtin = (float('-inf') if sis_trivially_broken(beta, q, n, s.secpar)
                 else sis_linf_core_svp(q, n, m, beta)[0])
      cost, note = estimator_sis(beta, q, n, m, s.secpar)
    status = "ok" if cost is not None and s.secpar < cost else (
      "FAILS" if cost is not None else "NO RESULT")
    rows.append([src, what, "2^%.1f" % builtin,
                 "-" if cost is None else "2^%.1f" % cost, note, status])
    progress("  %s %s: built-in 2^%.1f, estimator %s  (%.1fs)"
             % (src, what, builtin, "-" if cost is None else "2^%.1f" % cost, time.time() - t0))
  return rows, all(r[5] == "ok" for r in rows)


def print_confirmation(rows):
  rows = [r[:5] + [color_status(r[5])] for r in rows]
  print(tabulate(rows, headers=["source", "problem", "built-in", "estimator", "estimator method",
                                "status"], tablefmt="simple_outline"), "\n")


# ---------------------------------------------------------------------------
# KAHE bounds  [KAHE]  (search side; rigorous upper bounds via interval arithmetic)
# ---------------------------------------------------------------------------

def kahe_smoothing_inverse_eps(secpar, rho, mu):
  """1/eps for the smoothing parameter eta_eps in [KAHE].  The privacy proof
  (lem:kahe_add_private) uses the HintLWE lemma in rho-1 hybrids, each losing
  2(mu+1)*eps, so eps = 2^-secpar / (2 (rho-1) (mu+1)) keeps the total at 2^-secpar."""
  return 2 ** secpar * 2 * (rho - 1) * (mu + 1)


def kahe_sigma_100(n, inv_eps):
  """Smallest sigma = S/100 with sigma >= (2/sqrt(pi)) eta_eps(Z^n), using the standard
  bound eta_eps(Z^n) <= sqrt(ln(2n(1+1/eps))/pi)  (Micciancio-Regev, Lemma 3.3)."""
  eta = iv.sqrt(iv.log(ivi(2 * n * (1 + inv_eps))) / iv.pi)
  return iv_ceil(ivi(100) * (ivi(2) / iv.sqrt(iv.pi)) * eta)


# Subset-sum bound T_sigma (def:subsetsumbound, lem:subsetsumtail).  For sigma > 0,
#     K_sigma(tau) = ln E_{x <- D_sigma}[exp(tau max(x,0))]
#                  = ln( sum_x exp(tau max(x,0) - x^2/(2 sigma^2)) / sum_x exp(-x^2/(2 sigma^2)) ),
#     T_sigma(rho, delta) = min_{tau > 0} f(tau),  f(tau) = (rho K_sigma(tau) + ln(2/delta)) / tau.
# The minimiser tau is found in floating point; it is only a certificate: f(tau) >= T_sigma
# for every tau > 0, and f(tau) is evaluated rigorously (interval arithmetic, with explicit
# bounds on the truncated tails of both series).  So an inaccurate tau only costs tightness.
# tau is rounded to a multiple of 2^-TAU_BITS, so that it is exactly representable.

TAU_BITS = 20
_SUBSET_SUM_TAIL = 40      # truncate the series 40 sigma beyond the peak of their terms


def _kgf_pos_float(sigma, tau):
  """K_sigma(tau) in floating point (log-sum-exp; for the search only)."""
  c = 1.0 / (2.0 * sigma * sigma)
  xmax = int(math.ceil(tau * sigma * sigma + _SUBSET_SUM_TAIL * sigma))
  xmin = int(math.ceil(_SUBSET_SUM_TAIL * sigma))
  num = [tau * max(x, 0) - c * x * x for x in range(-xmin, xmax + 1)]
  den = [-c * x * x for x in range(-xmin, xmin + 1)]

  def lse(v):
    m = max(v)
    return m + math.log(math.fsum(math.exp(e - m) for e in v))
  return lse(num) - lse(den)


def find_subset_sum_tau(rho, sigma_100, ln_two_over_delta):
  """Approximate minimiser of f(tau) = (rho K_sigma(tau) + ln(2/delta)) / tau, rounded to a
  multiple of 2^-TAU_BITS.  f is unimodal (remark after def:subsetsumbound), so golden-
  section search on a bracket (0, hi] with f(hi) >= f(hi/2) finds it."""
  sigma = sigma_100 / 100.0
  L = float(ln_two_over_delta)

  def f(tau):
    return (rho * _kgf_pos_float(sigma, tau) + L) / tau
  hi = 1.0 / sigma
  while f(2 * hi) < f(hi):
    hi *= 2
  lo, hi = 0.0, 2 * hi
  g = (math.sqrt(5) - 1) / 2
  a, b = hi - g * (hi - lo), lo + g * (hi - lo)
  fa, fb = f(a), f(b)
  while hi - lo > 2.0 ** -(TAU_BITS + 2):
    if fa < fb:
      hi, b, fb = b, a, fa
      a = hi - g * (hi - lo)
      fa = f(a)
    else:
      lo, a, fa = a, b, fb
      b = lo + g * (hi - lo)
      fb = f(b)
  return max(1, round((lo + hi) / 2 * 2 ** TAU_BITS)) / 2 ** TAU_BITS


def _iv_series_tail(first, ratio):
  """Interval [0, first / (1 - ratio)] enclosing a series whose first term is at most
  `first` and whose consecutive terms have ratio at most `ratio` < 1."""
  assert ratio.b < 1
  return iv.mpf([0, (first / (ivi(1) - ratio)).b])


def kgf_pos_iv(sigma_100, tau):
  """Rigorous interval enclosing K_sigma(tau) for sigma = sigma_100/100 and the
  (exactly representable) float tau > 0.  Both series are summed exactly up to a cut-off
  X (resp. X+), and the remaining terms are enclosed by a geometric series: for x >= X+1
  the ratio of consecutive terms exp(-(2x+1)/(2 sigma^2)) (resp. times exp(tau)) is at
  most its value at x = X+1."""
  sigma = ivi(sigma_100) / ivi(100)
  c = ivi(1) / (ivi(2) * sigma * sigma)
  T = iv.mpf(tau)
  s = sigma_100 / 100.0
  X = int(math.ceil(_SUBSET_SUM_TAIL * s))
  Xp = int(math.ceil(tau * s * s + _SUBSET_SUM_TAIL * s))

  def g(x):
    return iv.exp(-c * ivi(x * x))

  def h(x):
    return iv.exp(T * ivi(x) - c * ivi(x * x))
  half = sum((g(x) for x in range(1, X + 1)), ivi(0))       # sum_{x=1}^{X} exp(-c x^2)
  tail_g = _iv_series_tail(g(X + 1), iv.exp(-c * ivi(2 * X + 3)))
  tail_h = _iv_series_tail(h(Xp + 1), iv.exp(T - c * ivi(2 * Xp + 3)))
  den = ivi(1) + ivi(2) * (half + tail_g)
  num = (half + tail_g) + sum((h(x) for x in range(0, Xp + 1)), ivi(0)) + tail_h
  return iv.log(num / den)


def subset_sum_f_iv(rho, sigma_100, two_over_delta, tau):
  """Rigorous interval enclosing f(tau) = (rho K_sigma(tau) + ln(2/delta)) / tau, an upper
  bound on T_sigma(rho, delta) (def:subsetsumbound).  2/delta is passed as an integer."""
  return (ivi(rho) * kgf_pos_iv(sigma_100, tau) + iv.log(ivi(two_over_delta))) / iv.mpf(tau)


_SUBSET_SUM_CACHE = {}


def subset_sum_bound(rho, sigma_100, two_over_delta, what=""):
  """(tau, interval f(tau)) with f(tau) >= T_sigma(rho, delta), tau an approximate
  minimiser.  Cached, since the KAHE search evaluates the same bounds repeatedly."""
  key = (rho, sigma_100, two_over_delta)
  if key not in _SUBSET_SUM_CACHE:
    t0 = time.time()
    tau = find_subset_sum_tau(rho, sigma_100, mpmath.log(two_over_delta))
    val = subset_sum_f_iv(rho, sigma_100, two_over_delta, tau)
    _SUBSET_SUM_CACHE[key] = (tau, val)
    if LOG_SEARCH:
      progress("  T_sigma(rho=%d, 2^-%.1f) %s: tau=%.6f, f(tau) <= %.3f  (%.2fs)"
               % (rho, log2(two_over_delta) - 1, what, tau, float(val.b), time.time() - t0))
  return _SUBSET_SUM_CACHE[key]


def kahe_nu(rho, sigma_100, n, secpar):
  """Smallest integer nu >= f(tau) >= T_sigma(rho, 2^-secpar / n)."""
  _, val = subset_sum_bound(rho, sigma_100, 2 ** (secpar + 1) * n, "for nu")
  return iv_ceil(val)


def kahe_q_floor(t, rho, sigma_100, eps_kahe, n, mu):
  """An integer z such that every q > z satisfies q > t (2 f(tau) + rho), hence
  q > t (2 T_sigma(rho, 2^-eps / (n mu)) + rho)."""
  _, val = subset_sum_bound(rho, sigma_100, 2 ** (eps_kahe + 1) * n * mu, "for q")
  return iv_floor_upper(ivi(t) * (ivi(2) * val + ivi(rho)))


# ---------------------------------------------------------------------------
# MSE bounds  [MSE]
# ---------------------------------------------------------------------------
#
# [MSE] (thm:multisetencoding): for capacity C, theta rows of delta cells, keys in
# Z_t^omega with t > C, the decoding failure probability is at most
#     C(C,2) t^-omega + sum_{s=2}^{C} C(C,s) p_delta(s)^theta  (+ PRF advantage),
# where p_delta(s) is the probability that s balls in delta bins leave no singleton bin.
# The first term (key collisions) depends on t, the second (stopping sets) does not.
# We give the second term all but 1/16 of the budget 2^-eps_mse.

_nosingleton_cache = {}

def nosingleton_counts(delta, N):
  """c[s] = number of maps [s] -> [delta] with no bin hit exactly once, for s <= N
  (inclusion-exclusion over the bins hit exactly once; exact integers)."""
  delta, N = int(delta), int(N)
  key = (delta, N)
  if key not in _nosingleton_cache:
    c = []
    for s in range(N + 1):
      tot, fall = int(0), int(1)            # Python ints; fall = s!/(s-j)!
      for j in range(0, min(s, delta) + 1):
        if j > 0:
          fall *= (s - j + 1)
        tot += (-1) ** j * math.comb(delta, j) * fall * (delta - j) ** (s - j)
      c.append(tot)
    _nosingleton_cache[key] = c
  return _nosingleton_cache[key]


def iblt_log2_bound(rho, theta, delta):
  """log2 of sum_{s=2}^{rho} C(rho,s) p_delta(s)^theta, computed exactly."""
  c = nosingleton_counts(delta, rho)
  rho, theta, delta = int(rho), int(theta), int(delta)
  num = int(sum(math.comb(rho, s) * c[s] ** theta * delta ** (theta * (rho - s))
                for s in range(2, rho + 1)))
  if num == 0:
    return float('-inf')
  return log2(num) - theta * rho * log2(delta)


def find_iblt_shape(rho, target_log2, verbose=False):
  """(theta, delta) minimising the number of cells theta*delta such that the
  stopping-set term of [MSE] is at most 2^target_log2.

  The search over theta and delta is iblt_common's (automatic range of theta,
  excluded by rigorous lower bounds; floating-point union bound if numpy is
  available).  The resulting delta is then confirmed with exact integers: the bound
  holds for delta and fails for delta - 1."""
  res = iblt_common.search_thetas(iblt_common.make_args(rho, 2.0 ** target_log2), None,
                                  simulate=False, report=False)
  if verbose:
    rows = [[th, pd, th * pd, "%.2f" % iblt_common.union_bound_log2(rho, th, pd)]
            for th, _, _, pd, _ in res["rows"] if pd is not None]
    print(tabulate(rows, headers=["theta", "delta", "cells", "log2 bound"],
                   tablefmt="simple_outline"), "\n")
  if res["provable"] is None:
    return None
  theta, delta = res["provable"]
  while iblt_log2_bound(rho, theta, delta) > target_log2:
    delta += 1
  while delta > 2 and iblt_log2_bound(rho, theta, delta - 1) <= target_log2:
    delta -= 1
  return theta, delta


def warn_if_simulation_infeasible(s, target):
  """Warn if --empirical would need an impractical number of trials per tested shape."""
  trials = math.ceil(s.sim_trials_factor / target)
  if trials > 1e9:
    print(red("WARNING: %.2e trials per tested shape; simulation at eps_MSE = %d is "
              "impractical." % (trials, s.eps_mse)))
  return trials


def find_empirical_iblt_shape(s, prov, target_log2, verbose=False):
  """--empirical: the shape found by simulation (iblt_empirical.empirical_shape) for the
  same stopping-set budget 2^target_log2, used only if it has fewer cells than the
  provable shape prov.  Returns (shape, info), where info describes the search and
  whether the empirical shape is used."""
  target = 2.0 ** target_log2
  workers = s.sim_workers or None
  trials = warn_if_simulation_infeasible(s, target)
  progress("  IBLT shape (empirical, by simulation; %d trials per tested shape; seed %d) ..."
           % (trials, s.seed))
  from iblt_empirical import empirical_shape
  emp = empirical_shape(s.cap, target, alpha=s.sim_alpha, trials_factor=s.sim_trials_factor,
                        resolution=s.sim_resolution, workers=workers, seed=s.seed,
                        verbose=verbose)
  pcells = prov[0] * prov[1]
  info = dict(mode="simulate", provable_shape=list(prov), target_log2=target_log2,
              used=False, result=emp)
  if emp is None:
    progress("  no empirical IBLT shape found; using the provable shape")
    return prov, info
  if emp["cells"] < pcells:
    progress("  empirical IBLT shape: theta=%d, delta=%d (%d cells < %d provable); "
             "using the empirical shape" % (emp["theta"], emp["delta"], emp["cells"], pcells))
    info["used"] = True
    return (emp["theta"], emp["delta"]), info
  progress("  empirical IBLT shape: theta=%d, delta=%d (%d cells >= %d provable); "
           "using the provable shape" % (emp["theta"], emp["delta"], emp["cells"], pcells))
  return prov, info


def nonce_chunks(s, bits_per_chunk):
  """[ABC] smallest xi_s with rho (Q + rho) 2^lambda <= 2^(xi_s * bits_per_chunk) (exact
  integers), i.e. rho (Q + rho) / |N| <= 2^-lambda for the nonce space
  N = {0,1}^(xi_s * bits_per_chunk) and Q = 2^log_queries.  The nonce is a bitstring,
  chunked like messages."""
  need = s.rho * (2 ** s.log_queries + s.rho) * 2 ** s.secpar
  xs = 0
  while 2 ** (xs * bits_per_chunk) < need:
    xs += 1
  return xs


def mse_params(s, t, shape, empirical=None):
  """Parameters of the multiset encoding over the field Z_t for a fixed IBLT shape, or
  None unless t is a prime larger than the capacity C (counters must not wrap around)."""
  if t <= s.cap or not is_prime(t):
    return None
  theta, delta = shape
  # smallest omega with C(C,2) t^-omega <= 2^-eps_mse / 16, i.e.
  # C(C,2) 2^(eps_mse + 4) <= t^omega (exact integers), for the capacity C
  pairs = math.comb(s.cap, 2)
  omega = 1
  while pairs * 2 ** (s.eps_mse + 4) > int(t) ** omega:
    omega += 1
  chunk_bits = int(t).bit_length() - 1      # floor(log2 t) = b
  xi_nonce = nonce_chunks(s, chunk_bits)
  xi = xi_nonce - (-s.msg_bits // chunk_bits)   # nonce + ceil(msg_bits / chunk_bits)
  return MseParams(theta=theta, delta=delta, omega=omega, xi=xi, chunk_bits=chunk_bits,
                   empirical=empirical, xi_nonce=xi_nonce)


def ng_mse_params(s, t):
  """Parameters of the Newton-Girard encoding over F = F_{t^omega}, or None unless t is
  a prime larger than the capacity C.  [NG]: 2^-eps_mse >= C^2 t^-omega, i.e. the smallest
  omega with C^2 2^eps_mse <= t^omega.  Simple chunking: each coordinate of an element of
  F = Z_t^omega carries b = floor(log2 t) message bits."""
  if t <= s.cap or not is_prime(t):
    return None
  omega = 1
  while s.cap ** 2 * 2 ** s.eps_mse > int(t) ** omega:
    omega += 1
  chunk_bits = int(t).bit_length() - 1
  xi_nonce = nonce_chunks(s, omega * chunk_bits)
  xi = xi_nonce - (-s.msg_bits // (omega * chunk_bits))
  return NgMseParams(rho=s.cap, omega=omega, xi=xi, chunk_bits=chunk_bits, xi_nonce=xi_nonce)


# ---------------------------------------------------------------------------
# KAHE + MSE search
# ---------------------------------------------------------------------------

def plaintext_prime(b, k):
  """Smallest prime t > 2^b with t = 1 mod 2^k', where k' = min(k, b - 1) keeps t below
  2^(b+1) for small b (k = 0: the smallest prime above 2^b).  NTT-friendliness of t is not
  needed by the paper; see the comment at the top."""
  k = min(k, b - 1)
  if k <= 0:
    return next_prime(2 ** b)
  step = 2 ** k
  t = (2 ** b // step + 1) * step + 1        # smallest value > 2^b with t = 1 mod 2^k
  while not is_prime(t):
    t += step
  return t


def two_adicity(t):
  """Largest v with 2^v dividing t - 1."""
  v, m = 0, int(t) - 1
  while m % 2 == 0 and m > 0:
    m //= 2
    v += 1
  return v


def kahe_candidate(s, b, shape, empirical=None):
  """Candidate for the plaintext modulus t = plaintext_prime(b, s.t_ntt_log), or None if
  the MSE conditions fail.  mu only depends on t and the encoding, so sigma
  (which depends on mu through the smoothing parameter) and nu are computed per candidate."""
  t = plaintext_prime(b, s.t_ntt_log)
  if s.mse == "NG":
    mse = ng_mse_params(s, t)
  else:
    mse = mse_params(s, t, shape, empirical)
  if mse is None:
    return None
  mu = -(-mse.elems // s.n)                 # ceil(elems / n)
  sigma_100 = kahe_sigma_100(s.n, kahe_smoothing_inverse_eps(s.secpar, s.rho, mu))
  nu = kahe_nu(s.rho, sigma_100, s.n, s.secpar)
  q = find_ntt_friendly_prime(s.n, kahe_q_floor(t, s.rho, sigma_100, s.eps_kahe, s.n, mu))
  return KaheParams(t=t, q=q, mu=mu, sigma_100=sigma_100, nu=nu, mse=mse)


def find_kahe_params(s, verbose):
  """Among the candidate plaintext moduli t (see kahe_candidate) with t > C, q from
  [KAHE] below 2^max_q_bits and RLWE with D_{sigma/2} 2^secpar hard, pick the one with
  the smallest ciphertext."""
  shape = emp = None
  if s.mse == "IBLT":
    progress("  IBLT shape (exact stopping-set bound) ...")
    target_log2 = -s.eps_mse + log2(15.0 / 16)
    shape = find_iblt_shape(s.cap, target_log2, verbose=verbose)
    if shape is None:
      print("[MSE] no IBLT shape found.")
      return None
    progress("  IBLT shape: theta=%d, delta=%d" % shape)
    if s.empirical:
      shape, emp = find_empirical_iblt_shape(s, shape, target_log2, verbose=verbose)

  b_min = int(s.cap).bit_length()          # 2^b_min > C, so t > 2^b_min > C

  def ok(c):
    return (c is not None and log2(c.q) < s.max_q_bits
            and rlweIsHard(s.n, c.q, c.sigma / 2, s.secpar))

  # q grows with t, so feasibility is monotone in b: binary search for the largest b
  b_max = None
  lo, hi = b_min, s.max_q_bits
  while lo <= hi:
    b = (lo + hi) // 2
    if ok(kahe_candidate(s, b, shape, emp)):
      b_max = b
      lo = b + 1
    else:
      hi = b - 1
  if b_max is None:
    c = kahe_candidate(s, b_min, shape, emp)
    print("[KAHE] no t with q < 2^%d and hard RLWE (smallest q needed: %.1f bits)."
          % (s.max_q_bits, log2(c.q)))
    return None

  cands = [c for c in (kahe_candidate(s, b, shape, emp) for b in range(b_min, b_max + 1))
           if c is not None]
  best = min(cands, key=lambda c: kahe_ct_bits(s.n, c.mu, c.q))
  if not ok(best):
    print("[KAHE] best candidate unexpectedly fails the RLWE check.")
    return None

  if verbose:
    rows = [[c.b, str(c.t), c.mu, "%.1f" % log2(c.q), "%.2f" % c.sigma, c.mse.omega,
             c.mse.xi, fmt_bytes(kahe_ct_bits(s.n, c.mu, c.q))] for c in cands]
    print(tabulate(rows, headers=["log t", "t", "mu", "log q", "sigma", "omega", "xi", "ct size"],
                   tablefmt="simple_outline"), "\n")
  return best


# ---------------------------------------------------------------------------
# Leaf commitment  [LEAF], with modulus p from [ABC]
# ---------------------------------------------------------------------------

def leaf_hiding_log2(beta, kappa, mu, q, n, rho):
  """log2 of (rho-1) * ((4b+1)/(2b+1)^2)^(kappa n/2) * q^(n(mu+1)/2)."""
  return (log2(rho - 1) + 0.5 * kappa * n * (log2(4 * beta + 1) - 2 * log2(2 * beta + 1))
          + 0.5 * n * (mu + 1) * log2(q))


def leaf_beta(kappa, mu, q, n, rho, secpar_hide):
  """Smallest beta satisfying the hiding inequality of [LEAF] for target 2^-secpar_hide
  (floating point with a small margin; check_all verifies the result exactly)."""
  # Solve (4b+1)/(2b+1)^2 <= C for b, with a tiny margin against floating point error.
  target = secpar_hide + 0.01
  log2C = (-(2.0 * target) / (kappa * n) - (2.0 / (kappa * n)) * log2(rho - 1)
           - ((mu + 1) / kappa) * log2(q))
  if log2C > -900:
    C = 2.0 ** log2C
    beta = int(math.ceil((1 - C + math.sqrt(1 - C)) / (2 * C)))
  else:
    # C underflows as a float (huge q); then (4b+1)/(2b+1)^2 ~ 1/b gives b ~ 2^-log2C
    e = -log2C
    beta = int(math.ceil(2.0 ** (e - math.floor(e) + 52))) << (math.floor(e) - 52)
  beta = max(beta, 1)
  while leaf_hiding_log2(beta, kappa, mu, q, n, rho) > -secpar_hide:
    beta += max(1, beta >> 20)
  return beta


def leaf_q_ok(q, beta, rho):
  """2*beta < sqrt(q/2)  <=>  8 beta^2 < q   [LEAF]; and 2*beta_agg < q-1 (else SIS trivial)."""
  return 8 * beta ** 2 < q and 4 * beta * rho < q - 1


def find_leaf_params(s, min_p, verbose, mu=1):
  """Leaf commitment with message space R_p^mu and prime modulus p > min_p, p = 5 (mod 8),
  minimising the size of the gamma aggregated leaf openings."""
  found = []
  best_size = None
  max_q = 2 ** s.max_q_bits - 1

  for kappa in range(1, 41):
    low_q = max(min_p + 1, 4 * s.rho + 2)
    high_q = max_q
    best = None
    # binary search for the smallest q that admits a valid beta
    while low_q <= high_q:
      mid_q = (low_q + high_q) // 2
      beta = leaf_beta(kappa, mu, mid_q, s.n, s.rho, s.secpar_hide)
      if leaf_q_ok(mid_q, beta, s.rho):
        best = mid_q
        high_q = mid_q - 1
      else:
        low_q = mid_q + 1
    if best is None:
      continue

    # The smallest q with a valid beta need not make SIS_{R,q,kappa,2 rho beta} hard: for
    # large rho, 2 rho beta is then close to q/2.  Since beta grows only like q^(2/kappa),
    # a larger q helps, so search upwards (geometrically, then by bisection) for the
    # smallest q that also satisfies [LEAF]'s binding condition.
    def leaf_ok(q):
      beta = leaf_beta(kappa, mu, q, s.n, s.rho, s.secpar_hide)
      return (leaf_q_ok(q, beta, s.rho)
              and rsisIsHard(2 * beta * s.rho, q, s.n, kappa, s.secpar))   # [LEAF] SIS_{R,q,kappa,2 beta_agg}

    lo, hi = best - 1, best        # invariant: leaf_ok fails at lo (or lo is a lower bound), hi to test
    while hi <= max_q and not leaf_ok(hi):
      lo, hi = hi, hi + max(1, hi >> 2)
    if hi > max_q:
      continue
    while hi - lo > max(1, hi >> 12):
      mid = (lo + hi) // 2
      if leaf_ok(mid):
        hi = mid
      else:
        lo = mid

    # first valid prime at or above this bound; re-check everything for it
    q = find_minimally_splitting_prime(hi - 1)
    tries = 0
    while not leaf_ok(q) and tries < 100:
      q = find_minimally_splitting_prime(q)
      tries += 1
    if not leaf_ok(q) or q > max_q or q <= min_p:
      continue
    beta = leaf_beta(kappa, mu, q, s.n, s.rho, s.secpar_hide)

    beta_agg = beta * s.rho                     # [LEAF] beta_agg >= rho*beta
    size = s.gamma * leaf_opening_bits(s.n, kappa, beta_agg)
    if best_size is None or size < best_size:
      best_size = size
      found.append(LeafParams(p=q, beta=beta, beta_agg=beta_agg, kappa=kappa, mu=mu))

  if verbose and found:
    rows = [[l.beta, l.kappa, l.mu, l.p, l.beta_agg,
             fmt_bytes(leaf_opening_bits(s.n, l.kappa, l.beta)),
             fmt_bytes(leaf_opening_bits(s.n, l.kappa, l.beta_agg)),
             fmt_bytes(s.gamma * leaf_opening_bits(s.n, l.kappa, l.beta_agg))] for l in found]
    print(tabulate(rows, headers=["beta", "kappa", "mu", "p", "beta_agg", "opening",
                                  "agg. opening", "gamma agg. openings"],
                   tablefmt="simple_outline"), "\n")
  return found[-1] if found else None


# ---------------------------------------------------------------------------
# Chipmunk vector commitment  [AVC]
# ---------------------------------------------------------------------------

def avc_limbs(q, eta):
  """ceil(log_{2 eta + 1} q), computed exactly."""
  k, x = 0, 1
  while x < q:
    x *= 2 * eta + 1
    k += 1
  return k


def chipmunk_candidate(s, q, eta, xi, p):
  return ChipmunkParams(q=q, eta=eta, kappa=avc_limbs(q, eta), kappa_prime=avc_limbs(p, eta),
                        beta_agg=eta * s.rho, tau=s.tau, xi=xi)


def chipmunk_objective(s, c):
  """gamma aggregated openings + rho commitments."""
  return (s.gamma * chipmunk_opening_bits(s.n, c.tau, c.kappa, c.kappa_prime, c.xi, c.beta_agg)
          + s.rho * vc_com_bits(s.n, c.q))


def avc_sis_hard(s, c):
  # [AVC] SIS_{R,q,2*ceil(log q),2 beta_agg} and SIS_{R,q,xi*ceil(log q'),2 beta_agg};
  # the two coincide when 2 kappa = xi kappa' (e.g. xi = 2 and kappa = kappa')
  return all(rsisIsHard(2 * c.beta_agg, c.q, s.n, m, s.secpar)
             for m in sorted({2 * c.kappa, c.xi * c.kappa_prime}))


@functools.lru_cache(maxsize=None)
def ntt_prime_of_bits(n, bits):
  """Smallest NTT-friendly prime above 2^(bits-1) (cached: the Chipmunk search asks for the
  same bit lengths for every eta, and primes of thousands of bits are slow to find)."""
  return find_ntt_friendly_prime(n, 2 ** (bits - 1))


def find_hvc_params(s, xi, p, verbose):
  """For each arity 2*eta+1, find the smallest NTT-friendly prime q for which both SIS
  problems of [AVC] are hard, and keep the configuration with the smallest openings.
  The vector commitment message space is R_p^xi with p the leaf modulus.
  The opening size only grows with q, so for each eta we only search q up to the
  point where it can no longer beat the best configuration found so far."""
  found = []
  best_size = None
  # Every eta up to 64, then a geometric grid with 8 steps per doubling (ratio 2^(1/8)),
  # up to 2^23.  Small eta are tried exhaustively since they are usually optimal; above
  # 64 consecutive candidates differ by less than 9%.
  etas = sorted(set(list(range(1, 65))
                    + [int(round(2.0 ** (k / 8.0))) for k in range(6 * 8 + 1, 23 * 8 + 1)]))
  for eta in etas:
    beta_agg = eta * s.rho
    # smallest q worth trying: the trivial attack needs n*log2(q/(4 beta_agg+1)) >= secpar
    lo_bits = int(math.floor(log2(4 * beta_agg + 1) + float(s.secpar) / s.n)) + 1
    if lo_bits > s.max_q_bits:
      break
    # largest q worth trying: openings must still beat the best size so far
    # (the objective grows with q, so bisect on the bit length; 2^(bits-1) + 1 stands in for
    # the prime, which lies just above it, to avoid finding primes of thousands of bits)
    hi_bits = s.max_q_bits
    if best_size is not None:
      def beats_best(bits):
        return chipmunk_objective(s, chipmunk_candidate(s, 2 ** (bits - 1) + 1, eta, xi, p)) < best_size
      lo_b, hi_b, hi_bits = lo_bits, s.max_q_bits, lo_bits - 1
      while lo_b <= hi_b:
        mid = (lo_b + hi_b) // 2
        if beats_best(mid):
          hi_bits, lo_b = mid, mid + 1
        else:
          hi_b = mid - 1
    if hi_bits < lo_bits:
      continue
    # Candidate moduli: the smallest NTT-friendly prime of each bit length, computed only
    # when tested (with --max-q-bits in the thousands, computing all of them up front is
    # far too slow).  Several bit lengths can share a prime (e.g. 40961 is the smallest
    # such prime above both 2^14 and 2^15 for n = 2048); testing a prime twice is harmless.
    # Hardness grows with q, and the smallest candidate (just above the trivial-attack
    # bound) is usually already hard, so search upwards from the bottom (exponential
    # search: indices 0, 1, 3, 7, ...) and then bisect the last gap.
    bit_lengths = list(range(lo_bits, hi_bits + 1))

    def candidate(i):
      q = ntt_prime_of_bits(s.n, bit_lengths[i])
      if log2(q) >= s.max_q_bits:
        return None
      return chipmunk_candidate(s, q, eta, xi, p)

    best = None
    lo, step = 0, 1                     # invariant: candidates [:lo] are known not to be hard
    while lo < len(bit_lengths):
      i = min(lo + step - 1, len(bit_lengths) - 1)
      c = candidate(i)
      if c is not None and avc_sis_hard(s, c):
        best, hi = c, i - 1             # smallest hard candidate is in [lo..i]
        while lo <= hi:
          mid = (lo + hi) // 2
          c = candidate(mid)
          if c is not None and avc_sis_hard(s, c):
            best, hi = c, mid - 1
          else:
            lo = mid + 1
        break
      lo, step = i + 1, 2 * step
    if best is None:
      continue
    size = chipmunk_objective(s, best)
    if best_size is None or size < best_size:
      best_size = size
      found.append(best)

  if verbose and found:
    rows = [[c.eta, c.kappa, c.kappa_prime, c.beta_agg, c.q,
             fmt_bytes(chipmunk_opening_bits(s.n, c.tau, c.kappa, c.kappa_prime, c.xi, c.eta)),
             fmt_bytes(chipmunk_opening_bits(s.n, c.tau, c.kappa, c.kappa_prime, c.xi, c.beta_agg)),
             fmt_bytes(vc_com_bits(s.n, c.q)), fmt_bytes(chipmunk_objective(s, c))] for c in found]
    print(tabulate(rows, headers=["eta", "kappa", "kappa'", "beta_agg", "q", "opening",
                                  "agg. opening", "com", "gamma agg. op. + rho com"],
                   tablefmt="simple_outline"), "\n")
  return found[-1] if found else None


# ---------------------------------------------------------------------------
# Final check of every bound in the paper
#
# Independent of the search: every condition is transcribed directly from the paper
# and evaluated with exact integer arithmetic.  ln, sqrt and pi are enclosed in
# rigorous intervals, and a condition only passes if it holds for the whole interval.
# The MSE stopping-set term is recomputed with a different algorithm (exponential
# generating function) than the one used by the search (inclusion-exclusion).
# ---------------------------------------------------------------------------

def nosingleton_counts_egf(delta, N):
  """Number of maps [s] -> [delta] with no bin hit exactly once, for s <= N, as the
  coefficients of s! [x^s] (e^x - x)^delta, by square-and-multiply on exponential
  generating functions (binomial convolution of integer count sequences)."""
  delta, N = int(delta), int(N)
  binom = [[math.comb(s, j) for j in range(s + 1)] for s in range(N + 1)]

  def mul(a, b):
    return [sum(binom[s][j] * a[j] * b[s - j] for j in range(s + 1)) for s in range(N + 1)]

  base = [1, 0] + [1] * (N - 1)             # one bin: any number of balls except exactly one
  base = base[:N + 1]
  result = [1] + [0] * N                    # zero bins
  while delta:
    if delta & 1:
      result = mul(result, base)
    base = mul(base, base)
    delta >>= 1
  return result


def check_all(s, kahe, leaf, avc):
  """Returns (rows, all_ok) for every condition the theorems of the paper impose."""
  rows = []

  def chk(src, what, cond, val=""):
    rows.append([src, what, "ok" if cond else "FAILS", val])

  rho, gamma, n, lam = s.rho, s.gamma, s.n, s.secpar
  t, q, mu, nu, mse = kahe.t, kahe.q, kahe.mu, kahe.nu, kahe.mse
  S = kahe.sigma_100                        # sigma = S / 100
  p = leaf.p
  pi = iv.pi

  # ---- [KAHE] thm:kahe_secure ---------------------------------------------------
  chk("KAHE", "n power of two", n > 0 and n & (n - 1) == 0, n)
  chk("KAHE", "q prime", is_prime(q), q)
  # q > t (2 T_sigma(rho, 2^-eps / (n mu)) + rho), via T_sigma <= f(tau) for any tau > 0
  # (def:subsetsumbound); tau from the minimiser search, f(tau) enclosed rigorously.
  two_over_delta_q = 2 ** (s.eps_kahe + 1) * n * mu
  tau_q = find_subset_sum_tau(rho, S, mpmath.log(two_over_delta_q))
  f_q = subset_sum_f_iv(rho, S, two_over_delta_q, tau_q)
  bound = ivi(t) * (ivi(2) * f_q + ivi(rho))
  chk("KAHE", "q > t (2 T_sigma(rho, 2^-eps/(n mu)) + rho)", int_gt(q, bound),
      "T_sigma <= %.2f, q - ceil(bound) = %d" % (float(f_q.b), q - iv_ceil(bound)))
  # sigma >= (2/sqrt(pi)) eta_eps(Z^n) with eps = 2^-lambda / (2 (rho-1) (mu+1)) and
  # eta_eps(Z^n) <= sqrt(ln(2n(1+1/eps))/pi)
  #   <=>  S^2 pi^2 >= 4 * 100^2 * ln(2n(1+1/eps))
  inv_eps = 2 ** lam * 2 * (rho - 1) * (mu + 1)
  sig_lhs = ivi(S) * ivi(S) * pi * pi
  sig_rhs = ivi(4 * 100 ** 2) * iv.log(ivi(2 * n * (1 + inv_eps)))
  chk("KAHE", "sigma >= 2/sqrt(pi) eta_eps(Z^n), eps = 2^-lambda/(2(rho-1)(mu+1))",
      iv_ge(sig_lhs, sig_rhs),
      "%.2f >= %.4f (eps = 2^-%.1f)" % (S / 100.0, float(mpmath.sqrt(sig_rhs.b) / pi.a) / 100,
                                       log2(inv_eps)))
  # nu >= T_sigma(rho, 2^-lambda / n), via T_sigma <= f(tau) as above
  two_over_delta_nu = 2 ** (lam + 1) * n
  tau_nu = find_subset_sum_tau(rho, S, mpmath.log(two_over_delta_nu))
  f_nu = subset_sum_f_iv(rho, S, two_over_delta_nu, tau_nu)
  chk("KAHE", "nu >= T_sigma(rho, 2^-lambda/n)", int_ge(nu, f_nu),
      "%d >= %.2f" % (nu, float(f_nu.b)))
  chk("KAHE", "q does not divide t", t % q != 0)
  chk("KAHE", "RLWE_{R,q,D_{sigma/2}} hard", rlweIsHard(n, q, kahe.sigma / 2, lam))

  # the MSE only has to hold the elements of at most C <= rho senders (sec:cover); C = rho
  # without cover traffic.  The names in the check descriptions use rho for C = rho.
  cap = s.cap
  cn = "rho" if cap == rho else "C"
  if mse.kind == "IBLT":
    # ---- [MSE] thm:multisetencoding ------------------------------------------------
    g, d, om = mse.theta, mse.delta, mse.omega
    chk("MSE", "t prime, t > %s" % cn, is_prime(t) and t > cap, t)
    # C(C,2) t^-omega + sum_{s=2}^{C} C(C,s) (c_s / d^s)^g <= 2^-eps_MSE, multiplied
    # by the common denominator t^omega d^(g C) 2^eps_MSE (exact integers)
    c = nosingleton_counts_egf(d, cap)
    D = d ** (g * cap)
    stop_num = sum(math.comb(cap, k) * c[k] ** g * d ** (g * (cap - k)) for k in range(2, cap + 1))
    T = int(t) ** om
    coll_l2 = log2(math.comb(cap, 2)) - om * log2(t)
    stop_l2 = log2(stop_num) - log2(D) if stop_num else float('-inf')
    emp = mse.empirical if mse.empirical and mse.empirical["used"] else None
    if emp is None:
      mse_ok = (math.comb(cap, 2) * D + stop_num * T) * 2 ** s.eps_mse <= T * D
      chk("MSE", "C(%s,2) t^-omega + sum_s C(%s,s) p_delta(s)^theta <= 2^-eps_MSE" % (cn, cn),
          mse_ok,
          "2^%.1f + 2^%.1f <= 2^-%d" % (coll_l2, stop_l2, s.eps_mse))
    else:
      # --empirical: the budget 2^-eps_MSE is split as in the search, 1/16 for key
      # collisions (proven) and 15/16 for stopping sets (empirical shape, not proven)
      chk("MSE", "key collisions: C(%s,2) t^-omega <= 2^-eps_MSE / 16" % cn,
          math.comb(cap, 2) * 16 * 2 ** s.eps_mse <= T,
          "2^%.1f <= 2^-%.1f" % (coll_l2, s.eps_mse + 4))
      stop_ok = stop_num * 16 * 2 ** s.eps_mse <= 15 * D
      r = emp["result"]
      # short value for the table; the details go into the note printed at the end
      short = "simulated, %.0f%% UB 2^%.1f" % (100 * (1 - r["alpha"]), r["upper_log2"])
      detail = ("simulation: %d/%d failures, %.0f%% upper bound 2^%.1f"
                % (r["failures"], r["trials"], 100 * (1 - r["alpha"]), r["upper_log2"]))
      empirical_detail = "%s; union bound 2^%.1f" % (detail, stop_l2)
      # a shape found by the search always passed its simulation; --check-only records
      # the outcome of its own simulation in "passed"
      sim_ok = r.get("passed", True)
      rows.append(["MSE", "stopping sets: sum_s C(%s,s) p_delta(s)^theta <= 15/16 2^-eps_MSE" % cn,
                   "ok" if stop_ok else ("EMPIRICAL" if sim_ok else "FAILS"), short])
    xm = mse.xi - mse.xi_nonce
    chk("MSE", "message fits: xi - xi_s chunks of b bits, 2^b <= t", xm * mse.chunk_bits >= s.msg_bits
        and 2 ** mse.chunk_bits <= t, "%d x %d >= %d bits" % (xm, mse.chunk_bits, s.msg_bits))
    nonce_space = 2 ** (mse.xi_nonce * mse.chunk_bits)
    elems = g * d * (1 + om + mse.xi)
  else:
    # ---- [NG] thm:mse_newton -------------------------------------------------------
    om = mse.omega
    chk("NG", "t prime, t > %s" % cn, is_prime(t) and t > cap, t)
    # 2^-eps_MSE >= C^2 t^-omega  <=>  C^2 2^eps_MSE <= t^omega
    chk("NG", "%s^2 t^-omega <= 2^-eps_MSE" % cn, cap ** 2 * 2 ** s.eps_mse <= int(t) ** om,
        "2^%.1f <= 2^-%d" % (2 * log2(cap) - om * log2(t), s.eps_mse))
    chk("NG", "capacity of the encoding = %s" % cn, mse.rho == cap, mse.rho)
    xm = mse.xi - mse.xi_nonce
    chk("NG", "message fits: xi - xi_s chunks of omega*b bits, 2^b <= t",
        xm * om * mse.chunk_bits >= s.msg_bits and 2 ** mse.chunk_bits <= t,
        "%d x %d x %d >= %d bits" % (xm, om, mse.chunk_bits, s.msg_bits))
    nonce_space = 2 ** (mse.xi_nonce * om * mse.chunk_bits)
    elems = om * cap * (1 + mse.xi)

  # ---- [ABC] thm:broadcast --------------------------------------------------------
  chk("ABC", "rho >= 2", rho >= 2, rho)
  if cap != rho:
    chk("ABC", "2 <= C <= rho (sec:cover)", 2 <= cap <= rho, cap)
  # rho (Q + rho) / |N| <= 2^-lambda  <=>  rho (Q + rho) 2^lambda <= |N|  (exact integers)
  Q = 2 ** s.log_queries
  chk("ABC", "rho (Q + rho) / |N| <= 2^-lambda, N = {0,1}^l nonces", rho * (Q + rho) * 2 ** lam <= nonce_space,
      "Q = 2^%d, l = %d bits (%d chunks)" % (s.log_queries, nonce_space.bit_length() - 1, mse.xi_nonce))
  chk("ABC", "encodings fit into R_t^mu", mu * n >= elems, "%d >= %d" % (mu * n, elems))
  chk("ABC", "p prime", is_prime(p), p)
  chk("ABC", "p > max(gamma, 2 nu)", p > max(gamma, 2 * nu), "%d > %d" % (p, max(gamma, 2 * nu)))

  # ---- [LEAF] thm:leafcom ----------------------------------------------------------
  beta, kappa, lmu = leaf.beta, leaf.kappa, leaf.mu
  chk("LEAF", "p = 5 mod 8", p % 8 == 5)
  chk("LEAF", "2 beta < sqrt(p/2)", 8 * beta ** 2 < p, "beta=%d" % beta)
  chk("LEAF", "beta_agg >= rho beta", leaf.beta_agg >= rho * beta)
  # (rho-1) ((4b+1)/(2b+1)^2)^(kappa n/2) p^(n(mu+1)/2) <= 2^-lambda_hide, squared:
  # (rho-1)^2 (4b+1)^(kappa n) p^(n(mu+1)) 2^(2 lambda_hide) <= (2b+1)^(2 kappa n)
  hide_ok = ((rho - 1) ** 2 * (4 * beta + 1) ** (kappa * n) * int(p) ** (n * (lmu + 1))
             * 2 ** (2 * s.secpar_hide) <= (2 * beta + 1) ** (2 * kappa * n))
  h = leaf_hiding_log2(beta, kappa, lmu, p, n, rho)
  chk("LEAF", "hiding inequality <= 2^-lambda_hide, lambda_hide = lambda + tau", hide_ok,
      "2^%.1f <= 2^-%d" % (h, s.secpar_hide))
  chk("LEAF", "SIS_{R,p,kappa,2 beta_agg} hard", rsisIsHard(2 * leaf.beta_agg, p, n, kappa, lam))

  # ---- [AVC] Chipmunk ----------------------------------------------------------------
  qh, eta, base = avc.q, avc.eta, 2 * avc.eta + 1
  chk("AVC", "q, q' = p prime", is_prime(qh) and is_prime(p))
  chk("AVC", "message space R_p^xi, xi = mu_leaf + 1", avc.xi == lmu + 1, avc.xi)
  chk("AVC", "vector length 2^tau >= gamma", 2 ** avc.tau >= gamma, 2 ** avc.tau)
  chk("AVC", "beta_agg >= eta rho", avc.beta_agg >= eta * rho)
  chk("AVC", "kappa = ceil(log_{2 eta+1} q), kappa' = ceil(log_{2 eta+1} p)",
      base ** avc.kappa >= qh > base ** (avc.kappa - 1)
      and base ** avc.kappa_prime >= p > base ** (avc.kappa_prime - 1),
      "%d, %d" % (avc.kappa, avc.kappa_prime))
  chk("AVC", "SIS_{R,q,2 kappa,2 beta_agg} hard",
      rsisIsHard(2 * avc.beta_agg, qh, n, 2 * avc.kappa, lam))
  chk("AVC", "SIS_{R,q,xi kappa',2 beta_agg} hard",
      rsisIsHard(2 * avc.beta_agg, qh, n, avc.xi * avc.kappa_prime, lam))

  stats = {
    # (rho-1) hybrids, each losing 2(mu+1) eps with eps = 2^-lambda/(2(rho-1)(mu+1))
    "kahe_privacy_log2": -lam,
    "vc_hiding_log2": log2(gamma) + h,
    "key_norm_failure_log2": -lam,
    "kahe_decryption_failure_log2": -s.eps_kahe,
    "mse_decoding_failure_log2": -s.eps_mse,
    "masking_log2": log2(rho * (2 ** s.log_queries + rho)) - log2(nonce_space),
    "mse_uses_prf": mse.kind == "IBLT",
    "mse_empirical": any(r[2] == "EMPIRICAL" for r in rows),
  }
  if stats["mse_empirical"]:
    stats["mse_empirical_detail"] = empirical_detail
  if cap != rho:
    stats["capacity"] = cap
  # an empirically chosen IBLT shape is a deliberate choice, not a failure
  return rows, all(r[2] in ("ok", "EMPIRICAL") for r in rows), stats


def print_checks(rows, stats):
  rows = [r[:2] + [color_status(r[2])] + r[3:] for r in rows]
  print(tabulate(rows, headers=["source", "condition", "status", "value"],
                 tablefmt="simple_outline"), "\n")
  print("Concrete statistical terms:")
  print("  KAHE privacy (HintLWE smoothing, rho-1 hybrids): 2^%d (exact by choice of eps)"
        % stats["kahe_privacy_log2"])
  print("  VC hiding (gamma leaf hybrids):                  2^%.1f" % stats["vc_hiding_log2"])
  print("  key norm bound nu (KAHE correctness, 2nd cond.):  2^%d" % stats["key_norm_failure_log2"])
  print("  KAHE + MSE decryption/decoding failure:           2^%d + 2^%d%s"
        % (stats["kahe_decryption_failure_log2"], stats["mse_decoding_failure_log2"],
           " (+ PRF advantage)" if stats["mse_uses_prf"] else ""))
  print("  RO masking, rho (Q + rho) / |N|:                  2^%.1f" % stats["masking_log2"])
  if stats.get("mse_empirical"):
    print("  (the MSE stopping-set part is an empirical estimate, not proven)")
  if "capacity" in stats:
    print(yellow("Cover traffic (sec:cover): the encoding holds at most C = %d senders.  "
                 "Guaranteed output\ndelivery only holds against at most C - h corrupted "
                 "clients, where h is the number of\nhonest senders; keeping the number of "
                 "senders below C is up to the application." % stats["capacity"]))
  print()


# ---------------------------------------------------------------------------
# Final parameter table
# ---------------------------------------------------------------------------

def fmt_mod(x):
  """A modulus together with its size in bits, e.g. '57773  (2^15.82)'."""
  x = int(x)
  if x & (x - 1) == 0:
    return "%d  (2^%d)" % (x, x.bit_length() - 1)
  return "%d  (2^%.2f)" % (x, log2(x))


def print_boxed(title, header, groups):
  """Print one table with a title bar; groups is a list of (component, rows) where each
  row is [parameter, value, description].  Components are separated by rules."""
  cols = [len(h) for h in header]
  for comp, rows in groups:
    cols[0] = max(cols[0], len(comp))
    for r in rows:
      for i, c in enumerate(r):
        cols[i + 1] = max(cols[i + 1], len(c))
  inner = sum(cols) + 3 * (len(cols) - 1)
  if len(title) > inner:
    cols[-1] += len(title) - inner
    inner = len(title)

  def rule(l, m, r, h="─"):
    return l + m.join(h * (w + 2) for w in cols) + r

  def line(cells):
    return "│ " + " │ ".join(c.ljust(w) for c, w in zip(cells, cols)) + " │"

  out = ["┌" + "─" * (inner + 2) + "┐",
         "│ " + title.center(inner) + " │",
         rule("├", "┬", "┤"),
         line(header),
         rule("╞", "╪", "╡", "═")]
  for gi, (comp, rows) in enumerate(groups):
    for ri, r in enumerate(rows):
      out.append(line([comp if ri == 0 else ""] + r))
    out.append(rule("├", "┼", "┤") if gi < len(groups) - 1 else rule("└", "┴", "┘"))
  print("\n".join(out), "\n")


def mse_group(s, m):
  if m.kind == "IBLT":
    return ("MSE (IBLT)", [
      ["θ × δ", "%d × %d = %d" % (m.theta, m.delta, m.cells),
       "hash functions × cells per hash = cells"],
      ["ω", str(m.omega), "key length (elements of Z_t)"],
      ["ξ", "%d = %d + %d" % (m.xi, m.xi_nonce, m.xi - m.xi_nonce),
       "element length (elements of Z_t) = nonce + message chunks"],
      ["b", "%d bits" % m.chunk_bits, "message bits per element of Z_t (chunk size)"],
      ["ε_MSE", "2^-%d" % s.eps_mse, "decoding failure probability"],
    ] + ([["shape", "empirical (simulation)",
           "θ, δ chosen empirically, not proven (provable: %d × %d = %d)"
           % (m.empirical["provable_shape"][0], m.empirical["provable_shape"][1],
              m.empirical["provable_shape"][0] * m.empirical["provable_shape"][1])]]
         if m.empirical and m.empirical["used"] else []))
  return ("MSE (Newton–Girard)", [
    ["F", "F_{t^%d}" % m.omega, "keys and chunks (t prime; large t needs randomized root finding)"],
    ["ω", str(m.omega), "extension degree = key length (elements of Z_t)"],
    ["ξ", "%d = %d + %d" % (m.xi, m.xi_nonce, m.xi - m.xi_nonce),
     "element length (elements of F) = nonce + message chunks"],
    ["b", "%d bits" % m.chunk_bits, "message bits per element of Z_t (chunk size)"],
    ["", "%d bits" % (m.omega * m.chunk_bits), "message bits per element of F"],
    ["size", "%d × (1 + %d) = %d" % (m.rho, m.xi, m.rho * (1 + m.xi)),
     "elements of F per encoding: C (1 + ξ)" if m.rho != s.rho else
     "elements of F per encoding: ρ (1 + ξ)"],
    ["ε_MSE", "2^-%d" % s.eps_mse, "decoding failure probability (key collisions)"],
  ])


def print_parameters(s, kahe, leaf, avc, sizes):
  m = kahe.mse
  groups = [
    ("Setting", [
      ["ρ", str(s.rho), "number of clients" if s.cap != s.rho else
       "number of clients (MSE capacity)"],
    ] + ([["C", str(s.cap), "MSE capacity: at most C senders (sec:cover)"]]
         if s.cap != s.rho else []) + [
      ["γ", str(s.gamma), "number of servers"],
      ["n", str(s.n), "ring dimension of R = Z[X]/(X^n+1)"],
      ["λ", str(s.secpar), "security parameter"],
      ["|m|", "%d bytes" % s.msg_bytes, "message size"],
      ["Q", "2^%d" % s.log_queries, "bound on the adversary's random-oracle queries"],
    ]),
    ("KAHE", [
      ["t", fmt_mod(kahe.t), "plaintext modulus, messages in R_t^µ"],
      ["t − 1", "2^%d · odd" % two_adicity(kahe.t), "NTT over Z_t up to length 2^%d" % two_adicity(kahe.t)],
      ["q", fmt_mod(kahe.q), "ciphertext modulus, ciphertexts in R_q^µ"],
      ["µ", str(kahe.mu), "ring elements per ciphertext"],
      ["σ", "%.2f" % kahe.sigma, "Gaussian width of keys and errors"],
      ["ν", str(kahe.nu), "norm bound on sums of keys"],
      ["ε_KAHE", "2^-%d" % s.eps_kahe, "decryption failure probability"],
    ]),
    mse_group(s, m),
    ("Secret sharing", [
      ["p", fmt_mod(leaf.p), "Shamir field Z_p, VC message space R_p"],
      ["t_s", "%d of %d" % (s.gamma // 2 + 1, s.gamma), "reconstruction threshold"],
    ]),
    ("Leaf commitment", [
      ["p", "(as above)", "modulus"],
      ["β", str(leaf.beta), "randomness norm bound"],
      ["β_agg", str(leaf.beta_agg), "aggregated randomness norm bound (ρβ)"],
      ["κ", str(leaf.kappa), "randomness length (ring elements)"],
      ["µ", str(leaf.mu), "message length (ring elements)"],
    ]),
    ("Chipmunk VC", [
      ["q", fmt_mod(avc.q), "modulus"],
      ["η", str(avc.eta), "decomposition base 2η+1 = %d" % (2 * avc.eta + 1)],
      ["κ, κ'", "%d, %d" % (avc.kappa, avc.kappa_prime), "limbs per element of R_q, R_p"],
      ["τ", str(avc.tau), "tree depth (2^τ = %d leaves)" % (2 ** avc.tau)],
      ["ξ", str(avc.xi), "leaf length (ring elements of R_p)"],
      ["β_agg", str(avc.beta_agg), "aggregated opening norm bound (ρη)"],
    ]),
    ("Sizes", [
      ["ct", fmt_bytes(sizes.ct), "KAHE ciphertext"],
      ["com", fmt_bytes(sizes.com), "vector commitment"],
      ["d_i^j", fmt_bytes(sizes.opening), "opening, one client and server"],
      ["d_i", fmt_bytes(sizes.agg_opening), "aggregated opening of one server"],
      ["client", fmt_bytes(sizes.client_up), "client upload (ct + com + γ openings, excl. PKE)"],
      ["server", fmt_bytes(sizes.server_up), "server upload"],
      ["recipient", fmt_bytes(sizes.recipient_down), "recipient download (ρ ct + ρ com + γ d_i)"],
    ]),
  ]
  print_boxed("Final parameters", ["Component", "Parameter", "Value", "Description"], groups)


# ---------------------------------------------------------------------------
# Export (JSON)
# ---------------------------------------------------------------------------

def _plain(x):
  """Convert dataclasses and Sage numbers into JSON-serialisable Python objects."""
  if dataclasses.is_dataclass(x):
    return {f.name: _plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
  if isinstance(x, dict):
    return {str(k): _plain(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [_plain(v) for v in x]
  if isinstance(x, (bool, str)) or x is None:
    return x
  try:
    return int(operator.index(x))
  except TypeError:
    return float(x)


# Fields of the JSON file: only the parameters, no search options, sizes or check results.
JSON_SETTING_FIELDS = ("rho", "gamma", "n", "secpar", "msg_bytes", "eps_kahe", "eps_mse", "mse")


def write_json(path, s, kahe, leaf, avc):
  kahe_d = _plain(kahe)
  kahe_d["mse"].pop("empirical", None)
  sigma = kahe.sigma                 # sigma_100 is internal to the search
  kahe_d = {("sigma" if k == "sigma_100" else k): (sigma if k == "sigma_100" else v)
            for k, v in kahe_d.items()}
  data = {
    "generated": datetime.datetime.now().isoformat(timespec="seconds"),
    "command": " ".join(sys.argv),
    "setting": dict({k: _plain(getattr(s, k)) for k in JSON_SETTING_FIELDS},
                    **({"capacity": s.cap} if s.cap != s.rho else {})),
    "kahe": kahe_d,
    "leaf": _plain(leaf),
    "chipmunk": _plain(avc),
  }
  with open(path, "w") as f:
    json.dump(data, f, indent=2)
  progress("wrote %s" % path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def find_params(s, verbose=False):
  progress("KAHE/MSE parameters ...")
  kahe = find_kahe_params(s, verbose)
  if kahe is None:
    print("Could not find KAHE/MSE parameters.")
    return None
  # [ABC] Shamir field / VC message space modulus p > max(gamma, 2 nu)
  min_p = max(s.gamma, 2 * kahe.nu)
  progress("leaf commitment parameters (p > %d) ..." % min_p)
  leaf = find_leaf_params(s, min_p, verbose)
  if leaf is None:
    print("Could not find leaf commitment parameters with p > %d." % min_p)
    return None
  xi = leaf.mu + 1     # a leaf commitment (c^1, c^2) consists of mu+1 elements of R_p
  progress("Chipmunk vector commitment parameters ...")
  avc = find_hvc_params(s, xi, leaf.p, verbose)
  if avc is None:
    print("Could not find Chipmunk vector commitment parameters.")
    return None
  return kahe, leaf, avc


# ---------------------------------------------------------------------------
# --check-only: read a parameter set from JSON and check it
# ---------------------------------------------------------------------------

class ParamFileError(Exception):
  pass


PARAMS_SCHEMA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.schema.json")


def validate_params_json(data):
  """Validate data against params.schema.json (needs the jsonschema package).  Integers
  must be JSON integers: 3.0 is rejected, unlike in plain JSON Schema."""
  try:
    import jsonschema
  except ImportError:
    sys.exit("--check-only needs the jsonschema package (pip install jsonschema).")
  with open(PARAMS_SCHEMA) as f:
    schema = json.load(f)
  base = jsonschema.Draft202012Validator
  checker = base.TYPE_CHECKER.redefine(
    "integer", lambda _, v: isinstance(v, int) and not isinstance(v, bool))
  validator = jsonschema.validators.extend(base, type_checker=checker)(schema)
  errors = sorted(validator.iter_errors(data), key=lambda e: list(map(str, e.absolute_path)))
  if errors:
    raise ParamFileError("does not match %s:\n  " % os.path.basename(PARAMS_SCHEMA) + "\n  ".join(
      "%s: %s" % (".".join(map(str, e.absolute_path)) or "(top level)", e.message)
      for e in errors))


def read_params_json(path, args):
  """(Setting, KaheParams, LeafParams, ChipmunkParams) from a JSON file written by --json.
  The file must match params.schema.json.  The setting is taken from the file, except
  for the options --check-only honours (--log-queries, --empirical, --sim-*, --seed)."""
  try:
    with open(path) as f:
      data = json.load(f)
  except (OSError, ValueError) as e:
    raise ParamFileError("cannot read %s: %s" % (path, e))
  validate_params_json(data)

  sv = dict(data["setting"])
  mse_kind = sv.pop("mse")
  if sv.get("capacity") == sv["rho"]:
    sv.pop("capacity")
  kv = dict(data["kahe"])
  md = kv.pop("mse")
  # sigma is searched in steps of 0.01 and checked exactly as sigma_100 / 100
  sig = kv.pop("sigma")
  kv["sigma_100"] = int(round(sig * 100))
  if kv["sigma_100"] / float(100) != sig:
    raise ParamFileError("kahe.sigma = %r is not a multiple of 0.01" % (sig,))
  mse = MseParams(**md) if mse_kind == "IBLT" else NgMseParams(**md)
  kahe = KaheParams(mse=mse, **kv)
  leaf = LeafParams(**data["leaf"])
  avc = ChipmunkParams(**data["chipmunk"])

  s = Setting(mse=mse_kind, log_queries=args.log_queries, empirical=args.empirical,
              sim_alpha=args.sim_alpha, sim_trials_factor=args.sim_trials_factor,
              sim_resolution=args.sim_resolution, sim_workers=args.sim_workers, seed=args.seed,
              **sv)
  # the hiding bound uses lambda_hide = lambda + tau with the tree depth for gamma servers
  if avc.tau != s.tau:
    raise ParamFileError("chipmunk.tau = %d, but gamma = %d servers need tree depth %d"
                         % (avc.tau, s.gamma, s.tau))
  return s, kahe, leaf, avc


def iblt_provable_ok(s, kahe):
  """Whether the IBLT satisfies the [MSE] bound of thm:multisetencoding (as in check_all)."""
  mse, cap = kahe.mse, s.cap            # capacity C (sec:cover)
  g, d, om = mse.theta, mse.delta, mse.omega
  c = nosingleton_counts_egf(d, cap)
  D = d ** (g * cap)
  stop_num = sum(math.comb(cap, k) * c[k] ** g * d ** (g * (cap - k)) for k in range(2, cap + 1))
  T = int(kahe.t) ** om
  return (math.comb(cap, 2) * D + stop_num * T) * 2 ** s.eps_mse <= T * D


def simulate_iblt_shape(s, kahe):
  """--check-only --empirical: simulate the given IBLT shape for the stopping-set budget
  15/16 2^-eps_MSE used by the search.  Returns the info dict check_all expects."""
  from iblt_common import Simulator, _load_lib
  from iblt_empirical import evaluate
  m = kahe.mse
  target_log2 = -s.eps_mse + log2(15.0 / 16)
  target = 2.0 ** target_log2
  trials = warn_if_simulation_infeasible(s, target)
  progress("simulating the IBLT shape %d x %d (%d trials; seed %d) ..."
           % (m.theta, m.delta, trials, s.seed))
  if not _load_lib():
    print("warning: no C compiler found, using the slow pure-Python simulator", file=sys.stderr)
  sim = Simulator(s.sim_workers or os.cpu_count() or 1, s.seed)
  try:
    ok, f, n, ub, _ = evaluate(sim, s.cap, m.theta, m.delta, trials, target, s.sim_alpha)
  finally:
    sim.close()
  return dict(mode="simulate", provable_shape=None, target_log2=target_log2, used=True,
              result=dict(failures=f, trials=n, alpha=s.sim_alpha,
                          upper_log2=log2(ub) if ub > 0 else float('-inf'), passed=ok))


def check_only(args):
  """--check-only: check the parameter set in args.check_only; exit status 1 on failure."""
  if args.confirm_with_estimator and not HAVE_ESTIMATOR:
    sys.exit("--confirm-with-estimator needs the lattice estimator; run with "
             "'sage -python' from a directory where 'estimator' is importable.")
  try:
    s, kahe, leaf, avc = read_params_json(args.check_only, args)
  except ParamFileError as e:
    print(red("ERROR: %s: %s" % (args.check_only, e)))
    sys.exit(2)
  progress("checking the parameters in %s ..." % args.check_only)
  if args.empirical:
    if s.mse != "IBLT":
      print(yellow("note: --empirical only applies to the IBLT multiset encoding; ignored"))
    elif iblt_provable_ok(s, kahe):
      progress("IBLT shape satisfies the provable bound; no simulation needed")
    else:
      kahe = dataclasses.replace(kahe, mse=dataclasses.replace(
        kahe.mse, empirical=simulate_iblt_shape(s, kahe)))
  rows, all_ok, stats = check_all(s, kahe, leaf, avc)
  print_checks(rows, stats)
  if stats.get("mse_empirical"):
    print(yellow("NOTE: the MSE stopping-set condition was checked by simulation only "
                 "(status EMPIRICAL): %s." % stats["mse_empirical_detail"]) + "\n")
  confirmed = True
  if args.confirm_with_estimator:
    progress("confirming hardness with the lattice estimator ...")
    crow, confirmed = confirm_with_estimator(s, kahe, leaf, avc)
    print_confirmation(crow)
  if not all_ok:
    print(red("FAILED: some conditions of the paper fail for these parameters "
              "(see the condition table above)."))
  if not confirmed:
    print(red("FAILED: the lattice estimator does not confirm 2^%d hardness for every "
              "problem (see the estimator table above)." % s.secpar))
  if all_ok and confirmed:
    progress("all checks passed")
    sys.exit(0)
  sys.exit(1)


def parse_args(argv=None):
  ap = argparse.ArgumentParser(
    description="Generate and check parameters for Panetiere (anonymous broadcast).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  ap.add_argument("--rho", type=int, default=300, help="number of clients")
  ap.add_argument("--capacity", type=int, default=None,
                  help="MSE capacity C <= rho, i.e. the maximum number of senders when clients "
                       "without a message send cover traffic (sec:cover); guaranteed output "
                       "delivery then holds only against at most C - h corrupted clients, "
                       "h = number of honest senders (default: rho)")
  ap.add_argument("--gamma", type=int, default=8, help="number of servers")
  ap.add_argument("--n", type=int, default=2048, help="ring dimension (power of two)")
  ap.add_argument("--secpar", type=int, default=128, help="security parameter lambda")
  ap.add_argument("--msg-bytes", type=int, default=16 * 1024, help="message size in bytes")
  ap.add_argument("--eps-kahe", type=int, default=100, help="KAHE decryption failure 2^-eps")
  ap.add_argument("--eps-mse", type=int, default=40,
                  help="MSE decoding failure 2^-eps (only affects output delivery, not anonymity)")
  ap.add_argument("--max-q-bits", type=int, default=64, help="upper limit on all moduli")
  ap.add_argument("--t-ntt-log", type=int, default=20,
                  help="choose t = 1 mod 2^K (NTT-friendly plaintext modulus; 0: smallest "
                       "prime above 2^b)")
  ap.add_argument("--log-queries", type=int, default=64,
                  help="log2 of the bound Q on the adversary's random-oracle queries (thm:broadcast)")
  ap.add_argument("--mse", choices=["IBLT", "NG"], default="NG", type=str.upper,
                  help="multiset encoding: invertible Bloom lookup table or Newton-Girard")
  ap.add_argument("--empirical", action="store_true",
                  help="choose the IBLT shape by simulation instead of by the provable bound "
                       "(feasible up to about eps_MSE = 25), used only if it has fewer cells "
                       "than the provable shape; with --check-only, check an IBLT shape that "
                       "misses the provable bound by simulation")
  ap.add_argument("--sim-alpha", type=float, default=0.05,
                  help="--empirical: confidence 1 - alpha of the simulation tests")
  ap.add_argument("--sim-trials-factor", type=float, default=10.0,
                  help="--empirical: trials per tested shape = factor * 2^eps_MSE")
  ap.add_argument("--sim-resolution", type=float, default=0.01,
                  help="--empirical: relative resolution of the search over delta")
  ap.add_argument("--sim-workers", type=int, default=0,
                  help="--empirical: simulation processes (0: number of CPUs)")
  ap.add_argument("--seed", type=int, default=argparse.SUPPRESS,
                  help="--empirical: simulation seed (default: random)")
  ap.add_argument("--verbose", action="store_true", help="print all candidates of each search")
  ap.add_argument("--log-search", action="store_true",
                  help="print every hardness estimate (RLWE and SIS) made during the search")
  ap.add_argument("--confirm-with-estimator", action="store_true",
                  help="re-check the final parameters with the lattice estimator (needs Sage; "
                       "slow) and warn if it does not confirm them")
  ap.add_argument("--json", metavar="PATH", help="write the parameters as JSON")
  ap.add_argument("--check-only", metavar="JSON_FILE",
                  help="do not generate parameters: read them from JSON_FILE (as written by "
                       "--json; needs the jsonschema package) and run all checks; exits with status "
                       "1 if a check fails, 2 if the file is invalid. All "
                       "other options are ignored except --log-queries, --empirical, --sim-*, "
                       "--seed and --confirm-with-estimator")
  return ap.parse_args(argv)


def main(argv=None):
  global LOG_SEARCH
  args = parse_args(argv)
  if args.confirm_with_estimator and not HAVE_ESTIMATOR:
    sys.exit("--confirm-with-estimator needs the lattice estimator; run with "
             "'sage -python' from a directory where 'estimator' is importable.")
  LOG_SEARCH = args.log_search
  if getattr(args, "seed", None) is None:   # random simulation seed unless --seed is set
    args.seed = secrets.randbelow(2 ** 63)
  if args.check_only:
    return check_only(args)
  if args.empirical and args.mse != "IBLT":
    sys.exit("--empirical only applies to the IBLT multiset encoding (--mse IBLT).")
  if args.mse == "IBLT" and iblt_common.np is None:
    print(yellow("WARNING: numpy is not installed; the IBLT shape search falls back to exact "
                 "integer arithmetic,\n         which can be very slow for large rho (minutes at "
                 "rho = 1000).  Install it with: pip install numpy") + "\n", flush=True)
  if args.capacity is not None and not 2 <= args.capacity <= args.rho:
    sys.exit("--capacity must satisfy 2 <= C <= rho.")
  capacity = None if args.capacity in (None, args.rho) else args.capacity
  s = Setting(rho=args.rho, capacity=capacity, gamma=args.gamma, n=args.n, secpar=args.secpar,
              msg_bytes=args.msg_bytes, eps_kahe=args.eps_kahe, eps_mse=args.eps_mse,
              max_q_bits=args.max_q_bits, t_ntt_log=args.t_ntt_log,
              log_queries=args.log_queries, mse=args.mse, empirical=args.empirical,
              sim_alpha=args.sim_alpha, sim_trials_factor=args.sim_trials_factor,
              sim_resolution=args.sim_resolution, sim_workers=args.sim_workers, seed=args.seed)

  res = find_params(s, args.verbose)
  if res is None:
    sys.exit(1)
  kahe, leaf, avc = res

  progress("checking all bounds ...")
  rows, all_ok, stats = check_all(s, kahe, leaf, avc)
  print_checks(rows, stats)

  confirmed = True
  if args.confirm_with_estimator:
    progress("confirming hardness with the lattice estimator ...")
    crow, confirmed = confirm_with_estimator(s, kahe, leaf, avc)
    print_confirmation(crow)

  sizes = compute_sizes(s, kahe, leaf, avc)
  print_parameters(s, kahe, leaf, avc, sizes)

  # warnings last, so they are not scrolled away by the parameter table
  if not all_ok:
    print(red("WARNING: some conditions of the paper FAIL for these parameters "
              "(see the condition table above).") + "\n")
  if stats.get("mse_empirical"):
    print(yellow("NOTE: the IBLT shape %d x %d was chosen empirically (--empirical).\n"
                 "      Only the MSE stopping-set condition is therefore not proven (status "
                 "EMPIRICAL);\n      all other conditions are checked as usual.\n      %s."
                 % (kahe.mse.theta, kahe.mse.delta, stats["mse_empirical_detail"]))
          + "\n")
  if not confirmed:
    print(red("WARNING: the lattice estimator does not confirm 2^%d hardness for every "
              "problem (see the estimator table above)." % s.secpar) + "\n")

  # 1: exit status 1 and no JSON file if a check or the estimator fails
  if not (all_ok and confirmed):
    if args.json:
      print(red("Not writing %s, since the parameters fail the checks." % args.json))
    sys.exit(1)
  if args.json:
    write_json(args.json, s, kahe, leaf, avc)
  progress("done")
  return s, kahe, leaf, avc


if __name__ == "__main__":
  main()
