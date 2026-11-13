"""Shared code for iblt_empirical.py and iblt_extrapolate.py.

Model: the IBLT used in the paper (sec:mse_iblt).  Each of the rho elements is
inserted into one uniformly random cell in each of the theta rows (the cells
only depend on the element's random key), and decoding peels cells containing
exactly one element.  Decoding fails iff peeling gets stuck before all rho
elements are recovered.

Key collisions are NOT included, neither in the simulation (elements are
assumed to have distinct keys) nor in the bounds.  Their probability depends on
the key space (at most C(rho,2) t^-omega for keys in Z_t^omega) and can be
computed exactly, so add it separately to the failure probability.

Contents:
  - the simulation core (a small C function compiled on first use, cached next
    to this file under a name containing a hash of its source; pure-Python
    fallback without a C compiler) and a multiprocessing driver;
  - Clopper-Pearson confidence bounds;
  - the exact union bound on stopping sets from the paper / generate_params.py,
        sum_{s=2}^{rho} C(rho,s) p_delta(s)^theta,
    where p_delta(s) is the probability that s balls in delta bins leave no
    singleton (numpy if available: floating point without cancellation, agrees
    with exact integer arithmetic to about 1e-12 in log2; otherwise exact and
    slow for rho beyond a few hundred), and the smallest provable delta;
  - two rigorous lower bounds on the failure probability, used to exclude thetas
    without simulation:
      * pair collisions: two elements sharing all theta cells can never be
        peeled; Bonferroni over the C(rho,2) pairs gives
        Pr[fail] >= N p - (N p)^2 / 2 with p = delta^-theta (excludes small thetas);
      * no singleton: if no row has a singleton cell, peeling cannot start;
        Pr[fail] >= p_delta(rho)^theta (excludes large thetas);
  - the search over theta and the result table.  The thetas are visited from
    the one with the fewest estimated cells outwards, down to theta = 2 and up
    until the no-singleton bound excludes every larger theta; shapes with at
    least as many cells as the best one found so far are not considered.
"""

import ctypes
import functools
import hashlib
import math
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import tempfile
import threading
import time
import types

try:
    import numpy as np
except ImportError:  # the exact (slow) union bound is used instead
    np = None

# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------

C_SOURCE = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

/* xoshiro256** */
static inline uint64_t next(uint64_t *s) {
  uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
  s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
  return r;
}

static inline uint64_t splitmix(uint64_t *x) {
  uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

/* uniform in [0, n) without modulo bias (Lemire) */
static inline uint32_t uniform(uint64_t *s, uint32_t n) {
  uint64_t m = (uint64_t)(uint32_t)(next(s) >> 32) * n;
  uint32_t l = (uint32_t)m;
  if (l < n) {
    uint32_t th = (uint32_t)(-n) % n;
    while (l < th) { m = (uint64_t)(uint32_t)(next(s) >> 32) * n; l = (uint32_t)m; }
  }
  return (uint32_t)(m >> 32);
}

/* Runs `trials` independent IBLT experiments with rho elements, theta rows and
   delta cells per row.  Failures that leave more than s0 elements stuck are
   also counted separately in *large.  Stops early once more than `maxfail`
   failures or more than `maxlarge` large failures occurred.  Writes the number
   of trials actually run to *done and returns the failures. */
long long simulate(int rho, int theta, int delta, long long trials,
                   unsigned long long seed, long long maxfail, int s0, long long maxlarge,
                   long long *done, long long *large) {
  int cells = theta * delta;
  int *cell = malloc(sizeof(int) * (size_t)rho * theta);
  int *cnt = malloc(sizeof(int) * cells);
  long long *acc = malloc(sizeof(long long) * cells);
  int *stack = malloc(sizeof(int) * ((size_t)cells + (size_t)rho * theta));
  char *gone = malloc(rho);
  uint64_t s[4], x = seed;
  for (int i = 0; i < 4; i++) s[i] = splitmix(&x);
  long long fails = 0, nlarge = 0, t;
  for (t = 0; t < trials && fails <= maxfail && nlarge <= maxlarge; t++) {
    memset(cnt, 0, sizeof(int) * cells);
    memset(acc, 0, sizeof(long long) * cells);
    memset(gone, 0, rho);
    for (int i = 0; i < rho; i++)
      for (int r = 0; r < theta; r++) {
        int c = r * delta + (int)uniform(s, (uint32_t)delta);
        cell[i * theta + r] = c; cnt[c]++; acc[c] += i;
      }
    int sp = 0, rec = 0;
    for (int c = 0; c < cells; c++) if (cnt[c] == 1) stack[sp++] = c;
    while (sp) {
      int c = stack[--sp];
      if (cnt[c] != 1) continue;
      int i = (int)acc[c];               /* the only element left in this cell */
      if (gone[i]) continue;
      gone[i] = 1; rec++;
      for (int r = 0; r < theta; r++) {
        int cc = cell[i * theta + r];
        cnt[cc]--; acc[cc] -= i;
        if (cnt[cc] == 1) stack[sp++] = cc;
      }
    }
    if (rec < rho) { fails++; if (rho - rec > s0) nlarge++; }
  }
  *done = t; *large = nlarge;
  free(cell); free(cnt); free(acc); free(stack); free(gone);
  return fails;
}
"""

_lib = None


def _load_lib():
    """Compile the C core once (cached next to this script or in a temp dir)."""
    global _lib
    if _lib is not None:
        return _lib
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        _lib = False
        return _lib
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, tempfile.gettempdir()):
        tag = hashlib.sha256(C_SOURCE.encode()).hexdigest()[:12]
        so = os.path.join(d, f".iblt_sim_{tag}.so")  # recompiled whenever the C source changes
        src = os.path.join(d, f".iblt_sim_{tag}.c")
        try:
            if not os.path.exists(so):
                with open(src, "w") as f:
                    f.write(C_SOURCE)
                subprocess.run([cc, "-O3", "-shared", "-fPIC", "-o", so, src], check=True)
            lib = ctypes.CDLL(so)
            lib.simulate.restype = ctypes.c_longlong
            lib.simulate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong,
                                     ctypes.c_ulonglong, ctypes.c_longlong, ctypes.c_int,
                                     ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong),
                                     ctypes.POINTER(ctypes.c_longlong)]
            _lib = lib
            return _lib
        except (OSError, subprocess.CalledProcessError):
            continue
    _lib = False
    return _lib


def _simulate_py(rho, theta, delta, trials, seed, maxfail, s0, maxlarge):
    """Pure-Python fallback (slow), same interface as the C core."""
    rng = random.Random(seed)
    fails = large = 0
    t = 0
    while t < trials and fails <= maxfail and large <= maxlarge:
        t += 1
        cell = [[r * delta + rng.randrange(delta) for r in range(theta)] for _ in range(rho)]
        cnt = [0] * (theta * delta)
        acc = [0] * (theta * delta)
        for i, cs in enumerate(cell):
            for c in cs:
                cnt[c] += 1
                acc[c] += i
        stack = [c for c in range(theta * delta) if cnt[c] == 1]
        gone = [False] * rho
        rec = 0
        while stack:
            c = stack.pop()
            if cnt[c] != 1:
                continue
            i = acc[c]
            if gone[i]:
                continue
            gone[i] = True
            rec += 1
            for cc in cell[i]:
                cnt[cc] -= 1
                acc[cc] -= i
                if cnt[cc] == 1:
                    stack.append(cc)
        if rec < rho:
            fails += 1
            if rho - rec > s0:
                large += 1
    return fails, t, large


def _worker(args):
    """One batch of trials in a worker process (C core, or the Python fallback)."""
    rho, theta, delta, trials, seed, maxfail, s0, maxlarge = args
    lib = _load_lib()
    if lib:
        done, large = ctypes.c_longlong(0), ctypes.c_longlong(0)
        f = lib.simulate(rho, theta, delta, trials, seed, maxfail, s0, maxlarge,
                         ctypes.byref(done), ctypes.byref(large))
        return f, done.value, large.value
    return _simulate_py(rho, theta, delta, trials, seed, maxfail, s0, maxlarge)


def _exit_with_parent(parent):
    """Pool initializer: exit the worker as soon as its parent process is gone.  If the
    parent is killed (e.g. SIGTERM or SIGKILL), multiprocessing cannot shut the pool
    down, and idle workers would otherwise wait for the next task forever (each holds
    the write end of the task pipe, so it never sees EOF)."""
    def watch():
        while os.getppid() == parent:
            time.sleep(1)
        os._exit(1)
    threading.Thread(target=watch, daemon=True).start()


class Simulator:
    """Runs trials in rounds across worker processes, with early rejection."""

    def __init__(self, workers, seed):
        self.workers = workers
        self.seed = seed
        self.calls = 0
        self.pool = None  # created on first use

    def run(self, rho, theta, delta, trials, maxfail=1 << 62, s0=None, maxlarge=1 << 62,
            round_size=200000):
        """Runs up to `trials` trials and returns (failures, trials run, large
        failures), where large failures leave more than s0 elements stuck
        (default s0 = rho: none).  Stops early once failures > maxfail or large
        failures > maxlarge; since each round runs in parallel, the returned
        counts can then exceed these limits."""
        if self.pool is None and self.workers > 1:
            self.pool = mp.Pool(self.workers, initializer=_exit_with_parent,
                                initargs=(os.getpid(),))
        s0 = rho if s0 is None else s0
        fails = done = large = 0
        while done < trials and fails <= maxfail and large <= maxlarge:
            n = min(round_size * self.workers, trials - done)
            per = [n // self.workers + (1 if k < n % self.workers else 0) for k in range(self.workers)]
            jobs = []
            for k, m in enumerate(per):
                if m == 0:
                    continue
                self.calls += 1
                jobs.append((rho, theta, delta, m, (self.seed * 1000003 + self.calls) & (2**64 - 1),
                             maxfail - fails, s0, maxlarge - large))
            res = self.pool.map(_worker, jobs) if self.pool else [_worker(j) for j in jobs]
            for f, d, lg in res:
                fails += f
                done += d
                large += lg
        return fails, done, large

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def binom_cdf(k, n, q):
    """P[Bin(n, q) <= k] for small k (summed in log space)."""
    if q <= 0:
        return 1.0
    if q >= 1:
        return 1.0 if k >= n else 0.0
    lq, l1q = math.log(q), math.log1p(-q)
    total = 0.0
    for i in range(0, k + 1):
        lt = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lq + (n - i) * l1q
        total += math.exp(lt)
    return min(total, 1.0)


def cp_upper(k, n, alpha):
    """One-sided Clopper-Pearson upper confidence bound on the failure rate."""
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


@functools.lru_cache(maxsize=None)
def max_failures(n, target, alpha):
    """Largest k such that k failures in n trials still certify rate <= target."""
    k = -1
    while cp_upper(k + 1, n, alpha) <= target:
        k += 1
    return k


# ---------------------------------------------------------------------------
# Union bound (upper bound on the failure probability)
# ---------------------------------------------------------------------------

_nosingleton_cache = {}


def nosingleton_count(delta, s):
    """Number of maps [s] -> [delta] with no bin hit exactly once (exact, by
    inclusion-exclusion over the bins hit exactly once)."""
    tot, fall = 0, 1
    for j in range(0, min(s, delta) + 1):
        if j > 0:
            fall *= (s - j + 1)
        tot += (-1) ** j * math.comb(delta, j) * fall * (delta - j) ** (s - j)
    return tot


def nosingleton_counts(delta, N):
    """[nosingleton_count(delta, s) for s = 0..N] (cached)."""
    key = (delta, N)
    if key not in _nosingleton_cache:
        _nosingleton_cache[key] = [nosingleton_count(delta, s) for s in range(N + 1)]
    return _nosingleton_cache[key]


_probs_cache = {}
_grid_cache = {}


def _grid(rho):
    """Per-rho index arrays for nosingleton_probs: the lower triangle k <= s, k,
    s - k, log C(s,k), and the chain p_1, p_2, p_4, ... (filled on demand)."""
    if rho not in _grid_cache:
        s_idx = np.arange(rho + 1)
        lgs = np.array([math.lgamma(i + 1) for i in range(rho + 1)])
        S, K = np.meshgrid(s_idx, s_idx, indexing="ij")
        lower = K <= S
        Kc, SK = np.where(lower, K, 0), np.where(lower, S - K, 0)
        p1 = np.ones(rho + 1)
        p1[1] = 0.0
        _grid_cache[rho] = (lower, Kc, SK, lgs[S] - lgs[Kc] - lgs[SK], [p1])
    return _grid_cache[rho]


def nosingleton_probs(delta, rho):
    """p_delta(s) for s = 0..rho in floating point (numpy).  Splitting the delta
    bins into groups of a and b bins, the balls in the first group are
    Binomial(s, a/(a+b)) and both groups must be singleton-free:
        p_{a+b}(s) = sum_k Binom(s, a/(a+b))(k) p_a(k) p_b(s-k).
    All terms are nonnegative (no cancellation), so this is numerically stable.
    p_delta is combined from p_1, p_2, p_4, ... along the binary digits of delta
    (the chain of powers of two is shared between calls), O(rho^2 log delta)."""
    key = (delta, rho)
    if key in _probs_cache:
        return _probs_cache[key]
    lower, Kc, SK, logC, chain = _grid(rho)

    def combine(pa, a, pb, b):
        r = a / (a + b)
        w = np.exp(np.where(lower, logC + Kc * math.log(r) + SK * math.log1p(-r), -np.inf))
        return (w * pb[SK]) @ pa

    acc, acc_n = None, 0
    d, k = delta, 0
    while d:
        while len(chain) <= k:
            chain.append(combine(chain[-1], 1 << (len(chain) - 1), chain[-1], 1 << (len(chain) - 1)))
        if d & 1:
            pw, pw_n = chain[k], 1 << k
            acc, acc_n = (pw, pw_n) if acc is None else (combine(acc, acc_n, pw, pw_n), acc_n + pw_n)
        d >>= 1
        k += 1
    _probs_cache[key] = acc
    return acc


@functools.lru_cache(maxsize=None)
def _log2_binomials(rho):
    """log2 C(rho, s) for s = 0..rho (numpy array)."""
    return np.array([math.lgamma(rho + 1) - math.lgamma(i + 1) - math.lgamma(rho - i + 1)
                     for i in range(rho + 1)]) / math.log(2)


def union_bound_log2(rho, theta, delta, s_max=None):
    """log2 of sum_{s=2}^{s_max} C(rho,s) p_delta(s)^theta (default s_max = rho).

    With s_max = rho this is the union bound over all stopping sets, an upper
    bound on the decoding failure probability.  With s_max < rho it bounds the
    probability that decoding gets stuck with at most s_max elements left (the
    stuck set is then a stopping set of size 2..s_max).  Floating point via
    nosingleton_probs if numpy is available, otherwise exact integer arithmetic."""
    s_max = rho if s_max is None else min(s_max, rho)
    if np is None:
        c = nosingleton_counts(delta, rho)
        num = sum(math.comb(rho, s) * c[s] ** theta * delta ** (theta * (rho - s))
                  for s in range(2, s_max + 1))
        return float("-inf") if num == 0 else math.log2(num) - theta * rho * math.log2(delta)
    p = nosingleton_probs(delta, rho)[2:s_max + 1]
    with np.errstate(divide="ignore"):
        t = _log2_binomials(rho)[2:s_max + 1] + theta * np.log2(p)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return float("-inf")
    m = t.max()
    return float(m + math.log2(np.sum(np.exp2(t - m))))


def smallest_delta(bound_log2, target_log2, start, limit=None):
    """Smallest delta >= 2 with bound_log2(delta) <= target_log2, for a bound
    that is decreasing in delta (doubling from start, then binary search).
    With a limit, returns the limit if the bound is not reached below it."""
    lo, hi = 2, max(2, start)
    while bound_log2(hi) > target_log2:
        if limit is not None and hi >= limit:
            return limit
        lo, hi = hi + 1, 2 * hi
        if limit is not None:
            hi = min(hi, limit)
    while lo < hi:
        mid = (lo + hi) // 2
        if bound_log2(mid) <= target_log2:
            hi = mid
        else:
            lo = mid + 1
    return lo


def provable_delta(rho, theta, target, start):
    """Smallest delta whose union bound is at most target."""
    return smallest_delta(lambda d: union_bound_log2(rho, theta, d), math.log2(target), start)


# ---------------------------------------------------------------------------
# Rigorous lower bounds on the failure probability (used to exclude thetas
# without simulation; no monotonicity assumptions)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def peeling_factor(theta):
    """Asymptotic peeling threshold: decoding fails w.h.p. once
    rho > c*_theta * (theta*delta), with c*_theta = min_{x>0} x / (theta (1-e^-x)^(theta-1))
    (2-core threshold of random theta-uniform hypergraphs).  Returns 1/c*_theta,
    i.e. the minimum number of cells per element (2, 1.22, 1.30, 1.43, ... for theta = 2, 3, 4, 5)."""
    if theta <= 2:
        return 2.0
    best = float("inf")
    for i in range(1, 20001):
        x = i / 1000
        best = min(best, x / (theta * (1 - math.exp(-x)) ** (theta - 1)))
    return 1 / best


@functools.lru_cache(maxsize=None)
def nosingle_log2(delta, s):
    """log2 p_delta(s), the probability that s balls in delta bins leave no
    singleton (exact)."""
    if s == 0:
        return 0.0
    tot = nosingleton_count(delta, s)
    return float("-inf") if tot <= 0 else math.log2(tot) - s * math.log2(delta)


def pair_excluded(rho, theta, cap, target):
    """True if every delta <= cap fails, by the probability that two elements
    share all theta cells (then neither can be peeled).  Bonferroni over k of
    the N = C(rho,2) pairs: Pr[fail] >= k p - (k p)^2 / 2 with p = delta^-theta;
    k = N gives N p - (N p)^2 / 2 if N p < 1, and k = floor(1/p) gives >= 3/8
    otherwise.  Both lower bounds are decreasing in delta, so delta = cap is the
    worst case."""
    if cap < 2:
        return rho >= 2
    x = math.exp(math.log(math.comb(rho, 2)) - theta * math.log(cap))
    lb = 0.375 if x >= 1 else x - x * x / 2
    return lb > target


def nosingleton_excluded(rho, theta, cap, target):
    """True if every delta <= cap fails, by the probability that no row has a
    singleton cell at all (then peeling cannot start): Pr[fail] >= p_delta(rho)^theta."""
    t = math.log2(target)
    for delta in range(cap, 0, -1):  # large delta is the likely counterexample
        if theta * nosingle_log2(delta, rho) <= t:
            return False
    return True


def excluded(rho, theta, max_cells, target):
    """Reason why no shape with this theta and theta*delta <= max_cells can
    reach the target, or None."""
    cap = max_cells // theta
    if pair_excluded(rho, theta, cap, target):
        return "pair collision bound"
    if cap <= rho and nosingleton_excluded(rho, theta, cap, target):
        return "no-singleton bound"
    return None


def all_larger_excluded(rho, theta, max_cells, target):
    """True if every theta' > theta is excluded by the no-singleton bound, i.e.
    for every delta <= max_cells // (theta+1) and the largest theta' with
    theta' * delta <= max_cells, p_delta(rho)^theta' > target."""
    t = math.log2(target)
    for delta in range(max_cells // (theta + 1), 0, -1):
        if (max_cells // delta) * nosingle_log2(delta, rho) <= t:
            return False
    return True


# ---------------------------------------------------------------------------
# Command line and the search over theta (shared by both scripts)
# ---------------------------------------------------------------------------

def parse_range(s):
    """'3-8' -> [3, ..., 8], '3,4,6' -> [3, 4, 6]."""
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def make_args(rho, target, thetas=None, no_bound=False, quiet=True):
    """Arguments for search_thetas when used as a library (target is the failure
    probability, e.g. 2.0 ** -eps)."""
    return types.SimpleNamespace(rho=rho, target=target, thetas=thetas, no_bound=no_bound,
                                 quiet=quiet)


def provable_shape(rho, target, thetas=None, quiet=True):
    """Smallest provable shape (theta, delta), i.e. with the fewest cells such that the
    union bound sum_{s=2}^{rho} C(rho,s) p_delta(s)^theta is at most target, over all
    theta (automatic range, see search_thetas).  Uses the floating-point union bound
    if numpy is available; callers that need a rigorous result should confirm it with
    exact arithmetic."""
    res = search_thetas(make_args(rho, target, thetas, quiet=quiet), None, simulate=False,
                        report=False)
    return res["provable"]


def common_arguments(ap):
    """Options shared by iblt_empirical.py and iblt_extrapolate.py."""
    ap.add_argument("--rho", type=int, required=True, help="number of elements (capacity)")
    ap.add_argument("--eps", type=float, default=20,
                    help="target decoding failure probability 2^-eps (default 20)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="upper confidence bounds on failure rates are at confidence 1 - alpha "
                         "(default 0.05, i.e. 95%%)")
    ap.add_argument("--thetas", default=None,
                    help="only try these numbers of rows, e.g. 3-8 or 3,4,6 (default: all, the range "
                         "is determined automatically)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="number of parallel simulation processes (default: number of CPUs)")
    ap.add_argument("--seed", type=int, default=1,
                    help="random seed; runs with the same seed and number of workers are "
                         "reproducible (default 1)")
    ap.add_argument("--no-bound", action="store_true",
                    help="skip the union bound (no provable column; faster for large rho)")
    ap.add_argument("--quiet", action="store_true",
                    help="only print the result table, not every simulated shape")


def search_thetas(args, solve, head=(), title="", simulate=True, notes=(), report=True):
    """Visits the thetas, calls solve(theta, provable_delta, max_cells) for each
    theta that is not excluded, and prints the result table (unless report is
    False).

    args needs the attributes rho, target, thetas (None: automatic), no_bound and
    quiet; see make_args for use as a library.

    solve returns None (no shape with at most max_cells cells, or none found) or
    (delta, cols) with len(cols) == len(head); max_cells is None for the first
    theta.  The thetas are visited from the one with the fewest estimated cells
    outwards: downwards to theta = 2, upwards until the no-singleton bound
    excludes every larger theta (for both the solved and the provable shapes).
    A theta is excluded without calling solve if a rigorous lower bound on the
    failure probability exceeds the target for every shape smaller than the best
    one so far.  Returns the rows (theta, delta, cols, provable_delta, reason) if
    report is True, and otherwise a dict with the rows, the best solved shape
    ("best": (theta, delta, cols) or None) and the best provable shape
    ("provable": (theta, delta) or None)."""
    rho, T = args.rho, args.target
    N = math.comb(rho, 2)

    def estimate(theta):
        # rough cell count from three necessary conditions on delta, each rounded
        # up to an integer: pair collisions (C(rho,2) delta^-theta <= T, matters
        # for small theta), the asymptotic peeling threshold (large theta), and
        # some row having a singleton (p_delta(rho)^theta <= T, matters when delta
        # is small, e.g. for small rho; Poisson approximation
        # p_delta(rho) ~ (1 - lam e^-lam)^delta with lam = rho/delta, since this is
        # only used to order the thetas)
        d_pair = math.ceil((N / T) ** (1 / theta))
        d_peel = math.ceil(peeling_factor(theta) * rho / theta)

        def nosingle_approx_log2(d):
            lam = rho / d
            return theta * d * math.log2(1 - lam * math.exp(-lam))

        # (the approximation is poor for delta >> rho, where this condition never
        # binds anyway; the limit stops the search there)
        d = max(d_pair, d_peel)
        if nosingle_approx_log2(d) > math.log2(T):  # only then the third condition matters
            d = smallest_delta(nosingle_approx_log2, math.log2(T), d, limit=max(4 * d, 4 * rho))
        return theta * d

    if args.thetas:
        order = sorted(parse_range(args.thetas), key=lambda th: (estimate(th), th))
        start = None
    else:
        start = min(range(2, 65), key=lambda th: (estimate(th), th))
        order = None
    if not args.quiet:
        print("order: " + (", ".join(map(str, order)) if order else
                           f"from theta = {start} outwards (fewest estimated cells) until the "
                           f"remaining thetas are excluded"), flush=True)

    best_cells = None   # smallest solved shape so far
    best_prov = None    # smallest provable shape so far

    def stop_limit():
        # thetas are visited until no shape with fewer cells than both results
        # is possible for the remaining ones
        if (simulate and best_cells is None) or (not args.no_bound and best_prov is None):
            return None
        vals = [c for c in (best_cells if simulate else None,
                            None if args.no_bound else best_prov) if c is not None]
        return max(vals) if vals else None

    def candidates():
        if order is not None:
            yield from order
            return
        yield start
        down, upw = start - 1, start + 1
        while True:
            lim = stop_limit()
            go_up = upw <= 64 if lim is None else not all_larger_excluded(rho, upw - 1, lim - 1, T)
            if not go_up and down < 2:
                return
            if go_up:
                yield upw
                upw += 1
            if down >= 2:
                yield down
                down -= 1

    rows = []
    for theta in candidates():
        res = reason = pdelta = None
        if not args.no_bound:
            # skip the provable search if a rigorous lower bound already shows that
            # it cannot beat the best provable shape (union bound >= Pr[fail])
            if best_prov is None or not excluded(rho, theta, best_prov - 1, T):
                pdelta = provable_delta(rho, theta, T, math.ceil(peeling_factor(theta) * rho / theta))
                if best_prov is None or theta * pdelta < best_prov:
                    best_prov = theta * pdelta
        if simulate:
            if best_cells is not None:
                reason = excluded(rho, theta, best_cells - 1, T)
            if reason:
                if not args.quiet:
                    print(f"  theta = {theta}: excluded without simulation ({reason})", flush=True)
            else:
                if not args.quiet:
                    print(f"  theta = {theta}", flush=True)
                # only shapes strictly smaller than the best one so far
                res = solve(theta, pdelta, None if best_cells is None else best_cells - 1)
                if res is None and not args.quiet:
                    print(f"    theta={theta:2d}: no shape with fewer than {best_cells} cells found"
                          if best_cells is not None else f"    theta={theta:2d}: no shape found",
                          flush=True)
        elif not args.quiet:
            print(f"  theta = {theta}: provable delta "
                  f"{pdelta if pdelta is not None else 'excluded (lower bound)'}", flush=True)
        delta = cols = None
        if res is not None:
            delta, cols = res
            best_cells = theta * delta
        rows.append((theta, delta, cols, pdelta, reason))
    rows.sort(key=lambda r: r[0])
    if not report:
        sol = [r for r in rows if r[1] is not None]
        prov = [r for r in rows if r[3] is not None]
        b = min(sol, key=lambda r: (r[0] * r[1], r[0])) if sol else None
        bp = min(prov, key=lambda r: (r[0] * r[3], r[0])) if prov else None
        return {"rows": rows, "best": (b[0], b[1], b[2]) if b else None,
                "provable": (bp[0], bp[3]) if bp else None}

    full_head = ["theta", "delta", "cells", "cells/rho"] + list(head) + ["delta", "cells", "cells/rho"]
    split = 4 + len(head)
    table = []
    for theta, delta, cols, pdelta, reason in rows:
        if delta is None:
            e = ["-" if not simulate else "excluded" if reason else "pruned"] + [""] * (split - 2)
        else:
            e = [str(delta), str(theta * delta), f"{theta * delta / rho:.2f}"] + list(cols)
        pr = (["-" if args.no_bound else "excluded", "", ""]) if pdelta is None else \
            [str(pdelta), str(theta * pdelta), f"{theta * pdelta / rho:.2f}"]
        table.append([str(theta)] + e + pr)
    w = [max(len(r[i]) for r in [full_head] + table) for i in range(len(full_head))]

    def fmt(r):
        left = "  ".join(x.ljust(w[i]) if x in ("pruned", "excluded") else x.rjust(w[i])
                         for i, x in enumerate(r[:split]))
        right = "  ".join(x.rjust(w[i]) for i, x in enumerate(r[split:], split))
        return left + " | " + right

    lw = sum(w[:split]) + 2 * (split - 1)
    print("\n" + title.center(lw) + " | " + "provable (union bound <= target)")
    print(fmt(full_head))
    for r in table:
        print(fmt(r))
    sol = [r for r in rows if r[1] is not None]
    if sol:
        b = min(sol, key=lambda r: r[0] * r[1])
        print(f"\nsmallest {title.split()[0].lower()}: theta={b[0]}, delta={b[1]} ({b[0] * b[1]} cells, "
              f"{b[0] * b[1] / rho:.2f} rho)")
    prov = [r for r in rows if r[3] is not None]
    if prov:
        b = min(prov, key=lambda r: r[0] * r[3])
        print(f"{'' if sol else chr(10)}smallest provable:  theta={b[0]}, delta={b[3]} "
              f"({b[0] * b[3]} cells, {b[0] * b[3] / rho:.2f} rho)")
    if rows:
        print("\npruned: no shape below the size limit was found; excluded: ruled out by a rigorous "
              "lower bound on the failure probability (no simulation / no union bound search)")
        for line in notes:
            print(line)
        if not args.thetas and stop_limit() is not None:
            print(f"all theta > {max(r[0] for r in rows)} are excluded by the no-singleton bound "
                  f"(for both columns); all theta >= 2 below were considered")
    return rows
