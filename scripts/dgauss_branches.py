#!/usr/bin/env python3
# Check that the discrete Gaussian sampler (src/dgauss.rs) compiles without branches
# on the sample.
# Written by Claude (Anthropic).
#
# Usage, from anywhere in the repository:
#   python3 scripts/dgauss_branches.py                                   # x86_64 and aarch64
#   python3 scripts/dgauss_branches.py --target aarch64-unknown-linux-gnu
#
# Needs the Rust targets (e.g. rustup target add aarch64-unknown-linux-gnu), but no
# linker, since only the library's assembly is emitted.  CI runs it with the latest
# stable toolchain; run it locally after changing the sampler: what the compiler does
# with the kernels depends on both.  It builds with RUSTFLAGS="", like CI, so a local
# target-cpu=native doesn't change what is checked.
#
# For each target, the script builds the library in release mode with --emit asm and
# reads the functions that handle the sample: dgauss::count (the kernels' dispatch,
# with the scalar kernel normally inlined), dgauss::count_avx2 on x86_64, and
# dgauss::with_signs (the sign).  Functions of dgauss that these call are checked as
# well, and any other call is reported.  Every conditional branch must be one of
#   - a loop branch: it jumps backward, and its flags come from comparing or counting
#     a register against an immediate, e.g. `cmpq $1504, %rax` or `subs x2, x2, #24`;
#   - the run-time CPU feature check: its flags come from a value loaded from, or
#     returned by, std_detect.
# Anything else is printed with its context, and the script fails.  This is a
# heuristic over the form of the assembly: it can't prove that a branch doesn't depend
# on the sample, and a new compiler may produce harmless branches it doesn't recognize.
# Read what it reports.

import argparse
import glob
import os
import re
import subprocess
import sys

TARGETS = ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"]
# The mangled name of a function in dgauss: _ZN 9panetiere 6dgauss <len><name> 17h<hash> E
FUNCTION = re.compile(r"^(_ZN9panetiere6dgauss(\d+)(\w+?)17h[0-9a-f]{16}E):")
ROOTS = {"x86_64": {"count", "count_avx2", "with_signs"}, "aarch64": {"count", "with_signs"}}

ARCH = {
  "x86_64": {
    "branch": re.compile(r"^j(?!mp\b)[a-z]+\s+(\S+)$"),
    "setter": re.compile(r"^(cmp|test|sub|add|and|or|xor|inc|dec|neg)[bwlq]?\s"),
    "counter": re.compile(r"^(cmp|sub|add)[bwlq]?\s+\$-?\d+,\s*%\w+$|^(inc|dec)[bwlq]?\s+%\w+$"),
    "call": re.compile(r"^call[q]?\s+\*?(\S+)"),
    "comment": re.compile(r"\s*#.*$"),
  },
  "aarch64": {
    "branch": re.compile(r"^(?:b\.[a-z]+|cbn?z\s+\w+,|tbn?z\s+\w+,\s*#\d+,)\s*(\S+)$"),
    "setter": re.compile(r"^(cmp|cmn|subs|adds|tst|ands|ccmp|ccmn)\s"),
    "counter": re.compile(r"^(cmp|cmn)\s+[wx]\d+,\s*#-?\d+$|^(subs|adds)\s+[wx]\d+,\s*[wx]\d+,\s*#-?\d+$"),
    "call": re.compile(r"^blr?\s+(\S+)"),
    "comment": re.compile(r"\s*//.*$"),
  },
}


def functions(asm, comment):
  """name -> list of lines (instructions and labels) for every function of dgauss."""
  lines = asm.split("\n")
  out = {}
  for i, line in enumerate(lines):
    m = FUNCTION.match(line)
    if not m:
      continue
    name = m.group(3)[:int(m.group(2))]
    end = next(j for j in range(i, len(lines)) if lines[j].startswith(".Lfunc_end"))
    body = []
    for x in lines[i + 1:end]:
      x = comment.sub("", x).strip()
      if x and (x.endswith(":") or not x.startswith(".")):
        body.append(x)
    out[name] = body
  return out


def check(body, name, rules, report):
  """Classify the conditional branches of one function; return the functions of dgauss
  it calls, and add anything unrecognized to report."""
  labels = {x[:-1]: i for i, x in enumerate(body) if x.endswith(":")}
  callees = set()
  for k, ins in enumerate(body):
    c = rules["call"].match(ins)
    if c:
      target = c.group(1)
      m = FUNCTION.match(target + ":")
      if m:
        callees.add(m.group(3)[:int(m.group(2))])
      elif "std_detect" not in target:
        report.append((name, f"call of {target}", body[max(0, k - 3):k + 1]))
      continue
    b = rules["branch"].match(ins)
    if not b:
      continue
    j = k - 1
    while j >= 0 and not rules["setter"].match(body[j]):
      j -= 1
    setter = body[j] if j >= 0 else "(none)"
    backward = labels.get(b.group(1), len(body)) < k
    if backward and rules["counter"].match(setter):
      kind = "loop"
    elif any("std_detect" in x for x in body[max(0, j - 6):k]):
      kind = "feature check"
    else:
      kind = None
    print(f"  {name}: {ins:24} {'back' if backward else 'fwd ':4}  flags from {setter:32} {kind or 'NOT RECOGNIZED'}")
    if kind is None:
      report.append((name, f"branch {ins!r}", body[max(0, j - 3):k + 1]))
  return callees


def main():
  ap = argparse.ArgumentParser(description="Check the sampler for branches on the sample.")
  ap.add_argument("--target", action="append", help="Rust target triple (default: %s)" % ", ".join(TARGETS))
  args = ap.parse_args()
  root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
  target_dir = os.path.join(root, "target", "dgauss-asm")
  env = dict(os.environ, RUSTFLAGS="")
  failed = False
  for target in args.target or TARGETS:
    arch = target.split("-")[0]
    if arch not in ARCH:
      sys.exit(f"no rules for {arch}")
    rules = ARCH[arch]
    deps = os.path.join(target_dir, target, "release", "deps")
    # Rebuild the library, so that the assembly is current: cargo only writes it when
    # it compiles, and doesn't compile what it considers up to date.
    subprocess.run(["cargo", "clean", "--quiet", "-p", "panetiere", "--release", "--target", target,
                    "--target-dir", target_dir], cwd=root, env=env, check=True)
    subprocess.run(["cargo", "rustc", "--quiet", "-p", "panetiere", "--release", "--lib", "--target", target,
                    "--target-dir", target_dir, "--", "--emit", "asm"], cwd=root, env=env, check=True)
    [path] = glob.glob(os.path.join(deps, "panetiere-*.s"))
    print(target)
    with open(path) as f:
      funcs = functions(f.read(), rules["comment"])
    report = []
    missing = ROOTS[arch] - funcs.keys()
    if missing:
      print(f"  functions not found: {sorted(missing)}")
      failed = True
    todo, done = sorted(ROOTS[arch] & funcs.keys()), set()
    while todo:
      name = todo.pop()
      if name in done:
        continue
      done.add(name)
      for callee in check(funcs[name], name, rules, report):
        if callee not in funcs:
          report.append((name, f"call of dgauss::{callee}, which is not in the assembly", []))
        elif callee not in done:
          todo.append(callee)
    print(f"  checked: {', '.join(sorted(done))}")
    for name, what, context in report:
      print(f"  {name}: unrecognized {what}; context:")
      for x in context:
        print("      " + x)
      failed = True
  sys.exit(1 if failed else 0)


if __name__ == "__main__":
  main()
