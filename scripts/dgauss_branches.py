#!/usr/bin/env python3
# Check that the kernels of the discrete Gaussian sampler (src/dgauss.rs) compile
# without branches on the sample.
# Written by Claude (Anthropic).
#
# Usage, from anywhere in the repository:
#   python3 scripts/dgauss_branches.py                                   # x86_64 and aarch64
#   python3 scripts/dgauss_branches.py --target aarch64-unknown-linux-gnu
#
# Needs the Rust targets (e.g. rustup target add aarch64-unknown-linux-gnu), but no
# linker, since only the library's assembly is emitted.  Run it after changing the
# sampler or the toolchain: what the compiler does with the kernels depends on both.
#
# For each target, the script builds the library in release mode with --emit asm and
# reads dgauss::count (with the scalar kernel inlined) and, on x86_64,
# dgauss::count_avx2.  Every conditional branch in them must be one of
#   - a loop branch: its flags come from comparing or counting a register against an
#     immediate, e.g. `cmpq $1504, %rax` or `subs x2, x2, #24`;
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
FUNCTIONS = re.compile(r"^(_ZN9panetiere6dgauss(?:5count|10count_avx2)17h[0-9a-f]+E):")

ARCH = {
  "x86_64": {
    "branch": re.compile(r"^j(?!mp\b)[a-z]+\s"),
    "setter": re.compile(r"^(cmp|test|sub|add|and|or|xor|inc|dec|neg)[bwlq]?\s"),
    "counter": re.compile(r"^(cmp|sub|add)[bwlq]?\s+\$-?\d+,\s*%\w+$|^(inc|dec)[bwlq]?\s+%\w+$"),
    "comment": re.compile(r"\s*#.*$"),
  },
  "aarch64": {
    "branch": re.compile(r"^(b\.[a-z]+|cbn?z|tbn?z)\s"),
    "setter": re.compile(r"^(cmp|cmn|subs|adds|tst|ands|ccmp|ccmn)\s"),
    "counter": re.compile(r"^(cmp|cmn)\s+[wx]\d+,\s*#-?\d+$|^(subs|adds)\s+[wx]\d+,\s*[wx]\d+,\s*#-?\d+$"),
    "comment": re.compile(r"\s*//.*$"),
  },
}


def instructions(lines, comment):
  """The instructions of a function body, without comments, labels and directives."""
  out = []
  for line in lines:
    line = comment.sub("", line).strip()
    if line and not line.startswith(".") and not line.endswith(":"):
      out.append(line)
  return out


def check(asm, arch):
  rules = ARCH[arch]
  lines = asm.split("\n")
  found, bad = [], []
  for i, line in enumerate(lines):
    m = FUNCTIONS.match(line)
    if not m:
      continue
    end = next(j for j in range(i, len(lines)) if lines[j].startswith(".Lfunc_end"))
    body = instructions(lines[i + 1:end], rules["comment"])
    name = "count_avx2" if "count_avx2" in m.group(1) else "count"
    found.append(name)
    for k, ins in enumerate(body):
      if not rules["branch"].match(ins):
        continue
      j = k - 1
      while j >= 0 and not rules["setter"].match(body[j]):
        j -= 1
      setter = body[j] if j >= 0 else "(none)"
      context = body[max(0, j - 3):k + 1]
      if rules["counter"].match(setter):
        kind = "loop"
      elif any("std_detect" in x for x in body[max(0, j - 6):k]):
        kind = "feature check"
      else:
        kind = None
      print(f"  {name}: {ins:24} flags from {setter:32} {kind or 'NOT RECOGNIZED'}")
      if kind is None:
        bad.append((name, ins, context))
  return found, bad


def main():
  ap = argparse.ArgumentParser(description="Check the sampler's kernels for branches on the sample.")
  ap.add_argument("--target", action="append", help="Rust target triple (default: %s)" % ", ".join(TARGETS))
  args = ap.parse_args()
  root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
  target_dir = os.path.join(root, "target", "dgauss-asm")
  failed = False
  for target in args.target or TARGETS:
    arch = target.split("-")[0]
    if arch not in ARCH:
      sys.exit(f"no rules for {arch}")
    deps = os.path.join(target_dir, target, "release", "deps")
    # Rebuild the library, so that the assembly is current: cargo only writes it when
    # it compiles, and doesn't compile what it considers up to date.
    subprocess.run(["cargo", "clean", "--quiet", "-p", "panetiere", "--release", "--target", target,
                    "--target-dir", target_dir], cwd=root, check=True)
    subprocess.run(["cargo", "rustc", "--quiet", "-p", "panetiere", "--release", "--lib", "--target", target,
                    "--target-dir", target_dir, "--", "--emit", "asm"], cwd=root, check=True)
    [path] = glob.glob(os.path.join(deps, "panetiere-*.s"))
    print(target)
    with open(path) as f:
      found, bad = check(f.read(), arch)
    expected = {"count", "count_avx2"} if arch == "x86_64" else {"count"}
    if set(found) != expected:
      print(f"  expected the functions {sorted(expected)}, found {sorted(found)}")
      failed = True
    for name, ins, context in bad:
      print(f"  {name}: unrecognized branch {ins!r}; context:")
      for x in context:
        print("      " + x)
      failed = True
  sys.exit(1 if failed else 0)


if __name__ == "__main__":
  main()
