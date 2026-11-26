#!/usr/bin/env python3
"""summarise sim results and draw each round as a timeline.

usage:
  sim/analyze.py target/sim/results.jsonl                 # text summary per round
  sim/analyze.py target/sim/results.jsonl --html out.html # plus a timeline page
  sim/analyze.py target/sim/sweep.jsonl --csv             # one row per round, for sweeps
"""

import argparse
import csv
import html
import io
import json
import re
import statistics
import sys

COMPUTE = "compute"
SEND = "send"
RECEIVE = "receive"


def kind(span):
    name = span["name"]
    if "cpu_us" in span:
        return COMPUTE
    if name.startswith("upload_"):
        return SEND
    return RECEIVE


def group(name):
    """download_post_17 -> download_post"""
    return re.sub(r"_\d+$", "", name)


def ms(span):
    return (span["end_us"] - span["start_us"]) / 1000


def mbit_per_s(span):
    secs = (span["end_us"] - span["start_us"]) / 1e6
    if not span.get("bytes") or secs <= 0:
        return None
    return span["bytes"] * 8 / secs / 1e6


def busy_ms(spans):
    """time covered by at least one span"""
    total, end = 0, None
    for s, e in sorted((s["start_us"], s["end_us"]) for s in spans):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total / 1000


def round_bounds(report):
    phases = report["phases"]
    return phases[0]["start_us"], phases[-1]["end_us"]


def load(paths):
    reports = []
    for path in paths:
        with open(path) as f:
            reports.extend(json.loads(line) for line in f if line.strip())
    return reports


def summarise(report, out):
    start, end = round_bounds(report)
    setup = report["setup"]
    label = f" [{report['label']}]" if report["label"] else ""
    out.write(
        f"round {report['round']}{label}: {setup['servers']} servers, "
        f"{setup['clients']} clients, {setup['messages']} messages of "
        f"{setup['message_bytes']} bytes, total {(end - start) / 1000:.0f} ms\n"
    )
    out.write("  phases: " + ", ".join(f"{p['name']} {ms(p):.0f} ms" for p in report["phases"]) + "\n\n")

    groups = {}
    for node in report["nodes"]:
        role = re.sub(r"-\d+$", "", node["node"])
        for span in node["spans"]:
            groups.setdefault((role, group(span["name"])), []).append((node["node"], span))

    header = f"  {'step':<34}{'n':>5}{'median ms':>11}{'max ms':>9}{'slowest':>12}{'MB':>9}{'Mbit/s':>9}{'cores':>7}\n"
    out.write(header)
    for (role, name), items in groups.items():
        spans = [s for _, s in items]
        durations = [ms(s) for s in spans]
        slowest = max(items, key=lambda i: ms(i[1]))[0]
        total_bytes = sum(s.get("bytes") or 0 for s in spans)
        rates = [r for r in map(mbit_per_s, spans) if r is not None]
        cores = [s["cpu_us"] / 1000 / ms(s) for s in spans if "cpu_us" in s and ms(s) > 0]
        out.write(
            f"  {role + ' ' + name:<34}{len(spans):>5}{statistics.median(durations):>11.1f}"
            f"{max(durations):>9.1f}{slowest:>12}"
            f"{total_bytes / 1e6 if total_bytes else 0:>9.2f}"
            f"{statistics.median(rates) if rates else 0:>9.1f}"
            f"{statistics.median(cores) if cores else 0:>7.1f}\n"
        )

    out.write("\n  share of the round each role spends busy (median over nodes):\n")
    roles = {}
    for node in report["nodes"]:
        role = re.sub(r"-\d+$", "", node["node"])
        roles.setdefault(role, []).append(busy_ms(node["spans"]) / ((end - start) / 1000))
    for role, shares in roles.items():
        out.write(f"    {role:<10}{statistics.median(shares) * 100:>5.0f}%\n")

    out.write("\n  whole-process use per node since the previous report:\n")
    out.write(f"    {'role':<10}{'n':>4}{'cpu ms':>9}{'max':>9}{'peak MB':>9}{'max':>9}\n")
    for role, nodes in by_role(report).items():
        cpu = [n["usage"]["cpu_us"] / 1000 for n in nodes]
        peak = [n["usage"]["peak_rss_bytes"] / 1e6 for n in nodes if "peak_rss_bytes" in n["usage"]]
        out.write(
            f"    {role:<10}{len(nodes):>4}{statistics.median(cpu):>9.0f}{max(cpu):>9.0f}"
            f"{statistics.median(peak) if peak else 0:>9.0f}{max(peak) if peak else 0:>9.0f}\n"
        )
    out.write("\n")


def by_role(report):
    roles = {}
    for node in report["nodes"]:
        if "usage" in node:
            roles.setdefault(re.sub(r"-\d+$", "", node["node"]), []).append(node)
    return roles


def csv_rows(reports):
    rows = []
    for r in reports:
        start, end = round_bounds(r)
        row = {"label": r["label"], "round": r["round"], **r["setup"], "total_ms": (end - start) / 1000}
        for p in r["phases"]:
            row[f"{p['name']}_ms"] = round(ms(p), 1)
        spans = {}
        for node in r["nodes"]:
            for span in node["spans"]:
                spans.setdefault(group(span["name"]), []).append(span)
        median = lambda name, f: round(statistics.median(map(f, spans[name])), 1) if name in spans else None
        row["post_kb"] = median("upload_post", lambda s: s["bytes"] / 1e3)
        row["recipient_download_mb"] = round(
            sum(s["bytes"] for n in ("download_post", "download_entry") for s in spans.get(n, [])) / 1e6, 2
        )
        for name in ("client_round", "unseal", "server_round", "recover", "decode"):
            row[f"{name}_ms"] = median(name, ms)
        for role, nodes in by_role(r).items():
            row[f"{role}_cpu_ms"] = round(max(n["usage"]["cpu_us"] for n in nodes) / 1000, 1)
            peaks = [n["usage"]["peak_rss_bytes"] for n in nodes if "peak_rss_bytes" in n["usage"]]
            row[f"{role}_peak_mb"] = round(max(peaks) / 1e6, 1) if peaks else None
        rows.append(row)
    return rows


def lanes(spans):
    """stack overlapping spans of one node into sub-rows"""
    ends, placed = [], []
    for span in sorted(spans, key=lambda s: s["start_us"]):
        for i, e in enumerate(ends):
            if span["start_us"] >= e:
                ends[i] = span["end_us"]
                placed.append((i, span))
                break
        else:
            ends.append(span["end_us"])
            placed.append((len(ends) - 1, span))
    return len(ends), placed


def timeline_svg(report):
    start, end = round_bounds(report)
    width, label_w, lane_h, gap = 960, 110, 10, 6
    plot_w = width - label_w - 16
    x = lambda us: label_w + (us - start) / max(end - start, 1) * plot_w

    rows, y = [], 28
    for node in report["nodes"]:
        n, placed = lanes(node["spans"])
        rows.append((node["node"], y, n, placed))
        y += max(n, 1) * lane_h + gap
    height = y + 8

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="timeline of round {report["round"]}">']
    last_label = -1e9
    for p in report["phases"]:
        px = x(p["start_us"])
        parts.append(f'<line x1="{px:.1f}" x2="{px:.1f}" y1="14" y2="{height}" class="grid"/>')
        # short phases would print on top of the next label, so they go unlabelled
        if px - last_label > 70 and x(p["end_us"]) - px > 20:
            parts.append(f'<text x="{px + 3:.1f}" y="12" class="axis">{html.escape(p["name"])}</text>')
            last_label = px
    total_ms = (end - start) / 1000
    parts.append(f'<text x="{width - 16}" y="12" class="axis" text-anchor="end">{total_ms:.0f} ms</text>')

    for name, top, _, placed in rows:
        parts.append(f'<text x="{label_w - 8}" y="{top + lane_h - 1}" class="label" text-anchor="end">{html.escape(name)}</text>')
        for lane, span in placed:
            x0, x1 = x(span["start_us"]), x(span["end_us"])
            w = max(x1 - x0, 1.5)
            tip = f"{name} · {span['name']} · {ms(span):.1f} ms"
            if span.get("bytes"):
                tip += f" · {span['bytes'] / 1e3:.1f} kB"
                rate = mbit_per_s(span)
                if rate:
                    tip += f" · {rate:.1f} Mbit/s"
            if "cpu_us" in span and ms(span) > 0:
                tip += f" · {span['cpu_us'] / 1000 / ms(span):.1f} cores"
            parts.append(
                f'<rect x="{x0:.1f}" y="{top + lane * lane_h:.1f}" width="{w:.1f}" height="{lane_h - 2}" '
                f'rx="2" class="{kind(span)}" data-tip="{html.escape(tip)}"/>'
            )
    parts.append("</svg>")
    return "\n".join(parts)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sim timeline</title>
<style>
:root { color-scheme: light; --surface: #fcfcfb; --text: #0b0b0b; --muted: #52514e; --grid: #e4e3df;
  --compute: #2a78d6; --send: #eb6834; --receive: #1baf7a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --surface: #1a1a19; --text: #f0efec; --muted: #c3c2b7; --grid: #383835;
  --compute: #3987e5; --send: #d95926; --receive: #199e70; } }
:root[data-theme="dark"] { color-scheme: dark; --surface: #1a1a19; --text: #f0efec; --muted: #c3c2b7;
  --grid: #383835; --compute: #3987e5; --send: #d95926; --receive: #199e70; }
body { background: var(--surface); color: var(--text); font: 14px/1.4 system-ui, sans-serif; margin: 0; padding: 16px; }
h2 { font-size: 15px; margin: 24px 0 4px; } p { color: var(--muted); margin: 0 0 8px; }
.legend { display: flex; gap: 16px; color: var(--muted); margin-bottom: 8px; }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 2px;
  margin-right: 6px; vertical-align: -1px; background: var(--c); }
.chart { overflow-x: auto; } svg { width: 100%; min-width: 640px; height: auto; display: block; }
.label, .axis { fill: var(--muted); font-size: 10px; } .grid { stroke: var(--grid); stroke-width: 1; }
.compute { fill: var(--compute); } .send { fill: var(--send); } .receive { fill: var(--receive); }
rect:hover { stroke: var(--text); stroke-width: 1; }
pre { color: var(--muted); font-size: 12px; overflow-x: auto; }
#tip { position: fixed; pointer-events: none; background: var(--surface); color: var(--text); border: 1px solid var(--grid);
  border-radius: 4px; padding: 4px 8px; font-size: 12px; display: none; }
</style></head><body>
<div class="legend"><span style="--c: var(--compute)">compute</span><span style="--c: var(--send)">sending</span><span style="--c: var(--receive)">receiving</span></div>
BODY
<div id="tip"></div>
<script>
const tip = document.getElementById("tip");
document.addEventListener("mousemove", (e) => {
  const t = e.target.dataset && e.target.dataset.tip;
  if (!t) { tip.style.display = "none"; return; }
  tip.textContent = t; tip.style.display = "block";
  tip.style.left = Math.min(e.clientX + 12, innerWidth - tip.offsetWidth - 8) + "px";
  tip.style.top = (e.clientY + 12) + "px";
});
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="jsonl files written by the coordinator")
    parser.add_argument("--html", help="write a timeline page here")
    parser.add_argument("--csv", action="store_true", help="print one csv row per round instead")
    args = parser.parse_args()

    reports = load(args.results)
    if args.csv:
        rows = csv_rows(reports)
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        return

    for report in reports:
        summarise(report, sys.stdout)

    if args.html:
        sections = []
        for report in reports:
            text = io.StringIO()
            summarise(report, text)
            label = f" · {html.escape(report['label'])}" if report["label"] else ""
            sections.append(
                f"<h2>Round {report['round']}{label}</h2>"
                f'<div class="chart">{timeline_svg(report)}</div>'
                f"<pre>{html.escape(text.getvalue())}</pre>"
            )
        with open(args.html, "w") as f:
            f.write(PAGE.replace("BODY", "\n".join(sections)))
        print(f"timeline written to {args.html}", file=sys.stderr)


if __name__ == "__main__":
    main()
