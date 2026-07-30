#!/usr/bin/env python3
"""FlexNPUSim runner — the one-command way to run a simulation.

    tools/run.py --config <hw-or-full-config.json> --network <workload.csv>
    tools/run.py --scenario <design>:<network>          # e.g. nvdla-large:resnet18
    ... [--label NAME] [--report]

Resolves the repo root and binary itself, streams the simulator's own
per-layer progress, and leaves results in out/<label>.csv.
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BINARY = os.path.join(ROOT, "build", "sim", "flexnpusim")


def main() -> int:
    ap = argparse.ArgumentParser(description="FlexNPUSim runner")
    ap.add_argument("--config", help="hw spec or full config JSON")
    ap.add_argument("--network", help="workload CSV (model/sw/...)")
    ap.add_argument("--scenario", help="<design>:<network> shorthand")
    ap.add_argument("--label", default="run", help="output name (out/<label>.csv)")
    ap.add_argument("--report", action="store_true",
                    help="also write the four-section report to out/<label>.report.txt")
    args = ap.parse_args()

    if not os.path.exists(BINARY):
        sys.exit(f"error: simulator not built ({BINARY}) — "
                 "run: cmake -B build -S . && cmake --build build -j")
    if not args.scenario and not (args.config and args.network):
        ap.error("give --scenario, or both --config and --network")

    out_csv = os.path.join(ROOT, "out", f"{args.label}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    cmd = [BINARY, "-o", out_csv]
    if args.scenario:
        cmd += ["-scenario", args.scenario]
    else:
        cmd += ["-hw_conf", args.config, "-network", args.network]
    if args.report:
        cmd += ["-report", os.path.join(ROOT, "out", f"{args.label}.report.txt")]

    # Simulator paths are repo-root relative — always run from ROOT.
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        return rc

    with open(out_csv) as fh:
        last = [ln for ln in fh.read().splitlines() if ln.strip()][-1]
    print(f"\nresult row : {last}")
    print(f"result csv : {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
