#!/usr/bin/env python3
"""One-shot: a PyTorch model → network CSV → flexnpusim → parsed results.

Proves the full pytorch-controlled pipeline as a SINGLE flow:

    torch.nn.Module
        │  pytorch_frontend.write_network_csv  (workload: shapes)
        ▼
    network CSV  ── -hw_conf <hw.json> (mapping + system) ──►  flexnpusim
        │  (compiles CSV → DNN image internally, runs on the SystemC system
        │   with the configured DRAM tier — dramsim3 by default)
        ▼
    Results (cycles / MACs / completed)

So "everything is controlled by conf (hw.json) + pytorch (model)" is one command.
"""

import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_frontend import write_network_csv, HW_DEFAULTS  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # src/compiler/pytorch -> repo root


def run_model(model, input_shape, hw_json,
              sim_bin=None, hw=None, timeout=300):
    """Run a torch.nn.Module through the simulator. Returns a result dict."""
    sim_bin = sim_bin or os.path.join(_REPO, "build/sim/flexnpusim")
    if not os.path.exists(sim_bin):
        raise FileNotFoundError(f"sim binary not found: {sim_bin} (build it first)")
    hw_json = hw_json if os.path.isabs(hw_json) else os.path.join(_REPO, hw_json)

    tmp = tempfile.mkdtemp(prefix="pt_run_")
    csv_path = os.path.join(tmp, "net.csv")
    out_path = os.path.join(tmp, "result.csv")
    rows = write_network_csv(model, input_shape, csv_path, hw or HW_DEFAULTS)

    proc = subprocess.run(
        [sim_bin, "-hw_conf", hw_json, "-network", csv_path, "-o", out_path],
        capture_output=True, text=True, timeout=timeout, cwd=_REPO)
    blob = proc.stdout + proc.stderr

    def grab(label, cast=int):
        m = re.search(rf"{label}\s*:?\s*([0-9]+)", blob)
        return cast(m.group(1)) if m else None

    completed = "Completed:  YES" in blob or "Completed: YES" in blob
    return {
        "layers": len(rows),
        "csv": csv_path,
        "exit_code": proc.returncode,
        "cycles": grab("Cycles"),
        "macs": grab("MACs"),
        "completed": completed,
        "dram": (re.search(r"DRAM model:\s*(\S+)", blob) or [None, None])[1]
                if re.search(r"DRAM model:\s*(\S+)", blob) else None,
    }


if __name__ == "__main__":
    import torch.nn as nn

    # MobileNet-style block — exercises standard/depthwise/pointwise conv + pool.
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=2, padding=1),
        nn.BatchNorm2d(16), nn.ReLU(),
        nn.Conv2d(16, 16, 3, 1, 1, groups=16),   # depthwise
        nn.BatchNorm2d(16), nn.ReLU6(),
        nn.Conv2d(16, 32, 1, 1, 0),              # pointwise
        nn.BatchNorm2d(32), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 10),
    )
    hw = sys.argv[1] if len(sys.argv) > 1 else "model/hw/npu/nvdla-large.json"
    r = run_model(model, (1, 3, 32, 32), hw)

    print("=== PyTorch → CSV → flexnpusim (one flow) ===")
    print(f"  hw conf:   {hw}")
    print(f"  layers:    {r['layers']}")
    print(f"  DRAM:      {r['dram']}")
    print(f"  cycles:    {r['cycles']}")
    print(f"  MACs:      {r['macs']}")
    print(f"  completed: {r['completed']}   exit={r['exit_code']}")
    ok = r["completed"] and r["exit_code"] == 0 and r["cycles"]
    print("PASS: full pytorch-controlled pipeline" if ok else "FAIL")
    sys.exit(0 if ok else 1)
