#!/usr/bin/env python3
"""User-input mode: run the simulator from latency-model parameters given DIRECTLY.

FlexNPUSim has two ways to obtain the per-layer latency descriptor that drives
the latency model:

  * COMPILER mode (scripts/pytorch_frontend.py): DERIVE the descriptor
    (F^I, F^W, F^O, l, n_min, n_max) automatically from a PyTorch model +
    hardware mapping (k/g/s, tile, dataflow).

  * USER-INPUT mode (this module): the user supplies that descriptor DIRECTLY.
    The simulator's CSV front-end has "oracle override" columns
    (f_i, f_w, f_o, l_pass, n_max, n_min, tau_i); when present they REPLACE the
    compiler-derived values, so the latency model runs with your exact numbers.

The workload shape columns are still required (the memory side fetches the real
tensors), but the compute-side descriptor comes straight from you.

Usage:
    from latency_descriptor import write_descriptor_csv
    layers = [dict(name="conv1", type="conv2d",
                   input_h=224, input_w=224, input_c=3,
                   kernel_h=3, kernel_w=3, kernel_c=3, kernel_count=64,
                   stride=1, padding=1,
                   f_i=648, f_w=288, f_o=3528, l=4, n_max=4, n_min=1, tau_i=1)]
    write_descriptor_csv(layers, "net.csv")
    # then: build/sim/flexnpusim -hw_conf <hw.json> -network net.csv -o out.csv
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_frontend import HEADER, NO_CONNECT, HW_DEFAULTS, _row  # noqa: E402

# Oracle-override columns the CSV front-end reads (frontend_network_csv.cpp).
ORACLE_COLS = ["f_i", "f_w", "f_o", "l_pass", "n_max", "n_min", "tau_i"]
USER_HEADER = HEADER + ORACLE_COLS

_REQUIRED = ("type", "input_h", "input_w", "input_c",
             "f_i", "f_w", "f_o", "l", "n_max", "n_min")


def write_descriptor_csv(layers, path, hw=HW_DEFAULTS):
    """Write a user-input-mode network CSV from explicit per-layer descriptors."""
    rows = []
    prev = NO_CONNECT
    for i, L in enumerate(layers):
        missing = [k for k in _REQUIRED if k not in L]
        if missing:
            raise ValueError(f"layer {i} missing fields: {missing}")
        base = _row(i, L.get("name", f"layer{i}"), L["type"], L["type"], prev,
                    L["input_h"], L["input_w"], L["input_c"],
                    L.get("kernel_h", 1), L.get("kernel_w", 1),
                    L.get("kernel_c", L["input_c"]),
                    L.get("kernel_count", L["input_c"]),
                    L.get("stride", 1), L.get("padding", 0), hw)
        oracle = [L["f_i"], L["f_w"], L["f_o"], L["l"],
                  L["n_max"], L["n_min"], L.get("tau_i", 1)]
        rows.append(base + oracle)
        prev = i
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(USER_HEADER)
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    import tempfile

    # The paper's running convolution layer: descriptor (648, 288, 3528, 4, 4, 1).
    layers = [
        dict(name="running_conv", type="conv2d",
             input_h=9, input_w=9, input_c=4,
             kernel_h=3, kernel_w=3, kernel_c=4, kernel_count=8,
             stride=1, padding=0,
             f_i=648, f_w=288, f_o=3528, l=4, n_max=4, n_min=1, tau_i=1),
    ]
    path = os.path.join(tempfile.gettempdir(), "user_descriptor.csv")
    rows = write_descriptor_csv(layers, path)

    def fail(m):
        print("FAIL:", m); sys.exit(1)

    if len(rows[0]) != len(USER_HEADER):
        fail("row width != header")
    # oracle values land in the trailing columns, in order
    o = rows[0][len(HEADER):]
    if o != [648, 288, 3528, 4, 4, 1, 1]:
        fail(f"oracle columns wrong: {o}")
    print(f"PASS: user-input descriptor CSV → {path}")
    print(f"  columns: ...,{','.join(ORACLE_COLS)}")
    print(f"  running_conv descriptor (F^I,F^W,F^O,l,n_max,n_min,tau_i) = {o}")
