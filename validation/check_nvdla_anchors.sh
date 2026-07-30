#!/usr/bin/env bash
# NVDLA perf-model anchors — the regression gate.
#
# The simulator's compute term reproduces NVDLA's official performance model
#   cycles = ceil(Cin,64)*Ho*Wo*ceil(Cout,16)*Kh*Kw / 1024   (nv_full, INT16)
# from hardware constants alone (no calibration):
#   googlenet_conv2_3x3 : sim 338,800 vs perf model 338,688 (+0.03%)
#   alexnet_conv5       : sim 146,023 vs perf model 146,016 (+0.005%)
# This script pins the sim-side numbers byte-exactly. Run from the repo root.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/sim/flexnpusim"
HW="$ROOT/model/hw/npu/nvdla-large.json"
NET_DIR="$ROOT/model/sw/cnn/nvdla-large"

check() { # name csv expected_compute_cycles
    local name="$1" csv="$2" expect="$3"
    local line
    line=$(cd "$ROOT" && FLEXNPUSIM_LAYER_CSV=1 "$BIN" -hw_conf "$HW" \
               -network "$csv" -o "$ROOT/out/anchor_$name.csv" 2>&1 |
           grep '@LAYER' | head -1)
    local got
    got=$(echo "$line" | cut -d, -f13)
    if [ "$got" = "$expect" ]; then
        echo "PASS  $name  compute=$got"
    else
        echo "FAIL  $name  compute=$got (expected $expect)"
        return 1
    fi
}

rc=0
check googlenet_conv2 "$NET_DIR/googlenet_conv2_3x3.csv" 338800 || rc=1
check alexnet_conv5   "$NET_DIR/alexnet_conv5.csv"       146023 || rc=1
exit $rc
