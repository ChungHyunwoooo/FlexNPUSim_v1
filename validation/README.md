# validation/

**Validation gate — genuine NVDLA performance-model agreement.**

The simulator's compute model reproduces NVDLA's official performance model
(the `nvdla-hw` release spreadsheets) from hardware constants alone —
Atomic-C = 64, Atomic-K = 16, so 1024 INT16 MAC/cycle on `nv_full` — with no
calibration or fitting:

| workload | simulator | NVDLA perf model | error |
|---|---|---|---|
| googlenet_conv2_3x3 | 338,800 | 338,688 | +0.03% |
| alexnet_conv5 | 146,023 | 146,016 | +0.005% |

The perf-model formula (`ceil(Cin,64)·Ho·Wo·ceil(Cout,16)·Kh·Kw / 1024`) was
itself verified against the spreadsheet's own table values before being used
as the reference. NVDLA's RTL *testbench traceplayer* cycle counts are ~44–51x
these numbers — a testbench artifact, not a valid comparison target.

## Running the gate

```sh
./validation/check_nvdla_anchors.sh     # from the repo root, after a build
```

Pins the two simulator-side numbers byte-exactly; every structural change to
the compute/tiling path must keep this green (configs that don't touch the
modeled path must not move them at all).

The anchor workloads ship in `model/sw/cnn/nvdla-large/`
(`googlenet_conv2_3x3.csv`, `alexnet_conv5.csv` — oracle-style rows extracted
from the NVDLA verification networks). The full research-side comparison
(RTL trace analysis, sweep results, plots) lives in the research tree,
not this repository.
