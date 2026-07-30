# PyTorch frontend → FlexNPUSim (conf + pytorch control)

Drive the simulator from a PyTorch model. This is the "pytorch" half of
"everything controlled via **conf** (hw JSON) + **pytorch** (model)".

```
torch.nn.Module ──pytorch_frontend──► network CSV ──flexnpusim──► results
                  (workload: shapes)   (+ hw JSON:    (compiles CSV→DNN image
                                         mapping)       internally, runs on the
                                                        SystemC system w/ dramsim3)
```

## Files

| script | role |
|---|---|
| `pytorch_frontend.py` | `torch.nn.Module` → network CSV. Forward hooks capture per-layer I/O shapes; maps Conv2d (incl. **depthwise** when `groups==in_channels`), Linear, MaxPool/AvgPool/**AdaptiveAvgPool**, ReLU. **BatchNorm is folded** (not emitted), matching inference. HW mapping (`pe_type/k/g/s/tile`) passed via `hw=` (default systolic NVDLA-class). |
| `run_pytorch_model.py` | One command: model → CSV → `flexnpusim` → parsed `{cycles, macs, completed}`. |
| `regenerate_networks.py` | Regenerate VGG16 / MobileNet-v1 CSVs from canonical PyTorch definitions (a *trusted* source replacing hand-authored CSVs) → `model/sw/cnn/regenerated/`. |

## Usage

```bash
# Run a PyTorch model end-to-end (uses dramsim3 by default)
python3 scripts/run_pytorch_model.py                      # MobileNet-style demo
python3 scripts/run_pytorch_model.py model/hw/npu/midap.json   # different hw conf

# Regenerate the paper workloads from PyTorch definitions
python3 scripts/regenerate_networks.py
build/sim/flexnpusim -hw_conf model/hw/npu/nvdla-large.json \
    -network model/sw/cnn/regenerated/vgg16.csv -o result.csv
```

```python
# Programmatic
import torch.nn as nn
from scripts.pytorch_frontend import write_network_csv
write_network_csv(my_model, (1, 3, 224, 224), "net.csv")
```

## Notes / scope

- **Sequential** (VGG, MobileNet) and **residual/branching** (ResNet) models are
  both supported: a `torch.fx` graph trace captures functional `add`/`cat` as
  "residual"/"concat" rows with real i1/i2 connectivity. Forward-hook tracing is
  the fallback for models fx cannot symbolically trace.
- The sim compiles the CSV → DNN image internally (`-network <csv>`); the
  separate `dnn_image_compiler` tool takes a network **JSON** (different path).
- Memory model: `dramsim3` is the default trusted cycle-accurate tier. The
  analytic `bank_model` is a faster approximation; its controller-pipeline
  residual is tunable via `MemoryControllerConfig::command_latency_cyc`.
