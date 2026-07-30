English | [한국어](README.ko.md)

# FlexNPUSim

A configurable SystemC-based NPU system simulator. One binary simulates an
NPU + DMA + AXI bus + DRAM system driven by a hardware spec (JSON) and a DNN
workload (CSV), and reports latency, DRAM traffic, PE utilization, and a
per-layer compute-vs-memory bottleneck verdict — fast enough (seconds per
network) to sweep accelerator × memory-system design spaces that RTL
simulation cannot reach.

**Validation** — every hardware parameter is derived from public specs;
nothing is fitted to a reference. The compute model reproduces NVDLA's
official performance model from hardware constants alone (alexnet_conv5
+0.005%, googlenet_conv2 +0.03%), and full-network runs match the authors'
reference simulator (MIDAP, three networks within 3.4%) and published RTL
measurements (NullHop sparse VGG-16, +3.1%) — six checkpoints, 1.6% average
error, zero calibration parameters. Reproduce with
`./validation/run_validation.sh`; per-number provenance lives in
`validation/results.md`.

## Quick start

```sh
cmake -B build -S . && cmake --build build -j$(nproc)

# hardware spec (or full config JSON) + workload CSV; paths resolve from repo root
build/sim/flexnpusim -hw_conf model/hw/npu/nvdla-large.json \
    -network model/sw/cnn/nvdla-large/resnet18.csv -o out/run.csv
# add -report out/report.txt for the per-layer bottleneck table

# or the shorthand runner / the validation gate
python3 tools/run.py --scenario nvdla-large:resnet18
./validation/check_nvdla_anchors.sh
```

Inputs: a **hardware model** (`model/hw/npu/*.json` — PE array, buffers,
dataflow, AXI, DRAM) and a **software model** — a workload CSV
(`model/sw/<class>/<design>/<network>.csv`, classes `cnn|attention`; two
modes: oracle per-layer factors or auto-derived from layer dims). PyTorch
models convert via `src/compiler/pytorch/`. Run everything from the
repository root — config-internal paths are root-relative. Outputs land in
`out/` (gitignored).

## Layout

```
src/
├── systemc/     SystemC cycle-accurate components (RTL-corresponding)
│   ├── bus/     AXI transport library, arbiters, topology wiring
│   ├── npu/     NPU controller (tile execution, GB residency)
│   ├── memory/  DRAM models (bank model, DRAMsim3 backend)
│   ├── dma/  processor/
├── model/       performance & functional models
│   ├── latency/   count-based latency model (the compute-side abstraction)
│   └── function/  OpenCV-backed functional model (PyTorch 1:1)
├── compiler/    workload generation & scheduling
│   ├── frontend/   hw JSON + network CSV loaders, request builder
│   ├── dnn_image/  network → dnn image (descriptor stream) compiler
│   └── pytorch/    PyTorch model → network CSV pipeline (Python)
├── system/      top-level assembly (flexnpusim_system) + performance report
└── common/      config loader, types, logging, tile descriptor

model/hw/    hardware models (JSON): npu/, memory/ (DRAM ini), system/ (reserved)
model/sw/    software models — workload CSVs by class: cnn/, attention/
examples/    baseline.json — full-block config reference
tools/       run.py — one-command runner (--scenario design:network)
validation/  reference-checkpoint harness (run_validation.sh), 16-config smoke
             test, SCALE-Sim cross-check, DSE regeneration matrix
tests/       contract tests (ctest), regenerated spec-first per module
docs/reference/  authoritative per-subsystem reference (how the code behaves);
             start at docs/reference/README.md — or docs/adding-a-new-npu.md
docs/research/  background research notes
external/    dramsim3 / systemc / spdlog (fetched), nlohmann, systemc_axi (legacy)
out/         simulator outputs (gitignored)
```

## Build

```sh
cmake -B build -S . && cmake --build build -j$(nproc)
```

Requires CMake ≥ 3.16 and a C++17 compiler. SystemC 2.3.4, spdlog, and
DRAMsim3 are fetched automatically when no checkout exists under `external/`.
OpenCV (core/dnn/imgproc) is optional and enables the functional model.

## Authors

- Hyunwoo Chung — Kwangwoon University
- Joonhwan Yi — Kwangwoon University

## Acknowledgment

This work was supported by Institute of Information & Communications Technology
Planning & Evaluation (IITP) grant funded by the Korea government (MSIT)
(RS-2022-II220998 / 2022-0-00998, Development of open platform for artificial
intelligent semiconductor chip, 2022년도 제1차 정보통신·방송
기술개발사업 및 표준개발지원사업(과학기술정보통신부 공고 제2021-1022호)).

## License

MIT — see [LICENSE](LICENSE).
