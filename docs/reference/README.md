# FlexNPUSim — Technical Reference

FlexNPUSim is a cycle-level NPU design-space-exploration simulator (C++17 +
SystemC + DRAMSim3). It answers "how many cycles, how much DRAM traffic, and
where is the bottleneck" for a fixed-dataflow accelerator running a given
network, from a hardware description and a per-layer workload — without an RTL
model.

This `reference/` tree is the authoritative description of **how the current
code behaves**, one directory per subsystem — including each subsystem's design
rationale and derivations. Validation results and config provenance live in
`../../validation/`; background research notes in `../research/`.

## The simulation pipeline

```
  hw JSON  +  network CSV                                 report
 (model/hw/npu) (model/sw/cnn)                          (cycles, MACs,
        │             │                                  util, DRAM traffic)
        ▼             ▼                                          ▲
   ┌─────────────────────┐   ┌───────────────────────┐   ┌──────────────┐
   │  compile            │──▶│  run (SystemC)         │──▶│  report      │
   │  network CSV -> DNN │   │  NpuController drives  │   │  per-layer   │
   │  image (per-function│   │  DMA / compute (LM) /  │   │  roofline    │
   │  LatencyDescriptor) │   │  memory timing / DRAM  │   │              │
   └─────────────────────┘   └───────────────────────┘   └──────────────┘
```

1. **Configure.** One hardware JSON (`model/hw/npu/<design>.json`) describes the
   compute array, buffers, DMA, AXI, and DRAM; one network CSV
   (`model/sw/cnn/<design>/<net>.csv`) lists each layer as one or more function
   descriptors. Both are parsed into a single `FlexNpuSimConfig`.
2. **Compile.** The compiler turns each layer's function descriptor into a
   `LatencyDescriptor` in the DNN image — the per-function workload the
   simulator replays (operand counts F^I/F^W/F^O, pipeline depth `l`, issue
   widths `n_min`/`n_max`, prefetch thresholds, a tile schedule, and the
   per-operand connectivity `operand_src`).
3. **Run.** `flexnpusim_system` assembles the SystemC model (NpuController + DMA
   engines + memory slave + a DRAM timing tier). The controller streams DMA into
   the global buffer, feeds the **latency model** (`latency-model/`) which
   produces per-cycle timing from operand counts, and charges DRAM through the
   selected **memory timing** model.
4. **Report.** `performance_report.cpp` emits total cycles, MACs, effective
   GOPS, PE utilization, DRAM traffic, and a per-layer memory-vs-compute
   roofline breakdown.

## Two abstraction planes

The simulator separates **timing** from **function**:

- **Timing** (the latency model) is purely count-based: it abstracts an RTL
  compute unit by operand-count ratios and never looks at tensor values. This is
  what determines cycles.
- **Function** (the function model, `model/function/`, OpenCV/PyTorch-parity) is
  value-based and optional: it computes actual tensor values when a functional
  run is requested. Value-dependent timing (e.g. activation sparsity / zero-skip)
  crosses into the timing plane only as a **count correction** measured at
  preprocessing — the sole value→count bridge.

## Subsystem map

| Directory | Source | Status |
|-----------|--------|--------|
| [`latency-model/`](latency-model/) | `src/model/latency/` | documented |
| [`processor/`](processor/) | `src/systemc/processor/` (host CPU: NPU boot/poll) | documented |
| [`common/`](common/) | `src/common/` (config_loader, types, tile_descriptor) | documented |
| [`compiler/`](compiler/) | `src/compiler/` (dnn_image, tiling, network_csv, ISA) | documented |
| [`function-model/`](function-model/) | `src/model/function/` | documented |
| [`npu/`](npu/) | `src/systemc/npu/` (container: controller, DMA, UFB, GB) | documented |
| [`bus/`](bus/) | `src/systemc/bus/`, `src/systemc/dma/` (AXI, DMA engine) | documented |
| [`memory/`](memory/) | `src/systemc/memory/` (DRAM tiers, timing, peak_bw) | documented |
| [`system/`](system/) | `src/system/` (flexnpusim_system, performance report) | documented |

## Per-subsystem document types

Each subsystem directory uses a fixed vocabulary; only the documents the
subsystem needs are created.

| File | Contents |
|------|----------|
| `README.md` | what the subsystem is + a map of its documents |
| `modeling.md` | the modeling method / abstraction / rationale (non-trivial models only) |
| `implementation.md` | code walkthrough — the mechanism, step by step |
| `parameters.md` | the configuration the subsystem consumes, as tables |
| `tests.md` | the test contracts — what each test asserts and why |
| `<topic>.md` | subsystem-specific extras (e.g. the latency model's `chain.md`) |
