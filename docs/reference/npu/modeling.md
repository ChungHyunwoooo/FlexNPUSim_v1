# NPU — Modeling Method

This document explains *what the NPU models and why*. The code lives in
`src/systemc/npu/`; the compute-timing core it drives is the latency model
([`../latency-model/`](../latency-model/)).

## The NPU is a container of sub-blocks

The NPU is modelled the way the hardware is built — a container that instances
its sub-blocks and wires them:

```
SC_MODULE(Npu)                        (npu.h)
 ├── NpuGlobalBuffer gb_               on-chip SRAM (byte store + presence model)
 ├── DmaChannel rdma_ch_, wdma_ch_     DMA engine + command signals (dma_channel.h)
 ├── Ufb ufb_                          compute datapath = PE-array timing (ufb.h)
 └── NpuController controller_         the sequencer — drives the three above
```

The **controller** is the conductor: registers, the FSM, and the read/compute/
write orchestration. It DRIVES the DMA, UFB, and GB (it holds them by reference);
it does not own them. This split matters — in hardware the compute unit (PE
array), the DMA, and the SRAM are distinct blocks the controller sequences, and
the model keeps that boundary.

## How the NPU runs a network

The host processor programs the NPU over its MMIO register bus (the AXI slave):
write `DNNSA` (the DNN-image base in DRAM), write `NPUCR = START`, then poll
`NPUSR` until `IDLE` and read the `PERF_CNT*` counters. On START the controller:

1. **Loads the DNN image** from DRAM (`load_dnn_image`) and parses it into a
   stream of per-layer function descriptors (`DnnImageLoader`).
2. **Runs each layer** through a three-stage pipeline that overlaps in time:
   - **read** — DMA the layer's input/weight operands from DRAM into the GB,
   - **compute** — stream those operands into the UFB cycle by cycle,
   - **write** — DMA the output back to DRAM.
3. **Signals completion** — sets `NPUSR.IDLE` and freezes the perf counters.

Read, compute, and write are separate SC_THREADs that run concurrently and
hand off through events. That concurrency is the point: a layer's compute
overlaps the next tile's DMA, so the layer time is the *larger* of the two, not
their sum (the roofline below).

## Timing is a per-layer roofline over an overlapped pipeline

Per layer, `time = max(memory_cycles, compute_cycles)`. Load and compute overlap;
whichever term dominates sets the layer time. The **compute** term is produced by
the UFB (the latency model); the **memory** term is the DMA traffic priced by the
selected DRAM timing tier ([`../memory/`](../memory/), once documented). The
controller resolves the overlap once per layer after streaming its tiles.

## The UFB: operand-count compute, not values

The UFB is the PE array's timing model. It holds an `LmModel` and, each cycle,
answers "given this many input/weight operands, how many output passes complete?"
— purely in integer operand counts, never tensor values (see
[`../latency-model/modeling.md`](../latency-model/modeling.md)). A multi-function
packet (e.g. NVDLA `conv→CACC→SDP`) becomes an `LmChain` of per-stage UFBs.

The **feed loop** — reading available operands from the GB and streaming them into
the UFB — is the CONTROLLER's job (`run_layer_compute`), not the UFB's. In
hardware the datapath sequencing that connects SRAM to the PE array is control,
so the model draws the UFB boundary at the compute unit and leaves the GB→PE
feed to the controller. That is why the UFB separates cleanly.

## Two execution paths, one predicate

A layer runs on one of two paths, chosen by `layer_runs_tiled` (a `TileEnable`
layer, a fused activation, or any layer whose footprint exceeds the GB):

- **Tiled** (`run_tiled_layer`) — the layer is streamed tile by tile; each tile's
  timing is the analytic `tile_compute_cycles` (compute-bound vs feed-bound, the
  max), run inline so a large tensor never overflows the GB streaming buffer.
- **Non-tiled** (the read/compute/write pipeline) — the whole layer fits the GB;
  the compute thread drives the UFB operand-by-operand as DMA fills the GB.

Both paths use the same predicate, so a layer is owned by exactly one — the
asymmetry that once left compute waiting forever for a tile-executed layer's data
is gone.

## The global buffer and inter-layer retention

`NpuGlobalBuffer` is the on-chip SRAM: a byte store plus a presence model
(scratchpad, or an L2 cache with per-line hit/miss). `GbState` tracks it at the
operand level (input/weight/output regions) — that is what the compute feed loop
reads to know how many operands have arrived. Under layer fusion, a layer's
output can stay resident and serve the next layer's input on-chip (the retention
ledger), so a fused activation never round-trips DRAM.

## What it models — and what it does not

Models: the register/FSM control path, an overlapped read/compute/write pipeline
with real AXI DMA, operand-count compute timing (the UFB) with multi-function
chaining, a per-layer memory-vs-compute roofline, GB scratchpad/cache presence,
tiled and non-tiled execution, and inter-layer GB retention.

Does not model: tensor values (that is the function model, `model/function/`,
crossing into timing only as a sparsity count correction), the exact PE-array
wiring (reduced to the latency model's `k·g·n·l`), or a registered/two-phase
inter-block datapath — the threads communicate through shared operand state, so
the cycle count is not fully robust to SystemC process order (see the hardware-
faithfulness note in the design history).
