# NPU

The NPU is the accelerator — the block that actually runs the network. It is
modelled as a **container** (`SC_MODULE(Npu)`) that instances its sub-blocks and
wires them:

- **NpuController** — the sequencer: MMIO registers, the FSM, and the read/
  compute/write orchestration. Drives the other three.
- **DmaChannel ×2** (rDMAC / wDMAC) — a DmaEngine plus its command signals;
  real AXI transactions between DRAM and the global buffer.
- **Ufb** — the unit function block: the PE array's operand-count timing model.
- **NpuGlobalBuffer** — on-chip SRAM (byte store + scratchpad/cache presence).

The host [processor](../processor/) programs it over the register bus (write the
DNN-image base, START, poll for IDLE); the NPU streams the layers and reports
cycles, MACs, and DRAM traffic through its performance counters.

- Source: `src/systemc/npu/` (`npu.h`, `npu_controller*.{h,cpp}`, `dma_channel.h`,
  `ufb.h`, `model/npu_global_buffer.*`, `model/gb_state.h`)
- Drives: the [latency model](../latency-model/) (per layer / per chain stage)
- Consumed by: `flexnpusim_system` (instances one `Npu`)

## Why it exists

A full-system run needs the accelerator itself — the thing that fetches operands,
computes, and writes results, with realistic DMA and on-chip buffering. Modelling
it as a container of a controller + DMA + compute unit + SRAM reproduces the
timing (DMA/compute overlap, GB pressure, tiling) without an RTL model, and keeps
the sub-block boundaries the hardware has.

## Documents

| File | Contents |
|------|----------|
| [`modeling.md`](modeling.md) | the method: the container structure, the register/FSM control path, the overlapped read/compute/write pipeline, the per-layer roofline, the UFB (operand-count compute) vs the controller's feed loop, tiled vs non-tiled paths, GB + inter-layer retention |
| [`isa-trm.md`](isa-trm.md) | the register/descriptor interface (ISA TRM): register map, descriptor formats, programming sequence |
| [`reuse-scheduling-gap.md`](reuse-scheduling-gap.md) | **known gap (not yet implemented)** — reuse-aware scheduling notes from the MIDAP comparison |

The controller implementation is split across `npu_controller*.cpp` by
responsibility (regs, dma, read, tile, compute, write, timeline); the compute
core is documented separately under [`../latency-model/`](../latency-model/).

## One-paragraph summary

On START the controller loads the DNN image from DRAM, parses it into per-layer
descriptors, and runs each layer through three concurrent stages — DMA-read
operands into the GB, feed them into the UFB (the PE-array latency model) cycle
by cycle, DMA-write the output — so compute overlaps DMA and the layer time is
`max(memory, compute)`. Layers that fit the GB take the streaming pipeline; larger
or tile-enabled layers take the analytic tiled path; both are selected by one
predicate. The UFB is the compute unit (operand counts, never values); the GB→PE
feed loop is the controller's dataflow sequencing — the hardware-faithful boundary
that lets the UFB, DMA, and GB sit as the controller's siblings under `Npu`.
