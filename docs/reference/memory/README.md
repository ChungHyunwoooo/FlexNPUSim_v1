# Memory

The memory subsystem is the simulator's **external DRAM** and its timing. It is
the AXI slave that DMA reads operands from and writes results to, plus the
memory-controller timing model that decides *when* each transaction completes.

It is modelled as three layers wrapped in one SystemC module:

- **`MemoryAxiSlaveV2<Spec>`** — the wrapper. Owns a byte store
  (`DenseRamBackend`), a signal-level AXI slave (`axi::Slave<Spec>`), and one
  DRAM timing model, and wires the timing model into the slave through the
  `MemPort` interface.
- **`axi::Slave<Spec>`** — real, signal-level AXI (ready/valid on all five
  channels). No latency formula lives here; it *drives* the timing model like a
  CPU drives a DRAM controller.
- **`DramTiming`** — the memory-controller timing model, one of four fidelity
  **tiers**. Its internals may approximate, but its interface is the real
  hardware one: backpressure / submit / tick / completion.

- Source: `src/systemc/memory/wrapper/memory_axi_slave_v2.h`,
  `src/systemc/memory/dram/` (tiers + factory),
  `src/systemc/memory/model/` (layer-level peak-bw model)
- Interface into the slave: `src/systemc/bus/axi/slave/axi_slave.h`
  (`Slave<Spec>::MemPort`)
- Consumed by: `flexnpusim_system` (instances one `MemoryAxiSlaveV2`,
  `src/system/flexnpusim_system.cpp:575`)

## Why it exists

A full-system run needs a memory whose timing is realistic but adjustable. The
AXI slave and DMA around the memory are always real hardware (ready/valid +
occupancy); only the memory-controller timing is allowed to approximate. Keeping
that timing behind a narrow `DramTiming` interface lets a run trade fidelity for
speed — a fixed-latency stub, a bandwidth roofline, a row-buffer model, or a
full DRAMSim3 cycle model — by changing one config key (`dram.tier`), with no
change to the slave or the DMA.

The DRAM model owns **only** memory timing. Tiling and the read/compute/write
overlap are the [controller](../npu/)'s job; the byte-vs-compute roofline is the
[report](../latency-model/)'s.

## The tiers

`dram.tier` selects one of four models, increasing fidelity and cost. The first
three share the analytical **bank-occupancy engine** (`AnalyticalDram`) and
differ only in their per-access latency; the fourth wraps DRAMSim3.

| Tier | Latency of one access | Concurrency | Models | Use when |
|------|-----------------------|-------------|--------|----------|
| `ideal` | fixed `read/write_latency_cyc` (0 = free) | `banks` in parallel | nothing but a constant | a latency floor; isolating compute from memory |
| `bandwidth` | `base + size / bandwidth_bytes_per_cycle` | `banks` in parallel | shared-bus transfer time (MIDAP-style) | a bandwidth roofline without a full DRAM model |
| `bank` | row-buffer hit/miss (`row_hit_cyc` / `row_miss_cyc`), writes `write_latency_cyc` | `banks` in parallel | open-row locality across banks | address-pattern sensitivity, cheaply |
| `cycle` | DRAMSim3 (driven loaded) | DRAMSim3 queues/banks/refresh | real JEDEC DDR timing | the DSE / new-result runs; highest fidelity |

A separate, **layer-level** model (`peak_bw`, selected by `dram.tiled_dma_timing`,
not `dram.tier`) charges a whole tiled layer's memory time analytically instead
of per burst — see [`modeling.md`](modeling.md).

## Documents

| File | Contents |
|------|----------|
| [`modeling.md`](modeling.md) | the timing semantics of each tier, the bank-occupancy engine, the transaction lifecycle through `MemPort` (read pre-fetch at AR vs write submit at WLAST), the ordering guarantees, and the layer-level `peak_bw` model |
| [`parameters.md`](parameters.md) | the `dram` config block per tier, the DRAMSim3 `.ini` presets in `model/hw/memory/`, and the clock conversion |

## One-paragraph summary

The AXI slave accepts an AR/AW only while the DRAM model's `will_accept()` is
true (bank occupancy is the backpressure), pre-fetches read beats from the byte
store at AR arrival, commits write beats to the store as they land, and `submit()`s
each transaction to the model tagged with a token — a read at AR, a write once
WLAST is seen. The model advances one cycle per `tick()`; the slave drains
finished tokens with `pop_completed()` and only then streams the response, in
strict AR/AW-arrival order. Which model runs is `dram.tier`: `ideal`,
`bandwidth`, and `bank` are one shared occupancy engine with different per-access
latencies, and `cycle` hands the traffic to DRAMSim3.
