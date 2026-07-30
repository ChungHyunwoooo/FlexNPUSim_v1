# Design: system composition config (`model/hw/system/`)

**Status: draft — structure reserved, implementation pending.**

## Goal

Describe the *system construction* declaratively: which components exist and
how they are wired. The composition file carries **structure only** — names,
kinds, spec pointers, connections, and the address map. Every electrical /
protocol / sizing parameter (AXI protocol and width, arbitration, DMA channel
counts, buffer and DRAM sizing) stays in each component's **hw conf**, not here
(user direction 2026-07-10). Target expressiveness, with examples:

- **two memory lanes** — the NPU reaches two DRAM controllers over separate
  links
- **no bus** — a master wired directly to a slave (point-to-point); expressed
  *structurally* by targeting a slave port instead of a bus component, not by a
  protocol flag
- **mixed protocols** — data ports speak AXI4, the register port AXI4-Lite;
  each port's protocol comes from its component's hw conf (see
  `bus-protocols.md`), so the wiring only has to connect compatible ports

## Current state (what actually exists, 2026-07-10)

No declarative composition loader exists today. The earlier bus-topology JSONs,
`topology_loader` / `build_system` / `component_registry`, their regression
fixtures, and tests were all retired with the unused assembly machinery (git
history has them; `model/hw/system/` now holds only this design's README).

The whole system shape is hand-wired in `src/system/flexnpusim_system.cpp`
(§4 "SystemC wiring", ~line 644):

- **npu** (`NpuController`) — owns L2/GB internally; exposes `rdma_*`/`wdma_*`
  command/status **ports** and a register slave.
- **rdmac / wdmac** (`DmaEngine`) — instantiated as **top-level siblings** of
  the NPU, commanded over the exposed `rdma_*`/`wdma_*` signals plus a
  `connect_dma()` pointer back-channel. Extra channels (`read_channels` /
  `write_channels` > 1) are spawned but tied to idle dummy signals (Phase 1).
- **memory** (`MemoryAxiSlaveV2`) — a single DRAM slave (`NumS` is hard-wired 1).
- **data bus** (`InterconnectAxiBusV2<NumM,1>`) — the R4 `Bus` needs `NumM` as a
  compile-time parameter, so the main `switch`es on the master count and
  instantiates the matching specialization ({1,2,3,4,5,6,8,10,12,16}).
- **cpu** (`CpuDriver`) — a fixed test driver (writes DNNSA/START, polls
  NPUSR), not a real processor.
- **reg_bus** — a bare `AXI_SIGNALS<NpuRegConfig>` bundle (no arbiter) shared by
  `npu.bind()` and `cpu.bind()`: the point-to-point control path.

`TopologyConfig` (`src/common/flexnpu_config.h`, parsed by
`config_loader.cpp::parse_topology` with an address-overlap check) still exists
but is **vestigial**: the only consumer beyond parse/validate is the per-master
qos/weight/priority read at `flexnpusim_system.cpp:863`. Its
masters/slaves/links/interconnect fields do not drive assembly.

## Proposed schema (v2 — Qsys-shaped)

Three ingredients, exactly like assembling blocks in Quartus Qsys:
**components** (the blocks), **connections** (the wires), **mmio** (the address
map). Structure only — protocol and sizing come from each block's hw conf.

```jsonc
{
  "components": [
    { "name": "cpu0", "kind": "processor" },               // control master; no DMA yet
    { "name": "npu0", "kind": "npu",
      "spec": "model/hw/npu/nvdla-large.json" },            // read/write_channels live in the spec →
                                                            // npu0 exposes npu0.m0, npu0.m1, and npu0.reg
    { "name": "noc0", "kind": "bus" },                      // protocol / width / arbitration from hw conf
    { "name": "ddr0", "kind": "memory",
      "spec": "model/hw/memory/LPDDR4_2400_x16.ini" }       // backing size from the memory hw conf
  ],
  "connections": [
    { "from": "cpu0.m",  "to": "npu0.reg" },   // control path — point-to-point (no bus), AXI4-Lite by the reg port
    { "from": "npu0.m0", "to": "noc0" },        // NPU-internal RDMA data
    { "from": "npu0.m1", "to": "noc0" },        // NPU-internal WDMA data
    { "from": "noc0",    "to": "ddr0.s" }
  ],
  "mmio": {
    "ddr0":     { "base": "0x8000_0000", "size": "0x8000_0000" },  // decode / route span
    "npu0.reg": { "base": "0x4000_0000", "size": "0x0001_0000" }   // NPUCR..PERF_CNT
  }
}
```

Modelling rules (the corrections this structure forces):

- **The NPU owns its DMA.** The RDMA/WDMA engines are *internal* to `npu0`,
  controlled by the NPU's own command issue stage — they are **not** top-level
  components. Channel count comes from the NPU spec
  (`hw.dma.read_channels` / `write_channels`); the NPU exposes one data master
  port per channel (`npu0.m0..mk`). A top-level `dma` kind is **reserved for a
  processor's DMA** (a separate master), designed-for but not instantiated
  today.
- **A component may expose several master ports** (`npu0.m0`, `npu0.m1`). The
  loader counts the master edges entering a bus and dispatches to the compiled
  `Bus<NumM, NumS>` specialization.
- **Protocol is a property of the port, not the wire.** `npu0.reg` is AXI4-Lite
  because the NPU IP says so (hw conf); `npu0.m*` are AXI4. The elaborator
  connects compatible ports and inserts a bridge otherwise; the wiring never
  *chooses* a protocol.
- **direct vs bussed is structural.** An edge to a bus component goes through
  arbitration; an edge straight to a slave port (`cpu0.m → npu0.reg`) is
  point-to-point. There is no `"direct"` flag.
- **Memory has two sizes.** The mmio `size` is the decode / route span
  (`DRAM_ROUTE_SPAN`); the physical backing (`MEM_SIZE`) is a memory hw-conf
  property. Keep them distinct; mmio ranges must not overlap.

**Ownership law (verified by the 2026-07-10 ownership audit):** every connection is a single-owner
edge — one master port drives one signal bundle. In this structure that holds
*by construction*: `cpu0.m→npu0.reg`, `npu0.m0/m1→noc0`, and `noc0→ddr0.s` each
have one writer per direction. The WDMA `SC_MANY_WRITERS` relaxation was never a
composition property — it was an NPU-**internal** two-requester issue (the tile
executor + write_thread) that the old sibling-DMA wiring wrongly exposed at top
level. Encapsulating the DMA inside the NPU behind a single internal DMA_KICK
issuer restores `SC_ONE_WRITER` as a byproduct (and retires the `wdma_in_use_`
mutex). See processor-control-model.md §Structural note.

## Implementation path

1. **Encapsulate the DMA in the NPU.** ✅ *Phase 1a done (2026-07-10):* channel-0
   `rdmac`/`wdmac` are now `NpuController` children; `rdma_*`/`wdma_*` are internal
   signals; a single internal `wdma_issue_thread` drives WDMA (`SC_ONE_WRITER`
   restored, `wdma_in_use_` mutex and `connect_dma()` removed). The system
   assembly hands the channel-0 transports via `attach_dma_transports()` (the AXI
   master stack still lives there). Verified byte-identical: anchors
   39,244 / 146,080, mobilenet 26,167, resnet18 × 9 designs.
   *Phase 1b (pending):* move the AXI master stack into the NPU so it exposes
   `reg` + `m0..mk` as ports (needs the DataSpecT templating / type-erasure
   decision).
2. **Schema + loader.** Parse components / connections / mmio; protocol and
   sizing read from each component's hw conf, not the composition file.
3. **Registry + elaboration.** A factory per `kind`; count master edges per bus
   and dispatch to `Bus<NumM, NumS>`; the memory factory builds slave + address
   map from mmio + backing. `build_system()` owns the whole shape and the report
   generation (the LayerPerfRecord four-section formatter) extracts into its own
   module. `flexnpusim_system.cpp` stops hand-wiring.
4. **Gates.** The default single-lane system reproduces the anchors
   byte-for-byte; the two-lane and processor-driven forms arrive with their own
   examples (need `NumS>1` specializations and the processor control model).

## Open questions

- **`NumS > 1`** (two memory lanes) needs `Bus<NumM, NumS>` specializations and
  range routing across slaves; the current dispatcher only varies `NumM` with
  `NumS==1`. Start range-partition; stride-interleave needs
  `dram_address_mapper`.
- Whether `noc0`'s params are read from the global `axi` / `topology` hw-conf
  blocks or move into a per-bus spec file once buses multiply.
- Where the memory backing size lives in the hw conf (the DRAM `.ini` is timing
  only today; `MEM_SIZE` is a hard-coded constant in the main).
