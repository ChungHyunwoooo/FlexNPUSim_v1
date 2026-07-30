# Design: adding bus protocols beside AXI4

**Status: draft — structure reserved, implementation pending.**

## Where the protocol boundary already is

The transport layer was built protocol-agnostic; adding a protocol means
filling in one directory, not refactoring callers.

- `src/systemc/bus/capabilities.h` — `Capabilities` (max outstanding, max
  beats/burst, max burst bytes) + `ModelingMode` (Full vs LatencyOnly).
  Callers size their burst chunking and pipelining from `caps()` and never
  name a protocol.
- `src/systemc/bus/memory_mapped_transport.h` — the caller-facing interface.
- `src/systemc/bus/common/` — protocol-neutral primitives in live use:
  `fsm_base.h`, `transfer_profile.h`. (A transport factory and spare
  handshake/ticket/queue primitives existed as unused scaffolding and were
  retired — git history `c6c4acb^..` has them; revive or redesign when the
  second protocol lands.)
- `src/systemc/bus/axi/` — the AXI4 implementation (spec/, master/, slave/,
  bus/ with 5 arbitration policies). Signal width is compiled as a
  ceiling (`FLEXNPUSIM_AXI_MAX_DATA_W`, AXI4 max 1024b); the runtime width is
  a config knob below it.

## Layout rule

One protocol = one directory: `src/systemc/bus/<proto>/` implementing master,
slave, and (if N×M) bus modules against the same transport surface, plus a
`caps()` mapping. Registered in `transport_factory.h` and selectable per link
via the topology `protocol` field (`system-topology.md`).

## Candidate protocols, in order of value

1. **AXI4-Lite** — single-beat, no bursts, no IDs; `caps = {1 outstanding,
   1 beat, bus-width bytes}`. Wanted first because it is the natural register
   /control path and the replacement for the legacy `external/systemc_axi`
   interfaces (see `processor-control-model.md`). Smallest possible new-proto
   exercise of the boundary. Its registration seam (a transport factory) gets
   (re)introduced at that point.
2. **AXI4-Stream** — no addresses; models feature-map streaming between
   accelerator stages. Needs a small interface extension (stream vs
   memory-mapped), which is exactly the generalization worth proving.
3. **AHB-Lite** — single outstanding, fixed pipeline; enables MCU-class NPU
   comparisons where AXI is not plausible hardware.

## Implementation path

1. Extract the implicit transport surface into an explicit docs/interface
   check-list (what a `<proto>/` must provide) — derived from what
   `axi/` exposes and what `flexnpusim_system.cpp` + DMA engines consume.
2. Implement `axi4lite/` (a ready/valid handshake helper can be revived
   from history or written fresh; no ticket pool).
3. Wire `protocol` through a (re)introduced transport factory + topology
   links.
4. Gate: existing 61 tests + anchors untouched with AXI4 defaults; a new
   signal-level test set under `tests/systemc/bus/<proto>/` mirroring the
   AXI ones (master single/multi, slave roundtrip, integration).

## Non-goals (for now)

Coherent protocols (ACE/CHI) and PCIe: nothing in the current NPU models
issues coherent or posted-transaction traffic, so they would be dead weight.
