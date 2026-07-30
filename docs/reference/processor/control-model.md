# Design: processor as a compiler-driven control model (testbench-like)

**Status: draft — structure reserved, implementation pending.**

## Goal

The processor should behave like an RTL testbench driver: the **compiler**
takes the execution source, **disassembles** it into an explicit command
stream, and the **processor model replays** that stream against the NPU —
register writes, DMA kicks, polls/waits — with realistic issue timing.
No hidden orchestration inside the simulator main.

## Current state (what this replaces)

- The register/control path runs on the **legacy** `external/systemc_axi`
  library: `AXI_master_if<NpuRegConfig>` is instantiated in
  `src/system/flexnpusim_system.cpp` (register bus) and
  `AXI_slave_if<NpuRegConfig>` inside `src/systemc/npu/npu_controller.h`.
  The root CMake keeps a `systemc_axi` INTERFACE target alive solely for
  this; its documented death condition is this design landing.
- Driving logic (start layers, watchdog progress lines) is hard-coded C++
  in the main/controller — not visible, not reorderable, not a program.
- `src/systemc/processor/` has a behavioral model + AXI master wrapper but
  no notion of "execute a program". Nothing references it today (the
  production register path bypasses it entirely) — it is retained as the
  starting material for this design (user decision, 2026-07-10).

## Operating forms (user requirement, 2026-07-10)

Both forms are first-class and must stay runnable:

1. **Standalone NPU** — today's form: the harness drives the NPU directly
   (no processor in the elaborated system). Fast, default for DSE sweeps.
2. **Processor-driven subsystem** — a processor component executes the
   command stream and drives the NPU **through its register interface over
   the bus** (write NPUCR/DNNSA..., poll NPUSR), exactly like firmware on a
   real SoC. This is the form the command IR below exists for, and the
   topology config (system-topology.md) selects between the two.

## Structural note — the WDMA one-driver violation (2026-07-10)

The E115 crash fix (commit 0bc50d6) relaxed the NPU->WDMA command signals
to SC_MANY_WRITERS because two requesters (the tiled executor in
read_thread, and write_thread) drive them directly, serialized only by a
cooperative mutex. In RTL terms that is a missing command **issue stage**:
one wire, one driver — multiple requesters belong behind a mux/queue.
The command-stream sequencer designed here IS that single issuer: when it
lands, DMA_KICK becomes the only producer, the relaxation is reverted, and
SC_ONE_WRITER is restored as a checked invariant. (Same root: with several
write channels configured, both requesters kick only channel 0 today —
the issue stage is also where multi-channel dispatch belongs.)

**Resolved (2026-07-10):** the issue stage landed *inside the NPU*, not in the
processor. The DMA is now an NPU-internal child driven by a single
`wdma_issue_thread` (requesters post descriptors), so SC_ONE_WRITER is already
restored — ahead of, and independent of, this processor sequencer. The host CPU
only programs NPU registers; the NPU issues its own DMA_KICKs, matching real
NVDLA-style control. The command stream below still drives the NPU at the
register level; it does not own the WDMA issue stage. Multi-channel dispatch
remains future work.

## Command IR (v1)

A flat, replayable stream — deliberately testbench-shaped:

```
REG_WRITE  <addr> <value>          # program a register
REG_READ   <addr> -> <dst>         # read back (scoreboarding)
DMA_KICK   <engine> <descriptor>   # launch a transfer
WAIT_REG   <addr> <mask> <value>   # poll until (reg & mask) == value
WAIT_EVT   <event-id>              # interrupt-style wait
BARRIER                            # all outstanding commands retire
```

Produced per layer (or per tile schedule) by the compiler; consumed in order
by the processor control model with configurable issue latency and poll
interval. Text form first (`.cmd`, one command per line, comments) so streams
are diffable and hand-writable in tests; a packed binary form can follow the
dnn-image pattern (`compiler/dnn_image`) if size ever matters.

## Pipeline

```
execution source (network CSV / dnn image)
        │  compiler lowering (new: src/compiler/cmdstream/)
        ▼
command stream (.cmd)  ──disassembler view (same module, reverse direction)
        │  processor control model replay (src/systemc/processor/)
        ▼
register writes / DMA kicks over an AXI4-Lite master (bus-protocols.md)
        ▼
NPU controller reacts purely to register state — no C++ back-channel
```

The "disassemble" direction matters: given an existing execution source (a
dnn image, or later a captured register trace), the same module prints the
command stream — this is how NVDLA-style register traces become replayable
stimuli for validation.

## Implementation path

1. Command IR struct + text parser/printer + unit tests (pure C++, core lib).
2. Processor control model: replay engine in `src/systemc/processor/`,
   issuing through the existing transport (AXI4-Lite once available; the
   legacy register path as a bridge until then).
3. Compiler lowering: `run_tiled_layer`'s schedule, expressed as commands
   (start with one layer = REG_WRITEs + DMA_KICK + WAIT_REG done-poll).
4. Parity gate: NVDLA anchors (39,244 / 146,080) and the demo numbers must
   be reproduced when driven by the command stream instead of the built-in
   sequencer.
5. Remove `external/systemc_axi` + the root `systemc_axi` INTERFACE target.

## Non-goals (v1)

Instruction-set simulation (no ALU/pipeline model — the processor is a
command sequencer with timing), interrupts beyond WAIT_EVT, multi-core.
