# Processor

The processor is the host CPU model — the agent that **starts the simulation**:
it hands the DNN image base to the NPU, asserts START, polls to completion, and
reads back the performance counters. It does not run a compute workload; the NPU
does all the work. The processor is the boot/control path.

It is one header-only SC_MODULE built on the DSL-lab `fractal_processor` idiom:
the driver firmware runs **natively** on the SystemC host, memory-mapped accesses
are issued as **manual AXI transactions**, and CPU-side latency is added
explicitly from an instruction-class model, not simulated microarchitecture.

- Source: `src/systemc/processor/processor.h` (class `Processor`)
- Consumed by: `flexnpusim_system` (one instance, bound to the NPU register bus)
- Replaces: the former `CpuDriver` + `ProcessorModel` + `ProcessorAxiMaster`
  (consolidated into this one module)

## Why it exists

A full-system run needs a host that programs the accelerator and waits for it.
The role is small and its functional behaviour is fixed (write a few registers,
poll a status bit), so it is modelled as native host code with hand-inserted bus
transactions rather than an instruction-set simulator. Timing is layered on top
as a per-instruction-class count model so the CPU-side cost is explicit and
config-tunable without modelling a pipeline.

## Documents

| File | Contents |
|------|----------|
| [`modeling.md`](modeling.md) | the method: the boot firmware, native execution + manual AXI, the internal(cache)/external(bus) split, `software_delay` (disassembly-derived counts) and `mem_access` (statistical cache), the config knobs, and what is scaffolding vs exercised |
| [`control-model.md`](control-model.md) | the host-side control design: boot/poll protocol, register interface, why the CPU is modeled natively |
| [`latency-derivation.md`](latency-derivation.md) | how the `software_delay` instruction-class counts were derived from firmware disassembly |

Derivation of the concrete `software_delay` counts (firmware disassembly →
instruction classes) lives in `latency-derivation.md`.

## One-paragraph summary

`Processor::run()` is the firmware: after a startup delay it writes `DNNSA`
(image base) and `NPUCR=START`, then polls `NPUSR` until `IDLE` with a per-layer
watchdog, and finally reads the `PERF_CNT*` counters — exposing `cnt0..3` and
`completed` for the report. Register accesses are **external**: real AXI
transactions on the register bus (`reg_read`/`reg_write` →
`master_if->read_single`/`write_single`), whose latency is the bus handshake plus
NPU-slave response. Compute between accesses is charged by
`software_delay(store, load, jmp, mul, div, gen)`, whose `jmp/mul/div/gen` counts
are hand-derived from the firmware disassembly; its `store/load` args drive
`mem_access()`, a purely statistical cache hit/miss penalty for **internal**
data-memory accesses that never touches the bus. The two planes — internal cache
latency and external bus transactions — are decoupled, matching
`fractal_processor`.
