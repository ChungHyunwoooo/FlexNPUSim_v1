# Processor — Modeling Method

This document explains *what the processor models and why*. The code is one
header, `src/systemc/processor/processor.h` (class `Processor`); the concrete
`software_delay` count derivation is in
`latency-derivation.md`.

## The role: simulation start point

The processor is the host CPU that boots the NPU. Its whole job is:

1. write the DNN image base into `DNNSA` (the image is preloaded in DRAM),
2. assert `NPUCR = START`,
3. poll `NPUSR` until the `IDLE` bit is set (reactive, with a per-layer
   watchdog), then
4. read the `PERF_CNT*` performance counters and stop the simulation.

It never runs a compute kernel — the NPU does the work. So the model does not
simulate a processor microarchitecture; it runs the firmware natively and adds
CPU-side latency as an explicit annotation.

## The idiom: native execution + manual bus + annotated latency

Modelled after the DSL-lab `fractal_processor`. Three separable ideas:

- **Native execution.** `Processor::run()` *is* the firmware, compiled into the
  simulator and executed on the host. Functional correctness comes for free —
  there is no instruction interpreter.
- **Manual bus transactions.** The firmware's MMIO accesses (`NPU[reg] = …`)
  cannot be real pointer dereferences on the host, so each is hand-written as an
  AXI transaction (`reg_read`/`reg_write`). Disassembly gives *timing*, not the
  addresses/data — those are compile-time constants here, so keeping them manual
  costs nothing.
- **Annotated latency.** Compute cost is charged by `software_delay(...)` from
  instruction-class counts; memory cost by `mem_access()` from a cache model.

## Internal vs external: two decoupled planes

The central modeling decision. An access is one of two kinds, and they are
priced by two independent mechanisms:

| Plane | What | Mechanism | Bus transaction? | Cache? |
|-------|------|-----------|:---:|:---:|
| **External** | peripheral / MMIO (the NPU registers) | `reg_read` / `reg_write` → real AXI | **yes** | no |
| **Internal** | the CPU's own data-memory load/stores | `software_delay` `store`/`load` args → `mem_access()` | no | **yes** (statistical) |

An **external** access generates a real signal-level AXI transaction on the
register bus; its latency *is* the AXI handshake plus the NPU slave's response
time, already advanced by the transaction itself. No cache penalty is applied —
device registers are non-cacheable. An **internal** access never touches the
bus; it is priced by the statistical cache model only.

This decoupling matches `fractal_processor`, where the cache model and the bus
transactions are separate code paths. Fusing them (charging a cache miss on top
of every MMIO transaction) would double-count MMIO latency and emit phantom
transactions on cache "hits"; the split avoids both.

## `software_delay(store, load, jmp, mul, div, gen)`

Charges the compute of a firmware region as instruction-class counts × per-class
cost:

```
cycles = jmp·cost_jump + mul·cost_mul + div·cost_div + gen·cost_generic
```

The `jmp/mul/div/gen` counts are **hand-derived from the AArch64 disassembly** of
an ARM-host mirror of the firmware region (memory instructions excluded — those
are the two planes above). The manual counts are the deliberate choice: the
firmware is tiny and fixed, so a one-time disassembly read is cheaper than an
automatic objdump-parsing pipeline (deferred). The `store/load` args are the
region's **internal** data-memory accesses; each drives one `mem_access()`.

## `mem_access()` — statistical feedback cache

One internal data-memory access, no bus. It is a feedback model: if the running
hit ratio is above the target `1 − cache_miss_ratio`, this access is charged as a
**miss** (`miss_latency`, pulling the ratio down); otherwise a **hit**
(`hit_latency`). Over many accesses the realised miss fraction converges to
`cache_miss_ratio`. There is no address or tag lookup — the abstraction is a
target miss rate, not a cache geometry.

## `reg_read` / `reg_write` — external transactions

Thin wrappers over `master_if->read_single` / `write_single`, which drive the
full AXI channel handshakes (AW/W/B, AR/R) on the register bus. The NPU's AXI
slave picks the write up beat-by-beat and detects START; a read blocks until the
slave responds. These are genuine transactions — the NPU completing a run is the
proof they are not shortcut.

## Parameters

All knobs load from the hw JSON `processor` section into `config::ProcessorCfg`
(`src/common/flexnpu_config.h`), applied by `Processor::configure()`:

| Key | Default | Meaning |
|-----|:---:|---------|
| `cost_jump` | 5 | branch/jump cycles |
| `cost_mul` | 4 | integer multiply cycles |
| `cost_div` | 10 | integer divide cycles |
| `cost_generic` | 1 | ALU/mov/etc. cycles |
| `cache_hit_latency` | 1 | cycles on an L1 hit |
| `cache_miss_latency` | 100 | cycles on a miss |
| `cache_miss_ratio` | 1.0 | target internal-memory miss fraction |

`cpu_cycles` accumulates the CPU-side total (internal only: `software_delay`
compute + `mem_access`). External MMIO latency is not in `cpu_cycles` — it is the
real transaction's own sim-time.

## What it models — and what it does not

Models: the NPU boot/poll control flow with real bus transactions, per-region
compute latency from instruction-class counts, and a statistical internal-memory
cache — all config-tunable.

Does not model: a processor pipeline / cache geometry / branch prediction (the
counts are static per region), or automatic disassembly (`software_delay` counts
are hand-entered). The internal-cache plane and the `cost_mul`/`cost_div` knobs
are **scaffolding**: the current firmware only pokes NPU registers (all external)
with no internal data memory, so those paths are built and correct but never
exercised until the processor's role expands to timing a real CPU workload.
Reported "Cycles" remains the NPU performance counter (`cnt0`); the CPU-side
`cpu_cycles` is surfaced only under `FLEXNPUSIM_VERBOSE`.
