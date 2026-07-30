# System

The system layer is the **top-level assembly** — the one function that turns a
pair of input files into a finished run. It parses the CLI, loads the hardware
config, elaborates the whole SystemC model (clock, `Npu`, `Processor`, the
per-master AXI stacks, the N×M data bus, the memory slave and its DRAM timing
tier), starts the kernel, and formats the report.

Everything else in the reference tree is a block this layer instances and wires;
nothing here computes timing itself. It is the seam between the OS process
(`argc/argv`, files, exit code) and the simulation.

- Source: `src/system/` (`flexnpusim_system.{h,cpp}`, `performance_report.{h,cpp}`,
  `scenario_spec.{h,cpp}`, `packet_execution_policy.{h,cpp}`)
- Entry symbol: `flexnpu_sim::system::run_flexnpusim_system(argc, argv)`,
  called from `src/flexnpusim_main.cpp`'s `sc_main`
- Instances: one [`Npu`](../npu/), one [`Processor`](../processor/), one
  `MemoryAxiSlaveV2` + a [DRAM tier](../memory/), one N×M AXI [bus](../bus/)
- Consumes: `FlexNpuSimConfig` (the parsed hw JSON) and the compiled `DnnImage`

## Why it exists

A run needs one place that owns the whole object graph: the SystemC modules must
be constructed in the right order, bound to each other, handed the config values
they each need, and kept alive for the kernel's lifetime. Concentrating that in a
single `run_flexnpusim_system` keeps every timing block (NPU, bus, DRAM,
processor) free of process-level concerns — argument parsing, file I/O, and the
report all live here, out of the model.

## The assembly

```
  argv ─▶ CLI parse ─▶ scenario_spec ─▶ load_flexnpu_config ──▶ FlexNpuSimConfig
                                              │
                       compile (network CSV) ─┴─▶ DnnImage ──▶ preload into memory
                                              │
                                              ▼   SystemC elaboration
        ┌──────────┐   reg bus    ┌────────────────────┐
        │ Processor │◀────────────▶│ Npu (controller +  │
        │ (host CPU)│  MMIO/poll   │  DMA + UFB + GB)   │
        └──────────┘              └──────────┬─────────┘
                                    ch0 rDMAC │ ch0 wDMAC  (+ idle fillers)
                                  ┌───────────┴───────────┐
                                  │ per-master V2 AXI stack│  Master+AxiTransport
                                  └───────────┬───────────┘  ×total_masters
                                    N×M data bus (InterconnectAxiBusV2<W,NumM,1>)
                                              │
                                  ┌───────────┴───────────┐
                                  │ MemoryAxiSlaveV2 + DRAM│  make_dram_timing()
                                  └───────────────────────┘
                                              │
                     sc_start() ─▶ counters ─▶ report / CSV ─▶ exit code
```

The `Processor` boots the `Npu` over the register bus and polls it to completion;
the `Npu` streams the network through its DMA/compute/write pipeline, driving AXI
transactions across the data bus to the DRAM-backed memory slave. When the
processor sees IDLE (or trips its watchdog) it reads the performance counters and
calls `sc_stop()`; control returns for reporting.

## Documents

| File | Contents |
|------|----------|
| [`implementation.md`](implementation.md) | the `run_flexnpusim_system` walkthrough: the CLI surface, config precedence, how master/bus width select the bus template, watchdog termination, and where `performance_report` gets each number |
| [`topology.md`](topology.md) | the system-composition (topology) config design: masters, slave ranges, links, and the single-owner connection law |

The subsystem's four translation units:

| File | Role |
|------|------|
| `flexnpusim_system.{h,cpp}` | `run_flexnpusim_system` — the assembly itself |
| `scenario_spec.{h,cpp}` | `resolve_scenario_to_paths` — a `<design>:<network>` shorthand → hw JSON + network CSV under a model root |
| `performance_report.{h,cpp}` | `write_performance_report` — the five-section EDA-style `-report` text (pure formatting) |
| `packet_execution_policy.{h,cpp}` | `PacketExecutionPolicy::decide` — a domain policy (bypass input/output DMA for packet-internal function execution); consumed by `NpuController`, not by the assembly |

## One-paragraph summary

`run_flexnpusim_system` parses the CLI, resolves a scenario or an explicit
`-hw_conf`/`-network` pair, loads the hardware JSON into one `FlexNpuSimConfig`,
compiles the network CSV into a `DnnImage`, and elaborates the SystemC model: a
clock, one `Npu`, a `Processor` host, one per-master AXI stack per DMA channel,
an N×M `InterconnectAxiBusV2` chosen at compile time by `total_masters`, and a
memory slave fronting a config-selected DRAM tier. It preloads the image into
DRAM, runs `sc_start()` until the processor stops the kernel, then reads the four
NPU performance counters and emits a one-line CSV, a banner, and (with `-report`)
a per-layer roofline breakdown where each layer's latency is
`max(mem_cycles, cmp_cycles)`.
