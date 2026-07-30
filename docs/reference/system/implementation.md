# System — Implementation

How `flexnpu_sim::system::run_flexnpusim_system` (in
`src/system/flexnpusim_system.cpp`) turns `argc/argv` into a run. It is the body
called from `sc_main` (`src/flexnpusim_main.cpp`); it constructs every SystemC
module, keeps them alive for the kernel's lifetime, and returns the process exit
code (`0` on completion, `1` otherwise).

## CLI surface

The argument loop (`flexnpusim_system.cpp:184-208`) accepts exactly these flags —
there is no `-bus`, `-l2`, or `-dram` knob; every hardware value comes from the
JSON, not the command line.

| Flag | Meaning |
|------|---------|
| `-hw_conf <hw.json>` | the unified hardware config (required unless `-scenario`) |
| `-network <net.csv>` | the per-layer network CSV (required unless `-scenario`) |
| `-scenario <design:network>` | shorthand resolved to the two paths above (see below) |
| `-model_root <dir>` | root hint for scenario resolution (else the CWD) |
| `-o <result.csv>` | append the one-line result row to this file |
| `-timeline <trace.csv>` | per-cycle NPU timeline trace (passed to `Npu`) |
| `-report <report.txt>` | write the five-section EDA-style report |
| `-h` / `--help` | usage |

`-scenario` is resolved by `resolve_scenario_to_paths`
(`src/system/scenario_spec.cpp`): it splits `design:network` (also `/` or `_`, plus
three aliases), then probes `<root>/model/hw/npu/<design>.json` and
`<root>/model/sw/{cnn,attention}/<design>/<network>.csv`. An explicit
`-hw_conf`/`-network` always wins; failure lists every path it tried.

## Config precedence

The hw JSON is the single source of hardware truth — the C++ preset factories are
gone, so a valid FlexNPUSim config is mandatory (`flexnpusim_system.cpp:307-311`).
The file is loaded by `load_flexnpu_config` **twice**: once early into `flex_early`
(so `hw.preset_name`, compute overrides, and the GB access mode take effect before
image generation and sparsity preprocessing), and once more to pull the AXI/DMA/
buffer knobs used to build the DRAM tier and the AXI stacks.

Runtime locals default in code, then get overwritten from `flex_early`:

| Local | Default | Config field |
|-------|---------|--------------|
| `bus_bits` | 64 | `axi.data_width_bits` (validated: power of two in [8,1024], ≤ compiled `FLEXNPUSIM_DMA_AXI_DATA_W`) |
| `l2_kb` | 256 | `hw.buffers.global.capacity_kb` |
| `dram_type` | DDR4-2400 | `dram.tier` |
| `sys_clock_ns` | 10.0 | `hw.clock_cycle_ns` |
| `db_enabled` | false | `hw.buffers.global.double_buffer != "none"` |
| `l2_mode` / line | scratchpad / 64 | `hw.buffers.global.l2_mode` / `l2_cache_line_bytes` |

A canonical `FlexNpuSimConfig cfg` is then built with `flex_early` as the **base**
(every parsed field rides along) and the CLI-derived values layered on top
(`flexnpusim_system.cpp:491-514`); `npu.set_system_config(cfg)` hands it to the NPU.
Config-honesty warnings (`:292-303`) print one stderr line for fields the runtime
silently ignores (`weight_reuse_across_tiles`, `accumulator.port_bytes_per_cycle`).

## Bus width and master count → template instance

The data-bus AXI spec is `DataSpecT = axi::Spec<AXI4, 4, 32,
FLEXNPUSIM_DMA_AXI_DATA_W, 0>` — the **signal** is compiled at the maximum data
width, while `bus_bits` is the **runtime** beat width (beat bytes = `bus_bits/8`,
carried into each master's `Capabilities`). One per-master V2 AXI stack
(`Master<DataSpecT>` + `AxiTransport` + an `AXI_SIGNALS` bundle) is built per DMA
channel in read-first order: reads `0..N_r-1`, then writes `N_r..N_r+N_w-1`
(`:611-618`). Channel-0 read/write transports go to the NPU's internal engines
(`npu.attach_dma_transports`); channels `1..N-1` feed top-level idle-filler
`DmaEngine`s.

`total_masters = read_channels + write_channels` (default `1+1 = 2`). Because the
R4 bus takes its master count as a **template parameter**, the assembly switches on
`total_masters` and instantiates the matching `InterconnectAxiBusV2<DataSpecT,
NumM, 1>` through `build_data_bus_n<NumM>` (`:660-674`). Supported counts are
`{1,2,3,4,5,6,8,10,12,16}`; the DSE sweep uses `2 (1+1)`, `3 (2+1)`, `6 (4+2)`,
`12 (8+4)`. Any other count throws. Arbitration comes from
`axi.default_arbit_policy` (five policies, default RoundRobin) with per-master
`qos/weight/priority` from `topology.masters` by order.

The single memory slave is `MemoryAxiSlaveV2<DataSpecT>` — a 64 MB dense backing
at `dram_base`, wrapping the DRAM timing object from
`make_dram_timing(flex_early.dram, sys_clock_ns)`. The tier is chosen by
`dram.tier`: `ideal | bandwidth | bank | cycle` (`cycle` hands an ini to DRAMSim3);
`sys_clock_ns` drives the DRAM→sys cycle conversion in the slave.

## Elaboration and run

After building the object graph the assembly preloads DRAM — `memset` `0xAA`, then
`memcpy` the compiled `DnnImage` at `dram_base` (`:685-688`) — and calls
`sc_start()` with **no time bound** (`:698`). The run ends when the `Processor`
calls `sc_stop()`.

### Watchdog / termination

Termination is owned by the `Processor` SC_THREAD (`src/systemc/processor/
processor.h:104-153`), not by this file. The host writes the DNN-image base and
`NPUCR_START`, then polls `NPUSR` every 10 ms:

- `NPUSR_IDLE` set → `completed = true`, break (normal completion).
- otherwise a per-layer watchdog counts stalls; it resets whenever the reported
  layer index advances, and trips after `20000` polls (~200 s sim time) on the
  same layer — printing `NPU timeout at layer N` with `completed = false`.

Either way the thread reads the four 64-bit performance counters and calls
`sc_stop()`. `run_flexnpusim_system` returns `cpu.completed ? 0 : 1`.

## Where the report numbers come from

Two counter sources feed the output:

1. **Run totals** — the NPU's `PERF_CNT0..3`, read by the Processor over the reg
   bus into `cpu.cnt0..cnt3` = **cycles, MACs, mem-read bytes, mem-write bytes**.
   These drive the stderr `RESULTS` banner and the one-line CSV (stdout + `-o`),
   whose derived columns are `fps = 1e9/(cycles·clk_ns)`,
   `util% = 100·MACs/(cycles·peak_mac)`, `dram_bw_gbps = (rd+wr)/(cycles·clk_ns)`,
   and `bus_util%` against a one-beat-per-cycle ceiling (`bus_bits/8 / clk_ns`).
   `peak_mac = ops_per_pass · num_output_lanes · parallel_passes`.

2. **Per-layer records** — `npu.controller().layer_records`
   (`NpuController::LayerPerfRecord`), passed to `write_performance_report`
   (`src/system/performance_report.cpp`) only when `-report` is given, along with
   `perf_read_txns` / `perf_write_txns`.

The report has five sections: configuration, performance summary, per-layer
breakdown, bottleneck analysis, and the output stream (partial-output vs boundary
write). The per-layer **roofline verdict** is a direct term comparison with no
threshold: `mem_bound = memory_cycles > compute_cycles`, and each layer's
`latency = max(mem_cycles, cmp_cycles)` (`performance_report.cpp:129,177`). Under
zero-skipping (`hw.compute.zero_skipping == "on"`) `util%` may exceed 100% — it
then reads as dense-MAC speedup, not PE occupancy, and the report labels it so.

The one-line CSV schema is:

```
npu,network,bus_bits,l2_kb,dram,resolution,dataflow,double_buf,threshold,
cycles,macs,mem_rd,mem_wr,completed,fps,util_pct,dram_bw_gbps,bus_util_pct
```

Appends to an existing `-o` file are schema-aware: the header is written only for a
new file, and the derived-metrics columns are emitted only if the existing header
already declares `dram_bw_gbps` (`:776-802`).
