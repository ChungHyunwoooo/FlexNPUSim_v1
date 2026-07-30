# Memory — Parameters

All memory configuration is one JSON block, `dram`, parsed into
`config::DramConfig` (`src/common/flexnpu_config.h:267-295`,
`config_loader.cpp::parse_dram`). `dram.tier` selects the fidelity tier; the
remaining fields are read only by the tier(s) that need them (the factory,
`src/systemc/memory/dram/factory.cpp`, passes each tier exactly its fields).

## Selecting the tier

| Field | Values | Default | Meaning |
|-------|--------|---------|---------|
| `tier` | `ideal` \| `bandwidth` \| `bank` \| `cycle` | `cycle` | Which `DramTiming` model to build. An unknown value throws. |

## Fields per tier

Cycle-valued fields (`*_cyc`) are in **NPU/system cycles** and pass through the
AXI slave unchanged. `banks` sets the parallel-server count (backpressure beyond
it) for all three internal tiers.

### `ideal`

| Field | Default | Meaning |
|-------|---------|---------|
| `read_latency_cyc` | 100 | Cycles until a read completes. 0 = free (ideal memory). |
| `write_latency_cyc` | 30 | Cycles until a write completes. |
| `banks` | 8 | Transactions in flight before backpressure. |

### `bandwidth` (MIDAP-style roofline)

| Field | Default | Meaning |
|-------|---------|---------|
| `read_latency_cyc` | 100 | Base read latency, before the byte term. |
| `write_latency_cyc` | 30 | Base write latency. |
| `bandwidth_bytes_per_cycle` | 0 | Shared-bus bandwidth. Adds `size / bw` to the latency. `0` = unlimited (base latency only). |
| `include_writes` | `true` | Whether writes pay the byte term (MIDAP `INCLUDE_DRAM_WRITE=False` → `false`). Writes always pay `write_latency_cyc` regardless. |
| `banks` | 8 | Parallel servers. |

### `bank` (row-buffer locality)

| Field | Default | Meaning |
|-------|---------|---------|
| `banks` | 8 | Bank count — both the decode fan-out and the parallelism. |
| `line_bytes` | 8192 | Row-buffer (DRAM row) size; the decode granularity. |
| `row_hit_cyc` | 20 | Read latency when the addressed row is already open. |
| `row_miss_cyc` | 45 | Read latency when it is not. |
| `write_latency_cyc` | 30 | Write latency (writes have no hit/miss split, but still leave the row open). |

### `cycle` (DRAMSim3)

| Field | Default | Meaning |
|-------|---------|---------|
| `ini` | `""` | Path to a DRAMSim3 `.ini` preset (project-root relative). Empty ⇒ DRAMSim3's built-in `DDR4_8Gb_x8_2400.ini`. |
| `output_dir` | `""` | Optional DRAMSim3 stats output dir (`""` ⇒ `out`). |

The `.ini` presets ship in `model/hw/memory/` (see that directory's `README.md`),
copied verbatim from upstream DRAMSim3 `configs/`:

| Preset | DDR | Data rate |
|--------|-----|-----------|
| `DDR4_3200_x8.ini` | DDR4 | 3200 MT/s |
| `DDR3_1600_x16.ini` | DDR3 | 1600 MT/s |
| `DDR3_1866_x16.ini` | DDR3 | 1866 MT/s |
| `LPDDR4_2400_x16.ini` | LPDDR4 | 2400 MT/s |

Referenced from a hardware JSON as `dram.ini` (the current key; the
`model/hw/memory/README.md` example predates this and shows the old
`dram_dramsim3_ini` / `memory.timing_tier` spelling). A typical `cycle` block:

```json
"dram": { "tier": "cycle", "ini": "model/hw/memory/LPDDR4_2400_x16.ini" }
```

## Clock conversion

`make_dram_timing(cfg, sys_clock_ns)` (`factory.cpp`) receives the NPU/sim clock
period from `hw.clock_cycle_ns` (`flexnpusim_system.cpp:318`). The internal tiers'
latencies are already in system cycles, so they pass through unchanged; only the
`cycle` tier uses `sys_clock_ns` — to convert its DRAM `tCK` into
`dram_per_sys_ = sys_clock_ns / tCK` DRAM cycles advanced per `tick()`.

## Layer-level roofline (separate from `tier`)

| Field | Values | Default | Meaning |
|-------|--------|---------|---------|
| `tiled_dma_timing` | `sim` \| `peak_bw` | `sim` | How the controller's **tiled** path charges layer memory time. `sim` uses the elapsed DMA/AXI/DRAM sim time; `peak_bw` charges `read_latency_cyc + bytes / bandwidth_bytes_per_cycle` (honoring `include_writes`) analytically and suppresses per-burst DMA timing. See [`modeling.md`](modeling.md). |

`peak_bw` reuses `read_latency_cyc`, `bandwidth_bytes_per_cycle`, and
`include_writes` from the same `dram` block; when `bandwidth_bytes_per_cycle` is
0 it falls back to the AXI bus width (`layer_timing_strategy.cpp:15-23`). It is
independent of `tier` — e.g. `midap_peakbw.json` pairs `tier: bandwidth` with
`tiled_dma_timing: peak_bw`.
