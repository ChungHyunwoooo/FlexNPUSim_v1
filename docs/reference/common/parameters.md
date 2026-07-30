# Common — Parameters

The full `FlexNpuSimConfig` surface (`src/common/flexnpu_config.h`), block by
block, as the loader parses it (`src/common/config_loader.cpp`). Fields not
listed keep their struct default. The `dram` block is documented separately in
[`../memory/parameters.md`](../memory/parameters.md); the per-layer/latency knobs
are in [`../latency-model/parameters.md`](../latency-model/parameters.md).

## `hw` (root)

| Field | Default | Meaning |
|-------|---------|---------|
| `preset_name` | `""` | Label for logging (e.g. `"nvdla-large"`). |
| `clock_cycle_ns` | 10.0 | NPU/system clock period. Also the DRAM `cycle`-tier conversion base. |
| `dataflow` | `WS` | `WS` \| `IS` \| `OS` — weight / input / output stationary. |

## `hw.npu_mapping` — the compute array (`ComputeCfg`)

Parsed by `parse_compute` into `hw.compute`, then bridged to `PeConfig` by
`pe_config_from`. The array is a three-level nest of PEs; three keys give its
geometry `(k, g, s)`:

| Key | Symbol | Default | Meaning |
|-----|--------|---------|---------|
| `ops_per_pass` | **k** | 16 | PEs per PE-group — the reduction width of one pass. |
| `parallel_passes` | **g** | 1 | PE-groups per PE-set — outputs produced per cycle by one set. |
| `num_output_lanes` | **s** | 16 | PE-sets — independent output lanes. |

Peak issue width is `n_max = g·s` and `n_min = g` (`PeConfig::n_max/n_min`,
`types.h`); the array holds `k·g·s` MACs. E.g. a 16×16 Gemmini is
`(16, 1, 16)`; NVDLA `nv_full` (Atomic-C 64, Atomic-K 16) is `(64, 1, 16)` =
1024 MACs.

Remaining `npu_mapping` keys:

| Key | Default | Meaning |
|-----|---------|---------|
| `unit_type` | `systolic` | `systolic` \| `adder_tree` \| `spatial`. Selects `PeType` (adder_tree ⇒ log-depth issue latency; else systolic). |
| `element_size_bytes` | 2 | Bytes per operand (INT8=1, FP16=2, FP32=4). The byte↔operand divisor. |
| `local_psum_buffer_kb` | 16 | Local partial-sum buffer. |
| `threshold_ratio` | 0.0 | Tile-load strategy: scales `prefetch_in`. 1.0 = preload (full first-pass data before compute); <1.0 = streaming. 0 = unset ⇒ treated as 1.0. |
| `zero_skipping` | `auto` | `on` \| `off` \| `auto` (auto ⇒ off). Gates zero-skip preprocessing. |
| `sparsity_model` | `off` | How activation sparsity is modelled — see below. |
| `sparsity_detect_width` | 0 | `lm`-model dense-scan width (operands/cycle). 0 ⇒ `ops_per_pass·num_output_lanes`. |
| `sparsity_overhead_cyc` | 0 | `overhead`-model lump latency per layer. |
| `enable_granularity` | `auto` | Issue-enable structure → `enable_gran`. `lockstep` (all accumulators enable together, NVDLA CACC) \| `independent` (per-accumulator, starvation narrows to `n_min`) \| `auto` (lockstep iff `n_min == n_max`). |
| `max_compute_stall_cycles` | 10000 | Deadlock watchdog: bail the compute loop after this many no-progress cycles. |

### `sparsity_model` — the four semantics

Activation sparsity enters as a per-layer skip fraction from the network CSV
(`preprocess_type=ZeroSkipping`, `preprocess_param0 = skip_permille`). With
`keep = (1000 − skip_permille) / 1000`, the four values differ in *what* they
scale and *what cost* they add (`flexnpu_config.h:49-67`;
`npu_controller_tile.cpp:130-153` for removal, `:599-622` for the mechanism cost):

| Value | Input fetch bytes (F^I) | Weight fetch bytes (F^W) | Compute F^O (MAC passes) | Extra cost |
|-------|:---:|:---:|:---:|------------|
| `off` (legacy) | ×keep | — | — | none — memory only. |
| `remove` | ×keep | ×keep | ×keep | none — the *ideal* benefit (fewer fetches, fewer passes). |
| `lm` | ×keep | ×keep | ×keep | a dense-scan stage: `+issue_latency` fill, then `max(., ⌈F^I / detect_width⌉)` — chain-overlapped, bottlenecks only when the scan is slower than the MAC. |
| `overhead` | ×keep | ×keep | ×keep | `+sparsity_overhead_cyc` per layer (a constant, pessimistic bound). |

So `off` shrinks only the input DMA; `remove`/`lm`/`overhead` also drop skipped
operands from the weight fetch and the compute (`output_write_operands`, the
compute F^O — not the write bytes); `lm` and `overhead` then add the mechanism's
own cost on top of that ideal benefit.

## `hw.global_buffer` — the on-chip pool (`GlobalBufferCfg`)

The L2 global buffer: capacity, partitioning, ports, and residency policy.
Selected keys:

| Key | Default | Meaning |
|-----|---------|---------|
| `capacity_kb` | 512 | GB size. |
| `partition_mode` | `fixed` | `fixed` \| `shared` \| `flexible`. |
| `input_/weight_/output_partition_kb` | 0 | Per-region caps (0 = uncapped). |
| `read_/write_port_bytes_per_cycle` | 0 | GB↔PE port widths (0 = unlimited); also settable via `hw.fetch_bitwidth`/`write_bitwidth` in bits. |
| `backpressure_enabled` | `true` | GB fullness stalls DMA. |
| `output_location` | `gb` | Where psum lives so it is not double-counted: `gb` \| `accumulator` (NVDLA CACC) \| `pe_local` (MIDAP OS). |
| `input_/weight_/output_compression_ratio` | 1.0 | Effective capacity = physical × ratio; also scales boundary bytes (>1 = compressed, Nullhop). Composes multiplicatively with zero-skip. |
| `weight_gb_stationary` | `dataflow` | `dataflow` \| `resident` (loaded once, reused across tiles) \| `streamed`. |
| `layer_fusion` | `none` | `auto` (fuse when output fits GB) \| `none`. Keeps a layer's output resident for the next layer (alias `inter_layer_retention`). |
| `model_spill` | `false` | Opt-in: derive the output tile from runtime GB capacity (too-small GB forces re-tiling / psum round-trips). Default keeps the compiler-baked tile. |
| `l2_mode` | `""` | `""` scratchpad \| `cache` (hit/miss per `l2_cache_line_bytes`). |
| `double_buffer` | `none` | `none` \| `ping_pong` \| `group`. |

## `hw.buffer` — PE-internal + accumulator (`BuffersCfg`)

PE-internal operand buffers fed from the GB, given as `{capacity, unit}` where
`unit` is `kb` (default) or `operands` (`buffer_cap_kb`). These become the
latency model's buffer caps (see the latency-model parameters doc).

| Key | Default | Meaning |
|-----|---------|---------|
| `input` | 0 = unlimited | Input operand buffer (`pe_buffers.input_buffer_capacity_kb`). |
| `weight` | 0 = unlimited | Weight operand buffer. |
| `output` | 0 = unlimited | Accumulator / pass-completion buffer (`accumulator.capacity_kb`). |
| `per_function` | — | Per-function-type overrides (`"conv2d"`, `"activation"`, …), inheriting unset fields from the defaults — one latency model ↔ one function descriptor, so units may have different internal buffers. |

## `hw.dma` — DMA engine (`DmaCfg`)

`profile` expands into the fields below (`resolve_profile`); explicit keys then
override per field.

| Key | Default | Meaning |
|-----|---------|---------|
| `profile` | `generic` | Named preset filling the per-field defaults. |
| `read_channels` / `write_channels` | 1 / 1 | DMA channel counts. |
| `buffer_size_kb` | 16 | Internal FIFO depth → the outstanding window. |
| `max_burst_length` | 8 | Beats per burst. |
| `max_issuing_reads` / `max_issuing_writes` | 0 | AXI issuing capability (0 = derive). |

## `axi` — transport (`AxiConfig`)

| Key | Default | Meaning |
|-----|---------|---------|
| `protocol` | `AXI4` | `AXI3` \| `AXI4`. |
| `data_width_bits` | 64 | Bus width (beat bytes = width/8). |
| `addr_width_bits` / `id_width_bits` | 32 / 4 | Address / ID widths. |
| `max_burst_length` | 256 | Beats (AXI3 = 16, AXI4 = 256). |
| `max_outstanding_reads` / `max_outstanding_writes` | 8 / 4 | Outstanding transaction windows. |
| `modeling_mode` | `Full` | `Full` \| `LatencyOnly`. |
| `default_arbit_policy` | `RoundRobin` | Bus arbitration. |

## Other top-level blocks

| Block | Struct | Purpose |
|-------|--------|---------|
| `network` | `NetworkConfig` | Workload source (`csv` or `json`, one required) + input dims. |
| `dram` | `DramConfig` | External memory — see [`../memory/parameters.md`](../memory/parameters.md). |
| `topology` | `TopologyConfig` | Externalized N×M crossbar (masters/slaves/links). |
| `address_map` | `AddressMapCfg` | MMIO/DRAM/SRAM bases and sizes (defaults mirror `memory_map::`). |
| `processor` | `ProcessorCfg` | Host-CPU firmware latency model — see [`../processor/`](../processor/). |
| `functional` | `FunctionalConfig` | Value-based functional run (OFM cache, seed). |
| `compiler` | `CompilerCfg` | `manual` \| `compile` + tiling `strategy`. |
| `sim` / `logging` / `dse` | `SimConfig` / `LogConfig` / `DseSweepConfig` | Output paths, log levels, sweep axes. |
