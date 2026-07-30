# Common

`common/` is the shared foundation every other subsystem builds on: the
**configuration tree**, its **loader**, and the **base types** (memory map, NPU
registers, FSM states, PE geometry, tile-schedule descriptors). It is plumbing —
it holds no timing or functional model of its own; it defines the vocabulary the
rest of the simulator speaks.

- Source: `src/common/`
- Consumed by: everything — the compiler, the controller, the latency model,
  the memory subsystem, and `flexnpusim_system` all read `FlexNpuSimConfig` and
  the types below.

## What lives here

| File | Contents |
|------|----------|
| `flexnpu_config.h` | `FlexNpuSimConfig` — the canonical, user-facing config tree (`hw`, `network`, `axi`, `topology`, `dram`, `address_map`, `processor`, …). Single source of truth for all settings since R2. Also the config→core bridges (`pe_config_from`, `dataflow_from`, `effective_threshold_ratio`). |
| `config_loader.{h,cpp}` | `load_flexnpu_config()` — the JSONC → `FlexNpuSimConfig` parser. One small, defensive parser per section; `$include` resolution; typo guard; cross-section validation. |
| `types.h` | Base types (`addr_t`, `data32_t`), the system memory map (`memory_map::`), the NPU register file and FSM states (`npu_reg::`, `NpuState`), and the compute-array geometry (`PeConfig`, `PeType`, `Dataflow`). |
| `tile_descriptor.{h,cpp}` | The tile-execution model types — `PassDescriptor`, tile/pass schedules, per-tile memory traffic. How a layer is decomposed into GB-sized tiles and channel-slice passes. |
| `json.h` | A minimal JSONC value type (comments allowed) used by the loader. |
| `debug_log.h` | The `FLEXNPU_LOG` logging macros / channel levels. |

## Config loading — how a JSON becomes a `FlexNpuSimConfig`

`load_flexnpu_config(path)` (`config_loader.cpp:475`) reads one JSONC file,
resolves `$include`s recursively, and runs one parser per top-level section.
Rules that hold across all sections:

- **Defaults from the struct.** A missing optional field keeps the default
  declared in `flexnpu_config.h`. The only hard requirement is a workload source
  (`network.csv` or `network.json`), and only when `require_network_source` is
  true — hardware-only JSONs (`model/hw/npu/*.json`) load with it false.
- **Typo guard.** `warn_unknown_keys` reports (once) any key a section parser
  does not recognize, so a misspelled knob is caught instead of silently
  ignored.
- **Profile then override.** `hw.dma.profile` expands into per-field defaults
  (`resolve_profile`); explicit `hw.dma.*` keys then override each field.
- **Aliases.** `hw.global_buffer.inter_layer_retention` is the old spelling of
  `layer_fusion`; both are read, the new key wins.
- **Derived overrides.** `hw.fetch_bitwidth` / `hw.write_bitwidth` (bits) set the
  GB port byte-widths when present.

The hardware JSON's `npu_mapping` block is parsed into `hw.compute`
(`ComputeCfg`); the `pe_config_from` / `dataflow_from` bridges then map it onto
the `PeConfig` / `Dataflow` core types the compiler and latency model consume.

## Documents

| File | Contents |
|------|----------|
| [`parameters.md`](parameters.md) | the full `FlexNpuSimConfig` surface as tables — `hw.npu_mapping` (the k/g/s array geometry), buffers, `global_buffer`, `dma`, `axi`, and the `sparsity_model` values with their exact byte/operand semantics |

There is no `modeling.md` here: `common/` is plumbing, not a model. The `dram`
block is documented under [`../memory/parameters.md`](../memory/parameters.md);
the compute/latency knobs the latency model reads are under
[`../latency-model/parameters.md`](../latency-model/parameters.md).
