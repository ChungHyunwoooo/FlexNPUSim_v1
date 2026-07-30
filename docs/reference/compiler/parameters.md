# Compiler — Parameters

The compiler's configuration is the **network CSV**. One row is one function
(a layer may span several rows / functions). Columns are matched by
`find_column` against a list of aliases (case-insensitive, after normalizing
spaces/`-`/`/`/`.` to `_`), so the primary name below is canonical and the
alias column lists the other accepted spellings. All values are unsigned
integers unless noted. Column source: `network_csv.cpp:307`–`:359`.

`4294967295` (= `0xFFFFFFFF` = `UINT32_MAX`) is the "none" sentinel used by the
connectivity columns.

## Required columns

The row is dropped if any of these is absent or unparseable
(`network_csv.cpp:365`).

| Column | Aliases | Meaning |
|--------|---------|---------|
| `layer_id` | `id` | Layer index. Rows sharing a `layer_id` form one layer packet. |
| `layer_type` | `type`, `op`, `operation` | Op type. `function_type` is used if `layer_type` is absent. Accepts `conv2d`, `dwconv`, `fc`, `pool`, `activation`, `matmul`, `eltwise`, … |
| `input_h` | `in_h` | Input feature-map height. |
| `input_w` | `in_w` | Input feature-map width. |
| `input_c` | `in_c` | Input channels. |
| `tile_h` | `t_h`, `th` | Output-tile height (clamped to ≥1). `tile_h < input_h` implicitly enables tiling. |
| `tile_w` | `t_w`, `tw` | Output-tile width. |

## Geometry columns (optional; typed defaults)

| Column | Aliases | Default | Meaning |
|--------|---------|---------|---------|
| `layer_name` | `name` | `layer_<id>` | Debug label. |
| `kernel_h` | `k_h`, `kh` | 1 | Kernel height. |
| `kernel_w` | `k_w`, `kw` | 1 | Kernel width. |
| `kernel_c` | `k_c`, `kc` | `input_c` | Kernel input-channel depth (per-filter). |
| `kernel_count` | `out_c`, `k_count`, `kernel_n` | `input_c` | Number of filters (= output channels for conv). |
| `stride` | `s` | 1 | Conv/pool stride (≥1). |
| `padding` | `pad`, `p` | 0 | Zero padding. |
| `dilation` | `d` | 1 | Dilation (≥1). |

Depthwise/pool/activation/element-wise rows have their kernel fields normalized
to the op's real footprint after parsing (`network_csv.cpp:423`).

## Function graph columns (optional)

| Column | Aliases | Default | Meaning |
|--------|---------|---------|---------|
| `function_id` | `fid` | auto per-layer running index | This function's id within its layer. |
| `function_type` | `ftype`, `func_type` | derived from `layer_type` | Fine-grained op (`preprocess`, `residual`, `concat`, …); sets `FunctionDescriptor.type`. |
| `i1_connect` | `input1_connect` | previous function's id | Producer `function_id` of operand 0 (→ `operand_src[0]`). |
| `i2_connect` | `input2_connect` | none (`0xFFFFFFFF`) | Producer of operand 1 (→ `operand_src[1]`). |
| `input_addr` | `input_address` | producer-chained | Explicit input DRAM address; overrides address chaining. |
| `kernel_addr` | `kernel_address` | cursor-allocated | Explicit weight address. |
| `output_addr` | `output_address` | cursor-allocated | Explicit output address. |
| `enable_granularity` | `eg` | inherit hw | Per-layer 𝓔 override: `lockstep` \| `independent` \| `grouped:N`. Encoded `(mode<<24)|N` into `LatencyDescriptor.enable_gran` (`network_csv.cpp:525`). |

## PE-mapping oracle columns — override the hw JSON

These columns replace the per-row `PeConfig` that was seeded from the hardware
JSON's `hw.compute` block. When present they win; when absent the hw-JSON preset
stands (`network_csv.cpp:481`). This lets one CSV pin a layer to a specific
mapping without editing the hardware description — the "oracle" the paper's
per-layer analysis needs.

| Column | Aliases | Overrides (`PeConfig`) | Meaning |
|--------|---------|------------------------|---------|
| `pe_type` | `petype` | `pe_type` | `systolic` (`l = k`) or `addertree` (`l = 1+⌈log2 k⌉`). |
| `k` | `ops_per_pass` | `ops_per_pass` | Unit ops one pass handles; sets `p = ⌈n̄/k⌉`. |
| `g` | `parallel_passes` | `parallel_passes` | Passes issued in parallel per output; `n_min = g`. |
| `s` | `num_output_lanes`, `num_pes`, `pes_count` | `num_output_lanes` | Independent output lanes; `n_max = g·s`. |
| `psum_kb` | `psum_capacity_kb` | `partial_sum_buffer_kb` | Local partial-sum capacity (KB). `0` forces spill ratio `δ = 1` (full writeback). |

Precedence for the PE mapping: **per-layer-index override > per-layer-type
override > CSV `k`/`g`/`s`/`psum_kb`/`pe_type` > hw-JSON `hw.compute` preset**.
The first two apply only on the generator's programmatic path
(`resolve_pe_config`, `dnn_image_generator.cpp:71`); on the CSV path the row
columns are the override.

## Latency oracle columns (Mode 2) — override the derived descriptor

If **all six** core columns parse, the compiler-derived `LatencyDescriptor` is
discarded and these values are used verbatim (`network_csv.cpp:539`, applied at
`:679`). The three `tau/post` columns are individually optional. This is the
paper §III 6-parameter descriptor, supplied directly.

| Column | Aliases | Target field | Meaning |
|--------|---------|--------------|---------|
| `f_i` | `F_I`, `fetch_operand0`, `total_input_operands` | `fetch_operand0` (`F^I`) | Total input operands the layer feeds. |
| `f_w` | `F_W`, `fetch_operand1`, `total_weight_operands` | `fetch_operand1` (`F^W`) | Total weight operands. |
| `f_o` | `F_O`, `write_output`, `total_output_writes` | `write_output` (`F^O`) | Pass-completion budget (`p·|O|`). |
| `l_pass` | `issue_latency`, `pipeline_depth` | `issue_latency` (`l`) | Issue-to-completion pipeline depth. |
| `n_max` | `max_outputs_per_cycle` | `n_max` | Pass-issue width upper bound. |
| `n_min` | `min_outputs_per_cycle` | `n_min` | Positive pass-issue width floor. |
| `tau_i` | `threshold_input` | `prefetch_in` | GB-resident input gate before compute. |
| `tau_w` | `threshold_weight` | `prefetch_wt` | GB-resident weight gate. |
| `post_cycles` | `post_completion_cycles` | `post_completion_cycles` | Layer-end overhead cycles. |

Required-together rule: all six of `f_i/f_w/f_o/l_pass/n_max/n_min` must be
present for the oracle to engage; a partial set is ignored and the derivation
stands.

## R9 tile-shape columns (optional)

Any present column sets `FuncFlag::TileEnable` and populates `TileShape`
(`network_csv.cpp:353`, `:510`). Axis meaning is op-specific (see
`dnn_image_format.h:165`).

| Column | `TileShape` field | Meaning |
|--------|-------------------|---------|
| `t_r` | `t_r` | Output-row tile. |
| `t_c` | `t_c` | Output-col tile. |
| `t_oc` | `t_oc` | Output-channel tile (MatMul: `T_N`). |
| `t_ic` | `t_ic` | Reduction tile (MatMul: `T_K`). |
| `t_kern` | `t_kern` | Kernel tile = `k_H·k_W`. |
| `intra_order` | `intra_order` | `TileTraversal` enum value. |

## Preprocess columns — the zero-skip bridge

| Column | Aliases | Default | Meaning |
|--------|---------|---------|---------|
| `preprocess_type` | `preprocess`, `pre_type` | `none` | `none` or `zero_skipping` (aliases `nullhop`, `zeroskip`). Selects a `PreprocessType`. |
| `preprocess_param0` | `pre_param0`, `skip_permille` | 0 | **Zero-skip permille of the layer's input stream.** Carried as `FunctionDescriptor.preprocess_param0`; the tile executor scales input fetch by `keep/1000` (`dnn_image_format.h:224`). |
| `preprocess_param1` | `pre_param1` | 0 | Reserved for preprocessor extension. |

`preprocess_param0` is the compile-time, static form of zero-skipping (a fixed
per-mille the author sets). The **measured** form — running the function model
to profile real activation sparsity — is the `apply_sparsity` value→count
bridge documented under [`../function-model/`](../function-model/); it patches
`fetch_operand0` at run assembly and is orthogonal to this column.

## Config inputs (not CSV)

Read once from `FlexNpuSimConfig` before the per-row loop
(`network_csv.cpp:638`), and used as the derivation defaults / bounds:

| Input | Source | Use |
|-------|--------|-----|
| `PeConfig` preset | `pe_config_from(hw.compute)` | Per-row PE mapping seed (CSV `k`/`g`/`s`/`psum_kb`/`pe_type` override it). |
| `Dataflow` | `dataflow_from(hw.dataflow)` | Routes resident vs. streamed operands (WS/IS/OS). |
| GB capacity (KB) | `hw.buffers.global.capacity_kb` | Tile-schedule budget. |
| threshold ratio | `effective_threshold_ratio(hw.compute)` | Prefetch scaling (`1.0` = preload). |
| `address_map.dram_size` | memory map | Image-footprint bound (linker check, `network_csv.cpp:652`). |
</content>
