# Compiler — Implementation

The mechanism, step by step, from a network CSV to a `DnnImage`. The entry point
is `generate_network_from_layer_spec_csv`
(`src/compiler/frontend/loaders/network_csv.cpp:634`); descriptor derivation is
`DnnImageGenerator::set_tiled_latency_params`
(`src/compiler/dnn_image/dnn_image_generator.cpp:1120`).

## 1. Parse the CSV into rows

`load_layer_spec_csv` (`network_csv.cpp:279`) reads the file once:

- The first non-blank, non-`#` line is the **header**; each column name is
  normalized (spaces/`-`/`/`/`.` → `_`) into a name→index map
  (`normalize_header`, `network_csv.cpp:98`).
- `find_column` (`network_csv.cpp:106`) resolves each logical field from a list
  of accepted aliases, case-insensitively — e.g. the reduction-op count column
  is any of `ops_per_pass` / `k` (`network_csv.cpp:322`). This is why the same
  loader accepts both terse and readable CSVs.
- Six columns are **required** (`network_csv.cpp:365`): `layer_id`, a type
  column (`layer_type` or `function_type`), `input_h/w/c`, and `tile_h/w`.
  A row missing any of these — or whose `layer_id`/type fails to parse — is
  skipped.
- Each data row becomes a `UnifiedLayerEntry` (`network_csv.cpp:16`) holding the
  parsed `LayerSpec` geometry, the per-row `PeConfig`, optional R9 `TileShape`,
  the preprocess selector, and the optional oracle block.

Type tokens are mapped by `parse_layer_type_token` / `parse_function_type_token`
(`network_csv.cpp:124`, `:152`) — both accept many spellings (`conv`, `dwconv`,
`fc`, `eltwise`, …). Depthwise, pooling, activation and element-wise rows get
their kernel-shape fields corrected to the op's real footprint
(`network_csv.cpp:423`) — notably a depthwise filter is `K_H·K_W·1` with
`kernel_count = C_in`, so its weight count cannot balloon by `C_in×`.

## 2. Order rows and wire connectivity

After all rows are read, `load_layer_spec_csv` runs three ordering/wiring passes
over the vector:

1. **Stable sort** by `(layer_id, has_function_id, function_id, row_order)`
   (`network_csv.cpp:555`) so a layer's functions are contiguous and ordered.
2. **Auto `function_id` + `operand_src`** (`network_csv.cpp:565`): rows without
   an explicit `function_id` get a per-layer running index; a function whose
   `i1_connect` is unset defaults to the previous function's id (a linear chain),
   and `i2_connect` defaults to "none" (`0xFFFFFFFF`).
3. **Producer→consumer address chaining** (`network_csv.cpp:584`): a `cursor`
   high-water allocator places every input/weight/output tensor. If a row has no
   explicit `input_addr`, its input address is set to the DRAM address its
   producer (`i1_connect`) wrote — in-packet first, else the previous packet's
   last output. Without this, every row would get a fresh input buffer and
   inter-layer retention could never observe a producer's output.

The final `cursor` is the image's DRAM footprint. `generate_network_...`
(`network_csv.cpp:652`) rejects it if it exceeds `address_map.dram_size`, the way
a linker reports a region overflow.

## 3. Derive the latency descriptor (Mode A)

For each ordered row, `generate_network_...` calls
`set_tiled_latency_params(spec, tile_h, tile_w, l2_kb, dataflow, pe)`
(`network_csv.cpp:674`). The hardware inputs come from the config once, up front
(`network_csv.cpp:638`): `pe_config_from`, `dataflow_from`,
`hw.buffers.global.capacity_kb`, and `effective_threshold_ratio`. Inside
`set_tiled_latency_params`:

1. **Output shape** `out_h/out_w/out_c` from the op type and geometry
   (`dnn_image_generator.cpp:1131`).
2. **Per-output ops** `n̄ = estimate_per_output_ops(spec)` — for Conv2D,
   `K_H·K_W·C_in` (`dnn_image_generator.cpp:35`).
3. **Pass count** `p = ⌈n̄ / k⌉` where `k = ops_per_pass`
   (`dnn_image_generator.cpp:1160`).
4. **`F^O = p · |O|`** — `write_output = p · out_h·out_w·out_c`
   (`dnn_image_generator.cpp:1163`). `F^O` counts pass **completions**, not
   pass-steps (see the derivation doc §1).
5. **Resident vs. streamed operands, routed by dataflow**
   (`dnn_image_generator.cpp:1191`):
   - `WS`: weight resident once, `F^W = |W| = K_H·K_W·K_C·K_count`
     (`:1193`); input streamed, `F^I = schedule.total_layer_input_bytes/4`
     (`:1195`).
   - `IS`: input resident at its im2col footprint
     `out_h·out_w·C_in·K_H·K_W` (`:1204`); weight streamed from the schedule
     (`:1207`).
   - `OS`: both streamed from the schedule (`:1211`).
   - `MatMul` skips the tile schedule — both operands load once (`:1166`).
   The streamed byte totals come from `compute_tile_schedule`
   (`dnn_image_generator.cpp:1175`), which sums per-tile refetch (halo + channel
   re-reads); `total_layer_{input,kernel}_bytes` live on `TileSchedule`
   (`src/common/tile_descriptor.h:236`).
6. **Spill ratio `δ`** (`dnn_image_generator.cpp:1222`): only when `p >
   parallel_passes`. `psum_capacity_elems = partial_sum_buffer_kb·1024/4`;
   `concurrent = min(|O|, n_max)`. `δ = 1` if no psum buffer, `1 −
   psum_cap/concurrent` if the concurrent outputs overflow it, else `0`.
7. **Writeback reload** (`dnn_image_generator.cpp:1239`): `δ · (p −
   parallel_passes) · |O|` added to the streamed operand — input for WS, weight
   for IS, both for OS.
8. **Prefetch gates** per op type (`dnn_image_generator.cpp:1258`) — for Conv2D,
   `prefetch_in = prefetch_wt = K_H·K_W·C_in` (one kernel window).
9. **`n_max = parallel_passes·num_output_lanes`, `n_min = parallel_passes`**
   (`dnn_image_generator.cpp:1289`; `PeConfig::n_max/n_min`,
   `src/common/types.h:134`).
10. **Pipeline depth `l`** from PE type (`dnn_image_generator.cpp:1293`):
    systolic → `l = k`; adder-tree → `l = 1 + ⌈log2 k⌉`.
11. `post_completion_cycles = 0` here — that field is filled only by the CSV
    oracle (`dnn_image_generator.cpp:1303`).

Unhandled op types (Activation/ElementWise) fall through to
`set_default_latency_params` (`dnn_image_generator.cpp:1310`), the non-tiled
form of the same computation.

> The formulas above are cross-checked, number by number, against the simulator
> in [`../latency-model/f-derivation.md`](../latency-model/f-derivation.md).
> That doc also records the four known code↔paper notation gaps (`F^O` for
> `g>1`, `reload = (p−g)`, the `F^I` traffic level, `δ` provenance). Treat the
> code lines cited here as the authority for what the compiler emits.

## 4. Oracle override (Mode 2)

If a CSV row supplies all six of `f_i`, `f_w`, `f_o`, `l_pass`, `n_max`, `n_min`
(detected at `network_csv.cpp:539`), the derived descriptor is discarded and the
CSV values are written straight into the spec **after**
`set_tiled_latency_params` runs (`network_csv.cpp:679`): `fetch_operand0/1`,
`write_output`, `issue_latency`, `n_max/n_min`, plus optional `tau_i`/`tau_w`
(→ `prefetch_in`/`prefetch_wt`) and `post_cycles`. This is how the validation
harness pins a descriptor to the paper's published 6 parameters instead of the
compiler's derivation.

## 5. Build the packet image

Each row is converted into a `FunctionRecord` (`network_csv.cpp:699`) carrying
the geometry, the finished `LatencyDescriptor`, `operand_src[4]` connectivity,
`preprocess_type`/`preprocess_param0/1`, `ops_per_output`, `sparsity_mode`, and
the tile shape + `flags`. Tiling is flagged three ways
(`network_csv.cpp:741`): an explicit R9 `TileShape`, else legacy
`tile_h/w < input dim` (sets `FuncFlag::TileEnable` and a `t_r/t_c/t_kern`
shape), else non-tiled.

`build_dnn_image_packets` (`dnn_image_packet.cpp`, declared at
`dnn_image_packet.h:45`) serializes the records into the canonical layout —
`DnnImageHeader` then, per layer packet, a `LayerPacketHeader` followed by its
`FunctionDescriptor`s, with any weight blobs collected into a trailing weight
section (`dnn_image_format.h:8`). The result is returned as `UnifiedCsvModel`
(`image`, `net_cfg`, `num_layers`, `num_layer_packets`).

## 6. Sparsity preprocessing (optional, after emission)

For zero-skipping NPUs, `flexnpusim_system` calls `apply_sparsity(image, net_cfg)`
after the image is built (`src/system/flexnpusim_system.cpp:389`). This is the one
value→count bridge; it is owned by the [function model](../function-model/) and
patches each descriptor's `fetch_operand0`/`prefetch_in` in place.
</content>
