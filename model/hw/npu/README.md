# Hardware JSON Inputs

- Path rule: `model/hw/npu/<npu>.json`
- Strict JSON (no comments — `json.load` and the C++ JSONC loader both accept these).
- Full schema as tables: `docs/reference/common/parameters.md`; latency-model contract:
  `docs/reference/latency-model/parameters.md`. Reference sources per validated
  accelerator: `validation/results.md`.

The latency model is a generic streaming compute stage
(`fetch → issue → generate → write`). The JSON exposes its contract directly.

## `hw` block

| field | meaning |
|---|---|
| `preset_name` | display label (no C++ presets) |
| `clock_cycle_ns` | clock period |
| `dataflow` | `WS` \| `IS` \| `OS` |
| `npu_mapping` | **Mode 2** structural source (physical HW → compiler-derived descriptor) |
| `buffer` | LM-internal operand buffers `{input, weight, output}` |
| `fetch_bitwidth` / `write_bitwidth` | GB↔PE port width (bits/cycle, `0` = unlimited) |
| `global_buffer` | GB / L2 pool |
| `dma` | optional DMA knobs |

### `npu_mapping`
Physical array description the compiler maps onto the LM contract:
`unit_type` (`systolic` \| `adder_tree`), `ops_per_pass` (k), `parallel_passes`,
`num_output_lanes`, `element_size_bytes`, `local_psum_buffer_kb`, `pe_rows`,
`pe_cols`, `threshold_ratio`, `zero_skipping`, and (NVDLA) `enable_granularity`.

### `buffer`
Each of `input` / `weight` / `output` is `{ "capacity": N, "unit": "operands" | "kb" }`.
`capacity: 0` = unlimited (unit-agnostic). `output` maps to the accumulator.
Per-function-type overrides (inherit unset fields from the defaults):
```jsonc
"buffer": {
  "input":  { "capacity": 0, "unit": "operands" },
  "weight": { "capacity": 0, "unit": "operands" },
  "output": { "capacity": 16, "unit": "kb" },
  "per_function": { "activation": { "input": { "capacity": 1, "unit": "kb" } } }
}
```

### `global_buffer`
`capacity_kb`, `partition_mode`, `backpressure_enabled`, `output_location`
(`gb` \| `accumulator` \| `pe_local`), plus optional L2 / spill / retention knobs.

## Two modes

- **Mode 2 (white-box)**: provide `npu_mapping`; the compiler derives the
  per-layer descriptor (`F^I/F^W/F^O`, `issue_latency`, `min/max_output_per_cycle`,
  `threshold`) from HW structure + the network.
- **Mode 1 (black-box)**: omit structure and supply the per-layer descriptor
  directly in the DNN image CSV (`model/sw/cnn/<npu>/<network>.csv` columns
  `f_i, f_w, f_o, l_pass, n_max, n_min, tau_i, tau_w`). NPU internals stay hidden.

`axi` and `dram` blocks are unchanged (see `examples/baseline.json`).
