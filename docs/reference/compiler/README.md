# Compiler

The compiler is the simulator's front end: it turns a **network CSV** plus one
`FlexNpuSimConfig` into a **DNN image** — a flat binary the NPU model reads from
DRAM. Each network layer becomes one or more `FunctionDescriptor`s, and the
compiler's central job is to fill each descriptor's `LatencyDescriptor` — the
operand counts (`F^I`/`F^W`/`F^O`), pipeline depth `l`, issue widths
`n_min`/`n_max`, and prefetch gates — that the [latency model](../latency-model/)
replays as timing. No tensor values are touched here; the compiler works purely
in shapes and counts.

- Source: `src/compiler/` — `frontend/loaders/network_csv.{h,cpp}` (CSV → per-row
  `UnifiedLayerEntry`, ordering, address chaining, oracle override),
  `dnn_image/dnn_image_generator.{h,cpp}` (`set_tiled_latency_params` — the
  F-formula derivation, "Mode A"), `dnn_image/dnn_image_compiler.{h,cpp}`
  (`LayerSpec` → `LayerDescriptor`, plus the Mode B `add_layer_derived`
  mapping-param path), `dnn_image/dnn_image_packet.{h,cpp}` +
  `dnn_image_format.h` (serialize/parse the on-DRAM packet layout),
  `dnn_image/tile_search.{h,cpp}` + `tile_math.h` (output-tile selection)
- Consumes: the network CSV (`model/sw/cnn/<net>.csv`) + `FlexNpuSimConfig`
  (`hw.compute`, `hw.dataflow`, `hw.buffers.global`)
- Produces: `DnnImage` (a `std::vector<uint8_t>`) consumed by `flexnpusim_system`
  and parsed by `NpuController`
- F-formula derivation is documented in
  [`../latency-model/f-derivation.md`](../latency-model/f-derivation.md)
  (verified line-by-line against the simulator); this reference states **which
  code computes them**, not the derivation.

## Why it exists

The latency model is count-driven: it needs `F^I`/`F^W`/`F^O` and a few
compute-org parameters per function, not tensors. Something has to turn a
human-authored layer list into those counts, place every tensor in DRAM, and
emit a binary the hardware model can DMA and parse. That is the compiler. It
also carries the two "escape hatches" the validation workflow needs: an
**oracle override** (paper §III's 6 parameters, supplied directly in the CSV,
replacing the derived descriptor) and a per-layer **PE-mapping override**
(`pe_type`/`k`/`g`/`s`/`psum_kb`) that supersedes the hardware JSON.

## Compile pipeline

```
  network CSV                 FlexNpuSimConfig (hw.compute, hw.dataflow,
 (model/sw/cnn/*.csv)          hw.buffers.global)
        │                              │
        ▼                              ▼
  ┌───────────────────────────────────────────────┐
  │ load_layer_spec_csv                            │   network_csv.cpp
  │  parse rows → UnifiedLayerEntry[]              │
  │  defaults · stable-sort (layer_id, fid)        │
  │  auto function_id / operand_src chaining       │
  │  producer→consumer DRAM address chaining       │
  │  footprint ≤ address_map.dram_size (linker)    │
  └───────────────────────────────────────────────┘
        │  per row
        ▼
  ┌───────────────────────────────────────────────┐
  │ set_tiled_latency_params  (Mode A)             │   dnn_image_generator.cpp
  │  output shape · ops_per_output · passes p      │
  │  F^O = p·|O|                                   │
  │  tile schedule → F^I / F^W (dataflow-routed)   │
  │  δ spill → writeback reload                     │
  │  prefetch · n_min/n_max · l (pe_type)          │
  └───────────────────────────────────────────────┘
        │
        ▼  if CSV row carries f_i..n_min → replace derived fields
  ┌───────────────────────────────────────────────┐
  │ oracle override                                │   network_csv.cpp
  └───────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────┐
  │ FunctionRecord → build_dnn_image_packets       │   dnn_image_packet.cpp
  │  DnnImageHeader + [LayerPacketHeader +         │
  │  FunctionDescriptor×n] × packets               │
  └───────────────────────────────────────────────┘
        │
        ▼  optional, zero-skipping NPUs only
  ┌───────────────────────────────────────────────┐
  │ apply_sparsity (value→count bridge)            │   see function-model/
  └───────────────────────────────────────────────┘
        │
        ▼
     DnnImage  ─────────────────▶  NpuController (run)
```

## Documents

| File | Contents |
|------|----------|
| [`implementation.md`](implementation.md) | the code walkthrough: CSV parse → `UnifiedLayerEntry`, ordering + auto-`function_id` + address chaining, the `set_tiled_latency_params` descriptor build (line-cited F formulas), oracle override, packet emission |
| [`parameters.md`](parameters.md) | the network CSV column schema as a table — geometry, tile, the PE-mapping oracle columns (`pe_type`/`k`/`g`/`s`/`psum_kb`) and their precedence over the hw JSON, the Mode-2 6-parameter latency oracle, `preprocess_type`/`preprocess_param0` (zero-skip permille) |

## One-paragraph summary

`generate_network_from_layer_spec_csv` reads the CSV into `UnifiedLayerEntry`
rows, fills geometry defaults, orders by `(layer_id, function_id)`, auto-assigns
`function_id`/`operand_src`, chains DRAM addresses so a consumer reads its
producer's output, and bounds the image against the DRAM region like a linker.
Per row it calls `set_tiled_latency_params`, which sets `F^O = p·|O|`, routes
resident vs. streamed operands by dataflow (WS/IS/OS) via a tile schedule, adds
writeback-reload traffic scaled by the spill ratio `δ`, and fills
`l`/`n_min`/`n_max`/prefetch from the PE config. A row may carry an **oracle**
(`f_i…n_min`) that replaces the derived descriptor verbatim, or override the PE
mapping (`pe_type`/`k`/`g`/`s`/`psum_kb`) ahead of the hw JSON. The records are
serialized into the packet-format `DnnImage`.
</content>
</invoke>
