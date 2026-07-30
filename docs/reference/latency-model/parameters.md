# Latency Model — Parameters

The LM is parameterized by two structs: `LmParams` (per layer, from the DNN
image) and `LmHwParams` (per NPU, from the hardware config). All quantities are
**operand or event counts** — the only place bytes appear is `from_config`, the
byte↔operand boundary.

## `LmParams` — per-layer workload

Set by the compiler into the DNN-image `LatencyDescriptor`; read by
`LmModel::reset()` / the constructor. Definition in `latency_model.h`.

| Field | Symbol | Meaning | Consumed by the model |
|-------|--------|---------|-----------------------|
| `fetch_operand0` | F^I | total input operands | yes — ratios, all-fetched gate |
| `fetch_operand1` | F^W | total weight operands (`0` = weightless) | yes — weight side of the estimate |
| `write_output` | F^O | total pass-completion events | yes — ratios, `is_done`, remaining cap |
| `issue_latency` | l | pipeline depth (issue → completion-ready) | yes — delay line, `PipelineFill` |
| `n_min` | | positive issue-width floor per cycle | yes — lower-bound gate, `Underflow`/`Drain` |
| `n_max` | | issue-width ceiling per cycle | yes — estimate cap |
| `prefetch_in` | | GB-resident input gate before the first output | yes — prefetch gate |
| `prefetch_wt` | | GB-resident weight gate before the first output | yes — prefetch gate |

`LmParams` carries **only** what the model uses. The per-layer overhead
(`post_completion_cycles`) and issue-granularity override (`enable_gran`) live on
the DNN-image descriptor and are read by the controller directly, not through
`LmParams`.

## `LmHwParams` — per-NPU hardware

Two kinds of quantity, both properties of the LM's own hardware:

| Field | Meaning | `0` means |
|-------|---------|-----------|
| `input_buf_capacity` | input buffer (B^I) size, operands | unlimited |
| `weight_buf_capacity` | weight buffer (B^W) size, operands | unlimited |
| `output_buf_capacity` | pass-completion timing-queue size, events | unlimited |
| `max_input_ops_per_cycle` | input port width, operands/cycle | unlimited |
| `max_weight_ops_per_cycle` | weight port width, operands/cycle | unlimited |
| `max_output_ops_per_cycle` | output/timing-token service width, events/cycle | unlimited |

The port widths are the LM's **own port capacity** — the max operands the port
moves per cycle regardless of what feeds it (the global buffer for a standalone
or first-stage LM, an upstream stage for a chained one). They are not a property
of the GB connection.

## `LmHwParams::from_config` — the byte↔operand boundary

The factory that maps `FlexNpuSimConfig` onto `LmHwParams`. It is the single
place bytes are converted to operands (divide by `element_size_bytes`).

| Config source | → LmHwParams field |
|---------------|--------------------|
| `hw.buffers.global.read_port_bytes_per_cycle` | `max_input_ops_per_cycle`, `max_weight_ops_per_cycle` (input and weight share the GB read port) |
| `hw.buffers.global.write_port_bytes_per_cycle` | `max_output_ops_per_cycle` |
| `hw.buffers.pe_buffers.input_buffer_capacity_kb` | `input_buf_capacity` |
| `hw.buffers.pe_buffers.weight_buffer_capacity_kb` | `weight_buf_capacity` |
| `hw.buffers.accumulator.capacity_kb` | `output_buf_capacity` |

Notes:

- **Per-function buffer override.** Given a `function_type` ("conv2d",
  "activation", …), `from_config` uses `hw.buffers.pe_buffers_per_function[type]`
  if present — one LM is 1:1 with a function descriptor, and different units may
  have different internal buffers.
- **Port caps default to the GB widths.** This is the GB-fed wiring (a standalone
  LM or the first chain stage). A chained stage's input port is fed by the
  previous stage, not the GB read port; wiring that from the DNN-image
  `operand_src` is the controller's job (see [`chain.md`](chain.md)).
- **Warnings.** `from_config` warns once per process on an lm↔gb inconsistency: a
  port byte-width that is not a whole multiple of `element_size_bytes` (cannot
  carry whole operands), or a PE buffer larger than the whole global buffer.

## Validation — prefetch vs buffer

The `LmModel` constructor throws `std::invalid_argument` when a finite input or
weight buffer cannot hold the initial prefetch
(`prefetch_in > input_buf_capacity`, or `prefetch_wt > weight_buf_capacity`) —
the start threshold could never accumulate, deadlocking the layer. With unlimited
buffers (`0`) there is nothing to check, which is why the shipped configs (whose
`pe_buffers` are unset) never trip it.

## Related config the LM does not read directly

`hw.compute.threshold_ratio` (α) scales `prefetch_in`/`prefetch_wt` at compile
time; `hw.compute.element_size_bytes` is the byte↔operand divisor used only in
`from_config`. Both are hardware-config fields, resolved before the LM sees its
operand-count parameters.
