# Latency Model

The latency model (LM) is the simulator's timing core. It abstracts one RTL
compute unit as an **operand-arrival-driven queue**: given the number of
input/weight operands offered each cycle, it reports how many outputs the unit
produces, how many operands it accepted, and what discrete state it is in — all
in integer operand counts, never touching tensor values.

One LM instance corresponds to one **function descriptor**. A layer with several
descriptors (e.g. NVDLA `conv → CACC → SDP`, or `conv → activation → pooling`)
becomes a **chain** of LMs (see [`chain.md`](chain.md)).

- Source: `src/model/latency/latency_model.{h,cpp}`,
  `src/model/latency/latency_model_chain.h`
- Tests: `tests/model/test_latency_model.cpp`,
  `tests/model/test_latency_model_chain.cpp`
- Consumed by: `NpuController` (one LM per layer on the streaming path, a chain
  for a multi-descriptor packet).

## Why it exists

Cycle counts for a fixed-dataflow accelerator are determined by how fast
operands arrive versus how fast the unit can consume them and emit results — not
by the tensor values. Modeling that as a small queue (buffers, a per-cycle issue
width, a pipeline depth) reproduces the timing of an arbitrary compute unit
without an RTL model and without a specific PE-array geometry.

## Documents

| File | Contents |
|------|----------|
| [`modeling.md`](modeling.md) | the abstraction: why counts not values, the per-layer descriptor (F^I/F^W/F^O, `l`, `n_min`/`n_max`, prefetch), the exact-integer output relation, conservation, the roofline |
| [`implementation.md`](implementation.md) | the code: the 9-step `update()` cycle, `accept()`, derived buffer levels, the prefetch gate, output backpressure, the `LmState` classification |
| [`parameters.md`](parameters.md) | configuration: `LmParams` (per-layer) and `LmHwParams` (per-NPU) tables, `from_config` (the sole byte↔operand boundary), the prefetch-vs-buffer error |
| [`chain.md`](chain.md) | multi-descriptor packets: `LmChain`, inter-stage carry/backpressure, combinational forwarding, the NVDLA example, the current linear-connectivity assumption |
| [`tests.md`](tests.md) | the contract tests, each mapped to the property it guards |
| [`f-derivation.md`](f-derivation.md) | code-verified traffic formulas: F^O = p·|O|, δ-gated psum reload (p−g)·|O|, tile refetch — with worked numeric checks |
| [`hw-derivation-examples.md`](hw-derivation-examples.md) | deriving the six descriptor parameters (k, g, s → F/l/n) from real accelerator structures, end to end |

## One-paragraph summary

Each cycle the LM `accept()`s offered operands up to its port width and buffer
headroom (the caller re-offers the rest — backpressure), computes how many
outputs the accepted operands can support with the exact integer relation
`floor(F^I_seen · F^O / F^I) − issued`, gates the first issue on an initial
**prefetch** of operands (a conv kernel window), delays issued work by the
pipeline depth `l`, services completed passes through a timing queue bounded by
the accumulator capacity, and classifies the cycle (`PipelineFill`, `Prefetch`,
`Starved`, `Underflow`, `Active`, `Drain`, `Done`). It keeps no fractional state
and conserves exactly: after `F^O` issues it has consumed exactly `F^I`.
