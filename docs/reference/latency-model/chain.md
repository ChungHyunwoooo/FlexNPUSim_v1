# Latency Model — Chain

A layer often decomposes into several **function descriptors** — NVDLA runs
`conv → CACC → SDP`; a fused CNN layer runs `conv → activation → pooling`. Each
descriptor is one `LmModel`; the descriptors of one layer are chained so that a
stage's output operands stream into the next stage. `LmChain`
(`latency_model_chain.h`) is that cycle-streaming chain.

## Packet grouping

Consecutive function descriptors that share a `packet_layer_id` form one
**packet** (a multi-function layer). `NpuController::count_packet_group(idx)`
returns the packet size; the descriptors are chained in `function_id` order.
Descriptor 0 is the main LM; descriptors 1..N-1 are the chain stages. The network
CSV lists them per `layer_id`, e.g. ResNet-18 on NVDLA:

```
  layer_id  function_id  function_type
  0         0            conv2d       ─┐
  0         1            activation    ├─ one packet → a 3-LM chain
  0         2            pooling      ─┘
  1         0            conv2d       ─┐
  1         1            activation   ─┘  a 2-LM chain
```

So NVDLA `conv(0) → CACC(1) → SDP(2)` becomes a 3-LM chain in that order — the
order the CSV author wrote.

## How `LmChain` runs

`LmChain` holds the stage pointers and a per-stage **carry** — operands offered
to a stage but not yet accepted. `step(upstream, offer_weight)` runs one cycle:

```
  carry[0] += upstream                          // producer feeds stage 0
  for d in 0..N-1:
      r = stages[d].update(carry[d], offer_weight(d))
      carry[d] -= r.consumed_input              // un-accepted operands wait here
      carry[d+1] += r.writeback_outputs         // this stage's output → next stage's input
  return final stage's writeback
```

Three properties follow:

- **Backpressure and conservation.** A stage accepts only up to its port width
  and buffer headroom; the rest stays in `carry[d]` for next cycle. A middle
  stage with a finite B^I therefore back-pressures its producer instead of
  losing operands, and the chain conserves — `carry_total()` returns to `0` when
  drained.
- **Combinational forwarding.** Stage `d`'s output this cycle reaches stage `d+1`
  this cycle (no inter-stage register). Each stage's own pipeline depth `l`
  already delays its work, so a token still cannot cross the whole chain in one
  cycle — only matured tokens forward. Total chain latency is the sum of the
  stages' `l`, not `N` extra cycles.
- **Time is `max`, not sum.** Because stages overlap, a packet's cycle count is
  the bottleneck stage's, plus each stage's one-time pipeline fill — the
  "chained < sequential" contract, verified in [`tests.md`](tests.md).

Weights are offered per stage by the caller's `offer_weight(d)` (typically the
whole tensor once, on the first cycle); a weightless stage is offered `0`.

## Reuse across layers

`LmChain` is non-owning and rebuilt per layer; the `LmModel` stages are reused
across layers via `reset()`, which clears all counters. Each layer conserves and
completes on its own — the models keep no state across resets.

## Current limitation: linear connectivity

`LmChain` assumes **linear** flow: stage `d`'s output always feeds stage `d+1`'s
**input**, and weights always come from the caller (the global buffer). This is
correct for the shipped CNN/NVDLA packets (`conv → act → pool`, where each
stage's input is the previous stage's output and weights are external), and the
network CSV defaults `i1_connect` to the previous function id.

It is not the general case. The DNN image carries explicit per-operand sources
(`operand_src[0]` = `i1_connect` for input, `operand_src[1]` = `i2_connect` for
weight; `0xFFFFFFFF` = external/DRAM), which can express:

- a stage whose **output feeds the next stage's weight** (not input) — which
  `LmChain` cannot represent (it always forwards output → next input);
- a **non-adjacent** connection (a residual within a packet);
- a stage read from a **separate weight memory** rather than the GB.

Honoring `operand_src` — reading the real connectivity, wiring each stage's
input/weight from the right source, and validating that a producer's output
operand count matches the consumer's operand count — is **topology**, and belongs
to the caller. The LM stays a pure timing unit with no connection state; the
`NpuController` will own connectivity-aware wiring. Until then, packets are
assumed linear.
