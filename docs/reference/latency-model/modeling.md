# Latency Model — Modeling Method

This document explains *what the LM models and why*. The mechanics of the code
are in [`implementation.md`](implementation.md); the config that parameterizes
it is in [`parameters.md`](parameters.md).

## The count abstraction

The LM abstracts a compute unit by **operand counts**, never tensor values. Its
inputs each cycle are two integers — how many input operands and weight operands
are offered — and its outputs are integers — how many operands it accepted and
how many result-passes it produced. It holds no data, only counters.

This is deliberate. Cycle count on a fixed-dataflow accelerator is set by
operand *flow* (arrival rate, buffer occupancy, issue width, pipeline depth),
not by the numbers being multiplied. Two consequences:

- **Functional accuracy is not the LM's job.** It belongs to the function model
  (`model/function/`, OpenCV/PyTorch parity). The two planes are independent.
- **Value-dependent timing enters only as a count correction.** Activation
  sparsity (zero-skip) changes timing by removing operands; that removal is
  measured once at preprocessing (the sparsity profiler) and handed to the LM as
  smaller counts. The LM itself stays value-blind.

## The per-layer descriptor

The compiler reduces each function to a small descriptor the LM replays. The six
quantities the LM actually consumes:

| Symbol | Field | Meaning |
|--------|-------|---------|
| F^I | `fetch_operand0` | total input operands the layer reads |
| F^W | `fetch_operand1` | total weight operands the layer reads (`0` = weightless: pooling, activation) |
| F^O | `write_output` | total pass-completion events the layer emits (`passes_per_output × output elements`) |
| l | `issue_latency` | pipeline depth: cycles from issuing a pass to it becoming completion-ready |
| n_min | `n_min` | minimum positive issue width per cycle (the lower-bound emission gate) |
| n_max | `n_max` | maximum issue width per cycle (peak passes emitted per cycle) |

Two more ride in the descriptor but gate the *start*, not the steady state:
`prefetch_in`/`prefetch_wt` (see [Prefetch](#the-prefetch-gate-first-output)).

### F^O is passes, not final outputs

`write_output` counts **pass completions**, not finished output elements. A
partial-sum accumulation over `passes_per_output` passes emits `passes_per_output`
completion events per output element. This is why F^O drives the compute term:
the array is busy for as many passes as it emits, whether or not each pass
finalizes an output.

## Structure-agnostic compute

The LM never assumes a PE-array geometry. Peak throughput is
`MAC/cycle = k · g · n_max` where `k` = ops per pass, `g` = parallel passes,
`n_max` = output lanes, and `l` is the pipeline depth. A 16×16 systolic array,
a `k1·n128` spatial unit (Nullhop), and a `16·8·g2` adder-tree all reduce to the
same four numbers. There are no `pe_rows`/`pe_cols`; a mapping is described by
`(k, g, n, l)` in the network CSV, and subunit-specific overhead is modeled by
writing each subunit as its own descriptor and chaining them.

## The core relation: outputs from accepted operands

The invariant the whole model rests on: the number of output passes the operands
seen so far can support is

```
  supportable(t) = floor( F^I_accepted(t) · F^O / F^I )
```

and the number still *issuable* this cycle is `supportable(t) − issued`, capped
by `n_max` and by the same quantity on the weight side. This is exact integer
arithmetic — no floating-point ratio is stored.

**Why this form.** `F^O / F^I` is the output-per-input rate. Multiplying the
operands accepted so far by that rate gives the total passes those operands can
justify; subtracting what has already been issued gives what is left to issue.
It is algebraically identical to the older "buffer level × ratio" formulation
(`floor(x − integer) = floor(x) − integer`) but stays in integers and cannot
drift.

### Conservation

Consumption is *derived*, not tracked separately: the operands an LM has consumed
after issuing `I` passes are `floor(I · F^I / F^O)`, and the buffer level is
`accepted − consumed`. When `I` reaches `F^O` the consumed count is exactly F^I.
So over a layer the LM drains exactly F^I input and F^W weight operands and emits
exactly F^O passes — no operand is created or lost, even under finite buffers
(the un-accepted remainder waits at the supplier; see backpressure below).

## The prefetch gate: first output

A conv output cannot be computed until at least a full kernel window of input has
arrived; a pool output needs its window. The LM models this as a **prefetch
gate**: before its first issue it holds until `prefetch_in` input operands (and,
for a weighted unit, `prefetch_wt` weight operands) have accumulated. The
compiler sets `prefetch_in` to the per-tile input footprint (a kernel window for
conv, a window for pool), scaled by the preload ratio `α` (`threshold_ratio`:
`1.0` = wait for the whole window, `<1.0` = start streaming early).

The gate applies only before the first issue; once issuing begins it never
re-arms, and it releases early if all operands have already arrived (so a
prefetch larger than the layer's supply cannot deadlock). A prefetch larger than
a *finite* input buffer, however, can never accumulate — that is a config error
the model rejects at construction (see [`parameters.md`](parameters.md)).

## Backpressure: buffers and ports

The LM has two per-cycle limits on how many operands it takes in:

- **Port width** — `max_*_ops_per_cycle`, the operands the unit's input/weight
  port moves per cycle. This is the LM's own port capacity; it applies whether
  the port is fed by the global buffer (a standalone or first-stage LM) or by an
  upstream stage in a chain.
- **Buffer capacity** — `*_buf_capacity`, how many operands the internal buffer
  (B^I/B^W) holds at once.

Accepted = `min(offered, port width, buffer headroom)`. The un-accepted remainder
is **not stored and not dropped**: `update()` reports what it consumed, and the
caller re-offers `offered − consumed` next cycle. That is the backpressure path —
a slow consumer stalls its supplier, and conservation holds across it.

The output side is symmetric: a bounded pass-completion timing queue (sized from
the accumulator capacity) back-pressures issue when it and its retire backlog are
full, so a stalled writeback stops the array instead of building unbounded work.

## Layer timing is a roofline

Per layer, `time = max(memory_cycles, compute_cycles)`. Load and compute overlap
at the layer level; whichever term is larger sets the layer time, with no
threshold — the roofline verdict is a direct comparison of the two terms. The LM
supplies the compute term (`≈ ⌈F^O / effective issue width⌉ + l`); the memory
term comes from the memory-timing model (`memory/`).

## What it models — and what it does not

Models: operand-arrival timing, finite input/weight buffers and their
backpressure, a per-cycle issue width with a lower-bound gate, a pipeline fill,
the first-output prefetch, a bounded output/accumulator queue, and (via chaining)
multi-descriptor packets.

Does not model: tensor values (function model), the exact PE-array wiring
(reduced to `k·g·n·l`), or arbitrary inter-descriptor connectivity — the chain
currently assumes linear `stage → next stage input` flow; reading the DNN-image
`operand_src` to wire input/weight/non-adjacent connections is the controller's
responsibility and is not yet done (see [`chain.md`](chain.md)).
