# Latency Model — Tests

The tests are executable contracts: each guards one property of the model. They
run without SystemC (the LM is pure C++) and assert on `update()` return values.
Asserts stay live in tests (`-UNDEBUG`).

## `tests/model/test_latency_model.cpp`

The unit under test is a single `LmModel`. The shared fixture `small_layer()` is
F^I=64, F^W=32, F^O=32, l=2, n_min=1, n_max=4, prefetch_in=8, prefetch_wt=4.

| Test | Property | How |
|------|----------|-----|
| `test_conservation_and_bound` | conservation + compute bound | with an unlimited faithful supplier the model consumes exactly F^I and F^W, writes back exactly F^O, and reaches `Done`; cycles ≥ F^O / n_max |
| `test_prefetch_gate` | first output waits for the prefetch | offering 4 (< prefetch_in 8) then `0` produces no output for several cycles; once the threshold is met the layer completes and still conserves (writeback == F^O) |
| `test_prefetch_exceeds_capacity_throws` | deadlock config is rejected | constructing with `input_buf_capacity` = 4 < `prefetch_in` = 8 throws `std::invalid_argument` |
| `test_pipeline_fill` | pipeline warm-up | no write-back within the first `l` cycles; outputs emerge after |
| `test_nmin_gate_bursts` | the n_min lower-bound gate | trickling 1 operand/cycle into a unit with n_min=4 holds issue while the estimate is 1,2,3, then bursts 4 on the fourth |
| `test_clamps` | per-cycle acceptance limits | a port width of 2 bounds input acceptance to 2; a B^W of 4 bounds weight acceptance to 4; a full buffer then accepts 0 (back-pressure) |
| `test_buffer_size_backpressure` | finite buffer + conservation | a small B^I/B^W under sustained oversupply never accepts beyond capacity, the supplier stalls (consumed < offered), and the layer still drains exactly F^I/F^W and writes F^O |
| `test_zero_capacity_is_unlimited` | the `0` sentinel | capacity/port `0` accepts the full offer |
| `test_starvation_recovery` | starvation is recoverable | zero supply stalls without completing; resuming supply completes the layer |

## `tests/model/test_latency_model_chain.cpp`

The unit under test is `LmChain`. Chain stages use a 1:1 ratio
(F^O = F^I) so operand counts flow through unchanged.

| Test | Property | How |
|------|----------|-----|
| `test_conservation_with_backpressure` | operands cross the chain exactly once | a downstream stage with a finite B^I (=2) back-pressures via the carry; every operand still reaches the final stage exactly once, `carry_total()` returns to 0 |
| `test_overlap_beats_sequential` | streaming overlaps stages | a chained two-stage run takes meaningfully fewer cycles than running stage B only after stage A fully finishes (the "chained < sequential" contract) |
| `test_done_reflects_all_stages` | whole-chain completion | `done()` is false until every stage is done, even when the whole tensor is offered up front and the carry holds the remainder |
| `test_multiple_layers` | reset switches layers cleanly | a 3-stage chain (conv→act→pool shape) reused across several layer sizes via `reset()` conserves and completes on each layer, with no operands stranded between layers — the models keep no state across resets |

## What is not tested here

- **Anchors** (end-to-end cycle counts — MIDAP 920819, NVDLA googlenet_conv2
  338800 / alexnet_conv5 146023) are validated at the system level, not in these
  unit tests. They exercise the tiled analytic path, which does not call the
  streaming `update()` loop, so the LM unit tests and the anchors cover disjoint
  code.
- **State classification** (`LmState`) has no assertions — it has no production
  consumer yet (see [`implementation.md`](implementation.md)); tests will be
  added with the profiling redesign.
- **Connectivity** (non-linear / output-as-weight packets) is out of scope for
  `LmChain`; it belongs to the controller (see [`chain.md`](chain.md)).
