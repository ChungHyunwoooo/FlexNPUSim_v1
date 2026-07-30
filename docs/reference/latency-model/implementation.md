# Latency Model — Implementation

How the code in `src/model/latency/latency_model.{h,cpp}` realizes the model of
[`modeling.md`](modeling.md). The class is `LmModel`; the controller calls
`update()` once per simulated cycle.

## State

`LmModel` stores only counters — no fractional numbers, no tensor data:

| Member | Meaning |
|--------|---------|
| `cumulative_input_fetches_`, `cumulative_weight_fetches_` | operands accepted so far |
| `issued_outputs_` | passes issued so far (basis for derived consumption) |
| `generated_outputs_` | passes retired (observed) so far |
| `output_buf_level_` | occupancy of the pass-completion timing queue |
| `retire_backlog_` | passes that matured but could not enter a full timing queue |
| `pipeline_delay_queue_` | the `l`-deep issue→retire delay line |
| `cycle_count_` | cycles elapsed in this layer |

Input/weight **buffer levels are not stored** — they are derived on demand:

```
  consumed(op)  = floor(issued_outputs_ · F^op / F^O)      // op ∈ {input, weight}
  buffered(op)  = cumulative_fetches(op) − consumed(op)
```

Advancing `issued_outputs_` therefore debits the supply implicitly, at the exact
integer rate, with no separate bookkeeping.

## The `update()` cycle — 9 steps

`update(offered_input, offered_weight)` returns an `LmStepResult`
(`consumed_input`, `consumed_weight`, `outputs`, `writeback_outputs`, `state`).

1. **Accept.** `accept(offered, max_ops_per_cycle, buf_capacity, buffered())`
   returns `min(offered, port width, buffer headroom)` for input and weight; a
   `0` limit means unlimited. `cumulative_*_fetches_` advance by the accepted
   amount. The un-accepted remainder is neither stored nor dropped — the caller
   re-offers it (backpressure).

2. **Estimate + generation rule.** `estimate_outputs()` returns the issuable
   count from the cumulative-count relation (`min` over the input and weight
   sides, capped by `n_max`). `apply_generation_rule()` then gates it:
   - **prefetch gate** — if `prefetch_pending()`, issue `0` (still accumulating
     the initial prefetch);
   - **output backpressure** — cap by the timing-queue headroom
     (`output_buf_capacity − level − backlog`);
   - **tail drain** — once all operands have arrived, issue `min(remaining,
     n_max)` directly (conservation makes the supply floor unnecessary and
     avoids a last-event round-down deadlock);
   - **n_min gate** — otherwise issue only if the estimate reaches `n_min`, else
     hold.

3. **Record issue.** `issued_outputs_ += issue_outputs`. No explicit debit —
   consumption is derived (see State).

4. **Pipeline delay.** `issue_outputs` enters the `l`-deep
   `pipeline_delay_queue_`; the front (issued `l` cycles ago) pops as
   `pipeline_delayed_outputs`. This is the issue→retire phase shift; it models
   the unit's own latency, so a chain needs no extra inter-stage register.

5. **Enqueue completions.** `pipeline_delayed_outputs + retire_backlog_` try to
   enter the timing queue. If `output_buf_capacity` would overflow, the excess
   stays in `retire_backlog_` (never dropped — conservation).

6. **Service.** The timing queue drains up to `max_output_ops_per_cycle`
   (`writeback`), the count that leaves the LM this cycle.

7. **Retire.** `generated_outputs_ += cycle_outputs` (what entered the queue).

8. **Result.** Fill `outputs`, `writeback_outputs`.

9. **Classify.** `classify_state()` labels the cycle (see below).

`is_done()` is `generated_outputs_ >= F^O`.

## `accept()` — one clamp, two limits

```cpp
uint32_t accept(offered, max_ops_per_cycle, capacity = 0, buffered = 0) {
    if (max_ops_per_cycle) offered = min(offered, max_ops_per_cycle);   // port width
    if (capacity)          offered = min(offered, capacity > buffered ? capacity - buffered : 0);
    return offered;                                                     // buffer headroom
}
```

Input and weight each call it once (`Step 1`); the output service reuses it with
only the port limit (`Step 6`). `0` = unlimited for either limit. The function
returns *how much is accepted*; the caller owns the remainder.

## The prefetch gate

`prefetch_pending()` is the sole gate on the first issue:

```
  pending = (issued_outputs_ == 0)
            && ( (input  not all fetched && cumulative_input  < prefetch_in)
              || (weight  not all fetched && cumulative_weight < prefetch_wt) )
```

It stops applying once any pass has issued, and releases early once all operands
have arrived (guarding against a prefetch set larger than the supply). Both
`apply_generation_rule()` (to hold issue) and `classify_state()` (to label the
cycle `Prefetch`) call it, so gate and label always agree.

The constructor rejects a prefetch that a *finite* buffer can never hold
(`prefetch_in > input_buf_capacity`, or the weight equivalent) with
`std::invalid_argument` — that configuration would deadlock.

## Discrete states — `LmState`

`classify_state()` labels each cycle (checked in this order):

| State | Condition |
|-------|-----------|
| `Done` | `generated_outputs_ >= F^O` |
| `Prefetch` | `prefetch_pending()` — accumulating the initial prefetch, nothing issued yet |
| `PipelineFill` | within the first `l` cycles (warm-up) |
| `Starved` | no output this cycle (buffer empty or below one output's worth) |
| `Underflow` | `0 < matured < n_min` and input not all fetched (waiting for supply) |
| `Drain` | `0 < matured < n_min` and input all fetched (flushing the tail) |
| `Active` | output produced this cycle |

The classification is currently computed every cycle but has no production
consumer (the profiler that read it was removed); it is retained as the
foundation for the planned profiling redesign — a time-in-state histogram per
layer. One known coarseness: `Starved` conflates a truly empty buffer with a
buffer holding data below one output's worth.

## `reset()`

`reset(params)` zeroes every counter and re-initializes the `l`-deep delay line,
so one `LmModel` can replay successive layers. It does not re-run the
constructor's prefetch-vs-capacity check.
