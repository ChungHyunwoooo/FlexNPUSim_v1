/**
 * @file test_latency_model.cpp
 * @brief Contracts for the operand-arrival latency model — the paper's
 *        core abstraction: an RTL unit modeled purely by count ratios.
 *
 * Flow under test: fetched input/weight -> internal buffers (B^I/B^W)
 * -> issue/execute as queueing (l, n_min/n_max) -> write-back.
 *
 * Contracts:
 *  1. conservation & completion — with unlimited supply the model
 *     consumes exactly F^I and F^W, writes back exactly F^O, reaches Done
 *  2. compute bound — total cycles >= F^O / n_max
 *  3. prefetch gate — the model holds issue until the initial prefetch
 *     (prefetch_in/wt) worth of operands accumulates; a prefetch larger than
 *     a finite buffer is a config error
 *  4. pipeline fill — no write-back within the first l cycles
 *  5. n_min lower-bound gate — starved supply holds issue until the
 *     estimate reaches n_min (then bursts), except in tail drain
 *  6. rate/buffer limits — per-cycle acceptance respects the intake rate and
 *     buffer capacity (back-pressure to the supplier)
 *  7. starvation is recoverable — zero supply stalls, resume completes
 */

#include "model/latency/latency_model.h"

#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace flexnpu_sim;

static LmParams small_layer() {
    LmParams p;
    p.fetch_operand0  = 64;
    p.fetch_operand1 = 32;
    p.write_output   = 32;
    p.issue_latency        = 2;
    p.n_min = 1;
    p.n_max = 4;
    p.prefetch_in       = 8;
    p.prefetch_wt      = 4;
    return p;
}

static void test_conservation_and_bound() {
    // Acceptance is supplier-driven (the GB/controller offers each layer's
    // operands and no more) — the faithful supplier offers the remainder.
    LmModel lm(small_layer());
    uint64_t ci = 0, cw = 0, wb = 0, cycles = 0;
    while (!lm.is_done() && cycles < 1000) {
        auto r = lm.update(static_cast<uint32_t>(64 - ci),
                           static_cast<uint32_t>(32 - cw));
        ci += r.consumed_input;
        cw += r.consumed_weight;
        wb += r.writeback_outputs;
        ++cycles;
    }
    assert(lm.is_done());
    assert(ci == 64 && "model drains exactly F^I from the supplier");
    assert(cw == 32 && "model drains exactly F^W");
    assert(wb == 32 && "writes back exactly F^O");
    assert(cycles >= 32 / 4 && "compute bound: F^O / n_max");
}

static void test_prefetch_gate() {
    // The model holds issue until the initial prefetch (prefetch_in/wt) worth of
    // operands has accumulated — "a conv needs a kernel window before the first
    // output". Below the gate: no output; once met: it proceeds and completes.
    LmParams p = small_layer();       // prefetch_in=8, prefetch_wt=4, F^O=32
    LmModel lm(p);
    lm.update(4, 4);                  // cumulative input=4 (< prefetch_in 8)
    for (int c = 0; c < 5; ++c)
        assert(lm.update(0, 0).outputs == 0 && "held below the prefetch threshold");
    uint64_t wb = 0, guard = 0;
    while (!lm.is_done() && guard++ < 1000)
        wb += lm.update(64, 64).writeback_outputs;
    assert(lm.is_done() && wb == 32 && "issues and completes once prefetch is met");
}

static void test_prefetch_exceeds_capacity_throws() {
    // A finite input buffer that cannot hold the initial prefetch is a deadlock
    // config — the constructor must reject it.
    LmParams p = small_layer();       // prefetch_in = 8
    LmHwParams hw;
    hw.input_buf_capacity = 4;      // < prefetch_in → impossible
    bool threw = false;
    try { LmModel lm(p, hw); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw && "prefetch_in > input buffer capacity must error");
}

static void test_pipeline_fill() {
    LmModel lm(small_layer());   // l = 2
    uint32_t early_wb = 0;
    for (int c = 0; c < 2; ++c) early_wb += lm.update(64, 64).writeback_outputs;
    assert(early_wb == 0 && "no write-back inside the first l cycles");
    uint32_t later = 0;
    for (int c = 0; c < 4; ++c) later += lm.update(64, 64).writeback_outputs;
    assert(later > 0 && "outputs emerge after pipeline fill");
}

static void test_nmin_gate_bursts() {
    LmParams p;
    p.fetch_operand0  = 32;
    p.fetch_operand1 = 0;      // single-operand unit
    p.write_output   = 32;     // ratio 1:1
    p.issue_latency        = 1;
    p.n_min = 4;      // gate
    p.n_max = 8;
    p.prefetch_in       = 1;
    p.prefetch_wt      = 0;
    LmModel lm(p);

    // Trickle 1 input/cycle: estimates 1,2,3 hold; the 4th cycle bursts 4.
    uint32_t total_out = 0;
    for (int c = 0; c < 3; ++c) {
        auto r = lm.update(1, 0);
        total_out += r.outputs;
        assert(r.outputs == 0 && "estimate < n_min holds issue");
    }
    lm.update(1, 0);                 // 4th operand: estimate hits n_min, issues
    auto r5 = lm.update(0, 0);       // outputs are observed at retire (l=1 later)
    assert(r5.outputs == 4 && "accumulated estimate bursts at n_min");
}

static void test_clamps() {
    LmParams p = small_layer();
    p.n_min = 100;  // gate issue so the buffer stays full
    LmHwParams hw;
    hw.max_input_ops_per_cycle = 2;   // input port operands/cycle clamp
    hw.weight_buf_capacity = 4;  // B^W clamp
    LmModel lm(p, hw);

    auto r = lm.update(64, 64);
    assert(r.consumed_input == 2 && "port width bounds acceptance");
    assert(r.consumed_weight == 4 && "B^W bounds acceptance");
    // Nothing issued (estimate < n_min): the full buffer back-pressures.
    auto r2 = lm.update(0, 64);
    assert(r2.consumed_weight == 0 && "full buffer back-pressures supply");
}

static void test_buffer_size_backpressure() {
    // Finite PE input/weight buffers: sustained oversupply must back-pressure.
    // The model never accepts beyond buffer headroom (level <= capacity), the
    // unaccepted remainder waits at the supplier, and the layer still drains
    // exactly F^I/F^W and completes — conservation holds under backpressure.
    LmParams p = small_layer();          // F^I=64, F^W=32, F^O=32, n_max=4
    LmHwParams hw;
    hw.input_buf_capacity  = 8;             // small PE input buffer (operands)
    hw.weight_buf_capacity = 8;             // small PE weight buffer
    LmModel lm(p, hw);

    uint64_t ci = 0, cw = 0, wb = 0, cycles = 0;
    bool backpressure_seen = false;
    while (!lm.is_done() && cycles < 2000) {
        const uint32_t offer_i = static_cast<uint32_t>(64 - ci);
        const uint32_t offer_w = static_cast<uint32_t>(32 - cw);
        auto r = lm.update(offer_i, offer_w);
        assert(r.consumed_input  <= 8 && "input buffer bounds per-cycle intake");
        assert(r.consumed_weight <= 8 && "weight buffer bounds per-cycle intake");
        assert(lm.get_buffered_input()  <= 8.0 + 1e-9 && "B^I <= capacity");
        assert(lm.get_buffered_weight() <= 8.0 + 1e-9 && "B^W <= capacity");
        if (r.consumed_input < offer_i) backpressure_seen = true;  // supplier stalled
        ci += r.consumed_input;
        cw += r.consumed_weight;
        wb += r.writeback_outputs;
        ++cycles;
    }
    assert(lm.is_done());
    assert(backpressure_seen && "oversupply into a small buffer stalls the supplier");
    assert(ci == 64 && cw == 32 && "drains exactly F^I/F^W under backpressure");
    assert(wb == 32 && "writes back exactly F^O under backpressure");
}

static void test_zero_capacity_is_unlimited() {
    // size = 0 is the "unlimited" sentinel: no clamp, the full offer is taken.
    LmParams p = small_layer();
    LmHwParams hw;                        // all capacities/ports = 0
    LmModel lm(p, hw);
    auto r = lm.update(64, 32);
    assert(r.consumed_input  == 64 && "capacity 0 accepts the full input offer");
    assert(r.consumed_weight == 32 && "capacity 0 accepts the full weight offer");
}

static void test_starvation_recovery() {
    LmModel lm(small_layer());
    for (int c = 0; c < 4; ++c) lm.update(8, 8);     // warm up
    for (int c = 0; c < 5; ++c) lm.update(0, 0);     // starve
    assert(!lm.is_done());
    uint64_t wb = 0, guard = 0;
    while (!lm.is_done() && guard++ < 1000)
        wb += lm.update(64, 64).writeback_outputs;
    assert(lm.is_done() && "resume after starvation completes the layer");
}

int main() {
    test_conservation_and_bound();
    test_prefetch_gate();
    test_prefetch_exceeds_capacity_throws();
    test_pipeline_fill();
    test_nmin_gate_bursts();
    test_clamps();
    test_buffer_size_backpressure();
    test_zero_capacity_is_unlimited();
    test_starvation_recovery();
    std::cout << "test_latency_model: all contracts hold\n";
    return 0;
}
