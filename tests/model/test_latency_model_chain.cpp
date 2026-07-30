/**
 * @file test_latency_model_chain.cpp
 * @brief Contracts for the cycle-streaming latency-model chain.
 *
 * Contracts:
 *  1. conservation through the chain — the final stage writes back exactly
 *     its F^O, and no operands are lost between stages, even when a middle
 *     stage's B^I back-pressures (the carry holds them)
 *  2. overlap — streaming two chained stages costs meaningfully less than
 *     running them sequentially (completion overlap is the point of the
 *     formalization's chaining)
 *  3. done() reflects the whole chain
 */

#include "model/latency/latency_model_chain.h"

#include <cassert>
#include <iostream>

using namespace flexnpu_sim;

static LmParams stage_params(uint32_t f, uint32_t l) {
    LmParams p;
    p.fetch_operand0  = f;
    p.fetch_operand1 = 0;      // chain stages: input-driven units
    p.write_output   = f;      // 1:1 ratio
    p.issue_latency        = l;
    p.n_min = 1;
    p.n_max = 4;
    p.prefetch_in       = 1;
    return p;
}

static uint64_t run_chain(LmModel& a, LmModel& b,
                          uint32_t total, uint64_t* final_out = nullptr) {
    LmChain chain({&b});
    uint64_t fed = 0, out = 0, cycles = 0;
    while ((!a.is_done() || !chain.done()) && cycles < 10000) {
        const uint32_t offer = static_cast<uint32_t>(total - fed);
        auto ra = a.update(offer, 0);
        fed += ra.consumed_input;
        out += chain.step(ra.writeback_outputs, nullptr);
        ++cycles;
    }
    assert(chain.carry_total() == 0 && "chain drains completely");
    if (final_out) *final_out = out;
    return cycles;
}

static void test_conservation_with_backpressure() {
    const uint32_t F = 64;
    LmModel a(stage_params(F, 1), {}, "a");
    LmHwParams tight;
    tight.input_buf_capacity = 2;   // finite B^I on the downstream unit
    LmModel b(stage_params(F, 1), tight, "b");

    uint64_t out = 0;
    (void)run_chain(a, b, F, &out);
    assert(a.is_done() && b.is_done());
    assert(out == F && "every operand crosses the chain exactly once");
}

static void test_overlap_beats_sequential() {
    const uint32_t F = 64;
    // Chained run.
    LmModel a1(stage_params(F, 2), {}, "a1");
    LmModel b1(stage_params(F, 2), {}, "b1");
    const uint64_t chained = run_chain(a1, b1, F);

    // Sequential run: stage B starts only after stage A fully finishes.
    LmModel a2(stage_params(F, 2), {}, "a2");
    LmModel b2(stage_params(F, 2), {}, "b2");
    uint64_t fed = 0, buf = 0, cycles = 0;
    while (!a2.is_done() && cycles < 10000) {
        auto r = a2.update(static_cast<uint32_t>(F - fed), 0);
        fed += r.consumed_input;
        buf += r.writeback_outputs;
        ++cycles;
    }
    uint64_t fed_b = 0;
    while (!b2.is_done() && cycles < 10000) {
        auto r = b2.update(static_cast<uint32_t>(buf - fed_b), 0);
        fed_b += r.consumed_input;
        ++cycles;
    }
    const uint64_t sequential = cycles;

    assert(chained < sequential &&
           "streaming chain must overlap stage execution");
    // With 1:1 ratios and equal stages the overlap saves roughly one full
    // stage; demand at least a quarter of that to keep the assert robust.
    assert(chained * 4 <= sequential * 3 + 4);
}

static void test_done_reflects_all_stages() {
    LmModel a(stage_params(8, 1), {}, "a");
    LmModel b(stage_params(8, 1), {}, "b");
    LmChain chain({&a, &b});
    assert(!chain.done());
    uint64_t cycles = 0;
    uint64_t fed = 0;
    while (!chain.done() && cycles < 1000) {
        chain.step(static_cast<uint32_t>(8 - fed > 8 ? 8 : 8 - fed),
                   nullptr);
        fed = 8;  // whole tensor offered up front; carry holds the rest
        ++cycles;
    }
    assert(chain.done());
}

static void test_multiple_layers() {
    // A 3-stage chain (e.g. NVDLA conv -> act -> pool) reused across several
    // layers via reset(): each layer must conserve and complete on its own —
    // the models keep no state across resets, and the chain switches cleanly.
    LmModel s0(stage_params(1, 1), {}, "conv");
    LmModel s1(stage_params(1, 1), {}, "act");
    LmModel s2(stage_params(1, 2), {}, "pool");
    for (uint32_t F : {16u, 40u, 8u, 64u}) {
        s0.reset(stage_params(F, 1));
        s1.reset(stage_params(F, 1));
        s2.reset(stage_params(F, 2));
        LmChain chain({&s0, &s1, &s2});
        uint64_t out = 0, fed = 0, cycles = 0;
        while (!chain.done() && cycles < 10000) {
            const uint32_t offer = fed < F ? F - fed : 0;  // offered once; carry holds it
            out += chain.step(offer, nullptr);
            fed = F;
            ++cycles;
        }
        assert(chain.done() && "3-stage layer completes after reset");
        assert(chain.carry_total() == 0 && "no operands stranded between layers");
        assert(out == F && "conservation across the 3-stage chain, every layer");
    }
}

int main() {
    test_conservation_with_backpressure();
    test_overlap_beats_sequential();
    test_done_reflects_all_stages();
    test_multiple_layers();
    std::cout << "test_latency_model_chain: all contracts hold\n";
    return 0;
}
