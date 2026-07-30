/**
 * @file test_dma_model.cpp
 * @brief Burst-planning contracts for DmaModel (pure C++).
 *
 * The DMA engine executes whatever this planner emits, so the protocol-
 * legality rules live here:
 *  1. conservation — the plan's bursts cover exactly [addr, addr+size),
 *     contiguous, in order
 *  2. burst cap — no burst exceeds max_burst_bytes
 *  3. 4 KiB rule — no burst crosses a 4096-byte boundary (AXI page rule;
 *     the bus does not split, the planner must)
 */

#include "systemc/dma/model/dma_model.h"

#include <cassert>
#include <iostream>

using namespace flexnpu_sim;

static void check_plan(uint64_t addr, uint32_t size,
                       uint32_t bytes_per_beat, uint32_t max_beats) {
    DmaModel dma(DmaConfig(bytes_per_beat, max_beats));
    const auto plan = dma.submit_transfer(addr, /*dst=*/0, size, DmaMode::Read);
    const uint32_t max_burst_bytes = bytes_per_beat * max_beats;

    assert(!plan.bursts.empty());
    uint64_t cursor = addr;
    uint64_t total = 0;
    for (const auto& b : plan.bursts) {
        assert(b.address == cursor && "bursts must be contiguous, in order");
        assert(b.bytes > 0);
        assert(b.bytes <= max_burst_bytes && "burst cap");
        assert((b.address % 4096) + b.bytes <= 4096 && "4 KiB rule");
        cursor += b.bytes;
        total += b.bytes;
    }
    assert(total == size && "conservation");
    dma.complete_transfer();
}

int main() {
    // aligned, well within one page and one burst
    check_plan(0x1000, 256, 8, 64);
    // exactly one page starting on the boundary
    check_plan(0x2000, 4096, 8, 256);
    // unaligned start crossing several pages (the hard case)
    check_plan(0x1F40, 10000, 8, 256);
    // small max burst forces many splits
    check_plan(0x0FF8, 4096, 8, 2);
    // wide beats (64B) with an odd page offset
    check_plan(0x3FC0, 8192, 64, 16);

    std::cout << "test_dma_model: all contracts hold\n";
    return 0;
}
