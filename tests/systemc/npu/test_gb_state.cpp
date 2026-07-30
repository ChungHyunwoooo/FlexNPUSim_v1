/**
 * @file test_gb_state.cpp
 * @brief Contracts for the Global Buffer occupancy model (gb_state.h) —
 *        the structure that distinguishes NVDLA CBUF / MIDAP FMEM /
 *        Nullhop SRAM organizations.
 *
 * Contracts:
 *  1. region flow — feed clamps at effective capacity, consume clamps at
 *     availability
 *  2. sparse compression scales the EFFECTIVE capacity (Nullhop)
 *  3. Shared partition mode — input and weight contend for one physical
 *     pool (NVDLA CBUF); Fixed mode keeps regions independent
 *  4. psum placement — output occupies the pool only when it physically
 *     lives in the GB (Accumulator/PeLocal do not double-count)
 *  5. tile reset keeps the weight region (weights are reusable across
 *     tiles; the dataflow decides refetch, not the buffer)
 */

#include "systemc/npu/model/gb_state.h"

#include <cassert>
#include <iostream>

using namespace flexnpu_sim;

static void test_region_flow_clamps() {
    GbRegionState r;
    r.capacity = 100;
    assert(r.feed(150) == 100 && "feed clamps at capacity");
    assert(r.full());
    assert(r.consume(30) == 30);
    assert(r.available() == 70);
    assert(r.feed(50) == 30 && "freed slots reopen");
    assert(r.consume(1000) == 100 && "consume clamps at availability");
    assert(r.empty());
}

static void test_compression_scales_effective_capacity() {
    GbRegionState r;
    r.capacity = 100;
    r.compression = 2.0;                    // sparse feature maps
    assert(r.effective_capacity() == 200);
    assert(r.feed(500) == 200 && "compressed region holds 2x operands");
}

static void test_shared_pool_contention() {
    GbState gb;
    // 1 KB pool, 1B elements -> 1024 operands, one shared pool (CBUF).
    gb.configure(1, 1, 1, 1, 1, GbPartitionMode::Shared);
    gb.input.capacity = 1024;               // regions individually uncapped
    gb.weight.capacity = 1024;
    assert(gb.feed_input(800) == 800);
    assert(gb.feed_weight(800) == 224 && "weight is bounded by the shared pool");
    assert(gb.total_free() == 0);

    // Fixed mode: same feeds are independent.
    GbState fx;
    fx.configure(1, 1, 1, 1, 1, GbPartitionMode::Fixed);
    fx.input.capacity = 1024;
    fx.weight.capacity = 1024;
    assert(fx.feed_input(800) == 800);
    assert(fx.feed_weight(800) == 800 && "fixed regions do not contend");
}

static void test_psum_placement() {
    GbState gb;
    gb.configure(1, 1, 1, 1, 1, GbPartitionMode::Shared, GbOutputLocation::Gb);
    gb.output.capacity = 1024;
    gb.feed_output(400);
    assert(gb.total_used() == 400 && "GB-resident output counts");

    GbState acc;
    acc.configure(1, 1, 1, 1, 1, GbPartitionMode::Shared,
                  GbOutputLocation::Accumulator);
    acc.output.capacity = 1024;
    acc.feed_output(400);
    assert(acc.total_used() == 0 && "accumulator psum must not consume GB");
}

static void test_tile_reset_keeps_weight() {
    GbState gb;
    gb.configure(4, 1, 1, 1, 1);
    gb.input.capacity = gb.weight.capacity = gb.output.capacity = 1024;
    gb.feed_input(10);
    gb.feed_weight(20);
    gb.feed_output(30);
    gb.reset_for_tile();
    assert(gb.input.available() == 0 && gb.output.available() == 0);
    assert(gb.weight.available() == 20 && "weights persist across tiles");
    gb.reset_for_layer();
    assert(gb.weight.available() == 0);
}

int main() {
    test_region_flow_clamps();
    test_compression_scales_effective_capacity();
    test_shared_pool_contention();
    test_psum_placement();
    test_tile_reset_keeps_weight();
    std::cout << "test_gb_state: all contracts hold\n";
    return 0;
}
