/**
 * @file test_tile_descriptor.cpp
 * @brief Contract tests for common/tile_descriptor.{h,cpp}.
 *
 * Contracts under test (from the header's documented guarantees):
 *  1. decomposition — a layer that fits the GB is one tile / one pass;
 *     an oversized layer splits, and the tiles exactly cover the output
 *  2. conservation — lowering a schedule reproduces the layer totals:
 *     WS fetches the weight tensor exactly once, IS the input tensor
 *     exactly once, and the output operand sum is dataflow-invariant
 *  3. FetchTraffic arithmetic — reuse penalty formula, no penalty when
 *     parallel passes cover the reduction
 *  4. residency semantics — the Dataflow enum overload is exactly the
 *     OperandResidency mapping (WS={weight}, IS={input}, OS={})
 */

#include "common/tile_descriptor.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>

using namespace flexnpu_sim;

namespace {

TileSchedule small_schedule() {
    // 8x8x16 output, 16 input channels, 3x3 kernel — comfortably in 512KB.
    return compute_tile_schedule("small", 0, /*oh=*/8, /*ow=*/8, /*oc=*/16,
                                 /*ic=*/16, /*kh=*/3, /*kw=*/3, /*stride=*/1,
                                 /*pad=*/1, /*l2_kb=*/512, /*pe_rows=*/16,
                                 /*pe_cols=*/16);
}

TileSchedule big_schedule() {
    // 56x56x256 output, 512 input channels, 3x3 kernel vs a 64KB buffer.
    return compute_tile_schedule("big", 1, 56, 56, 256, 512, 3, 3, 1, 1,
                                 /*l2_kb=*/64, 16, 16);
}

uint64_t sum_in(const std::vector<TileTxn>& txns) {
    uint64_t s = 0;
    for (const auto& t : txns) s += t.input_fetch_operands;
    return s;
}
uint64_t sum_wt(const std::vector<TileTxn>& txns) {
    uint64_t s = 0;
    for (const auto& t : txns) s += t.weight_fetch_operands;
    return s;
}
uint64_t sum_out(const std::vector<TileTxn>& txns) {
    uint64_t s = 0;
    for (const auto& t : txns) s += t.output_write_operands;
    return s;
}

}  // namespace

static void test_decomposition() {
    auto small = small_schedule();
    assert(small.num_tiles == 1);
    assert(small.tiles.size() == 1);
    assert(small.tiles[0].num_passes == 1);

    auto big = big_schedule();
    assert(big.num_tiles > 1);
    // Tiles exactly cover the output plane (channels are full per tile).
    uint64_t area = 0;
    for (const auto& t : big.tiles) {
        area += static_cast<uint64_t>(t.output_height) * t.output_width;
        assert(t.output_channels == 256);
    }
    assert(area == 56ull * 56ull);

    // Pass slicing fires when the tile output fits its budget but the
    // per-channel input+kernel footprint exceeds the input/kernel budget:
    // a spatially small but channel-deep layer.
    auto deep = compute_tile_schedule("deep", 2, /*oh=*/4, /*ow=*/4, /*oc=*/16,
                                      /*ic=*/512, 3, 3, 1, 1, /*l2_kb=*/8,
                                      16, 16);
    bool multi_pass = false;
    for (const auto& t : deep.tiles)
        if (t.num_passes > 1) multi_pass = true;
    assert(multi_pass && "channel-deep layer vs 8KB must slice passes");
    // Documented limitation (tile_descriptor.cpp): when a tile's output
    // exceeds the output budget, splitting is not implemented and the tile
    // falls back to a single pass — pinned here so a fix shows up as a diff.
    bool fallback_single = true;
    for (const auto& t : big.tiles)
        if (t.num_passes != 1) fallback_single = false;
    assert(fallback_single);
}

static void test_lowering_conservation() {
    auto sched = big_schedule();
    const uint32_t in_total = 58u * 58u * 512u;        // padded input tensor
    const uint32_t wt_total = 3u * 3u * 512u * 256u;   // full weight tensor
    const uint32_t elem = 1;

    auto ws = lower_tile_schedule(sched, Dataflow::WS, in_total, wt_total, elem);
    auto is = lower_tile_schedule(sched, Dataflow::IS, in_total, wt_total, elem);
    auto os = lower_tile_schedule(sched, Dataflow::OS, in_total, wt_total, elem);
    assert(ws.size() == sched.num_tiles);

    // Resident operand is streamed exactly once (on the first tile).
    assert(sum_wt(ws) == wt_total);
    assert(ws[0].weight_fetch_operands == wt_total);
    for (size_t i = 1; i < ws.size(); ++i) assert(ws[i].weight_fetch_operands == 0);
    assert(sum_in(is) == in_total);

    // Output operand total is dataflow-invariant.
    const uint64_t out_ws = sum_out(ws);
    assert(out_ws == sum_out(is) && out_ws == sum_out(os));

    // OS refetches the weight per tile: strictly more than the WS single load.
    assert(sum_wt(os) > sum_wt(ws));
}

static void test_fetch_traffic_arithmetic() {
    FetchTraffic ft;
    ft.initial_input_bytes = 1000;
    ft.initial_weight_bytes = 500;
    ft.output_bytes = 100;
    ft.num_passes = 4;
    ft.pegs_per_pes = 1;
    ft.input_reuse_factor = 1.0;   // refetch across passes
    ft.weight_reuse_factor = 0.0;  // reused across passes

    assert(ft.compute_input_fetch() == 1000 + 100 * (4 - 1));
    assert(ft.compute_weight_fetch() == 500);
    assert(ft.compute_total_traffic() ==
           ft.compute_input_fetch() + ft.compute_weight_fetch() + 100);

    // Parallel passes covering the reduction -> no reload penalty.
    ft.pegs_per_pes = 4;
    assert(ft.compute_input_fetch() == 1000);
}

static void test_residency_equivalence() {
    auto sched = big_schedule();
    const uint32_t in_total = 58u * 58u * 512u;
    const uint32_t wt_total = 3u * 3u * 512u * 256u;

    const OperandResidency ws_res{/*input*/ false, /*weight*/ true};
    const OperandResidency is_res{true, false};
    const OperandResidency os_res{false, false};

    auto pairs = {
        std::make_pair(Dataflow::WS, ws_res),
        std::make_pair(Dataflow::IS, is_res),
        std::make_pair(Dataflow::OS, os_res),
    };
    for (const auto& [df, res] : pairs) {
        auto a = lower_tile_schedule(sched, df, in_total, wt_total, 1);
        auto b = lower_tile_schedule(sched, res, in_total, wt_total, 1);
        assert(a.size() == b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            assert(a[i].input_fetch_operands == b[i].input_fetch_operands);
            assert(a[i].weight_fetch_operands == b[i].weight_fetch_operands);
            assert(a[i].output_write_operands == b[i].output_write_operands);
        }
    }
}

int main() {
    test_decomposition();
    test_lowering_conservation();
    test_fetch_traffic_arithmetic();
    test_residency_equivalence();
    std::cout << "test_tile_descriptor: all contracts hold\n";
    return 0;
}
