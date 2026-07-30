/**
 * @file test_tile_search.cpp
 * @brief Contract tests for the selectable output-tile search (Phase-D S1).
 *
 * MinTiles: chosen tile fits, GB-monotone, whole-iff-fits.
 * Dataflow: each dataflow caps a DIFFERENT operand (the crux) — OS→accumulator
 * (output tile), WS→weight buffer (t_c + C-split), IS→feature buffer (input
 * tile) — and psum_home decides whether a C-split spills.
 */

#include "compiler/dnn_image/tile_search.h"

#include <cassert>
#include <iostream>

using namespace flexnpu_sim::compiler;
using flexnpu_sim::Dataflow;

static TileSearchInput base_layer() {
    TileSearchInput in;
    in.out_h = 56; in.out_w = 56; in.out_c = 64;
    in.in_c = 64;
    in.kernel_h = 3; in.kernel_w = 3;
    in.stride = 1;
    in.elem_bytes = 2;
    in.double_buffer = 2;
    return in;
}

// ---- MinTiles strategy ----------------------------------------------------
static void test_min_tiles() {
    auto mk = [](uint64_t gb) { auto in = base_layer(); in.gb_bytes = gb; return in; };
    const auto probe = mk(0);
    const uint64_t ws_whole =
        tile_working_set_bytes(probe, probe.out_h, probe.out_w, probe.out_c);
    const uint64_t ws_min = tile_working_set_bytes(probe, 1, 1, 1);

    // unbounded → single tile
    auto whole = search_output_tile(mk(0), TilingStrategy::MinTiles);
    assert(whole.fits && whole.num_tiles == 1 &&
           whole.t_h == 56 && whole.t_w == 56 && whole.t_c == 64);

    // num_tiles == 1 iff whole WS fits
    assert(search_output_tile(mk(ws_whole),     TilingStrategy::MinTiles).num_tiles == 1);
    assert(search_output_tile(mk(ws_whole - 1), TilingStrategy::MinTiles).num_tiles >= 2);

    // fit invariant + monotonicity across a budget sweep
    uint32_t prev = 0xFFFFFFFFu;
    for (uint64_t gb = ws_min; gb <= ws_whole + ws_min; gb += 4096) {
        auto r = search_output_tile(mk(gb), TilingStrategy::MinTiles);
        if (!r.fits) continue;
        assert(r.working_set_bytes <= gb);
        assert(r.num_tiles <= prev && "MinTiles: num_tiles non-increasing in GB");
        prev = r.num_tiles;
    }
    assert(prev == 1);

    // degenerate budget → flagged, fully tiled
    auto tiny = search_output_tile(mk(1), TilingStrategy::MinTiles);
    assert(!tiny.fits && tiny.t_h == 1 && tiny.t_w == 1 && tiny.t_c == 1);
    assert(tiny.num_tiles == 56u * 56u * 64u);
}

// ---- Dataflow strategy: Output-Stationary ---------------------------------
static void test_dataflow_os() {
    auto mk = [](uint64_t accum) {
        auto in = base_layer(); in.dataflow = Dataflow::OS;
        in.resident_cap_bytes = accum; return in;
    };
    uint32_t prev = 0xFFFFFFFFu;
    for (uint64_t cap = 16384; cap <= 1u << 21; cap *= 2) {
        auto r = search_output_tile(mk(cap));  // default strategy = Dataflow
        assert(r.fits && r.working_set_bytes <= cap &&
               "OS: output tile must fit the accumulator");
        assert(r.num_c_passes == 1 && !r.psum_spill &&
               "OS accumulates full C in place, never spills");
        assert(r.num_tiles <= prev && "OS: num_tiles non-increasing in accumulator");
        prev = r.num_tiles;
    }
    assert(prev == 1 && "large accumulator → single output tile");
}

// ---- Dataflow strategy: Weight-Stationary + C-split / psum_home ------------
static void test_dataflow_ws() {
    // Wide reduction so one output channel's full-C weights overflow the buffer.
    auto in = base_layer();
    in.dataflow = Dataflow::WS;
    in.in_c = 512;                       // 3*3*512*2 = 9216 B for one output ch
    in.resident_cap_bytes = 4096;        // < 9216 → must split C

    in.psum_resident = true;             // NVDLA/TPU/Gemmini: free re-accumulate
    auto res = search_output_tile(in);
    assert(res.num_c_passes == 3 && "ceil(9216/4096) = 3 reduction passes");
    assert(!res.psum_spill && "resident accumulator: C-split does not spill");
    assert(res.working_set_bytes <= in.resident_cap_bytes);

    in.psum_resident = false;            // MAERI/Eyeriss: C-split round-trips
    auto spill = search_output_tile(in);
    assert(spill.num_c_passes == 3 && spill.psum_spill &&
           "spill-buffer psum: a C-split forces a round-trip");

    // Roomy weight buffer → no split, t_c grows, spatial stays whole.
    auto in2 = base_layer(); in2.dataflow = Dataflow::WS;
    in2.resident_cap_bytes = 1u << 20;
    auto big = search_output_tile(in2);
    assert(big.num_c_passes == 1 && big.t_h == 56 && big.t_w == 56 &&
           big.t_c >= 1 && big.working_set_bytes <= in2.resident_cap_bytes);
}

// ---- Dataflow strategy: Input-Stationary ----------------------------------
static void test_dataflow_is() {
    auto mk = [](uint64_t feat) {
        auto in = base_layer(); in.dataflow = Dataflow::IS;
        in.resident_cap_bytes = feat; return in;
    };
    uint32_t prev = 0xFFFFFFFFu;
    for (uint64_t cap = 32768; cap <= 1u << 22; cap *= 2) {
        auto r = search_output_tile(mk(cap));
        assert(r.t_c == 64 && "IS streams all output channels (t_c = OC)");
        if (r.fits) assert(r.working_set_bytes <= cap);
        assert(r.num_tiles <= prev && "IS: num_tiles non-increasing in feature buffer");
        prev = r.num_tiles;
    }
}

// ---- The crux: same layer + budget, different dataflow → different tile ----
static void test_dataflow_distinguishes() {
    const uint64_t cap = 65536;
    auto ws = base_layer(); ws.dataflow = Dataflow::WS; ws.resident_cap_bytes = cap;
    auto os = base_layer(); os.dataflow = Dataflow::OS; os.resident_cap_bytes = cap;
    auto rws = search_output_tile(ws);
    auto ros = search_output_tile(os);
    const bool differ = rws.t_h != ros.t_h || rws.t_w != ros.t_w || rws.t_c != ros.t_c;
    assert(differ && "WS and OS must tile the same layer differently");
}

// ---- Unbounded + unimplemented strategies ---------------------------------
static void test_edges() {
    auto in = base_layer();  // no caps → unbounded
    auto whole = search_output_tile(in, TilingStrategy::Dataflow);
    assert(whole.num_tiles == 1 && whole.t_h == 56 && whole.t_w == 56 && whole.t_c == 64);

    bool threw = false;
    try { (void)search_output_tile(in, TilingStrategy::MinTraffic); }
    catch (const std::exception&) { threw = true; }
    assert(threw && "MinTraffic is declared but not yet implemented");
}

// ---- Name registry (config <-> enum) --------------------------------------
static void test_registry() {
    TilingStrategy s = TilingStrategy::MinTiles;
    assert(tiling_strategy_from_string("dataflow", s) && s == TilingStrategy::Dataflow);
    assert(tiling_strategy_from_string("min_traffic", s) && s == TilingStrategy::MinTraffic);
    assert(!tiling_strategy_from_string("bogus", s) && "unknown name rejected");
    assert(std::string(to_string(TilingStrategy::MinTiles)) == "min_tiles");
    // round-trip every name
    for (const char* n : {"dataflow", "min_tiles", "min_traffic", "min_latency"}) {
        TilingStrategy t;
        assert(tiling_strategy_from_string(n, t));
        assert(std::string(to_string(t)) == n);
    }
}

int main() {
    test_min_tiles();
    test_dataflow_os();
    test_dataflow_ws();
    test_dataflow_is();
    test_dataflow_distinguishes();
    test_registry();
    test_edges();
    std::cout << "test_tile_search: all contracts hold\n";
    return 0;
}
