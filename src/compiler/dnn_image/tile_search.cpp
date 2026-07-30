/**
 * @file tile_search.cpp
 * @brief Selectable output-tile search — see tile_search.h.
 */

#include "compiler/dnn_image/tile_search.h"

#include <algorithm>
#include <stdexcept>

namespace flexnpu_sim::compiler {

namespace {

inline uint32_t ceil_div(uint32_t a, uint32_t b) { return b ? (a + b - 1) / b : a; }
inline uint64_t ceil_div64(uint64_t a, uint64_t b) { return b ? (a + b - 1) / b : a; }

// An output extent `t` reads `t*stride + kernel - 1` inputs (convolution halo).
inline uint64_t input_extent(uint32_t t, uint32_t stride, uint32_t kernel) {
    return static_cast<uint64_t>(t) * stride + (kernel > 0 ? kernel - 1 : 0);
}

// SINGLE registry of strategy names — the one place to edit when adding a
// strategy (plus its enum value and dispatch case).
struct StrategyName { const char* name; TilingStrategy strat; };
constexpr StrategyName kStrategyNames[] = {
    {"dataflow",    TilingStrategy::Dataflow},
    {"min_tiles",   TilingStrategy::MinTiles},
    {"min_traffic", TilingStrategy::MinTraffic},
    {"min_latency", TilingStrategy::MinLatency},
};

}  // namespace

const char* to_string(TilingStrategy s) {
    for (const auto& e : kStrategyNames) if (e.strat == s) return e.name;
    return "dataflow";
}

bool tiling_strategy_from_string(const std::string& name, TilingStrategy& out) {
    for (const auto& e : kStrategyNames)
        if (name == e.name) { out = e.strat; return true; }
    return false;
}

uint64_t tile_working_set_bytes(const TileSearchInput& in,
                                uint32_t t_h, uint32_t t_w, uint32_t t_c) {
    const uint64_t elem = in.elem_bytes ? in.elem_bytes : 1;
    const uint64_t db   = in.double_buffer ? in.double_buffer : 1;
    const uint64_t out_bytes = static_cast<uint64_t>(t_h) * t_w * t_c * elem;
    const uint64_t in_h = input_extent(t_h, in.stride, in.kernel_h);
    const uint64_t in_w = input_extent(t_w, in.stride, in.kernel_w);
    const uint64_t in_bytes = in_h * in_w * in.in_c * elem;
    const uint64_t wt_bytes =
        static_cast<uint64_t>(in.kernel_h) * in.kernel_w * in.in_c * t_c * elem;
    return out_bytes + db * in_bytes + wt_bytes;
}

// ---------------------------------------------------------------------------
// Strategy: MinTiles — fewest tiles subject to the full working set <= gb_bytes.
// (A larger budget grows the feasible set, so the minimum can only fall.)
// ---------------------------------------------------------------------------
static TileSearchResult search_min_tiles(const TileSearchInput& in) {
    const uint32_t OH = std::max(1u, in.out_h);
    const uint32_t OW = std::max(1u, in.out_w);
    const uint32_t OC = std::max(1u, in.out_c);
    TileSearchResult r;

    if (in.gb_bytes == 0) {  // unbounded → whole layer
        r.t_h = OH; r.t_w = OW; r.t_c = OC;
        r.working_set_bytes = tile_working_set_bytes(in, OH, OW, OC);
        return r;
    }

    uint32_t best_h = 0, best_w = 0, best_c = 0, best_nt = 0;
    uint64_t best_size = 0;
    for (uint32_t t_c = 1; t_c <= OC; ++t_c) {
        const uint32_t nc = ceil_div(OC, t_c);
        if (best_nt != 0 && nc >= best_nt) continue;  // channel tiles alone lose
        for (uint32_t t_w = 1; t_w <= OW; ++t_w) {
            const uint64_t ws1 = tile_working_set_bytes(in, 1, t_w, t_c);
            const uint64_t ws2 = tile_working_set_bytes(in, 2, t_w, t_c);
            const uint64_t A = (ws2 > ws1) ? (ws2 - ws1) : 1;  // per-t_h slope
            const uint64_t B = (ws1 > A) ? (ws1 - A) : 0;      // WS at t_h=0
            if (in.gb_bytes <= B) continue;
            const uint64_t max_h = (in.gb_bytes - B) / A;
            if (max_h < 1) continue;
            const uint32_t t_h = static_cast<uint32_t>(std::min<uint64_t>(OH, max_h));
            const uint32_t nt = ceil_div(OH, t_h) * ceil_div(OW, t_w) * nc;
            const uint64_t sz = static_cast<uint64_t>(t_h) * t_w * t_c;
            if (best_nt == 0 || nt < best_nt || (nt == best_nt && sz > best_size)) {
                best_nt = nt; best_size = sz; best_h = t_h; best_w = t_w; best_c = t_c;
            }
        }
    }
    if (best_nt > 0) {
        r.t_h = best_h; r.t_w = best_w; r.t_c = best_c;
        r.working_set_bytes = tile_working_set_bytes(in, best_h, best_w, best_c);
        r.num_tiles = best_nt;
        return r;
    }
    r.t_h = 1; r.t_w = 1; r.t_c = 1;
    r.working_set_bytes = tile_working_set_bytes(in, 1, 1, 1);
    r.num_tiles = OH * OW * OC;
    r.fits = false;
    return r;
}

// ---------------------------------------------------------------------------
// Strategy: Dataflow — real-NPU blocking. The resident operand (chosen by
// dataflow) is held in its buffer; the tile is sized to that buffer. See
// docs/research/real-npu-tiling-algorithms.md.
// ---------------------------------------------------------------------------
static TileSearchResult search_dataflow(const TileSearchInput& in) {
    const uint32_t OH = std::max(1u, in.out_h);
    const uint32_t OW = std::max(1u, in.out_w);
    const uint32_t OC = std::max(1u, in.out_c);
    const uint32_t C  = std::max(1u, in.in_c);
    const uint64_t elem = in.elem_bytes ? in.elem_bytes : 1;
    const uint64_t cap  = in.resident_cap_bytes ? in.resident_cap_bytes : in.gb_bytes;

    TileSearchResult r;
    if (cap == 0) {  // unbounded → whole layer resident
        r.t_h = OH; r.t_w = OW; r.t_c = OC;
        r.working_set_bytes = tile_working_set_bytes(in, OH, OW, OC);
        return r;
    }

    switch (in.dataflow) {
    case Dataflow::OS: {
        // Output tile resident in the accumulator: t_h*t_w*t_c*elem <= cap.
        // Greedy channel-major fill (fewest output tiles). C reduces in place.
        const uint64_t budget = cap / elem;
        const uint32_t t_c = static_cast<uint32_t>(
            std::max<uint64_t>(1, std::min<uint64_t>(OC, budget)));
        const uint32_t t_w = static_cast<uint32_t>(
            std::max<uint64_t>(1, std::min<uint64_t>(OW, budget / t_c)));
        const uint32_t t_h = static_cast<uint32_t>(
            std::max<uint64_t>(1, std::min<uint64_t>(
                OH, budget / (static_cast<uint64_t>(t_c) * t_w))));
        r.t_h = t_h; r.t_w = t_w; r.t_c = t_c;
        r.working_set_bytes = static_cast<uint64_t>(t_h) * t_w * t_c * elem;
        r.num_tiles = ceil_div(OH, t_h) * ceil_div(OW, t_w) * ceil_div(OC, t_c);
        r.num_c_passes = 1;           // full C accumulated in place (resident)
        r.psum_spill = false;         // output-stationary never spills psum
        r.fits = (r.working_set_bytes <= cap);
        break;
    }
    case Dataflow::WS: {
        // Weights [Kh*Kw*C*t_c] resident in the weight buffer. If even one
        // output channel's full-C weights overflow, split C into passes; then
        // size t_c to the budget. Spatial streams (kept whole in S1).
        const uint64_t wt_full_1c =
            static_cast<uint64_t>(in.kernel_h) * in.kernel_w * C * elem;
        uint32_t c_pass = 1;
        if (wt_full_1c > cap) c_pass = static_cast<uint32_t>(ceil_div64(wt_full_1c, cap));
        const uint64_t c_slice = ceil_div64(C, c_pass);
        const uint64_t wt_per_tc =
            static_cast<uint64_t>(in.kernel_h) * in.kernel_w * c_slice * elem;
        uint32_t t_c = wt_per_tc ? static_cast<uint32_t>(cap / wt_per_tc) : OC;
        t_c = std::max(1u, std::min(OC, t_c));
        r.t_h = OH; r.t_w = OW; r.t_c = t_c;
        r.working_set_bytes = wt_per_tc * t_c;  // resident weight footprint
        r.num_tiles = ceil_div(OC, t_c);
        r.num_c_passes = c_pass;
        r.psum_spill = (c_pass > 1) && !in.psum_resident;
        r.fits = (r.working_set_bytes <= cap);
        break;
    }
    case Dataflow::IS: {
        // Input tile resident in the feature buffer: halo footprint * C <= cap.
        // Maximize output spatial (→ input tile); stream all output channels.
        // For fixed t_w the footprint is affine in t_h → closed-form max t_h.
        uint32_t best_h = 0, best_w = 0, best_nt = 0;
        uint64_t best_area = 0;
        for (uint32_t t_w = 1; t_w <= OW; ++t_w) {
            auto foot = [&](uint32_t t_h) {
                return input_extent(t_h, in.stride, in.kernel_h) *
                       input_extent(t_w, in.stride, in.kernel_w) * C * elem;
            };
            const uint64_t f1 = foot(1), f2 = foot(2);
            const uint64_t A = (f2 > f1) ? (f2 - f1) : 1;
            const uint64_t B = (f1 > A) ? (f1 - A) : 0;
            if (cap <= B) continue;
            const uint64_t max_h = (cap - B) / A;
            if (max_h < 1) continue;
            const uint32_t t_h = static_cast<uint32_t>(std::min<uint64_t>(OH, max_h));
            const uint32_t nt = ceil_div(OH, t_h) * ceil_div(OW, t_w);
            const uint64_t area = static_cast<uint64_t>(t_h) * t_w;
            if (best_nt == 0 || nt < best_nt || (nt == best_nt && area > best_area)) {
                best_nt = nt; best_area = area; best_h = t_h; best_w = t_w;
            }
        }
        if (best_nt > 0) {
            r.t_h = best_h; r.t_w = best_w; r.t_c = OC;
            r.working_set_bytes =
                input_extent(best_h, in.stride, in.kernel_h) *
                input_extent(best_w, in.stride, in.kernel_w) * C * elem;
            r.num_tiles = best_nt;
            r.fits = true;
        } else {
            r.t_h = 1; r.t_w = 1; r.t_c = OC;
            r.working_set_bytes =
                input_extent(1, in.stride, in.kernel_h) *
                input_extent(1, in.stride, in.kernel_w) * C * elem;
            r.num_tiles = OH * OW;
            r.fits = false;
        }
        r.num_c_passes = 1;   // C consumed in one sweep
        r.psum_spill = false;
        break;
    }
    }
    return r;
}

TileSearchResult search_output_tile(const TileSearchInput& in,
                                    TilingStrategy strategy) {
    switch (strategy) {
    case TilingStrategy::Dataflow: return search_dataflow(in);
    case TilingStrategy::MinTiles: return search_min_tiles(in);
    case TilingStrategy::MinTraffic:
    case TilingStrategy::MinLatency:
        throw std::logic_error(
            "tile_search: MinTraffic/MinLatency strategy not yet implemented "
            "(see docs/research/tile-search-objectives.md)");
    }
    return search_dataflow(in);
}

}  // namespace flexnpu_sim::compiler
