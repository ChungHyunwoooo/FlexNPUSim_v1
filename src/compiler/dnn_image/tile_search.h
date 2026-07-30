/**
 * @file tile_search.h
 * @brief Selectable output-tile search (compiler pass, Phase-D S1).
 *
 * Pure analytical function: given a layer's geometry + the HW budget, pick the
 * output tile so tiling — and the traffic derived from it — is decided ONCE by
 * the compiler instead of being re-guessed at runtime by divergent paths.
 *
 * The *how* is a selectable strategy (`TilingStrategy`):
 *   - Dataflow  : reproduces what real NPUs do — the resident operand (chosen
 *                 by dataflow: WS→weights, OS→output, IS→input) is held in its
 *                 buffer, and the tile is sized to that buffer. This is the
 *                 production/faithful path. See docs/research/real-npu-tiling-algorithms.md.
 *   - MinTiles  : research objective — fewest tiles subject to WS ≤ GB.
 *   - MinTraffic / MinLatency : research objectives (not yet implemented; see
 *                 docs/research/tile-search-objectives.md).
 *
 * No SystemC / runtime dependency: testable in isolation.
 */

#pragma once

#include <cstdint>
#include <string>

#include "common/types.h"  // Dataflow

namespace flexnpu_sim::compiler {

/// How the output tile is chosen. Default = faithful real-NPU blocking.
enum class TilingStrategy {
    Dataflow,    ///< real-NPU dataflow blocking (production, default)
    MinTiles,    ///< research: fewest tiles s.t. WS <= GB
    MinTraffic,  ///< research: least DRAM traffic (not yet implemented)
    MinLatency,  ///< research: least modeled cycles (not yet implemented)
};

/// Layer geometry + HW budget for one tiling decision. All dims in elements.
struct TileSearchInput {
    uint32_t out_h       = 1;   ///< output height
    uint32_t out_w       = 1;   ///< output width
    uint32_t out_c       = 1;   ///< output channels (K)
    uint32_t in_c        = 1;   ///< input (reduction) channels (C)
    uint32_t kernel_h    = 1;
    uint32_t kernel_w    = 1;
    uint32_t stride      = 1;
    uint32_t elem_bytes  = 1;   ///< bytes per operand (INT8=1, INT16=2, ...)
    Dataflow dataflow    = Dataflow::WS;

    /// Budget the RESIDENT operand must fit (Dataflow strategy): WS→weight
    /// buffer, OS→accumulator, IS→feature buffer. 0 → fall back to gb_bytes.
    uint64_t resident_cap_bytes = 0;
    /// General/streaming budget; the MinTiles working-set constraint. 0 =
    /// unbounded (whole layer, single tile).
    uint64_t gb_bytes    = 0;

    /// Partial-sum home (Dataflow strategy): true = resident accumulator
    /// (NVDLA/TPU/Gemmini/MIDAP/NullHop/Simba — a C-split re-accumulates for
    /// free); false = spill-and-refeed to a shared buffer (MAERI/Eyeriss — a
    /// C-split costs a buffer round-trip).
    bool     psum_resident = true;
    uint32_t double_buffer = 2; ///< streamed-operand double-buffer factor (>=1)
};

/// Chosen output tile + verification/derivation fields.
struct TileSearchResult {
    uint32_t t_h = 1;
    uint32_t t_w = 1;
    uint32_t t_c = 1;
    uint64_t working_set_bytes = 0;  ///< resident footprint of the chosen tile
    uint32_t num_tiles   = 1;        ///< ceil-product over the output tile axes
    uint32_t num_c_passes = 1;       ///< reduction (C) split into this many passes
    bool     psum_spill  = false;    ///< a C-split forces a partial-sum round-trip
    bool     fits = true;            ///< false iff even the minimal tile overflows
};

/// Working set (bytes) of one output tile under the MinTiles model: resident
/// output tile + double-buffered halo-aware input footprint + the tile weights.
uint64_t tile_working_set_bytes(const TileSearchInput& in,
                                uint32_t t_h, uint32_t t_w, uint32_t t_c);

/// Pick the output tile with the given strategy (default = Dataflow).
TileSearchResult search_output_tile(const TileSearchInput& in,
                                    TilingStrategy strategy = TilingStrategy::Dataflow);

// --- Name registry (config <-> enum) --------------------------------------
// SINGLE source of the string names. To add a strategy: add the enum value,
// one row in kStrategyNames (tile_search.cpp), and its case in the dispatch.
const char* to_string(TilingStrategy s);
/// Resolve a config string to a strategy. Returns false on an unknown name.
bool tiling_strategy_from_string(const std::string& name, TilingStrategy& out);

}  // namespace flexnpu_sim::compiler
