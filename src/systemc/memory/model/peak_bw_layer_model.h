/**
 * @file peak_bw_layer_model.h
 * @brief Layer-aggregate memory-timing model for the "peak_bw" mode.
 *
 * The per-transaction tiers (DramTiming: ideal/bandwidth/bank/cycle) charge
 * each burst. "peak_bw" instead models a fully-pipelined streaming
 * memory analytically at the LAYER level: the layer's memory time is one read
 * latency plus its byte traffic delivered at the bus bandwidth (per-burst
 * latency hidden). It lives in the memory subsystem — not the controller — so
 * that changing memory behavior is a config/model change, not a compute-side
 * edit. All parameters come from the dram config block.
 */

#pragma once

#include <algorithm>
#include <cstdint>

namespace flexnpu_sim {

struct PeakBwLayerModel {
    uint32_t bus_bytes_per_cycle = 8;    ///< bandwidth: bus width / 8
    uint32_t read_latency_cyc    = 100;  ///< single latency term (per-burst hidden)
    bool     include_writes      = true; ///< false = reads only
                                         ///  (matches MIDAP INCLUDE_DRAM_WRITE=False)

    /// Whole-layer memory cycles from aggregate read/write byte counts.
    uint64_t layer_cycles(uint64_t rd_bytes, uint64_t wr_bytes) const {
        const uint64_t bytes = include_writes ? rd_bytes + wr_bytes : rd_bytes;
        const uint64_t bpc   = std::max<uint32_t>(1u, bus_bytes_per_cycle);
        return read_latency_cyc + bytes / bpc;
    }
};

}  // namespace flexnpu_sim
