/**
 * @file layer_timing_strategy.cpp
 * @brief Factory: pick the layer memory-timing strategy from config.
 */

#include "systemc/memory/model/layer_timing_strategy.h"
#include "common/flexnpu_config.h"

#include <algorithm>

namespace flexnpu_sim {

std::unique_ptr<LayerTimingStrategy> make_layer_timing_strategy(
    const config::DramConfig& dram, uint32_t axi_bus_bytes_per_cycle) {
    if (dram.tiled_dma_timing == "peak_bw") {
        auto s = std::make_unique<PeakBwTiming>();
        s->model.bus_bytes_per_cycle = dram.bandwidth_bytes_per_cycle
            ? dram.bandwidth_bytes_per_cycle
            : std::max<uint32_t>(1u, axi_bus_bytes_per_cycle);
        s->model.read_latency_cyc = dram.read_latency_cyc;
        s->model.include_writes = dram.include_writes;
        return s;
    }
    return std::make_unique<SimDmaTiming>();
}

}  // namespace flexnpu_sim
