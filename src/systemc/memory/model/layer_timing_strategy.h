/**
 * @file layer_timing_strategy.h
 * @brief How a tiled layer's memory time is charged — selected by config.
 *
 * The config (dram.tiled_dma_timing) picks one implementation at construction;
 * the controller calls the uniform interface with no per-mode branch in the hot
 * path. This is the same polymorphic shape the DRAM tier factory uses.
 */

#pragma once

#include "systemc/memory/model/peak_bw_layer_model.h"

#include <cstdint>
#include <memory>

namespace flexnpu_sim {

namespace config { struct DramConfig; }

struct LayerTimingStrategy {
    virtual ~LayerTimingStrategy() = default;
    /// Whether per-transfer DMA advances sim time (false) or is suppressed so
    /// the layer memory time is charged analytically by mem_cycles() (true).
    virtual bool suppresses_dma() const = 0;
    /// Whole-layer memory cycles.
    virtual uint64_t mem_cycles(uint64_t rd_bytes, uint64_t wr_bytes,
                                uint64_t dma_elapsed_cyc) const = 0;
};

/// "sim": the DMA/AXI/DRAM transaction sim already advanced time; the memory
/// term is that elapsed time.
struct SimDmaTiming : LayerTimingStrategy {
    bool suppresses_dma() const override { return false; }
    uint64_t mem_cycles(uint64_t, uint64_t, uint64_t dma_elapsed) const override {
        return dma_elapsed;
    }
};

/// "peak_bw": suppress per-burst DMA timing; charge the layer's byte traffic at
/// the bus bandwidth plus one read latency (PeakBwLayerModel).
struct PeakBwTiming : LayerTimingStrategy {
    PeakBwLayerModel model;
    bool suppresses_dma() const override { return true; }
    uint64_t mem_cycles(uint64_t rd, uint64_t wr, uint64_t) const override {
        return model.layer_cycles(rd, wr);
    }
};

/// Factory: pick the strategy from config. Defined in the .cpp to keep the
/// DramConfig include out of this header.
std::unique_ptr<LayerTimingStrategy> make_layer_timing_strategy(
    const config::DramConfig& dram, uint32_t axi_bus_bytes_per_cycle);

}  // namespace flexnpu_sim
