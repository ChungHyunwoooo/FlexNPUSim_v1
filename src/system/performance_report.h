/**
 * @file performance_report.h
 * @brief EDA-style per-run performance report (the -report output).
 *
 * Formats the NPU's per-layer records plus the run counters into a five-section
 * text report (configuration / performance / per-layer / bottleneck / output
 * stream). Pure formatting — no simulation state beyond what is passed in.
 */

#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "systemc/npu/npu_controller.h"  // NpuController::LayerPerfRecord
#include "common/flexnpu_config.h"        // config::FlexNpuSimConfig

namespace flexnpu_sim {

void write_performance_report(
    std::ostream& os,
    const std::vector<NpuController::LayerPerfRecord>& recs,
    const config::FlexNpuSimConfig& cfg,
    const std::string& design,
    const std::string& network,
    const std::string& dataflow,
    const std::string& dram_name,
    uint32_t l2_kb,
    uint32_t bus_bits,
    double   clock_ns,
    uint64_t cycles,
    uint64_t macs,
    uint64_t mem_rd_bytes,
    uint64_t mem_wr_bytes,
    uint64_t read_txns,
    uint64_t write_txns,
    bool     completed);

}  // namespace flexnpu_sim
