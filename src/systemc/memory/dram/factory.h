/**
 * @file factory.h
 * @brief Create a DRAM timing model from config.
 */

#pragma once

#include <memory>

#include "systemc/memory/dram/dram_timing.h"

namespace flexnpu_sim {

namespace config { struct DramConfig; }

/// Build the DRAM model named by `cfg.tier`. `sys_clock_ns` is the NPU/sim clock
/// period so the internal tiers' cycle latencies pass through the AXI slave's
/// DRAM->sys conversion unchanged (the `cycle` tier reports the DRAM tCK instead).
std::unique_ptr<DramTiming> make_dram_timing(const config::DramConfig& cfg,
                                             double sys_clock_ns);

}  // namespace flexnpu_sim
