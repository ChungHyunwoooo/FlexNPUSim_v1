/**
 * @file dram_bandwidth.h
 * @brief Tier `bandwidth`: base latency + byte-transfer time (MIDAP-style).
 *
 * Each access finishes base_latency + size / bandwidth_bytes_per_cycle cycles
 * later — the byte term models the DRAM-internal transfer time. `include_writes`
 * decides whether writes pay it (MIDAP INCLUDE_DRAM_WRITE=False -> false).
 * bandwidth 0 = unlimited (base latency only). `banks` sets the parallelism.
 */

#pragma once

#include "systemc/memory/dram/analytical/analytical_dram.h"

namespace flexnpu_sim {

class DramBandwidth : public AnalyticalDram {
public:
    DramBandwidth(uint32_t read_latency_cyc, uint32_t write_latency_cyc,
                  uint32_t bandwidth_bytes_per_cycle, bool include_writes,
                  uint32_t banks);

    std::string name() const override { return "bandwidth"; }

protected:
    uint64_t access_latency(uint64_t address, uint32_t size,
                            bool is_write) override;

private:
    uint32_t read_cyc_;
    uint32_t write_cyc_;
    uint32_t bw_bpc_;          ///< bytes per cycle; 0 = unlimited
    bool     include_writes_;
};

}  // namespace flexnpu_sim
