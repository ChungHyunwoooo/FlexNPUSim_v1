/**
 * @file dram_ideal.h
 * @brief Tier `ideal`: fixed-latency memory over the bank-occupancy engine.
 *
 * Every access finishes a constant number of cycles later, independent of
 * address or size. A latency of 0 models an ideal, zero-cost memory. `banks`
 * sets how many accesses proceed in parallel (backpressure beyond that).
 */

#pragma once

#include "systemc/memory/dram/analytical/analytical_dram.h"

namespace flexnpu_sim {

class DramIdeal : public AnalyticalDram {
public:
    DramIdeal(uint32_t read_latency_cyc, uint32_t write_latency_cyc,
              uint32_t banks);

    std::string name() const override { return "ideal"; }

protected:
    uint64_t access_latency(uint64_t address, uint32_t size,
                            bool is_write) override;

private:
    uint32_t read_cyc_;
    uint32_t write_cyc_;
};

}  // namespace flexnpu_sim
