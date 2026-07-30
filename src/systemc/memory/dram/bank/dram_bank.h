/**
 * @file dram_bank.h
 * @brief Tier `bank`: bank + row-buffer (line) locality over the engine.
 *
 * Over the shared bank-occupancy engine, each access decodes to (bank, row):
 * a read to the bank's open row is a hit (row_hit_cyc), otherwise a miss
 * (row_miss_cyc). Writes take write_latency_cyc. Every access leaves its row
 * open, so a following access to the same row hits. `banks` sets both the
 * decode fan-out and the parallelism (accepts up to `banks` in flight).
 */

#pragma once

#include <cstdint>
#include <vector>

#include "systemc/memory/dram/analytical/analytical_dram.h"
#include "systemc/memory/dram/bank/bank_address.h"

namespace flexnpu_sim {

class DramBank : public AnalyticalDram {
public:
    struct Config {
        uint32_t banks             = 8;
        uint32_t line_bytes        = 8192;   ///< row-buffer (row) size
        uint32_t row_hit_cyc       = 20;
        uint32_t row_miss_cyc      = 45;
        uint32_t write_latency_cyc = 30;
    };

    explicit DramBank(const Config& cfg);

    std::string name() const override { return "bank"; }
    void reset() override;

protected:
    uint64_t access_latency(uint64_t address, uint32_t size,
                            bool is_write) override;

private:
    Config cfg_;
    BankDecoder decoder_;
    std::vector<uint64_t> open_row_;   ///< per-bank open row (NONE = UINT64_MAX)
};

}  // namespace flexnpu_sim
