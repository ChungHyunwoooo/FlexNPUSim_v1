#include "systemc/memory/dram/bank/dram_bank.h"

#include <algorithm>
#include <limits>

namespace flexnpu_sim {

namespace {
constexpr uint64_t kNoOpenRow = std::numeric_limits<uint64_t>::max();
}

DramBank::DramBank(const Config& cfg)
    : AnalyticalDram(cfg.banks),
      cfg_(cfg),
      decoder_(cfg.banks, cfg.line_bytes),
      open_row_(cfg.banks ? cfg.banks : 1, kNoOpenRow) {}

void DramBank::reset() {
    AnalyticalDram::reset();
    std::fill(open_row_.begin(), open_row_.end(), kNoOpenRow);
}

uint64_t DramBank::access_latency(uint64_t address, uint32_t /*size*/,
                                  bool is_write) {
    const BankAddress ba = decoder_.decode(address);
    const bool hit = (open_row_[ba.bank] == ba.row);
    open_row_[ba.bank] = ba.row;   // the access leaves this row open

    if (is_write) return cfg_.write_latency_cyc;
    return hit ? cfg_.row_hit_cyc : cfg_.row_miss_cyc;
}

}  // namespace flexnpu_sim
