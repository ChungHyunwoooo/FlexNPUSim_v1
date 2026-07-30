#include "systemc/memory/dram/ideal/dram_ideal.h"

namespace flexnpu_sim {

DramIdeal::DramIdeal(uint32_t read_latency_cyc, uint32_t write_latency_cyc,
                     uint32_t banks)
    : AnalyticalDram(banks),
      read_cyc_(read_latency_cyc), write_cyc_(write_latency_cyc) {}

uint64_t DramIdeal::access_latency(uint64_t /*address*/, uint32_t /*size*/,
                                   bool is_write) {
    return is_write ? write_cyc_ : read_cyc_;
}

}  // namespace flexnpu_sim
