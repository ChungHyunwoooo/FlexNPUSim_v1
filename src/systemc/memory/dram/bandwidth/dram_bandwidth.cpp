#include "systemc/memory/dram/bandwidth/dram_bandwidth.h"

namespace flexnpu_sim {

DramBandwidth::DramBandwidth(uint32_t read_latency_cyc, uint32_t write_latency_cyc,
                             uint32_t bandwidth_bytes_per_cycle, bool include_writes,
                             uint32_t banks)
    : AnalyticalDram(banks),
      read_cyc_(read_latency_cyc), write_cyc_(write_latency_cyc),
      bw_bpc_(bandwidth_bytes_per_cycle), include_writes_(include_writes) {}

uint64_t DramBandwidth::access_latency(uint64_t /*address*/, uint32_t size,
                                       bool is_write) {
    if (is_write)
        return write_cyc_ + (include_writes_ && bw_bpc_ ? size / bw_bpc_ : 0);
    return read_cyc_ + (bw_bpc_ ? size / bw_bpc_ : 0);
}

}  // namespace flexnpu_sim
