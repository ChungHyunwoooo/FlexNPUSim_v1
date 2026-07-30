#include "systemc/memory/dram/factory.h"

#include <stdexcept>

#include "common/flexnpu_config.h"
#include "systemc/memory/dram/ideal/dram_ideal.h"
#include "systemc/memory/dram/bandwidth/dram_bandwidth.h"
#include "systemc/memory/dram/bank/dram_bank.h"
#include "systemc/memory/dram/cycle/dram_cycle.h"

namespace flexnpu_sim {

std::unique_ptr<DramTiming> make_dram_timing(const config::DramConfig& d,
                                             double sys_clock_ns) {
    if (d.tier == "ideal")
        return std::make_unique<DramIdeal>(
            d.read_latency_cyc, d.write_latency_cyc, d.banks);

    if (d.tier == "bandwidth")
        return std::make_unique<DramBandwidth>(
            d.read_latency_cyc, d.write_latency_cyc,
            d.bandwidth_bytes_per_cycle, d.include_writes, d.banks);

    if (d.tier == "bank") {
        DramBank::Config c;
        c.banks             = d.banks;
        c.line_bytes        = d.line_bytes;
        c.row_hit_cyc       = d.row_hit_cyc;
        c.row_miss_cyc      = d.row_miss_cyc;
        c.write_latency_cyc = d.write_latency_cyc;
        return std::make_unique<DramBank>(c);
    }

    if (d.tier == "cycle")
        return std::make_unique<DramCycle>(d.ini, d.output_dir, sys_clock_ns);

    throw std::invalid_argument("Unknown DRAM tier: " + d.tier +
                                " (expected ideal | bandwidth | bank | cycle)");
}

}  // namespace flexnpu_sim
