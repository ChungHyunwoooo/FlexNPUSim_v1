#include "system/performance_report.h"

#include <algorithm>
#include <cstdio>
#include <iomanip>

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
    bool     completed)
{
    // Peak MAC/cycle = k (ops_per_pass) × n (output lanes) × g (parallel
    // passes). Omitting parallel_passes under-counts the peak for arrays that
    // run passes in parallel (e.g. Nullhop k1·n128·g1, or 16·8·g2=256),
    // pushing reported utilization above 100%.
    const uint64_t peak_mac =
        std::max<uint64_t>(1, static_cast<uint64_t>(cfg.hw.compute.ops_per_pass) *
                               cfg.hw.compute.num_output_lanes *
                               std::max(1u, cfg.hw.compute.parallel_passes));
    const double wall_ms = cycles * clock_ns / 1e6;
    const double util_all = cycles
        ? 100.0 * macs / (static_cast<double>(cycles) * peak_mac) : 0.0;
    const double gops = wall_ms > 0
        ? (2.0 * macs) / (cycles * clock_ns) : 0.0;  // GMAC*2/ns = GOPS
    const double bw_rd = cycles
        ? static_cast<double>(mem_rd_bytes) / (cycles * clock_ns) : 0.0;  // GB/s
    const double bw_wr = cycles
        ? static_cast<double>(mem_wr_bytes) / (cycles * clock_ns) : 0.0;  // GB/s
    const double inf_per_s = wall_ms > 0 ? 1000.0 / wall_ms : 0.0;  // inferences/s
    // With zero-skipping the array skips zero operands, so cycles reflect the
    // sparse execution while MACs stays the dense algorithmic count — util then
    // exceeds 100% and means dense-MAC speedup, NOT PE occupancy.
    const bool zero_skip = (cfg.hw.compute.zero_skipping == "on");
    // Peak data-bus bandwidth = one beat (bus_bits/8 B) per cycle. bus_util is
    // achieved (rd+wr) over this ceiling — the "how much of the bus did we use"
    // number that explains an achieved BW far below peak (unhidden per-beat
    // latency / poor pipelining), independent of the DRAM tier.
    const double peak_bus_bw = clock_ns > 0 ? (bus_bits / 8.0) / clock_ns : 0.0;  // GB/s
    const double bus_util = peak_bus_bw > 0
        ? 100.0 * (bw_rd + bw_wr) / peak_bus_bw : 0.0;

    os << "****************************************\n"
       << "Report : performance\n"
       << "Design : " << design << " / " << network << "\n"
       << "Tool   : flexnpusim\n"
       << "Status : " << (completed ? "COMPLETED" : "NOT COMPLETED") << "\n"
       << "****************************************\n\n";

    os << "1. Configuration Summary\n"
       << "------------------------\n"
       << "  dataflow            : " << dataflow << "\n"
       << "  PE (k x n_max)      : " << cfg.hw.compute.ops_per_pass
       << " x " << cfg.hw.compute.num_output_lanes
       << "  (peak " << peak_mac << " MAC/cycle, "
       << cfg.hw.compute.enable_gran << " enable)\n"
       << "  element size        : " << cfg.hw.compute.element_size_bytes << " B\n"
       << "  global buffer       : " << l2_kb << " KB"
       << " (partition " << cfg.hw.buffers.global.partition_mode
       << ", spill " << (cfg.hw.buffers.global.model_spill ? "on" : "off") << ")\n"
       << "  AXI                 : " << bus_bits << "b (max "
       << FLEXNPUSIM_DMA_AXI_DATA_W << "b), outstanding R/W "
       << cfg.axi.max_outstanding_reads << "/"
       << cfg.axi.max_outstanding_writes << "\n"
       << "  DRAM                : " << dram_name
       << " / tier=" << cfg.dram.tier << "\n"
       << "  clock               : " << clock_ns << " ns\n\n";

    os << "2. Performance Summary\n"
       << "----------------------\n"
       << "  total cycles        : " << cycles << "\n"
       << "  wall latency        : " << std::fixed << std::setprecision(3)
       << wall_ms << " ms\n"
       << "  inference rate      : " << std::setprecision(1) << inf_per_s << " inf/s\n"
       << "  MACs                : " << macs << "\n"
       << "  effective compute   : " << std::setprecision(2) << gops << " GOPS\n"
       << "  PE utilization      : " << util_all
       << (zero_skip ? " %  (dense-MAC basis; >100% = zero-skip speedup, not occupancy)\n"
                     : " %  (MACs/(cycles*peak))\n")
       << "  DRAM read traffic   : " << mem_rd_bytes << " B ("
       << mem_rd_bytes / 1.0e6 << " MB, " << bw_rd << " GB/s)\n"
       << "  DRAM write traffic  : " << mem_wr_bytes << " B ("
       << mem_wr_bytes / 1.0e6 << " MB, " << bw_wr << " GB/s)\n"
       << "  DRAM bus util       : " << bus_util << " %  ("
       << (bw_rd + bw_wr) << " of " << peak_bus_bw << " GB/s peak)\n"
       << "  DRAM transactions   : " << read_txns << " rd / " << write_txns << " wr"
       << "  (avg " << std::setprecision(0)
       << (read_txns ? mem_rd_bytes / static_cast<double>(read_txns) : 0.0) << " / "
       << (write_txns ? mem_wr_bytes / static_cast<double>(write_txns) : 0.0)
       << " B per xfer)\n" << std::setprecision(2) << "\n";

    os << "3. Per-Layer Breakdown\n"
       << "----------------------\n"
       << "  id  type     cycles      time%  util%   duty%  busU%  "
       << "cmp_cycles   mem_cycles   rd_bytes     wr_bytes    verdict\n";
    unsigned util_overflow = 0;
    for (const auto& r : recs) {
        const double t_pct = cycles
            ? 100.0 * r.total_cycles / cycles : 0.0;
        const double util = r.total_cycles
            ? 100.0 * r.macs / (static_cast<double>(r.total_cycles) * peak_mac) : 0.0;
        if (util > 100.5) util_overflow++;
        const double duty = (r.total_cycles && r.avg_output_ratio > 0)
            ? 100.0 * r.f_o / (r.avg_output_ratio * r.total_cycles) : 0.0;
        // Per-layer bus utilization: this layer's (rd+wr) bytes over its wall
        // time vs the one-beat-per-cycle peak — pinpoints the memory-heavy
        // layers (dwconv ~2%, FC/large-conv near the roofline knee).
        const double busu = (r.total_cycles && peak_bus_bw > 0)
            ? 100.0 * (r.mem_rd_bytes + r.mem_wr_bytes) /
                  (r.total_cycles * clock_ns) / peak_bus_bw : 0.0;
        // Roofline verdict by direct comparison of the two terms —
        // no threshold. Layer time = max(memory, compute); whichever
        // term set the max is the bound.
        const bool mem_bound = r.memory_cycles > r.compute_cycles;
        char line[224];
        std::snprintf(line, sizeof(line),
            "  %-3u %-8s %-11llu %5.1f  %5.1f  %6.1f  %5.1f  "
            "%-12llu %-12llu %-12llu %-11llu %s\n",
            r.layer_id, r.type.c_str(),
            static_cast<unsigned long long>(r.total_cycles),
            t_pct, util, duty, busu,
            static_cast<unsigned long long>(r.compute_cycles),
            static_cast<unsigned long long>(r.memory_cycles),
            static_cast<unsigned long long>(r.mem_rd_bytes),
            static_cast<unsigned long long>(r.mem_wr_bytes),
            mem_bound ? "memory-bound" : "compute-bound");
        os << line;
    }
    if (util_overflow && zero_skip)
        os << "\n  note: util% > 100 on " << util_overflow
           << " layer(s) is expected under zero-skipping — dense MACs over "
           << "sparse (zero-skipped) cycles = dense-MAC speedup, not a mapping "
           << "error.\n";
    else if (util_overflow)
        os << "\n  ** MISMATCH: util% > 100 on " << util_overflow
           << " layer(s) — the network CSV's per-layer mapping "
           << "(k/g/s) exceeds hw.compute's declared PE array. "
           << "Align hw.compute with the CSV mapping (or remove the "
           << "CSV k/g/s columns).\n";
    os << "\n";

    os << "4. Bottleneck Analysis\n"
       << "----------------------\n";
    uint64_t mem_cyc = 0, cmp_cyc = 0;
    for (const auto& r : recs)
        (r.memory_cycles > r.compute_cycles ? mem_cyc : cmp_cyc)
            += r.total_cycles;
    const uint64_t layer_sum = std::max<uint64_t>(1, mem_cyc + cmp_cyc);
    os << "  compute-bound time  : " << 100.0 * cmp_cyc / layer_sum << " %\n"
       << "  memory-bound time   : " << 100.0 * mem_cyc / layer_sum << " %\n"
       << "  top time consumers  :\n";
    auto sorted = recs;
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) {
                  return a.total_cycles > b.total_cycles; });
    for (size_t i = 0; i < sorted.size() && i < 3; ++i)
        os << "    #" << (i + 1) << "  layer " << sorted[i].layer_id
           << " (" << sorted[i].type << ")  "
           << sorted[i].total_cycles << " cycles  ("
           << std::setprecision(1)
           << 100.0 * sorted[i].total_cycles / cycles << "%)\n";
    os << "\n  note: latency = max(mem_cycles, cmp_cycles) per layer"
       << " (roofline); the verdict is whichever term set the max"
       << " — no threshold."
       << "\n        utilization = duty x issue-width x pass-width.\n";

    os << "\n5. Output Stream (partial output vs boundary write)\n"
       << "---------------------------------------------------\n"
       << "  id  F^O(pass cmpl)  psum_sram_wr  out_wr_bytes  psum_spill_wr  "
       << "psum_reload_rd\n";
    uint64_t tot_out = 0, tot_pwr = 0, tot_prd = 0;
    for (const auto& r : recs) {
        char line[160];
        std::snprintf(line, sizeof(line),
            "  %-3u %-15llu %-13llu %-13llu %-14llu %llu\n",
            r.layer_id,
            static_cast<unsigned long long>(r.f_o),
            static_cast<unsigned long long>(r.psum_sram_wr),
            static_cast<unsigned long long>(r.out_wr_bytes),
            static_cast<unsigned long long>(r.psum_wr_bytes),
            static_cast<unsigned long long>(r.psum_rd_bytes));
        os << line;
        tot_out += r.out_wr_bytes;
        tot_pwr += r.psum_wr_bytes;
        tot_prd += r.psum_rd_bytes;
    }
    os << "\n  totals: final output " << tot_out
       << " B, psum spill write " << tot_pwr
       << " B, psum reload " << tot_prd << " B\n"
       << "  note: F^O counts accumulator-side pass completions"
       << " (partial-output updates), NOT boundary writes; the"
       << "\n        boundary output-write stream = final stores"
       << " + spill (W^O). Spill appears only with model_spill"
       << " when a\n        tile's partials exceed the psum store"
       << " (output_location / accumulator capacity)."
       << "\n        psum_sram_wr = OFMAP-SRAM writes by psum residency:"
       << " resident psum (OS / accumulator / pe_local) writes"
       << "\n        the output once, a non-resident psum round-trips"
       << " it every pass (= F^O).\n";
}

}  // namespace flexnpu_sim
