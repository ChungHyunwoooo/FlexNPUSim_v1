#ifndef AXI_PROFILER_H
#define AXI_PROFILER_H

#include "systemc/bus/axi/axi_config.h"
#include "../axi_logger.h"
#include <systemc.h>
#include <map>
#include <fstream>
#include <iostream>

/**
 * AXI Protocol Profiler
 * Monitors: latency, bandwidth, contention, outstanding transactions
 */
template<typename CFG = AXI4_Default>
class AXIProfiler {
public:
    using addr_t = typename CFG::addr_t;
    using id_t = typename CFG::id_t;

    struct TransactionTimestamp {
        sc_time ar_start, r_end;
        sc_time aw_start, b_end;
        id_t transaction_id;
        addr_t address;
        unsigned int burst_length;
    };

    struct PerformanceCounters {
        uint64_t total_read_transactions = 0;
        uint64_t total_write_transactions = 0;
        uint64_t total_read_beats = 0;
        uint64_t total_write_beats = 0;

        sc_time total_read_latency;
        sc_time total_write_latency;
        sc_time max_read_latency;
        sc_time max_write_latency;
        sc_time min_read_latency = sc_max_time();
        sc_time min_write_latency = sc_max_time();

        uint64_t ar_contention_cycles = 0;
        uint64_t aw_contention_cycles = 0;
        uint64_t w_stall_cycles = 0;
        uint64_t r_stall_cycles = 0;
        uint64_t b_stall_cycles = 0;

        uint32_t max_outstanding_reads = 0;
        uint32_t max_outstanding_writes = 0;
        uint32_t current_outstanding_reads = 0;
        uint32_t current_outstanding_writes = 0;

        uint64_t total_read_bytes = 0;
        uint64_t total_write_bytes = 0;

        uint64_t resp_okay = 0;
        uint64_t resp_exokay = 0;
        uint64_t resp_slverr = 0;
        uint64_t resp_decerr = 0;
    };

    PerformanceCounters counters;
    std::map<id_t, TransactionTimestamp> active_reads;
    std::map<id_t, TransactionTimestamp> active_writes;

    void record_ar_handshake(id_t id, addr_t addr, unsigned int len) {
        TransactionTimestamp ts;
        ts.ar_start = sc_time_stamp();
        ts.transaction_id = id;
        ts.address = addr;
        ts.burst_length = len + 1;
        active_reads[id] = ts;

        counters.current_outstanding_reads++;
        if (counters.current_outstanding_reads > counters.max_outstanding_reads) {
            counters.max_outstanding_reads = counters.current_outstanding_reads;
        }
    }

    void record_r_complete(id_t id, sc_uint<2> resp) {
        if (active_reads.find(id) == active_reads.end()) return;

        auto& ts = active_reads[id];
        ts.r_end = sc_time_stamp();

        sc_time latency = ts.r_end - ts.ar_start;
        counters.total_read_latency += latency;
        if (latency > counters.max_read_latency) counters.max_read_latency = latency;
        if (latency < counters.min_read_latency) counters.min_read_latency = latency;

        counters.total_read_transactions++;
        counters.total_read_beats += ts.burst_length;
        counters.total_read_bytes += ts.burst_length * (CFG::DATA_W / 8);
        counters.current_outstanding_reads--;

        switch (resp.to_uint()) {
            case 0: counters.resp_okay++; break;
            case 1: counters.resp_exokay++; break;
            case 2: counters.resp_slverr++; break;
            case 3: counters.resp_decerr++; break;
        }

        active_reads.erase(id);
    }

    void record_aw_handshake(id_t id, addr_t addr, unsigned int len) {
        TransactionTimestamp ts;
        ts.aw_start = sc_time_stamp();
        ts.transaction_id = id;
        ts.address = addr;
        ts.burst_length = len + 1;
        active_writes[id] = ts;

        counters.current_outstanding_writes++;
        if (counters.current_outstanding_writes > counters.max_outstanding_writes) {
            counters.max_outstanding_writes = counters.current_outstanding_writes;
        }
    }

    void record_b_complete(id_t id, sc_uint<2> resp) {
        if (active_writes.find(id) == active_writes.end()) return;

        auto& ts = active_writes[id];
        ts.b_end = sc_time_stamp();

        sc_time latency = ts.b_end - ts.aw_start;
        counters.total_write_latency += latency;
        if (latency > counters.max_write_latency) counters.max_write_latency = latency;
        if (latency < counters.min_write_latency) counters.min_write_latency = latency;

        counters.total_write_transactions++;
        counters.total_write_beats += ts.burst_length;
        counters.total_write_bytes += ts.burst_length * (CFG::DATA_W / 8);
        counters.current_outstanding_writes--;

        switch (resp.to_uint()) {
            case 0: counters.resp_okay++; break;
            case 1: counters.resp_exokay++; break;
            case 2: counters.resp_slverr++; break;
            case 3: counters.resp_decerr++; break;
        }

        active_writes.erase(id);
    }

    void record_contention(bool arvalid, bool arready, bool awvalid, bool awready,
                          bool wvalid, bool wready, bool rvalid, bool rready,
                          bool bvalid, bool bready) {
        if (arvalid && !arready) counters.ar_contention_cycles++;
        if (awvalid && !awready) counters.aw_contention_cycles++;
        if (wvalid && !wready) counters.w_stall_cycles++;
        if (rvalid && !rready) counters.r_stall_cycles++;
        if (bvalid && !bready) counters.b_stall_cycles++;
    }

    void print_statistics() {
        AXI_LOG_INFO("");
        AXI_LOG_INFO("========== AXI Performance Statistics ==========");
        AXI_LOG_INFO("Simulation Time: {}", sc_time_stamp().to_string());
        AXI_LOG_INFO("");

        AXI_LOG_INFO("--- Transaction Counts ---");
        AXI_LOG_INFO("  Read Transactions:  {}", counters.total_read_transactions);
        AXI_LOG_INFO("  Write Transactions: {}", counters.total_write_transactions);
        AXI_LOG_INFO("  Read Beats:         {}", counters.total_read_beats);
        AXI_LOG_INFO("  Write Beats:        {}", counters.total_write_beats);
        AXI_LOG_INFO("");

        AXI_LOG_INFO("--- Latency Statistics ---");
        if (counters.total_read_transactions > 0) {
            sc_time avg_read = counters.total_read_latency / counters.total_read_transactions;
            AXI_LOG_INFO("  Avg Read Latency:  {}", avg_read.to_string());
            AXI_LOG_INFO("  Min Read Latency:  {}", counters.min_read_latency.to_string());
            AXI_LOG_INFO("  Max Read Latency:  {}", counters.max_read_latency.to_string());
        }
        if (counters.total_write_transactions > 0) {
            sc_time avg_write = counters.total_write_latency / counters.total_write_transactions;
            AXI_LOG_INFO("  Avg Write Latency: {}", avg_write.to_string());
            AXI_LOG_INFO("  Min Write Latency: {}", counters.min_write_latency.to_string());
            AXI_LOG_INFO("  Max Write Latency: {}", counters.max_write_latency.to_string());
        }
        AXI_LOG_INFO("");

        AXI_LOG_INFO("--- Bus Contention ---");
        AXI_LOG_INFO("  AR Contention Cycles: {}", counters.ar_contention_cycles);
        AXI_LOG_INFO("  AW Contention Cycles: {}", counters.aw_contention_cycles);
        AXI_LOG_INFO("  W Stall Cycles:       {}", counters.w_stall_cycles);
        AXI_LOG_INFO("  R Stall Cycles:       {}", counters.r_stall_cycles);
        AXI_LOG_INFO("  B Stall Cycles:       {}", counters.b_stall_cycles);
        AXI_LOG_INFO("");

        AXI_LOG_INFO("--- Outstanding Transactions ---");
        AXI_LOG_INFO("  Max Outstanding Reads:  {}", counters.max_outstanding_reads);
        AXI_LOG_INFO("  Max Outstanding Writes: {}", counters.max_outstanding_writes);
        AXI_LOG_INFO("");

        AXI_LOG_INFO("--- Bandwidth ---");
        sc_time sim_time = sc_time_stamp();
        if (sim_time > SC_ZERO_TIME) {
            double read_bw = (double)counters.total_read_bytes / sim_time.to_seconds();
            double write_bw = (double)counters.total_write_bytes / sim_time.to_seconds();
            AXI_LOG_INFO("  Read Bandwidth:  {:.2f} MB/s", read_bw / 1e6);
            AXI_LOG_INFO("  Write Bandwidth: {:.2f} MB/s", write_bw / 1e6);
            AXI_LOG_INFO("  Total Bandwidth: {:.2f} MB/s", (read_bw + write_bw) / 1e6);
        }
        AXI_LOG_INFO("");

        AXI_LOG_INFO("--- Response Statistics ---");
        AXI_LOG_INFO("  OKAY:    {}", counters.resp_okay);
        AXI_LOG_INFO("  EXOKAY:  {}", counters.resp_exokay);
        AXI_LOG_INFO("  SLVERR:  {}", counters.resp_slverr);
        AXI_LOG_INFO("  DECERR:  {}", counters.resp_decerr);
        AXI_LOG_INFO("================================================");
        AXI_LOG_INFO("");
    }

    void export_to_csv(const std::string& filename) {
        AXI_LOG_DEBUG("Exporting performance statistics to CSV: {}", filename);
        std::ofstream file(filename);
        if (!file.is_open()) {
            AXI_LOG_ERROR("Failed to open file for writing: {}", filename);
            return;
        }
        file << "Metric,Value\n";
        file << "total_read_transactions," << counters.total_read_transactions << "\n";
        file << "total_write_transactions," << counters.total_write_transactions << "\n";
        file << "max_outstanding_reads," << counters.max_outstanding_reads << "\n";
        file << "max_outstanding_writes," << counters.max_outstanding_writes << "\n";
        file << "ar_contention_cycles," << counters.ar_contention_cycles << "\n";
        file << "aw_contention_cycles," << counters.aw_contention_cycles << "\n";
        file << "w_stall_cycles," << counters.w_stall_cycles << "\n";
        file << "r_stall_cycles," << counters.r_stall_cycles << "\n";
        file << "b_stall_cycles," << counters.b_stall_cycles << "\n";
        file.close();
        AXI_LOG_INFO("Successfully exported performance statistics to {}", filename);
    }
};

// Empty profiler (zero overhead when disabled)
struct EmptyProfiler {
    template<typename... Args> void record_ar_handshake(Args...) {}
    template<typename... Args> void record_r_complete(Args...) {}
    template<typename... Args> void record_aw_handshake(Args...) {}
    template<typename... Args> void record_b_complete(Args...) {}
    template<typename... Args> void record_contention(Args...) {}
    void print_statistics() {}
    void export_to_csv(const std::string&) {}
};

#endif // AXI_PROFILER_H
