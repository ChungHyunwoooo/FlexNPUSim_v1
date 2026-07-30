#ifndef FLEXNPU_TRANSPORT_AXI_BUS_AXI_CONTENTION_PROFILE_H
#define FLEXNPU_TRANSPORT_AXI_BUS_AXI_CONTENTION_PROFILE_H

// Contention and utilization metrics for the R4 N×M AXI bus.
//
// Per-master × per-slave matrix captures how each master's traffic
// distributes across slaves and where it waits. Per-slave channel-level
// busy cycles let the profiler tell "cross-slave non-blocking" from
// "accidental serialization". Decode stats surface protocol violations
// (unmapped addresses, 4KB crossings, range overflows) the bus
// converted into DECERR responses.

#include <cstdint>
#include <vector>

namespace flexnpu_sim::transport::axi {

struct ContentionProfile {
    struct MasterStats {
        uint64_t ar_wait_cycles = 0;  ///< ARVALID high, no grant this cycle
        uint64_t aw_wait_cycles = 0;
        uint64_t w_wait_cycles  = 0;
        uint64_t ar_grants      = 0;  ///< AR accepted (any slave)
        uint64_t aw_grants      = 0;
        uint64_t hol_blocks     = 0;  ///< head-of-line: blocked by older
                                      ///< same-slave transfer in its queue
        std::vector<uint64_t> ar_grants_per_slave;   ///< size = NumS
        std::vector<uint64_t> aw_grants_per_slave;
        std::vector<uint64_t> wait_per_slave;        ///< cycles waiting per target slave
    };

    struct SlaveStats {
        uint64_t occupied_cycles = 0;        ///< any of the 5 channels active
        uint64_t ar_busy_cycles  = 0;        ///< handling AR this cycle
        uint64_t aw_busy_cycles  = 0;
        uint64_t w_busy_cycles   = 0;        ///< W beat in flight
        uint64_t r_busy_cycles   = 0;        ///< R beat in flight
        uint64_t b_busy_cycles   = 0;
    };

    struct DecodeStats {
        uint64_t unmapped_reads         = 0;
        uint64_t unmapped_writes        = 0;
        uint64_t boundary_4k_violations = 0;
        uint64_t slave_range_overflows  = 0;
    };

    /// Sampled once per simulation cycle: how many slaves had any of their
    /// five channels active simultaneously. The fraction of cycles where
    /// this is > 1 is a direct measure of cross-slave parallelism.
    struct ConcurrencyStats {
        std::vector<uint64_t> active_slaves_histogram;  ///< index = simultaneous count
    };

    ContentionProfile(unsigned num_masters, unsigned num_slaves)
        : masters(num_masters), slaves(num_slaves) {
        for (auto& m : masters) {
            m.ar_grants_per_slave.assign(num_slaves, 0);
            m.aw_grants_per_slave.assign(num_slaves, 0);
            m.wait_per_slave     .assign(num_slaves, 0);
        }
        concurrency.active_slaves_histogram.assign(num_slaves + 1, 0);
    }

    // --- Per-cycle recorders (called by Bus SC_THREADs) ---------------------

    void record_ar_wait   (unsigned m, unsigned s = ~0u) {
        if (m < masters.size()) {
            masters[m].ar_wait_cycles++;
            if (s < slaves.size()) masters[m].wait_per_slave[s]++;
        }
    }
    void record_aw_wait   (unsigned m, unsigned s = ~0u) {
        if (m < masters.size()) {
            masters[m].aw_wait_cycles++;
            if (s < slaves.size()) masters[m].wait_per_slave[s]++;
        }
    }
    void record_w_wait    (unsigned m) { if (m < masters.size()) masters[m].w_wait_cycles++; }

    void record_ar_grant  (unsigned m, unsigned s) {
        if (m < masters.size()) {
            masters[m].ar_grants++;
            if (s < slaves.size()) masters[m].ar_grants_per_slave[s]++;
        }
    }
    void record_aw_grant  (unsigned m, unsigned s) {
        if (m < masters.size()) {
            masters[m].aw_grants++;
            if (s < slaves.size()) masters[m].aw_grants_per_slave[s]++;
        }
    }
    void record_hol       (unsigned m) { if (m < masters.size()) masters[m].hol_blocks++; }

    void record_slave_channel(unsigned s, int ch /*0=AR,1=AW,2=W,3=R,4=B*/) {
        if (s >= slaves.size()) return;
        switch (ch) {
            case 0: slaves[s].ar_busy_cycles++; break;
            case 1: slaves[s].aw_busy_cycles++; break;
            case 2: slaves[s].w_busy_cycles ++; break;
            case 3: slaves[s].r_busy_cycles ++; break;
            case 4: slaves[s].b_busy_cycles ++; break;
            default: break;
        }
        slaves[s].occupied_cycles++;
    }

    void record_concurrency(unsigned active_slave_count) {
        if (active_slave_count < concurrency.active_slaves_histogram.size()) {
            concurrency.active_slaves_histogram[active_slave_count]++;
        }
    }

    void record_decode(bool is_read, int kind /*0=unmapped,1=4k,2=range*/) {
        switch (kind) {
            case 0: is_read ? decode.unmapped_reads++ : decode.unmapped_writes++; break;
            case 1: decode.boundary_4k_violations++; break;
            case 2: decode.slave_range_overflows++;  break;
            default: break;
        }
    }

    // --- Derived metrics ----------------------------------------------------

    /// Jain's fairness over combined AR+AW grants across masters.
    double fairness_index() const {
        if (masters.empty()) return 0.0;
        double sum = 0.0, sum_sq = 0.0;
        for (const auto& m : masters) {
            const double g = static_cast<double>(m.ar_grants + m.aw_grants);
            sum    += g;
            sum_sq += g * g;
        }
        if (sum == 0.0 || sum_sq == 0.0) return 0.0;
        return (sum * sum) / (static_cast<double>(masters.size()) * sum_sq);
    }

    uint64_t total_hol_blocks() const {
        uint64_t t = 0;
        for (const auto& m : masters) t += m.hol_blocks;
        return t;
    }

    /// Cycles with at least two slaves simultaneously active — the signature
    /// of a genuine crossbar (vs. serializing bus).
    uint64_t parallel_cycles() const {
        uint64_t t = 0;
        for (unsigned k = 2; k < concurrency.active_slaves_histogram.size(); ++k) {
            t += concurrency.active_slaves_histogram[k];
        }
        return t;
    }

    // --- Data members -------------------------------------------------------

    std::vector<MasterStats> masters;
    std::vector<SlaveStats>  slaves;
    DecodeStats              decode;
    ConcurrencyStats         concurrency;
};

}  // namespace flexnpu_sim::transport::axi

#endif  // FLEXNPU_TRANSPORT_AXI_BUS_AXI_CONTENTION_PROFILE_H
