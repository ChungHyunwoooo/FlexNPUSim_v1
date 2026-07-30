/**
 * @file analytical_dram.h
 * @brief Bank-occupancy engine shared by the ideal / bandwidth / bank tiers.
 *
 * Real memory-controller behavior with an approximate per-access latency:
 *   - `banks` parallel servers. A transaction enters only while a server is
 *     free (will_accept = in_flight < banks) — that IS the backpressure. When
 *     the AXI slave's outstanding window exceeds `banks`, the extra requests
 *     stall here; effective concurrency = min(outstanding, banks) emerges from
 *     occupancy, never computed.
 *   - Each accepted transaction finishes `access_latency()` cycles later. The
 *     shared read-data bus is NOT modeled here — the AXI R channel serializes
 *     data for real.
 *
 * Subclasses supply only the per-access latency (the approximate part):
 *   ideal      — fixed cycles.
 *   bandwidth  — fixed + size / bandwidth.
 *   bank       — row-buffer hit/miss (keeps open-row state).
 */

#pragma once

#include <cstdint>
#include <deque>

#include "systemc/memory/dram/dram_timing.h"

namespace flexnpu_sim {

class AnalyticalDram : public DramTiming {
public:
    explicit AnalyticalDram(uint32_t banks) : banks_(banks ? banks : 1) {}

    bool will_accept(uint64_t /*address*/, bool /*is_write*/) const override {
        return in_flight_ < banks_;
    }

    void submit(uint64_t id, uint64_t address, uint32_t size,
                bool is_write) override {
        const uint64_t lat = access_latency(address, size, is_write);
        pending_.push_back({id, now_ + lat});
        ++in_flight_;
    }

    void tick() override { ++now_; }

    bool pop_completed(uint64_t& id) override {
        for (auto it = pending_.begin(); it != pending_.end(); ++it) {
            if (it->done <= now_) {
                id = it->id;
                pending_.erase(it);
                --in_flight_;
                return true;
            }
        }
        return false;
    }

    void reset() override {
        pending_.clear();
        in_flight_ = 0;
        now_ = 0;
    }

protected:
    /// Access latency in system cycles (the approximate, tier-specific part).
    virtual uint64_t access_latency(uint64_t address, uint32_t size,
                                    bool is_write) = 0;

private:
    struct Pending { uint64_t id; uint64_t done; };

    uint32_t              banks_;
    uint32_t              in_flight_ = 0;
    uint64_t              now_ = 0;
    std::deque<Pending>   pending_;
};

}  // namespace flexnpu_sim
