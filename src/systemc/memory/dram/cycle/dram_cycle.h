/**
 * @file dram_cycle.h
 * @brief Tier `cycle`: DRAMSim3 driven loaded (real flow control).
 *
 * DRAMSim3 is a clock-driven, concurrent queue model. This tier drives it the
 * way its own frontend does (src/cpu.cc): submit transactions as they arrive,
 * tick the clock every cycle, and take completions from the callback. No
 * blocking drain — many transactions stay in flight, so bank parallelism,
 * queueing, and refresh are actually modelled.
 *
 * will_accept() maps to WillAcceptTransaction (backpressure), submit() breaks a
 * burst into bus-sized DRAMSim3 transactions, tick() advances the DRAM clock by
 * sys_clock/tCK cycles per system cycle, pop_completed() returns a burst's token
 * once all its transactions have completed.
 *
 * Built only with HAVE_DRAMSIM3 (always set by the build); without it the tier
 * completes every transaction immediately (0-latency stub).
 */

#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>

#include "systemc/memory/dram/dram_timing.h"

namespace dramsim3 { class MemorySystem; }

namespace flexnpu_sim {

class DramCycle : public DramTiming {
public:
    DramCycle(const std::string& ini_path, const std::string& output_dir,
              double sys_clock_ns);
    ~DramCycle() override;

    bool will_accept(uint64_t address, bool is_write) const override;
    void submit(uint64_t id, uint64_t address, uint32_t size,
                bool is_write) override;
    void tick() override;
    bool pop_completed(uint64_t& id) override;
    void reset() override;
    std::string name() const override { return "cycle"; }

private:
    void on_complete(uint64_t address);   // DRAMSim3 read/write callback target
    void step_one_dram_cycle();

    dramsim3::MemorySystem* mem_ = nullptr;
    uint32_t bus_bytes_    = 64;    ///< bytes per DRAMSim3 transaction
    double   dram_per_sys_ = 1.0;   ///< DRAM cycles advanced per system cycle
    double   dram_accum_   = 0.0;   ///< fractional-cycle carry
    uint32_t accept_cap_   = 32;    ///< in-flight bursts before backpressure

    struct Chunk { uint64_t id; uint64_t addr; bool is_write; };
    std::deque<Chunk> to_issue_;                                 ///< queued for DRAMSim3
    std::unordered_map<uint64_t, std::deque<uint64_t>> addr_wait_;  ///< addr -> FIFO of ids
    std::unordered_map<uint64_t, uint32_t> remaining_;           ///< id -> chunks left
    std::deque<uint64_t> done_;                                  ///< completed ids
    uint32_t in_flight_ids_ = 0;
};

}  // namespace flexnpu_sim
