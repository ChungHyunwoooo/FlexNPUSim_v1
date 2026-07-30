#include "systemc/memory/dram/cycle/dram_cycle.h"

#include <algorithm>
#include <iostream>

#ifdef HAVE_DRAMSIM3
#include "dramsim3.h"
#endif

namespace flexnpu_sim {

DramCycle::DramCycle(const std::string& ini_path, const std::string& output_dir,
                     double sys_clock_ns) {
#ifdef HAVE_DRAMSIM3
    std::string ini = ini_path;
    if (ini.empty()) {
#ifdef DRAMSIM3_CONFIG_DIR
        ini = std::string(DRAMSIM3_CONFIG_DIR) + "/DDR4_8Gb_x8_2400.ini";
#else
        ini = "DDR4_8Gb_x8_2400.ini";
#endif
    }
    mem_ = dramsim3::GetMemorySystem(
        ini, output_dir.empty() ? "out" : output_dir,
        [this](uint64_t addr) { on_complete(addr); },
        [this](uint64_t addr) { on_complete(addr); });
    if (mem_) {
        bus_bytes_ = static_cast<uint32_t>(mem_->GetBusBits() / 8)
                   * static_cast<uint32_t>(mem_->GetBurstLength());
        if (bus_bytes_ == 0) bus_bytes_ = 64;
        const double tck = mem_->GetTCK();
        dram_per_sys_ = (tck > 0.0) ? (sys_clock_ns / tck) : 1.0;
        const int q = mem_->GetQueueSize();
        if (q > 0) accept_cap_ = static_cast<uint32_t>(q);
        std::cout << "[dram/cycle] DRAMSim3: " << ini << " (tCK=" << tck
                  << "ns, " << dram_per_sys_ << " dram-cy/sys-cy)\n";
    }
#else
    (void)ini_path; (void)output_dir; (void)sys_clock_ns;
    std::cerr << "[dram/cycle] DRAMSim3 not built in — cycle tier is a 0-latency stub\n";
#endif
}

DramCycle::~DramCycle() {
#ifdef HAVE_DRAMSIM3
    delete mem_;
#endif
}

bool DramCycle::will_accept(uint64_t /*address*/, bool /*is_write*/) const {
    return in_flight_ids_ < accept_cap_;
}

void DramCycle::submit(uint64_t id, uint64_t address, uint32_t size,
                       bool is_write) {
    uint32_t remaining = size ? size : 1;
    uint64_t addr = address;
    uint32_t chunks = 0;
    while (remaining > 0) {
        const uint32_t chunk = std::min(remaining, bus_bytes_);
        to_issue_.push_back({id, addr, is_write});
        addr      += chunk;
        remaining -= chunk;
        ++chunks;
    }
    remaining_[id] = chunks;
    ++in_flight_ids_;
#ifndef HAVE_DRAMSIM3
    // Stub: no DRAMSim3 — complete immediately.
    to_issue_.clear();
    remaining_.erase(id);
    --in_flight_ids_;
    done_.push_back(id);
#endif
}

void DramCycle::tick() {
#ifdef HAVE_DRAMSIM3
    if (!mem_) return;
    dram_accum_ += dram_per_sys_;
    while (dram_accum_ >= 1.0) {
        step_one_dram_cycle();
        dram_accum_ -= 1.0;
    }
#endif
}

void DramCycle::step_one_dram_cycle() {
#ifdef HAVE_DRAMSIM3
    while (!to_issue_.empty() &&
           mem_->WillAcceptTransaction(to_issue_.front().addr,
                                       to_issue_.front().is_write)) {
        const Chunk c = to_issue_.front();
        to_issue_.pop_front();
        addr_wait_[c.addr].push_back(c.id);
        mem_->AddTransaction(c.addr, c.is_write);
    }
    mem_->ClockTick();   // completion callbacks fire here -> on_complete
#endif
}

void DramCycle::on_complete(uint64_t address) {
    auto it = addr_wait_.find(address);
    if (it == addr_wait_.end() || it->second.empty()) return;
    const uint64_t id = it->second.front();
    it->second.pop_front();
    if (it->second.empty()) addr_wait_.erase(it);

    auto rit = remaining_.find(id);
    if (rit == remaining_.end()) return;
    if (--rit->second == 0) {
        remaining_.erase(rit);
        done_.push_back(id);
        if (in_flight_ids_ > 0) --in_flight_ids_;
    }
}

bool DramCycle::pop_completed(uint64_t& id) {
    if (done_.empty()) return false;
    id = done_.front();
    done_.pop_front();
    return true;
}

void DramCycle::reset() {
#ifdef HAVE_DRAMSIM3
    if (mem_) mem_->ResetStats();
#endif
    to_issue_.clear();
    addr_wait_.clear();
    remaining_.clear();
    done_.clear();
    in_flight_ids_ = 0;
    dram_accum_ = 0.0;
}

}  // namespace flexnpu_sim
