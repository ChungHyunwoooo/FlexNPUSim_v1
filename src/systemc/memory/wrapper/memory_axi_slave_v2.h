/**
 * @file memory_axi_slave_v2.h
 * @brief Memory wrapper — axi::Slave + DenseRamBackend + a DRAM timing model.
 *
 * Internals:
 *   - DenseRamBackend: byte-addressable storage anchored at a base address.
 *   - axi::Slave<Spec>: signal-level AXI slave (real ready/valid).
 *   - DramTiming: the memory-controller timing model, driven through the
 *     slave's MemPort — will_accept (backpressure) / submit / tick /
 *     pop_completed. Concurrency and backpressure live in the model; the R
 *     channel serializes the read-data bus. No latency formula here.
 *
 * Binding (data-bus path in flexnpusim_system):
 *   MemoryAxiSlaveV2<Spec> memory(name, base, size, std::move(dram), ccfg);
 *   memory.clk(clk);
 *   bind_port_signal(&memory.port(), mem_bus_signals);
 *
 * Preload: memory.memory() returns the underlying std::vector<uint8_t>.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <systemc.h>
#include <vector>

#include "systemc/memory/dram/dram_timing.h"
#include "systemc/bus/axi/slave/axi_memory_backend.h"
#include "systemc/bus/axi/slave/axi_slave.h"

namespace flexnpu_sim {

template <class Spec>
class MemoryAxiSlaveV2 : public sc_core::sc_module {
public:
    sc_core::sc_in<bool> clk{"clk"};

    MemoryAxiSlaveV2(sc_core::sc_module_name name,
                     uint64_t base_addr,
                     uint32_t size_bytes,
                     std::unique_ptr<DramTiming> timing_model,
                     const ::axi::CommonConfig& ccfg = ::axi::CommonConfig{})
        : sc_core::sc_module(name),
          base_addr_(base_addr),
          size_bytes_(size_bytes),
          backend_(base_addr, size_bytes),
          slave_("slave", backend_, ccfg),
          timing_(std::move(timing_model))
    {
        slave_.clk(clk);
        install_mem_port();
    }

    // Port accessor: callers bind bus signals to &v2.port().
    SLAVE_PORTS<Spec>&       port()       { return slave_.port; }
    const SLAVE_PORTS<Spec>& port() const { return slave_.port; }

    // Dense storage accessor: preload writes raw bytes at base_addr offset 0.
    std::vector<uint8_t>&       memory()       { return backend_.data(); }
    const std::vector<uint8_t>& memory() const { return backend_.data(); }

    std::string timing_name() const {
        return timing_ ? timing_->name() : std::string("none");
    }

private:
    // Wire the DRAM timing model into the slave as its memory-controller port.
    void install_mem_port() {
        if (!timing_) return;   // no model -> zero-latency memory
        DramTiming* t = timing_.get();
        typename transport::axi::Slave<Spec>::MemPort mp;
        mp.will_accept   = [t](uint64_t a, bool w) { return t->will_accept(a, w); };
        mp.submit        = [t](uint64_t id, uint64_t a, uint32_t s, bool w) {
            t->submit(id, a, s, w);
        };
        mp.tick          = [t]() { t->tick(); };
        mp.pop_completed = [t](uint64_t& id) { return t->pop_completed(id); };
        slave_.set_mem_port(mp);
    }

    uint64_t base_addr_;
    uint32_t size_bytes_;

    transport::axi::DenseRamBackend backend_;
    transport::axi::Slave<Spec>     slave_;
    std::unique_ptr<DramTiming>     timing_;
};

}  // namespace flexnpu_sim
