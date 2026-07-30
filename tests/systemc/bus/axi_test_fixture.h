/**
 * @file axi_test_fixture.h
 * @brief Shared fixture for the signal-level AXI contract tests.
 *
 * DirectLink wires one real Master and one real Slave (DenseRamBackend)
 * through a single AXI_SIGNALS bundle — the smallest elaborated system that
 * exercises the five-channel protocol FSMs end to end.
 */

#pragma once

#include <systemc.h>

#include "systemc/bus/axi/axi_signal_ports.h"
#include "systemc/bus/axi/master/axi_master.h"
#include "systemc/bus/axi/slave/axi_memory_backend.h"
#include "systemc/bus/axi/slave/axi_slave.h"
#include "systemc/bus/axi/spec/axi_spec.h"

namespace axitest {

// Small compiled test spec: AXI4, 4-bit ID, 32-bit addr, 64-bit data.
using TestSpec = ::axi::Spec<::axi::Protocol::AXI4, 4, 32, 64, 0>;
using MasterT  = flexnpu_sim::transport::axi::Master<TestSpec>;
using SlaveT   = flexnpu_sim::transport::axi::Slave<TestSpec>;
using RamT     = flexnpu_sim::transport::axi::DenseRamBackend;

constexpr unsigned BEAT_BYTES = TestSpec::DATA_W / 8;

struct DirectLink {
    sc_core::sc_clock       clk;
    AXI_SIGNALS<TestSpec>   sig;
    RamT                    mem;
    MasterT                 master;
    SlaveT                  slave;

    DirectLink(uint64_t base, std::size_t bytes,
               const ::axi::CommonConfig& ccfg = ::axi::CommonConfig{})
        : clk("clk", 1, sc_core::SC_NS),
          mem(base, bytes),
          master("m0", ccfg),
          slave("s0", mem, ccfg) {
        master.clk(clk);
        slave.clk(clk);
        bind_port_signal(&master.port, sig);
        bind_port_signal(&slave.port, sig);
    }
};

/// Pattern beat: recognizable per-index payload.
inline TestSpec::data_t pattern(unsigned i) {
    return TestSpec::data_t(0xA0B0C0D000000000ull + i * 0x0101010101ull);
}

}  // namespace axitest
