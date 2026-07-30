/**
 * @file test_memory_slave_timing.cpp
 * @brief MemoryAxiSlaveV2 contract: the DRAM timing model delays the response
 *        and the raw datapath is preserved.
 *
 * Contracts:
 *  1. A read burst does not return until the memory model reports it complete —
 *     with a fixed-latency (ideal) model the read pays at least that latency.
 *  2. Data written through the wrapper reads back intact (the wrapper preserves
 *     the raw Slave/DenseRam datapath).
 */

#include <systemc.h>

#include "systemc/bus/axi/axi_signal_ports.h"
#include "systemc/bus/axi/master/axi_master.h"
#include "systemc/bus/axi/spec/axi_spec.h"
#include "systemc/memory/dram/dram_timing.h"
#include "systemc/memory/dram/factory.h"
#include "systemc/memory/wrapper/memory_axi_slave_v2.h"
#include "common/flexnpu_config.h"

#include <iostream>
#include <memory>
#include <vector>

using TestSpec = ::axi::Spec<::axi::Protocol::AXI4, 4, 32, 64, 0>;
using MasterT  = flexnpu_sim::transport::axi::Master<TestSpec>;
using flexnpu_sim::MemoryAxiSlaveV2;

static int failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::cerr << "FAIL: " << msg << "\n";                     \
            ++failures;                                               \
        }                                                             \
    } while (0)

static constexpr uint64_t BASE = 0x80000000ull;
static constexpr unsigned BEAT_BYTES = 8;
static constexpr unsigned DRAM_CYC = 50;   // fixed ideal-tier latency

static std::unique_ptr<flexnpu_sim::DramTiming> fixed_timing() {
    flexnpu_sim::config::DramConfig c;
    c.tier = "ideal";
    c.read_latency_cyc  = DRAM_CYC;
    c.write_latency_cyc = DRAM_CYC;
    return flexnpu_sim::make_dram_timing(c, /*sys_clock_ns=*/1.0);
}

SC_MODULE(Tb) {
    MasterT* m = nullptr;
    sc_core::sc_time* rd_lat = nullptr;

    SC_CTOR(Tb) { SC_THREAD(run); }

    void run() {
        // seed memory through the wrapper, then measure a read
        std::vector<TestSpec::data_t> d(8, TestSpec::data_t(0xCAFE));
        const std::vector<TestSpec::strb_t> strb(8, TestSpec::strb_t(0xFF));
        m->write_burst(BASE, BEAT_BYTES, d, strb);

        const sc_core::sc_time t0 = sc_core::sc_time_stamp();
        auto r8 = m->read_burst(BASE, 8, BEAT_BYTES);
        *rd_lat = sc_core::sc_time_stamp() - t0;

        CHECK(r8.size() == 8, "burst length");
        for (auto& b : r8) CHECK(b == TestSpec::data_t(0xCAFE), "payload intact");

        sc_core::sc_stop();
    }
};

int sc_main(int, char**) {
    sc_core::sc_clock clk("clk", 1, sc_core::SC_NS);
    MemoryAxiSlaveV2<TestSpec> mem("mem", BASE, 1 << 20, fixed_timing());
    MasterT master("m0");
    AXI_SIGNALS<TestSpec> sig;
    mem.clk(clk);
    master.clk(clk);
    bind_port_signal(&master.port, sig);
    bind_port_signal(&mem.port(), sig);

    sc_core::sc_time rd_lat;
    Tb tb("tb");
    tb.m = &master;
    tb.rd_lat = &rd_lat;
    sc_core::sc_start();

    CHECK(rd_lat >= sc_core::sc_time(DRAM_CYC, sc_core::SC_NS),
          "read must pay at least the memory-model latency");

    if (failures == 0)
        std::cout << "test_memory_slave_timing: all contracts hold\n";
    return failures == 0 ? 0 : 1;
}
