/**
 * @file test_axi_bus_routing.cpp
 * @brief Bus<Spec,2,2> contracts: address routing, DECERR synthesis,
 *        concurrent masters.
 *
 * Contracts:
 *  1. routing — each slave sees exactly the bytes addressed to its range
 *  2. DECERR — a request to an unmapped address is answered by the bus
 *     itself with resp=0x3 (never forwarded to a slave)
 *  3. two masters issuing concurrently both complete (no deadlock, data
 *     intact per master)
 */

#include "axi_test_fixture.h"

#include "systemc/bus/axi/bus/axi_bus.h"

#include <iostream>
#include <vector>

using namespace axitest;
namespace fta = flexnpu_sim::transport::axi;

static int failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::cerr << "FAIL: " << msg << "\n";                     \
            ++failures;                                               \
        }                                                             \
    } while (0)

using BusT = fta::Bus<TestSpec, 2, 2>;

static constexpr uint64_t S0_BASE = 0x10000000ull;
static constexpr uint64_t S1_BASE = 0x20000000ull;
static constexpr uint64_t RANGE   = 0x00010000ull;  // 64 KB each

SC_MODULE(Tb) {
    MasterT* m0 = nullptr;
    MasterT* m1 = nullptr;
    RamT*    ram0 = nullptr;
    RamT*    ram1 = nullptr;

    SC_CTOR(Tb) { SC_THREAD(run); }

    void write_beats(MasterT& m, uint64_t addr, unsigned n, unsigned seed) {
        std::vector<TestSpec::data_t> d;
        for (unsigned i = 0; i < n; ++i) d.push_back(pattern(seed + i));
        const std::vector<TestSpec::strb_t> strb(n, TestSpec::strb_t(0xFF));
        m.write_burst(addr, BEAT_BYTES, d, strb);
    }

    void run() {
        // -- 1. routing: bytes land in the right slave's backing store ----
        write_beats(*m0, S0_BASE + 0x100, 4, 0);
        write_beats(*m0, S1_BASE + 0x200, 4, 40);
        auto r0 = m0->read_burst(S0_BASE + 0x100, 4, BEAT_BYTES);
        auto r1 = m0->read_burst(S1_BASE + 0x200, 4, BEAT_BYTES);
        for (unsigned i = 0; i < 4; ++i) {
            CHECK(r0[i] == pattern(i), "slave0 payload");
            CHECK(r1[i] == pattern(40 + i), "slave1 payload");
        }
        // Backing stores diverge — routing was real, not aliased.
        CHECK(ram0->read(S0_BASE + 0x100, BEAT_BYTES) !=
                  ram1->read(S1_BASE + 0x200, BEAT_BYTES),
              "backends hold different data");

        // -- 2. DECERR for unmapped addresses ------------------------------
        std::vector<TestSpec::data_t> one{pattern(7)};
        const std::vector<TestSpec::strb_t> strb1{TestSpec::strb_t(0xFF)};
        const int wtk = m0->issue_write(0xF0000000ull, BEAT_BYTES, one, strb1);
        while (!m0->write_done(wtk)) sc_core::wait(m0->write_done_event(wtk));
        CHECK(m0->write_resp(wtk).to_uint() == 0x3u,
              "unmapped write must return DECERR");
        m0->release_write(wtk);

        // -- 3. concurrent masters complete with intact data ---------------
        const int a = m0->issue_read(S0_BASE + 0x100, 4, BEAT_BYTES, 0);
        const int b = m1->issue_read(S1_BASE + 0x200, 4, BEAT_BYTES, 0);
        CHECK(a >= 0 && b >= 0, "both masters issued");
        while (!m0->read_done(a)) sc_core::wait(m0->read_done_event(a));
        while (!m1->read_done(b)) sc_core::wait(m1->read_done_event(b));
        for (unsigned i = 0; i < 4; ++i) {
            CHECK(m0->peek_read(a)[i] == pattern(i), "m0 concurrent payload");
            CHECK(m1->peek_read(b)[i] == pattern(40 + i), "m1 concurrent payload");
        }
        m0->release_read(a);
        m1->release_read(b);

        sc_core::sc_stop();
    }
};

int sc_main(int, char**) {
    sc_core::sc_clock clk("clk", 1, sc_core::SC_NS);

    RamT ram0(S0_BASE, RANGE), ram1(S1_BASE, RANGE);
    MasterT m0("m0"), m1("m1");
    SlaveT s0("s0", ram0), s1("s1", ram1);

    std::vector<fta::MasterArbitCfg> mcfgs(2);
    std::array<BusT::SlaveRange, 2> ranges{{{S0_BASE, RANGE}, {S1_BASE, RANGE}}};
    BusT bus("bus", fta::ArbitPolicy::RoundRobin, mcfgs, ranges);

    AXI_SIGNALS<TestSpec> m0_sig, m1_sig, s0_sig, s1_sig;
    m0.clk(clk); m1.clk(clk); s0.clk(clk); s1.clk(clk); bus.clk(clk);
    bind_port_signal(&m0.port, m0_sig);
    bind_port_signal(&m1.port, m1_sig);
    bind_port_signal(&s0.port, s0_sig);
    bind_port_signal(&s1.port, s1_sig);
    bind_port_signal(&bus.m_ports[0], m0_sig);
    bind_port_signal(&bus.m_ports[1], m1_sig);
    bind_port_signal(&bus.s_ports[0], s0_sig);
    bind_port_signal(&bus.s_ports[1], s1_sig);

    Tb tb("tb");
    tb.m0 = &m0; tb.m1 = &m1; tb.ram0 = &ram0; tb.ram1 = &ram1;
    sc_core::sc_start();
    if (failures == 0) std::cout << "test_axi_bus_routing: all contracts hold\n";
    return failures == 0 ? 0 : 1;
}
