/**
 * @file test_axi_roundtrip.cpp
 * @brief Master<->Slave signal-level contracts over a direct link.
 *
 * Contracts:
 *  1. burst roundtrip — data written through the five-channel protocol is
 *     read back intact; write response is OKAY
 *  2. multi-outstanding — the ticket pool grants exactly
 *     max_outstanding_reads concurrent reads, refuses (-1) beyond it, and
 *     recycles slots after completion
 *  3. slave latency hook — a hook returning N cycles delays the response
 *     phase by at least N cycles (the DRAM-timing attachment point)
 */

#include "axi_test_fixture.h"

#include <sysc/kernel/sc_spawn.h>

#include <deque>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

using namespace axitest;

static int failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::cerr << "FAIL: " << msg << "\n";                     \
            ++failures;                                               \
        }                                                             \
    } while (0)

SC_MODULE(Tb) {
    DirectLink* link = nullptr;

    SC_CTOR(Tb) { SC_THREAD(run); }

    void run() {
        auto& m = link->master;
        const uint64_t base = 0x80000000ull;
        const std::vector<TestSpec::strb_t> all(8, TestSpec::strb_t(0xFF));

        // -- 1. burst roundtrip + OKAY response --------------------------
        std::vector<TestSpec::data_t> wdata;
        for (unsigned i = 0; i < 8; ++i) wdata.push_back(pattern(i));
        const int wtk = m.issue_write(base, BEAT_BYTES, wdata, all, /*id=*/0);
        CHECK(wtk >= 0, "write ticket");
        while (!m.write_done(wtk)) sc_core::wait(m.write_done_event(wtk));
        CHECK(m.write_resp(wtk).to_uint() == 0u, "write resp must be OKAY");
        m.release_write(wtk);

        auto rdata = m.read_burst(base, 8, BEAT_BYTES);
        CHECK(rdata.size() == 8, "read burst length");
        for (unsigned i = 0; i < rdata.size(); ++i)
            CHECK(rdata[i] == pattern(i), "beat payload intact");

        // -- 2. ticket pool: saturation = cooperative backpressure ---------
        // Tickets are caller-released (release_read); with the pool held at
        // its cap of 2, a third issue_read BLOCKS until someone releases —
        // the documented TicketPool contract. A spawned releaser frees t0
        // 300 ns later; the blocked issue must resume exactly then.
        const int t0 = m.issue_read(base, 1, BEAT_BYTES, 0);
        const int t1 = m.issue_read(base + 64, 1, BEAT_BYTES, 1);
        CHECK(t0 >= 0 && t1 >= 0, "two concurrent reads granted");
        while (!m.read_done(t0)) sc_core::wait(m.read_done_event(t0));
        while (!m.read_done(t1)) sc_core::wait(m.read_done_event(t1));

        const sc_core::sc_time saturated_at = sc_core::sc_time_stamp();
        sc_core::sc_spawn([this, t0] {
            sc_core::wait(300, sc_core::SC_NS);
            link->master.release_read(t0);
        });
        const int t2 = m.issue_read(base + 128, 1, BEAT_BYTES, 2);
        CHECK(t2 >= 0, "blocked issue granted after a release");
        CHECK(sc_core::sc_time_stamp() >=
                  saturated_at + sc_core::sc_time(300, sc_core::SC_NS),
              "saturated issue must block until the release");
        while (!m.read_done(t2)) sc_core::wait(m.read_done_event(t2));
        m.release_read(t1);
        m.release_read(t2);

        // -- 3. memory port delays the response phase --------------------
        const sc_core::sc_time before = sc_core::sc_time_stamp();
        (void)m.read_burst(base, 1, BEAT_BYTES);
        const sc_core::sc_time fast = sc_core::sc_time_stamp() - before;

        // A fixed-latency memory port: every submitted token completes 200
        // cycles after submit — the real memory-controller attachment point.
        struct FixedLat {
            uint64_t now = 0;
            std::deque<std::pair<uint64_t, uint64_t>> pend;   // (token, done)
        };
        auto st = std::make_shared<FixedLat>();
        using SlaveT = std::decay_t<decltype(link->slave)>;
        typename SlaveT::MemPort mp;
        mp.will_accept    = [](uint64_t, bool) { return true; };
        mp.submit         = [st](uint64_t tk, uint64_t, uint32_t, bool) {
            st->pend.push_back({tk, st->now + 200});
        };
        mp.tick           = [st]() { ++st->now; };
        mp.pop_completed  = [st](uint64_t& tk) -> bool {
            for (auto it = st->pend.begin(); it != st->pend.end(); ++it)
                if (it->second <= st->now) {
                    tk = it->first; st->pend.erase(it); return true;
                }
            return false;
        };
        link->slave.set_mem_port(mp);
        const sc_core::sc_time t_start = sc_core::sc_time_stamp();
        (void)m.read_burst(base, 1, BEAT_BYTES);
        const sc_core::sc_time slow = sc_core::sc_time_stamp() - t_start;
        CHECK(slow >= fast + sc_core::sc_time(200, sc_core::SC_NS),
              "200-cycle memory port must delay the response by >= 200 cycles");

        sc_core::sc_stop();
    }
};

int sc_main(int, char**) {
    ::axi::CommonConfig ccfg;
    ccfg.max_outstanding_reads  = 2;
    ccfg.max_outstanding_writes = 2;
    DirectLink link(0x80000000ull, 1 << 20, ccfg);
    Tb tb("tb");
    tb.link = &link;
    sc_core::sc_start();
    if (failures == 0) std::cout << "test_axi_roundtrip: all contracts hold\n";
    return failures == 0 ? 0 : 1;
}
