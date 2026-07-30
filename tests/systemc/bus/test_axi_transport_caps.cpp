/**
 * @file test_axi_transport_caps.cpp
 * @brief MemoryMappedTransport boundary contracts over AxiTransport.
 *
 * This is the protocol-agnostic seam (doc/design/bus-protocols.md): callers
 * hold a MemoryMappedTransport* and size their bursts from caps().
 *
 * Contracts:
 *  1. caps() echoes the configured limits verbatim
 *  2. a byte-level BurstReq write/read roundtrip preserves payload and
 *     reports Ok — protocol details (ID, strobes, beat packing) stay inside
 *  3. issue/wait multi-outstanding pattern completes out-of-order safely
 */

#include "axi_test_fixture.h"

#include "systemc/bus/axi/axi_transport.h"
#include "systemc/bus/memory_mapped_transport.h"

#include <cstring>
#include <iostream>
#include <vector>

using namespace axitest;
namespace ft = flexnpu_sim::transport;

static int failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::cerr << "FAIL: " << msg << "\n";                     \
            ++failures;                                               \
        }                                                             \
    } while (0)

SC_MODULE(Tb) {
    ft::MemoryMappedTransport* t = nullptr;   // callers only see this type

    SC_CTOR(Tb) { SC_THREAD(run); }

    void run() {
        // -- 1. caps echo ---------------------------------------------------
        const auto& c = t->caps();
        CHECK(c.max_outstanding_reads == 2, "caps outstanding reads");
        CHECK(c.max_beats_per_burst == TestSpec::MAX_BURST_LEN, "caps beats");
        CHECK(c.boundary_bytes == 4096, "caps 4KB boundary");

        // -- 2. byte-level roundtrip through the agnostic interface --------
        ft::BurstReq req;
        req.addr = 0x80000400ull;
        req.beats = 4;
        req.beat_size_log2 = 3;  // 8B beats
        std::vector<uint8_t> payload(req.total_bytes());
        for (size_t i = 0; i < payload.size(); ++i)
            payload[i] = uint8_t(0x5A ^ i);
        auto wr = t->write_burst(req, payload);
        CHECK(wr.resp == ft::TransportResp::Ok, "write resp Ok");

        auto rr = t->read_burst(req);
        CHECK(rr.resp == ft::TransportResp::Ok, "read resp Ok");
        CHECK(rr.bytes == payload.size(), "read byte count");
        CHECK(std::memcmp(rr.data, payload.data(), payload.size()) == 0,
              "payload preserved across the boundary");

        // -- 3. issue/wait multi-outstanding, waited in reverse order ------
        ft::BurstReq r2 = req;
        r2.addr = 0x80000800ull;
        std::vector<uint8_t> p2(r2.total_bytes(), 0x33);
        (void)t->write_burst(r2, p2);

        ft::Ticket ta = t->issue_read(req);
        ft::Ticket tb = t->issue_read(r2);
        CHECK(ta.valid() && tb.valid(), "two in-flight reads");
        auto rb = t->wait_read(tb);                 // reverse completion order
        CHECK(rb.data[0] == 0x33, "second burst payload");
        auto ra = t->wait_read(ta);
        CHECK(ra.data[0] == uint8_t(0x5A ^ 0), "first burst payload");

        sc_core::sc_stop();
    }
};

int sc_main(int, char**) {
    ::axi::CommonConfig ccfg;
    ccfg.max_outstanding_reads  = 2;
    ccfg.max_outstanding_writes = 2;
    DirectLink link(0x80000000ull, 1 << 20, ccfg);

    ft::Capabilities caps;
    caps.max_outstanding_reads  = 2;
    caps.max_outstanding_writes = 2;
    caps.max_beats_per_burst    = TestSpec::MAX_BURST_LEN;
    caps.max_burst_bytes        = TestSpec::MAX_BURST_LEN * BEAT_BYTES;
    caps.addr_alignment_bytes   = 1;
    caps.boundary_bytes         = 4096;
    flexnpu_sim::transport::axi::AxiTransport<TestSpec> transport(link.master, caps);

    Tb tb("tb");
    tb.t = &transport;
    sc_core::sc_start();
    if (failures == 0)
        std::cout << "test_axi_transport_caps: all contracts hold\n";
    return failures == 0 ? 0 : 1;
}
