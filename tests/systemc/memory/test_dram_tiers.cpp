/**
 * @file test_dram_tiers.cpp
 * @brief Contracts for the analytical DRAM tiers + the tier factory.
 *
 * The tiers expose the real memory-controller interface (will_accept / submit /
 * tick / pop_completed); a transaction finishes access_latency cycles after
 * submit. Contracts:
 *  1. ideal      — fixed latency, independent of address and size; read/write
 *     charges are separate; 0 = zero-cost memory
 *  2. bandwidth  — latency = base + size / bandwidth; include_writes gates the
 *     byte term for writes; bandwidth 0 = base only
 *  3. bank       — first touch of a row is a miss, a repeat is a cheaper hit; a
 *     write opens the row for a following read; lines interleave across banks
 *  4. backpressure — will_accept goes false once `banks` transactions are in
 *     flight (occupancy, not a formula)
 *  5. factory    — make_dram_timing dispatches by cfg.tier and rejects unknown
 *
 * (tier "cycle" needs a DRAMSim3 ini + CWD; the anchors exercise it.)
 */

#include "systemc/memory/dram/factory.h"
#include "systemc/memory/dram/dram_timing.h"
#include "common/flexnpu_config.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace flexnpu_sim;

static config::DramConfig cfg(const std::string& tier) {
    config::DramConfig c;
    c.tier = tier;
    return c;
}

// Submit one transaction and drain it, returning the completion latency (cycles).
static uint64_t latency(DramTiming& t, uint64_t addr, uint32_t size, bool w) {
    static uint64_t id = 0;
    assert(t.will_accept(addr, w));
    t.submit(++id, addr, size, w);
    uint64_t cyc = 0, out;
    while (!t.pop_completed(out)) { t.tick(); ++cyc; }
    return cyc;
}

static void test_ideal() {
    auto c = cfg("ideal");
    c.read_latency_cyc = 50;
    c.write_latency_cyc = 20;
    auto t = make_dram_timing(c, /*sys_clock_ns=*/1.0);
    assert(t->name() == "ideal");
    assert(latency(*t, 0x1000, 999, false) == 50);   // address/size independent
    assert(latency(*t, 0x0, 1, false) == 50);
    assert(latency(*t, 0xDEAD, 4096, true) == 20);

    auto z = cfg("ideal");                            // 0 = zero-cost memory
    z.read_latency_cyc = 0;
    auto tz = make_dram_timing(z, 1.0);
    assert(latency(*tz, 0, 4096, false) == 0);
}

static void test_bandwidth() {
    auto c = cfg("bandwidth");
    c.read_latency_cyc = 100;
    c.write_latency_cyc = 30;
    c.bandwidth_bytes_per_cycle = 26;
    c.include_writes = false;
    auto t = make_dram_timing(c, 1.0);
    assert(t->name() == "bandwidth");
    assert(latency(*t, 0, 260, false) == 100 + 10);   // 260/26 = 10
    assert(latency(*t, 0, 0, false) == 100);           // base only
    assert(latency(*t, 0, 260, true) == 30);           // writes excluded

    c.include_writes = true;
    auto tw = make_dram_timing(c, 1.0);
    assert(latency(*tw, 0, 260, true) == 30 + 10);

    c.bandwidth_bytes_per_cycle = 0;                   // unlimited
    auto tu = make_dram_timing(c, 1.0);
    assert(latency(*tu, 0, 1000, false) == 100);
}

static void test_bank() {
    auto c = cfg("bank");
    c.banks = 4;
    c.line_bytes = 1024;
    c.row_hit_cyc = 20;
    c.row_miss_cyc = 45;
    c.write_latency_cyc = 30;
    auto t = make_dram_timing(c, 1.0);
    assert(t->name() == "bank");

    assert(latency(*t, 0, 64, false) == 45);      // first touch bank0/row0 -> miss
    assert(latency(*t, 0, 64, false) == 20);      // repeat -> hit
    assert(latency(*t, 512, 64, false) == 20);    // same line -> hit
    assert(latency(*t, 1024, 64, false) == 45);   // line1/bank1 -> miss
    assert(latency(*t, 2048, 64, true) == 30);    // fixed write charge
    assert(latency(*t, 2048, 64, false) == 20);   // write opened the row -> hit
}

static void test_backpressure() {
    auto c = cfg("ideal");
    c.read_latency_cyc = 100;
    c.banks = 4;
    auto t = make_dram_timing(c, 1.0);
    for (uint64_t i = 0; i < 4; ++i) {
        assert(t->will_accept(0, false));
        t->submit(i, 0, 64, false);
    }
    assert(!t->will_accept(0, false));   // all 4 servers busy -> backpressure
    uint64_t out;
    while (!t->pop_completed(out)) t->tick();
    assert(t->will_accept(0, false));    // one freed
}

static void test_factory_rejects_unknown() {
    bool threw = false;
    try {
        make_dram_timing(cfg("nonsense"), 1.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw && "unknown tier must throw std::invalid_argument");
}

int main() {
    test_ideal();
    test_bandwidth();
    test_bank();
    test_backpressure();
    test_factory_rejects_unknown();
    std::cout << "test_dram_tiers: all contracts hold\n";
    return 0;
}
