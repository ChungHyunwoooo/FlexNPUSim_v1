/**
 * @file test_npu_global_buffer.cpp
 * @brief Contracts for NpuGlobalBuffer — the GB storage + presence model.
 *
 * Both operating modes are covered (cache mode had no coverage before this):
 *  1. Scratchpad — mark/has range containment, size-0 is trivially present,
 *     clear resets, the region ledger trims oldest but keeps recent entries.
 *  2. Cache — miss on a cold buffer; store makes a range resident; load is a
 *     hit and copies the exact bytes back; direct-mapped conflict evicts;
 *     hit/miss counters track outcomes.
 *  3. Mode switch clears presence.
 */

#include "systemc/npu/model/npu_global_buffer.h"

#include <cassert>
#include <iostream>

using namespace flexnpu_sim;
using Mode = NpuGlobalBuffer::Mode;

static void test_scratchpad_containment() {
    NpuGlobalBuffer gb(1024);
    assert(gb.mode() == Mode::Scratchpad && "default mode is scratchpad");
    assert(!gb.scratchpad_has_range(0, 100) && "cold buffer holds nothing");

    gb.scratchpad_mark_range(0, 256);
    assert(gb.scratchpad_has_range(0, 100)  && "sub-range is contained");
    assert(gb.scratchpad_has_range(0, 256)  && "exact range is contained");
    assert(!gb.scratchpad_has_range(0, 257) && "overflowing range is not");
    assert(!gb.scratchpad_has_range(200, 100) && "[200,300) not in [0,256)");
    assert(gb.scratchpad_has_range(100, 0)  && "size-0 is trivially present");

    gb.clear_presence();
    assert(!gb.scratchpad_has_range(0, 100) && "clear_presence resets the ledger");
}

static void test_scratchpad_ledger_trims_but_keeps_recent() {
    NpuGlobalBuffer gb(1u << 20);
    for (uint32_t i = 0; i < 5000; ++i)
        gb.scratchpad_mark_range(i * 8, 4);   // exceeds the 4096 cap → trims oldest 2048
    assert(gb.scratchpad_has_range(4999 * 8, 4) && "most recent mark survives the trim");
}

static void test_cache_store_load_roundtrip() {
    NpuGlobalBuffer gb(4096);
    gb.set_cache_line_size(64);
    gb.set_mode(Mode::Cache);

    // Write a pattern into buffer bytes [0,128) and publish it to DRAM 0x1000.
    for (int i = 0; i < 128; ++i) gb.data()[i] = static_cast<uint8_t>(i * 7 + 1);
    assert(!gb.cache_range_hit(0x1000, 128) && "range is cold before store");
    gb.cache_store_range(0x1000, /*buf_off=*/0, 128);
    assert(gb.cache_range_hit(0x1000, 128) && "range is resident after store");

    // Clobber buffer bytes [256,384) then load DRAM 0x1000 back into them.
    for (int i = 256; i < 384; ++i) gb.data()[i] = 0;
    assert(gb.cache_load_range(0x1000, 128, /*buf_off=*/256) && "resident range loads (hit)");
    for (int i = 0; i < 128; ++i)
        assert(gb.data()[256 + i] == static_cast<uint8_t>(i * 7 + 1) && "loaded bytes match stored");

    assert(gb.cache_hits() == 1 && gb.cache_misses() == 0 && "one hit, no miss so far");

    // A never-stored address misses.
    assert(!gb.cache_load_range(0x9000, 64, 0) && "cold address misses");
    assert(gb.cache_misses() == 1 && "miss counted");
}

static void test_cache_direct_mapped_eviction() {
    NpuGlobalBuffer gb(4096);      // 64 lines of 64 B
    gb.set_cache_line_size(64);
    gb.set_mode(Mode::Cache);

    gb.cache_store_range(0x1000, 0, 64);   // line 0, tag 1
    assert(gb.cache_range_hit(0x1000, 64));
    gb.cache_store_range(0x0000, 0, 64);   // line 0, tag 0 — evicts tag 1
    assert(gb.cache_range_hit(0x0000, 64) && "new tag resident");
    assert(!gb.cache_range_hit(0x1000, 64) && "conflicting tag evicted");
}

static void test_cache_reset_presence_keeps_storable() {
    NpuGlobalBuffer gb(4096);
    gb.set_mode(Mode::Cache);
    gb.cache_store_range(0x1000, 0, 64);
    assert(gb.cache_range_hit(0x1000, 64));
    gb.reset_presence();   // drops residency but must rebuild lines so stores work
    assert(!gb.cache_range_hit(0x1000, 64) && "reset drops residency");
    gb.cache_store_range(0x1000, 0, 64);
    assert(gb.cache_range_hit(0x1000, 64) && "store works again after reset");
}

static void test_mode_switch_clears_presence() {
    NpuGlobalBuffer gb(1024);
    gb.scratchpad_mark_range(0, 256);
    assert(gb.scratchpad_has_range(0, 100));
    gb.set_mode(Mode::Cache);
    // Presence was cleared on the switch; back to scratchpad shows nothing stale.
    gb.set_mode(Mode::Scratchpad);
    assert(!gb.scratchpad_has_range(0, 100) && "mode switch cleared the ledger");
}

int main() {
    test_scratchpad_containment();
    test_scratchpad_ledger_trims_but_keeps_recent();
    test_cache_store_load_roundtrip();
    test_cache_direct_mapped_eviction();
    test_cache_reset_presence_keeps_storable();
    test_mode_switch_clears_presence();
    std::cout << "test_npu_global_buffer: all contracts passed\n";
    return 0;
}
