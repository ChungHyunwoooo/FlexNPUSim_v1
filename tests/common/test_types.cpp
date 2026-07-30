/**
 * @file test_types.cpp
 * @brief Contract tests for common/types.h.
 *
 * Verifies the documented derivations, not the constant values themselves:
 *  - PeConfig derived rates (max/min outputs per cycle, execution units)
 *  - register map consistency (REG_SPACE ends the register file)
 *  - memory map regions do not overlap
 */

#include "common/types.h"

#include <cassert>
#include <cstdint>
#include <iostream>

using namespace flexnpu_sim;

static void test_pe_config_derivations() {
    // Default: 16 independent lanes, one pass each (NVDLA-like, g=1).
    PeConfig def;
    assert(def.n_max() == 16);
    assert(def.n_min() == 1);   // g=1 -> n_min=1
    assert(def.total_execution_units() == 16);

    // MAERI-like: g>1 parallel passes -> n_min follows g.
    PeConfig maeri;
    maeri.num_output_lanes = 8;
    maeri.parallel_passes  = 4;
    assert(maeri.n_max() == 32);
    assert(maeri.n_min() == 4);  // n_min = g
    assert(maeri.total_execution_units() == 32);
}

static void test_register_map_consistency() {
    // REG_SPACE is defined as "last register + 4".
    static_assert(npu_reg::REG_SPACE == npu_reg::PERF_CNT3_HI + 4,
                  "REG_SPACE must end at the last register");
    // The 64-bit perf counters expose {HI:LO} pairs at fixed distance.
    static_assert(npu_reg::PERF_CNT0_HI - npu_reg::PERF_CNT0 == 0x10,
                  "HI bank sits one 4-register bank above LO");
    static_assert(npu_reg::PERF_CNT3_HI - npu_reg::PERF_CNT3 == 0x10,
                  "HI bank sits one 4-register bank above LO");
}

static void test_memory_map_no_overlap() {
    struct Region { addr_t base; uint64_t size; };
    const Region regions[] = {
        {memory_map::ADDR_SRAM,      memory_map::SIZE_SRAM},
        {memory_map::ADDR_NPU,       memory_map::SIZE_NPU},
        {memory_map::ADDR_DRAM_CTRL, memory_map::SIZE_DRAM_CTRL},
        {memory_map::ADDR_PROC,      memory_map::SIZE_PROC},
        {memory_map::ADDR_DRAM,      memory_map::SIZE_DRAM},
    };
    const int n = sizeof(regions) / sizeof(regions[0]);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const bool disjoint =
                regions[i].base + regions[i].size <= regions[j].base ||
                regions[j].base + regions[j].size <= regions[i].base;
            assert(disjoint && "memory map regions must not overlap");
        }
    }
}

int main() {
    test_pe_config_derivations();
    test_register_map_consistency();
    test_memory_map_no_overlap();
    std::cout << "test_types: all contracts hold\n";
    return 0;
}
