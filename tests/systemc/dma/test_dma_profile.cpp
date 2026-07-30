/**
 * @file test_dma_profile.cpp
 * @brief Contracts for DMA profile presets + outstanding-window math (pure C++).
 *
 * The universal DMA model is configured by picking a profile (real IP preset)
 * whose values are grounded in vendor specs; JSON keys override per field.
 * These contracts pin the preset numbers and the FIFO→window derivation so a
 * profile change is a deliberate, reviewed edit — not an accident.
 */

#include "systemc/dma/model/dma_profile.h"

#include <cassert>
#include <iostream>

using namespace flexnpu_sim;

int main() {
    // NVDLA CDMA — burst 4, ~128-cyc (4 KiB) read buffer, 2 read paths.
    const auto cdma = resolve_profile("nvdla_cdma");
    assert(cdma.max_burst_beats == 4 && cdma.fifo_bytes == 4096);
    assert(cdma.read_channels == 2 && cdma.write_channels == 1);

    // ARM PL330 — QEMU reset defaults: rd/wr issuing 8, MFIFO 4096 B.
    const auto p330 = resolve_profile("arm_dma330");
    assert(p330.max_issuing_reads == 8 && p330.max_issuing_writes == 8);
    assert(p330.fifo_bytes == 4096 && p330.max_burst_beats == 16);

    // generic == historical DmaCfg defaults (no behavior change for legacy cfgs).
    const auto gen = resolve_profile("generic");
    assert(gen.max_burst_beats == 8 && gen.fifo_bytes == 16u * 1024);
    assert(gen.read_channels == 1 && gen.write_channels == 1);
    assert(gen.max_issuing_reads == 0 && gen.max_issuing_writes == 0);

    // Unknown name falls back to generic (never throws).
    const auto unknown = resolve_profile("does-not-exist");
    assert(unknown.max_burst_beats == gen.max_burst_beats &&
           unknown.fifo_bytes == gen.fifo_bytes);

    // outstanding_window: fifo/burst, capped by issuing, floored at 1, div0-safe.
    assert(outstanding_window(4096, 256, 0) == 16);   // 4096/256
    assert(outstanding_window(4096, 256, 8) == 8);    // issuing cap wins
    assert(outstanding_window(4096, 256, 32) == 16);  // cap above natural → natural
    assert(outstanding_window(128, 256, 0) == 1);     // burst > fifo → floor 1
    assert(outstanding_window(4096, 0, 0) == 1);      // guard div-by-zero

    std::cout << "test_dma_profile: all profile/window contracts hold\n";
    return 0;
}
