/**
 * @file dma_profile.h
 * @brief Universal DMA timing parameters + real-IP profile presets (pure C++).
 *
 * A "profile" (a real DMA IP preset) expands into DmaParams; individual JSON
 * keys under hw.dma then override per field. NO SystemC dependency so the
 * presets and the outstanding-window math are unit-testable without sc_start.
 *
 * Grounded on NVDLA CDMA and ARM DMA-330: both IPs sit on ONE axis —
 * outstanding / internal-FIFO depth — so a single parameter set covers both;
 * latency hiding is the emergent result of the in-flight window plus
 * memory-side overlap, not a separate knob.
 */

#pragma once

#include <cstdint>
#include <string_view>

namespace flexnpu_sim {

/**
 * @brief Resolved DMA timing parameters (one profile → one DmaParams).
 */
struct DmaParams {
    // Group A — burst shaping.
    uint32_t max_burst_beats    = 8;          ///< Max beats per burst.

    // Group B — the single latency-hiding axis: internal FIFO depth bounds how
    // many bursts stay in flight (= prefetch depth). max_issuing caps that to
    // the AXI issuing capability when the IP specifies one (0 = derive from FIFO).
    uint32_t fifo_bytes         = 16u * 1024; ///< Internal read/write buffer depth.
    uint32_t max_issuing_reads  = 0;          ///< AXI read issuing capability (0=derive).
    uint32_t max_issuing_writes = 0;          ///< AXI write issuing capability (0=derive).

    // Group C — channels (bus-sharing transfer contexts).
    uint32_t read_channels      = 1;
    uint32_t write_channels     = 1;
};

/**
 * @brief Expand a profile name into DmaParams.
 *
 * Grounded in vendor specs:
 *   "nvdla_cdma" — NVDLA CDMA: 512b data path, max burst 4, 64B-aligned reads,
 *                  read buffer sized for ~128-cyc latency (4 KiB @ 32 B/cyc).
 *                  Deep buffering → full-pipeline (peak_bw) extreme.
 *   "arm_dma330" — ARM PL330: read/write issuing capability 4, shallow example
 *                  MFIFO, per-channel microcode fetch. (rd/wr_cap: CRD register.)
 *   "generic"    — historical DmaCfg defaults; keeps behavior unchanged for
 *                  configs that do not set a profile.
 *
 * Unknown names fall back to generic.
 */
inline DmaParams resolve_profile(std::string_view name) {
    DmaParams p;  // generic defaults
    if (name == "nvdla_cdma") {
        p.max_burst_beats      = 4;
        p.fifo_bytes           = 4096;
        p.read_channels        = 2;   // DAT + WT paths
        p.write_channels       = 1;
    } else if (name == "arm_dma330") {
        // QEMU hw/dma/pl330.c reset defaults (data_width=64, data_buffer_dep=256
        // → MFIFO = 64/4 * 256 = 4096 B; rd_cap = wr_cap = 8). The MFIFO is a
        // configurable synthesis parameter, not inherently shallow.
        p.max_burst_beats      = 16;
        p.fifo_bytes           = 4096;
        p.max_issuing_reads    = 8;
        p.max_issuing_writes   = 8;
        p.read_channels        = 8;   // single shared AXI master (timing approx)
        p.write_channels       = 8;
    }
    return p;
}

/**
 * @brief In-flight burst window from FIFO depth.
 *
 * How many bursts the DMA can keep outstanding before it must drain =
 * fifo_bytes / burst_bytes, capped by the AXI issuing capability (when the
 * profile sets one), floored at 1. This is what makes buffer_size_kb a live
 * timing knob: deeper FIFO → larger window → more latency hidden once the
 * memory side overlaps requests.
 */
inline uint32_t outstanding_window(uint32_t fifo_bytes, uint32_t burst_bytes,
                                   uint32_t max_issuing) {
    uint32_t w = burst_bytes ? fifo_bytes / burst_bytes : 1u;
    if (w == 0) w = 1u;
    if (max_issuing && w > max_issuing) w = max_issuing;
    return w;
}

}  // namespace flexnpu_sim
