/**
 * @file dma_model.cpp
 * @brief DMA Behavioral Model Implementation
 *
 * Extracted from DmaEngine SC_MODULE - pure C++ behavioral core.
 * NO SystemC dependencies.
 */

#include "systemc/dma/model/dma_model.h"
#include <algorithm>

namespace flexnpu_sim {

// ============================================================================
// Public Interface
// ============================================================================

BurstPlan DmaModel::submit_transfer(uint64_t src_addr, uint64_t dst_addr,
                                     uint32_t size, DmaMode mode) {
    // Save transfer parameters
    current_src_addr = src_addr;
    current_dst_addr = dst_addr;
    current_size = size;
    current_mode = mode;

    // Transition to Setup state
    state = DmaState::Setup;

    // Calculate burst plan
    uint64_t base_addr = (mode == DmaMode::Read) ? src_addr : dst_addr;
    BurstPlan plan = calculate_bursts(base_addr, size);

    // Transition to Transfer state
    state = DmaState::Transfer;

    return plan;
}

void DmaModel::complete_transfer() {
    if (state != DmaState::Transfer && state != DmaState::Complete) {
        return;
    }

    // Create burst plan for statistics update
    BurstPlan plan = calculate_bursts(
        (current_mode == DmaMode::Read) ? current_src_addr : current_dst_addr,
        current_size
    );

    // Update statistics
    update_stats(plan);

    // Return to Idle
    state = DmaState::Idle;
}

void DmaModel::abort_transfer() {
    state = DmaState::Idle;
}

void DmaModel::reset() {
    state = DmaState::Idle;
    current_mode = DmaMode::Read;
    current_src_addr = 0;
    current_dst_addr = 0;
    current_size = 0;
    stats = DmaStats();
}

// ============================================================================
// Private Helpers
// ============================================================================

BurstPlan DmaModel::calculate_bursts(uint64_t base_addr, uint32_t size) const {
    // Real AXI DMA controllers (NVDLA CDMA, ARM PL330, Xilinx AXI DMA) are
    // spec-obligated to split bursts at the 4 KiB page boundary (AXI4 §A3.4.1)
    // and at max_burst_bytes. Model both here so BurstPlan matches the
    // sub-bursts the hardware would emit; downstream transport layers can
    // issue each entry verbatim without re-splitting.
    constexpr uint64_t PAGE_BYTES = 4096;

    BurstPlan plan;

    uint32_t remaining = size;
    uint64_t current_addr = base_addr;

    while (remaining > 0) {
        // Bytes until the next 4 KiB boundary from current_addr.
        const uint64_t page_end    = (current_addr & ~(PAGE_BYTES - 1)) + PAGE_BYTES;
        const uint64_t bytes_to_page = page_end - current_addr;

        // Size cap = min(remaining, max_burst_bytes, bytes_to_page).
        uint32_t burst_bytes = std::min(remaining, config.max_burst_bytes());
        if (static_cast<uint64_t>(burst_bytes) > bytes_to_page) {
            burst_bytes = static_cast<uint32_t>(bytes_to_page);
        }

        // AXI: the beat size (ARSIZE) requires the start address to be aligned
        // to it, and the beat-rounded read (beats*beat) must not cross the 4 KiB
        // boundary. Clamp the beat to the largest power of two dividing the
        // address (<= configured beat); with an address-aligned beat and a
        // page-limited size, ceil(bytes/beat)*beat never crosses the page.
        // (Without this, an unaligned start — e.g. the 32-byte image header —
        // makes a 64B-beat first burst over-read past the page → Boundary4K
        // drop → corruption for any bus wider than 256b.)
        uint32_t beat = config.bytes_per_beat;
        if (current_addr != 0) {
            const uint64_t addr_align = current_addr & (~current_addr + 1);
            while (beat > 1 && addr_align < beat) beat >>= 1;
        }
        uint32_t beats = (burst_bytes + beat - 1) / beat;
        if (beats == 0) beats = 1;
        if (beats > config.max_burst_beats) {
            beats = config.max_burst_beats;
            burst_bytes = beats * beat;
        }

        plan.add_burst(current_addr, beats, burst_bytes, beat);
        current_addr += burst_bytes;
        remaining -= std::min(remaining, burst_bytes);
    }

    return plan;
}

void DmaModel::update_stats(const BurstPlan& plan) {
    if (current_mode == DmaMode::Read) {
        stats.total_read_bytes += plan.total_bytes;
        stats.total_read_transfers++;
        stats.total_read_bursts += plan.total_bursts;
    } else {
        stats.total_write_bytes += plan.total_bytes;
        stats.total_write_transfers++;
        stats.total_write_bursts += plan.total_bursts;
    }
}

} // namespace flexnpu_sim
