/**
 * @file dma_model.h
 * @brief DMA Behavioral Model (Pure C++)
 *
 * Behavioral core extracted from DmaEngine SC_MODULE.
 * Manages burst calculation, transfer state, and statistics.
 * NO SystemC dependency - pure C++ for independent testing.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace flexnpu_sim {

/**
 * @brief DMA transfer mode
 */
enum class DmaMode : uint8_t {
    Read,   ///< Memory → L2 (read from memory)
    Write   ///< L2 → Memory (write to memory)
};

/**
 * @brief DMA transfer state
 */
enum class DmaState : uint8_t {
    Idle,      ///< Idle, ready for new transfer
    Setup,     ///< Setting up transfer parameters
    Transfer,  ///< Active transfer in progress
    Complete   ///< Transfer complete
};

/**
 * @brief DMA model configuration
 */
struct DmaConfig {
    uint32_t bytes_per_beat;   ///< Bytes per AXI beat (typically 8 for 64-bit)
    uint32_t max_burst_beats;  ///< Max beats per burst (typically 256 for AXI4)

    DmaConfig()
        : bytes_per_beat(8),
          max_burst_beats(256) {}

    DmaConfig(uint32_t bytes_beat, uint32_t max_burst)
        : bytes_per_beat(bytes_beat),
          max_burst_beats(max_burst) {}

    /**
     * @brief Get max burst size in bytes
     */
    uint32_t max_burst_bytes() const {
        return max_burst_beats * bytes_per_beat;
    }
};

/**
 * @brief Single burst descriptor
 */
struct BurstDescriptor {
    uint64_t address;    ///< Start address
    uint32_t beats;      ///< Number of beats
    uint32_t bytes;      ///< Total bytes in this burst
    uint32_t beat_bytes; ///< Per-burst beat size (AXI requires addr aligned to it)

    BurstDescriptor()
        : address(0), beats(0), bytes(0), beat_bytes(0) {}

    BurstDescriptor(uint64_t addr, uint32_t b, uint32_t sz, uint32_t beat_b)
        : address(addr), beats(b), bytes(sz), beat_bytes(beat_b) {}
};

/**
 * @brief Complete burst plan for a transfer
 */
struct BurstPlan {
    std::vector<BurstDescriptor> bursts;  ///< List of bursts
    uint32_t total_bytes;                 ///< Total transfer size
    uint32_t total_bursts;                ///< Number of bursts

    BurstPlan() : total_bytes(0), total_bursts(0) {}

    /**
     * @brief Add a burst to the plan
     */
    void add_burst(uint64_t address, uint32_t beats, uint32_t bytes,
                   uint32_t beat_bytes) {
        bursts.emplace_back(address, beats, bytes, beat_bytes);
        total_bytes += bytes;
        total_bursts++;
    }
};

/**
 * @brief DMA execution statistics
 */
struct DmaStats {
    uint64_t total_read_bytes;
    uint64_t total_write_bytes;
    uint64_t total_read_transfers;
    uint64_t total_write_transfers;
    uint64_t total_read_bursts;
    uint64_t total_write_bursts;

    DmaStats()
        : total_read_bytes(0), total_write_bytes(0),
          total_read_transfers(0), total_write_transfers(0),
          total_read_bursts(0), total_write_bursts(0) {}
};

/**
 * @brief Pure C++ DMA Behavioral Model
 *
 * Calculates burst plans, tracks transfer state, and maintains statistics.
 * Does NOT depend on SystemC or perform actual data movement.
 * Wrapper layer handles actual AXI transactions.
 *
 * Usage:
 *   DmaModel dma(config);
 *   auto plan = dma.submit_transfer(src_addr, dst_addr, size, DmaMode::Read);
 *   // Execute bursts from plan...
 *   dma.complete_transfer();
 */
class DmaModel {
public:
    explicit DmaModel(const DmaConfig& cfg = DmaConfig())
        : config(cfg),
          state(DmaState::Idle),
          current_mode(DmaMode::Read)
    {}

    /**
     * @brief Submit a DMA transfer request
     *
     * Calculates burst plan by splitting large transfers into
     * max_burst_bytes chunks.
     *
     * @param src_addr Source address
     * @param dst_addr Destination address
     * @param size Transfer size in bytes
     * @param mode Transfer mode (Read or Write)
     * @return Burst plan with all bursts
     */
    BurstPlan submit_transfer(uint64_t src_addr, uint64_t dst_addr,
                              uint32_t size, DmaMode mode);

    /**
     * @brief Mark current transfer as complete
     *
     * Updates statistics and returns to Idle state.
     */
    void complete_transfer();

    /**
     * @brief Abort current transfer
     */
    void abort_transfer();

    /**
     * @brief Reset DMA state
     */
    void reset();

    /**
     * @brief Check if DMA is idle
     */
    bool is_idle() const { return state == DmaState::Idle; }

    /**
     * @brief Check if DMA is busy
     */
    bool is_busy() const {
        return state == DmaState::Setup || state == DmaState::Transfer;
    }

    /**
     * @brief Get current state
     */
    DmaState get_state() const { return state; }

    /**
     * @brief Get execution statistics
     */
    const DmaStats& get_stats() const { return stats; }

    /**
     * @brief Get configuration
     */
    const DmaConfig& get_config() const { return config; }

private:
    DmaConfig config;
    DmaState state;
    DmaMode current_mode;
    DmaStats stats;

    uint64_t current_src_addr;
    uint64_t current_dst_addr;
    uint32_t current_size;

    /**
     * @brief Calculate bursts for a transfer
     *
     * Splits size into chunks of max_burst_bytes, with final
     * partial burst if needed.
     */
    BurstPlan calculate_bursts(uint64_t base_addr, uint32_t size) const;

    /**
     * @brief Update statistics for completed transfer
     */
    void update_stats(const BurstPlan& plan);
};

} // namespace flexnpu_sim
