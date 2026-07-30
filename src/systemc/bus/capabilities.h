/**
 * @file capabilities.h
 * @brief Protocol-agnostic capability descriptor and ModelingMode enum.
 *
 * Callers query capabilities via MemoryMappedTransport::caps() to size their
 * burst splitting and outstanding strategies without knowing which protocol
 * sits underneath.
 */

#pragma once

#include <cstdint>

namespace flexnpu_sim::transport {

/**
 * @brief Modeling fidelity selection per transport instance.
 *
 * - Full        : Cycle-accurate signal-level handshakes (RTL parity).
 * - LatencyOnly : Analytic latency (no per-cycle sc_signal toggling).
 *                 Ticket pool capacity is still honored — multi-outstanding
 *                 backpressure is structurally identical.
 */
enum class ModelingMode : uint8_t {
    Full        = 0,
    LatencyOnly = 1,
};

/**
 * @brief Protocol limits exposed to the caller.
 *
 * The caller uses these to chunk a logical transfer into protocol-legal
 * bursts, decide burst boundaries (e.g., AXI 4KB rule), and size the
 * concurrent in-flight pipeline.
 */
struct Capabilities {
    uint32_t max_outstanding_reads  = 1;
    uint32_t max_outstanding_writes = 1;
    uint32_t max_beats_per_burst    = 1;     ///< AXI3=16, AXI4=256, PCIe TLP-equivalent.
    uint32_t max_burst_bytes        = 64;    ///< Hard cap on a single burst.
    uint32_t addr_alignment_bytes   = 1;     ///< Required start-address alignment.
    uint32_t boundary_bytes         = 4096;  ///< Address space boundary a burst must not cross.

    bool     supports_out_of_order  = false; ///< Same ID returns must be in-order; cross-ID may reorder.
    bool     supports_unaligned     = true;  ///< Sub-beat addresses with byte enables.
    bool     supports_fixed_burst   = false; ///< BurstType::Fixed accepted.
    bool     supports_wrap_burst    = false; ///< BurstType::Wrap accepted.

    ModelingMode mode = ModelingMode::Full;
};

}  // namespace flexnpu_sim::transport
