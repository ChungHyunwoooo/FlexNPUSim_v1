/**
 * @file transfer_profile.h
 * @brief Per-burst latency breakdown — populated by R2 master, consumed by
 *        R11 ProfileSink (CSV / JSON / Markdown / ChromeTrace).
 */

#pragma once

#include "systemc/bus/memory_mapped_transport.h"

#include <cstdint>
#include <string>
#include <vector>

namespace flexnpu_sim::transport::common {

/// Lifecycle timestamps of one burst transfer (read or write).
struct TransferProfile {
    /// Identity
    uint32_t      ticket_slot   = 0;
    std::string   master_name;        ///< Topology-defined master ID.
    std::string   slave_name;         ///< Resolved at AR/AW dispatch.
    bool          is_read       = true;
    transport::addr_t addr      = 0;
    uint32_t      beats         = 0;
    uint32_t      beat_size_bytes = 0;
    transport::TransportResp resp = transport::TransportResp::Ok;

    /// Timestamps (sim ns).
    uint64_t      issue_ns       = 0;  ///< caller called issue_*
    uint64_t      ar_aw_ns       = 0;  ///< AR/AW handshake completed
    uint64_t      first_beat_ns  = 0;  ///< first R/W data beat sampled
    uint64_t      last_beat_ns   = 0;  ///< last beat / RLAST / WLAST
    uint64_t      complete_ns    = 0;  ///< caller observed wait_* return

    /// Convenience derived metrics.
    uint64_t handshake_latency_ns() const { return ar_aw_ns      - issue_ns; }
    uint64_t first_beat_latency_ns() const { return first_beat_ns - ar_aw_ns; }
    uint64_t data_phase_ns()         const { return last_beat_ns  - first_beat_ns; }
    uint64_t total_latency_ns()      const { return complete_ns   - issue_ns; }
};

}  // namespace flexnpu_sim::transport::common
