/**
 * @file burst_req.h
 * @brief Burst-level request descriptor passed across MemoryMappedTransport.
 *
 * Protocol-agnostic. Concrete transports (AXI/PCIe/...) interpret beat_size_log2
 * and BurstType per their own wire semantics. Per-protocol attributes (AXI ID,
 * PCIe TLP tag, NoC virtual channel) are auto-assigned inside the transport;
 * callers only supply the abstract hints in ReqAttr.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace flexnpu_sim::transport {

using addr_t = uint64_t;

enum class BurstType : uint8_t {
    Incr  = 0,  ///< Address increments per beat (AXI INCR, PCIe MRd/MWr).
    Fixed = 1,  ///< Address stays — streaming to same location (AXI FIXED).
    Wrap  = 2,  ///< Wrap at burst boundary (AXI WRAP).
};

/**
 * @brief Caller-controlled abstract attributes — common to all protocols.
 *
 * Protocol-specific fields (AXI ID, PCIe TLP tag, NoC VC) are NOT here;
 * the active transport assigns them internally from its ticket pool.
 */
struct ReqAttr {
    uint8_t qos   = 0;  ///< Priority hint (0=lowest). Mapped to AXI QOS / PCIe TC.
    uint8_t cache = 0;  ///< Cacheability hint. Mapped to AXI CACHE / PCIe attr.
    uint8_t prot  = 0;  ///< Privilege/secure hint. Mapped to AXI PROT.
};

/**
 * @brief Single burst transfer request.
 *
 * size_bytes() = beats * (1 << beat_size_log2).
 */
struct BurstReq {
    addr_t    addr            = 0;
    uint32_t  beats           = 1;   ///< 1..N (transport caps().max_beats_per_burst).
    uint8_t   beat_size_log2  = 3;   ///< 0=1B, 3=8B, 4=16B, ...
    BurstType type            = BurstType::Incr;
    ReqAttr   attr{};

    constexpr uint32_t beat_size_bytes() const {
        return 1u << beat_size_log2;
    }

    constexpr uint64_t total_bytes() const {
        return static_cast<uint64_t>(beats) * beat_size_bytes();
    }
};

}  // namespace flexnpu_sim::transport
