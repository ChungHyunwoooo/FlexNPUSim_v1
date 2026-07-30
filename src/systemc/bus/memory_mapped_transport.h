/**
 * @file memory_mapped_transport.h
 * @brief Runtime-polymorphic interface for AXI/PCIe-class memory-mapped buses.
 *
 * Callers (DmaEngine, Processor, NpuController) hold a MemoryMappedTransport*
 * and never see protocol specifics. Concrete transports (AxiTransport,
 * PcieTransport, ...) implement issue/wait + capabilities; per-protocol
 * detail (AXI ID assignment, TLP tagging) lives entirely inside.
 *
 * Pattern — multi-outstanding via issue+wait:
 *   std::vector<Ticket> in_flight;
 *   for (const auto& burst : plan) {
 *       in_flight.push_back(transport.issue_read(burst));
 *   }
 *   for (auto t : in_flight) {
 *       auto res = transport.wait_read(t);
 *       copy_to_l2(res.data, res.bytes);
 *   }
 *
 * Single-outstanding convenience:
 *   auto res = transport.read_burst(burst);
 */

#pragma once

#include "systemc/bus/burst_req.h"
#include "systemc/bus/capabilities.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace flexnpu_sim::transport {

/**
 * @brief Opaque ticket — handle to an in-flight transfer.
 *
 * Returned by issue_read/issue_write, consumed by wait_read/wait_write.
 * The transport interprets `id` per its own scoreboard (slot index, AXI ID,
 * PCIe tag — implementation choice). `valid()` distinguishes from a default-
 * constructed empty ticket.
 */
struct Ticket {
    uint32_t id = 0xFFFFFFFFu;

    constexpr bool valid() const { return id != 0xFFFFFFFFu; }

    static constexpr Ticket invalid() { return Ticket{}; }
};

/**
 * @brief Response code — protocol responses normalized.
 */
enum class TransportResp : uint8_t {
    Ok           = 0,
    SlaveError   = 1,  ///< AXI SLVERR / PCIe completer abort.
    DecodeError  = 2,  ///< AXI DECERR / PCIe unsupported request.
    ExclusiveOk  = 3,
    Other        = 0xFF,
};

/**
 * @brief Read result — bytes returned by the slave.
 *
 * `data` is owned by the transport; it remains valid until the caller's next
 * wait_read on a different ticket OR until the caller copies it out. Concrete
 * transports may reuse a per-ticket buffer.
 */
struct ReadResult {
    const uint8_t* data  = nullptr;
    uint64_t       bytes = 0;
    TransportResp  resp  = TransportResp::Ok;
};

/**
 * @brief Write result — completion status.
 */
struct WriteResult {
    TransportResp  resp  = TransportResp::Ok;
};

class MemoryMappedTransport {
public:
    virtual ~MemoryMappedTransport() = default;

    /// Non-blocking issue. Returns a ticket; blocks (sc_thread suspend) only
    /// when the ticket pool is saturated (caller stall = backpressure).
    virtual Ticket issue_read (const BurstReq& req) = 0;
    virtual Ticket issue_write(const BurstReq& req,
                               const std::vector<uint8_t>& data,
                               const std::vector<uint8_t>& strobe = {}) = 0;

    /// Blocking wait on a specific ticket — returns when that transfer
    /// completes. Releases the ticket back to the pool.
    virtual ReadResult  wait_read (Ticket tk) = 0;
    virtual WriteResult wait_write(Ticket tk) = 0;

    /// Convenience — single-outstanding (issue + immediate wait).
    ReadResult read_burst(const BurstReq& req) {
        return wait_read(issue_read(req));
    }
    WriteResult write_burst(const BurstReq& req,
                            const std::vector<uint8_t>& data,
                            const std::vector<uint8_t>& strobe = {}) {
        return wait_write(issue_write(req, data, strobe));
    }

    /// Protocol/configuration limits the caller must respect.
    virtual const Capabilities& caps() const = 0;
};

}  // namespace flexnpu_sim::transport
