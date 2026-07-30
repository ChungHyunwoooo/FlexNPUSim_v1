/**
 * @file dram_timing.h
 * @brief Clock-driven DRAM timing model — a memory-controller front-end.
 *
 * This is one of the three places the simulator allows an approximate model
 * (the others: the latency model and the processor). Everything the AXI slave
 * and DMA do around it is real hardware (ready/valid + occupancy); the DRAM
 * model's INTERNALS may approximate, but its INTERFACE is the real hardware
 * one: accept (backpressure) / submit / tick / completion.
 *
 * The AXI slave drives it exactly like a CPU drives a DRAM controller:
 *   - will_accept(): can another transaction enter? (false = backpressure)
 *   - submit():      hand over an accepted transaction, tagged with a token.
 *   - tick():        advance one system cycle.
 *   - pop_completed(): drain finished transactions (returns their token).
 *
 * Concurrency (bank parallelism) and backpressure emerge from the model's own
 * occupancy — there is NO closed-form completion formula or min() outside it.
 * The shared read-data bus is serialized by the AXI R channel, not here.
 *
 * Four tiers, increasing fidelity: ideal / bandwidth / bank share the
 * analytical bank-occupancy engine (dram/analytical); cycle wraps DRAMSim3
 * driven loaded (dram/cycle).
 */

#pragma once

#include <cstdint>
#include <string>

namespace flexnpu_sim {

class DramTiming {
public:
    virtual ~DramTiming() = default;

    /// Backpressure: can the model accept a new transaction this cycle?
    /// False stalls the AXI accept handshake (real ready/valid).
    virtual bool will_accept(uint64_t address, bool is_write) const = 0;

    /// Hand over an accepted transaction. `id` is an opaque token the caller
    /// matches against pop_completed(). Precondition: will_accept() was true.
    virtual void submit(uint64_t id, uint64_t address, uint32_t size,
                        bool is_write) = 0;

    /// Advance one system cycle.
    virtual void tick() = 0;

    /// Drain one finished transaction; sets `id` to its submit() token and
    /// returns true, or returns false when none are ready this cycle.
    virtual bool pop_completed(uint64_t& id) = 0;

    /// Reset internal state between runs.
    virtual void reset() {}

    /// Human-readable model name (matches the tier key).
    virtual std::string name() const = 0;
};

}  // namespace flexnpu_sim
