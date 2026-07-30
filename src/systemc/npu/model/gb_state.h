/**
 * @file gb_state.h
 * @brief Global Buffer State — circular buffer pointer model
 *
 * Models the on-chip operand buffer as up to three regions (input / weight /
 * output) with config-driven capacity, partition policy, optional sparse
 * compression, and partial-output (psum) placement. Generalizes across:
 *   - NVDLA  CBUF:  shared pool (data↔weight), psum in a separate accumulator
 *   - MIDAP  FMEM:  fixed partitions, output-stationary psum in PE-local
 *   - Nullhop SRAM: fixed separate SRAMs, compressed feature maps
 *
 * Usage:
 *   GbState gb;
 *   gb.configure(total_kb, input_kb, weight_kb, output_kb, element_bytes,
 *                GbPartitionMode::Fixed, GbOutputLocation::Gb);
 *   gb.feed_input(count);          // DMA loads input (shared-pool aware)
 *   uint32_t avail = gb.input.available();  // compute checks
 *   gb.input.consume(count);       // PE consumes
 *   gb.feed_output(count);         // PE produces output
 *   gb.output.consume(count);      // wDMA writes to DRAM
 *
 * Backpressure:
 *   gb.input.full()  → DMA read stall
 *   gb.output.full() → PE writeback stall → compute stall (output_location==Gb)
 */

#pragma once

#include <cstdint>
#include <algorithm>
#include <string>

namespace flexnpu_sim {

// GB partition policy.
//   Fixed  : each region bounded only by its own capacity (MIDAP/Nullhop:
//            physically separate SRAMs). Default — partitioning is the norm.
//   Shared : regions draw from one total_capacity pool (NVDLA CBUF: input and
//            weight dynamically share the convolution buffer).
enum class GbPartitionMode : uint8_t { Fixed, Shared };

// Where partial outputs (psum) physically live, so they are not double-counted
// against the GB pool.
//   Gb          : in the GB output region (Nullhop: separate output SRAM).
//   Accumulator : in a dedicated accumulator buffer (NVDLA CACC).
//   PeLocal     : in PE-local registers (MIDAP output-stationary).
enum class GbOutputLocation : uint8_t { Gb, Accumulator, PeLocal };

inline GbPartitionMode gb_partition_mode_from_string(const std::string& s) {
    return (s == "shared" || s == "flexible") ? GbPartitionMode::Shared
                                              : GbPartitionMode::Fixed;
}

inline GbOutputLocation gb_output_location_from_string(const std::string& s) {
    if (s == "accumulator") return GbOutputLocation::Accumulator;
    if (s == "pe_local")    return GbOutputLocation::PeLocal;
    return GbOutputLocation::Gb;
}

struct GbRegionState {
    uint32_t capacity    = 0;    // physical partition (operands). 0 = unbounded
    double   compression = 1.0;  // effective = capacity × compression (sparse FM)
    uint32_t loaded      = 0;    // amount filled by upstream (write_ptr)
    uint32_t consumed    = 0;    // amount consumed by downstream (read_ptr)

    // Effective (logical) capacity after compression. Only meaningful when
    // capacity > 0; the unbounded case is gated on capacity == 0 by callers.
    uint32_t effective_capacity() const {
        if (capacity == 0) return 0;
        double eff = static_cast<double>(capacity) * compression;
        return (eff < 1.0) ? 1u : static_cast<uint32_t>(eff);
    }

    uint32_t available() const { return loaded - consumed; }

    uint32_t free_slots() const {
        if (capacity == 0) return UINT32_MAX;  // unbounded
        uint32_t cap  = effective_capacity();
        uint32_t used = available();
        return (used < cap) ? cap - used : 0;
    }

    bool full() const {
        if (capacity == 0) return false;  // unbounded never fills
        return available() >= effective_capacity();
    }

    bool empty() const { return loaded == consumed; }

    // Upstream loads count operands; clamped so effective capacity is not exceeded.
    uint32_t feed(uint32_t count) {
        uint32_t actual = std::min(count, free_slots());
        loaded += actual;
        return actual;
    }

    // Downstream consumes count operands.
    uint32_t consume(uint32_t count) {
        uint32_t actual = std::min(count, available());
        consumed += actual;
        return actual;
    }

    void reset() { loaded = 0; consumed = 0; }
};

struct GbState {
    GbRegionState input;
    GbRegionState weight;
    GbRegionState output;

    uint32_t         total_capacity  = 0;                       // operands. 0 = unbounded
    GbPartitionMode  partition_mode  = GbPartitionMode::Fixed;  // partitioned by default
    GbOutputLocation output_location = GbOutputLocation::Gb;

    // GB-resident occupancy. Output counts toward the pool only when it
    // physically lives in the GB (output_location == Gb).
    uint32_t total_used() const {
        uint32_t u = input.available() + weight.available();
        if (output_location == GbOutputLocation::Gb) u += output.available();
        return u;
    }

    uint32_t total_free() const {
        if (total_capacity == 0) return UINT32_MAX;
        uint32_t used = total_used();
        return (used < total_capacity) ? total_capacity - used : 0;
    }

    // Shared-pool-aware feed: in Shared mode the per-region feed is additionally
    // bounded by the remaining shared pool, so input and weight contend for one
    // physical buffer (NVDLA CBUF). In Fixed mode each region is independent.
    uint32_t feed_input(uint32_t count) {
        if (partition_mode == GbPartitionMode::Shared)
            count = std::min(count, total_free());
        return input.feed(count);
    }
    uint32_t feed_weight(uint32_t count) {
        if (partition_mode == GbPartitionMode::Shared)
            count = std::min(count, total_free());
        return weight.feed(count);
    }
    uint32_t feed_output(uint32_t count) {
        if (partition_mode == GbPartitionMode::Shared &&
            output_location == GbOutputLocation::Gb)
            count = std::min(count, total_free());
        return output.feed(count);
    }

    void reset_for_layer() { input.reset(); weight.reset(); output.reset(); }
    void reset_for_tile()  { input.reset(); output.reset(); }
    // Weights can be reused across tiles, so reset_for_tile leaves them intact.

    void configure(uint32_t total_kb, uint32_t input_kb,
                   uint32_t weight_kb, uint32_t output_kb,
                   uint32_t element_bytes,
                   GbPartitionMode mode = GbPartitionMode::Fixed,
                   GbOutputLocation out_loc = GbOutputLocation::Gb,
                   double input_compression  = 1.0,
                   double weight_compression = 1.0,
                   double output_compression = 1.0) {
        uint32_t elem = (element_bytes > 0) ? element_bytes : 1;
        total_capacity   = (total_kb * 1024) / elem;
        input.capacity   = (input_kb  > 0) ? (input_kb  * 1024) / elem : 0;
        weight.capacity  = (weight_kb > 0) ? (weight_kb * 1024) / elem : 0;
        output.capacity  = (output_kb > 0) ? (output_kb * 1024) / elem : 0;
        partition_mode   = mode;
        output_location  = out_loc;
        input.compression  = (input_compression  > 0.0) ? input_compression  : 1.0;
        weight.compression = (weight_compression > 0.0) ? weight_compression : 1.0;
        output.compression = (output_compression > 0.0) ? output_compression : 1.0;
    }
};

} // namespace flexnpu_sim
