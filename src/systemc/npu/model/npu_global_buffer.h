/**
 * @file npu_global_buffer.h
 * @brief The NPU global buffer (GB): byte storage + a selectable presence model.
 *
 * One physical buffer, two selectable operating modes (real designs pick one):
 *   Scratchpad — software-managed staging. Presence is an explicit ledger of
 *                marked ranges ("was this region staged into the GB?").
 *   Cache      — direct-mapped, line-granular tag/valid model. Resident lines
 *                can serve a read without a DRAM round-trip (load), and stores
 *                populate lines from the buffer.
 *
 * Replaces the ad-hoc "l2_*" members that used to live inside NpuController.
 * Pure C++ (no SystemC) so it is unit-testable in isolation.
 */

#pragma once

#include <cstdint>
#include <vector>

namespace flexnpu_sim {

class NpuGlobalBuffer {
public:
    enum class Mode : uint8_t { Scratchpad = 0, Cache = 1 };

    NpuGlobalBuffer() = default;
    explicit NpuGlobalBuffer(uint32_t size_bytes) { resize(size_bytes); }

    /// (Re)allocate the backing storage to size_bytes, zero-filled.
    void resize(uint32_t size_bytes);

    uint32_t       size() const { return size_; }
    uint8_t*       data()       { return storage_.data(); }
    const uint8_t* data() const { return storage_.data(); }

    Mode mode() const { return mode_; }
    /// Select the operating mode. Clears presence; builds cache metadata for Cache.
    void set_mode(Mode m);
    /// Cache line size in bytes (rounded up to a power of two, min 16).
    void set_cache_line_size(uint32_t bytes);

    /// Drop all presence state (region ledger + cache lines + hit/miss counts).
    void clear_presence();

    /// Reset presence for a new layer: clear the ledger and, in Cache mode,
    /// rebuild empty (invalid) cache lines so stores can repopulate them.
    void reset_presence();

    // ---- Scratchpad mode ----
    bool scratchpad_has_range(uint32_t addr, uint32_t size) const;
    void scratchpad_mark_range(uint32_t addr, uint32_t size);

    // ---- Cache mode ----
    /// True iff every line spanning [addr, addr+size) is resident (valid+tag).
    bool cache_range_hit(uint32_t addr, uint32_t size) const;
    /// On a hit, copy the resident bytes into storage_[buf_off..]; returns hit.
    /// Counts one hit (return true) or one miss (return false).
    bool cache_load_range(uint32_t addr, uint32_t size, uint32_t buf_off);
    /// Populate the addressed lines from storage_[buf_off..] (write-allocate).
    void cache_store_range(uint32_t addr, uint32_t buf_off, uint32_t size);

    uint64_t cache_hits() const   { return cache_hit_count_; }
    uint64_t cache_misses() const { return cache_miss_count_; }

private:
    void init_cache_metadata();
    bool cache_load_range_impl(uint32_t addr, uint32_t size, uint32_t buf_off);

    struct Region { uint32_t addr = 0, size = 0; };
    struct CacheLine { bool valid = false; uint32_t tag = 0; std::vector<uint8_t> bytes; };

    std::vector<uint8_t>   storage_;
    uint32_t               size_ = 0;
    Mode                   mode_ = Mode::Scratchpad;
    uint32_t               cache_line_bytes_ = 64;
    std::vector<Region>    regions_;
    std::vector<CacheLine> lines_;
    uint64_t               cache_hit_count_  = 0;
    uint64_t               cache_miss_count_ = 0;
};

}  // namespace flexnpu_sim
