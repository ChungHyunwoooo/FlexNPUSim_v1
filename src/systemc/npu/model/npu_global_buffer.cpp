#include "systemc/npu/model/npu_global_buffer.h"

#include <algorithm>
#include <cstring>

namespace flexnpu_sim {

void NpuGlobalBuffer::resize(uint32_t size_bytes) {
    size_ = size_bytes;
    storage_.assign(size_bytes, 0);
    if (mode_ == Mode::Cache) init_cache_metadata();
}

void NpuGlobalBuffer::init_cache_metadata() {
    uint32_t line = std::max<uint32_t>(16, cache_line_bytes_);
    if ((line & (line - 1)) != 0) {
        uint32_t p2 = 1;
        while (p2 < line) p2 <<= 1;
        line = p2;
    }
    cache_line_bytes_ = line;
    uint32_t num_lines = std::max<uint32_t>(1, size_ / cache_line_bytes_);
    lines_.assign(num_lines, CacheLine{});
    for (auto& cl : lines_) {
        cl.bytes.assign(cache_line_bytes_, 0);
    }
}

void NpuGlobalBuffer::set_mode(Mode m) {
    mode_ = m;
    clear_presence();
    if (mode_ == Mode::Cache) {
        init_cache_metadata();
    }
}

void NpuGlobalBuffer::set_cache_line_size(uint32_t bytes) {
    if (bytes == 0) bytes = 64;
    cache_line_bytes_ = bytes;
    if (mode_ == Mode::Cache) {
        init_cache_metadata();
    }
}

void NpuGlobalBuffer::clear_presence() {
    regions_.clear();
    lines_.clear();
    cache_hit_count_ = 0;
    cache_miss_count_ = 0;
}

void NpuGlobalBuffer::reset_presence() {
    clear_presence();
    if (mode_ == Mode::Cache) init_cache_metadata();
}

bool NpuGlobalBuffer::scratchpad_has_range(uint32_t addr, uint32_t size) const {
    if (size == 0) return true;
    uint64_t a0 = addr;
    uint64_t a1 = a0 + size;
    for (const auto& r : regions_) {
        uint64_t r0 = r.addr;
        uint64_t r1 = r0 + r.size;
        if (a0 >= r0 && a1 <= r1) return true;
    }
    return false;
}

void NpuGlobalBuffer::scratchpad_mark_range(uint32_t addr, uint32_t size) {
    if (size == 0) return;
    regions_.push_back({addr, size});
    if (regions_.size() > 4096) {
        regions_.erase(regions_.begin(), regions_.begin() + 2048);
    }
}

bool NpuGlobalBuffer::cache_range_hit(uint32_t addr, uint32_t size) const {
    if (size == 0) return true;
    if (lines_.empty()) return false;
    uint64_t line = cache_line_bytes_;
    uint64_t first = static_cast<uint64_t>(addr) / line;
    uint64_t last = (static_cast<uint64_t>(addr) + size - 1) / line;
    uint64_t n = lines_.size();
    for (uint64_t la = first; la <= last; ++la) {
        uint64_t idx = la % n;
        uint32_t tag = static_cast<uint32_t>(la / n);
        const auto& cl = lines_[idx];
        if (!cl.valid || cl.tag != tag) return false;
    }
    return true;
}

bool NpuGlobalBuffer::cache_load_range(uint32_t addr, uint32_t size, uint32_t buf_off) {
    const bool hit = cache_load_range_impl(addr, size, buf_off);
    (hit ? cache_hit_count_ : cache_miss_count_)++;
    return hit;
}

bool NpuGlobalBuffer::cache_load_range_impl(uint32_t addr, uint32_t size, uint32_t buf_off) {
    if (size == 0) return true;
    if (!cache_range_hit(addr, size)) return false;
    if (lines_.empty()) return false;

    uint64_t end = static_cast<uint64_t>(addr) + size;
    uint64_t cur = addr;
    uint32_t dst = buf_off;
    uint64_t n = lines_.size();

    while (cur < end) {
        uint64_t line_addr = cur / cache_line_bytes_;
        uint64_t idx = line_addr % n;
        uint32_t tag = static_cast<uint32_t>(line_addr / n);
        const auto& cl = lines_[idx];
        if (!cl.valid || cl.tag != tag) return false;

        uint32_t offset = static_cast<uint32_t>(cur % cache_line_bytes_);
        uint32_t chunk = static_cast<uint32_t>(
            std::min<uint64_t>(end - cur, cache_line_bytes_ - offset));
        std::memcpy(storage_.data() + dst,
                    cl.bytes.data() + offset,
                    chunk);
        cur += chunk;
        dst += chunk;
    }
    return true;
}

void NpuGlobalBuffer::cache_store_range(uint32_t addr, uint32_t buf_off, uint32_t size) {
    if (size == 0 || lines_.empty()) return;
    const uint8_t* src = storage_.data() + buf_off;

    uint64_t end = static_cast<uint64_t>(addr) + size;
    uint64_t cur = addr;
    uint32_t src_off = 0;
    uint64_t n = lines_.size();

    while (cur < end) {
        uint64_t line_addr = cur / cache_line_bytes_;
        uint64_t idx = line_addr % n;
        uint32_t tag = static_cast<uint32_t>(line_addr / n);
        auto& cl = lines_[idx];
        if (cl.bytes.size() != cache_line_bytes_) {
            cl.bytes.assign(cache_line_bytes_, 0);
        }
        cl.valid = true;
        cl.tag = tag;

        uint32_t offset = static_cast<uint32_t>(cur % cache_line_bytes_);
        uint32_t chunk = static_cast<uint32_t>(
            std::min<uint64_t>(end - cur, cache_line_bytes_ - offset));
        std::memcpy(cl.bytes.data() + offset, src + src_off, chunk);
        cur += chunk;
        src_off += chunk;
    }
}

}  // namespace flexnpu_sim
