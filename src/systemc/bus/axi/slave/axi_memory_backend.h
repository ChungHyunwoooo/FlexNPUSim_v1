#ifndef FLEXNPU_TRANSPORT_AXI_SLAVE_AXI_MEMORY_BACKEND_H
#define FLEXNPU_TRANSPORT_AXI_SLAVE_AXI_MEMORY_BACKEND_H

// Byte-addressable storage abstraction used by the signal-level AXI slave.
// Pure C++ (no SystemC): testable in unit scope, swappable for dense/sparse/
// external (DRAMSim3, memory-mapped I/O) backends.

#include <cstdint>
#include <map>
#include <vector>

namespace flexnpu_sim::transport::axi {

class MemoryBackend {
public:
    virtual ~MemoryBackend() = default;

    /// Returns `bytes` bytes starting at @p addr. Unmapped bytes read as zero.
    virtual std::vector<uint8_t> read(uint64_t addr, unsigned bytes) = 0;

    /// Writes under byte-strobe @p enable. `bytes.size()` must equal
    /// `enable.size()`.
    virtual void write(uint64_t addr,
                       const std::vector<uint8_t>& bytes,
                       const std::vector<bool>& enable) = 0;
};

/// Dense contiguous RAM anchored at a base address. Out-of-range accesses
/// read as zero and silently drop writes. Intended for DRAM-scale models
/// (preloaded images, weight tensors) where sparse-map overhead is
/// prohibitive.
class DenseRamBackend : public MemoryBackend {
public:
    DenseRamBackend(uint64_t base_addr, std::size_t size_bytes)
        : base_(base_addr), data_(size_bytes, 0) {}

    std::vector<uint8_t> read(uint64_t addr, unsigned bytes) override {
        std::vector<uint8_t> out(bytes, 0);
        for (unsigned i = 0; i < bytes; ++i) {
            const uint64_t a = addr + i;
            if (a >= base_ && a < base_ + data_.size()) {
                out[i] = data_[a - base_];
            }
        }
        return out;
    }

    void write(uint64_t addr,
               const std::vector<uint8_t>& bytes,
               const std::vector<bool>& enable) override {
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            const bool on = (i < enable.size()) ? enable[i] : true;
            if (!on) continue;
            const uint64_t a = addr + i;
            if (a >= base_ && a < base_ + data_.size()) {
                data_[a - base_] = bytes[i];
            }
        }
    }

          std::vector<uint8_t>& data()       { return data_; }
    const std::vector<uint8_t>& data() const { return data_; }
    uint64_t base_addr() const { return base_; }

private:
    uint64_t             base_;
    std::vector<uint8_t> data_;
};

/// Sparse byte-keyed RAM. Unwritten bytes read as zero. Intended for tests
/// and small-footprint slaves; use a dense backend for whole-DRAM models.
class RamBackend : public MemoryBackend {
public:
    std::vector<uint8_t> read(uint64_t addr, unsigned bytes) override {
        std::vector<uint8_t> out(bytes, 0);
        for (unsigned i = 0; i < bytes; ++i) {
            auto it = mem_.find(addr + i);
            if (it != mem_.end()) out[i] = it->second;
        }
        return out;
    }

    void write(uint64_t addr,
               const std::vector<uint8_t>& bytes,
               const std::vector<bool>& enable) override {
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            const bool on = (i < enable.size()) ? enable[i] : true;
            if (on) mem_[addr + i] = bytes[i];
        }
    }

          std::map<uint64_t, uint8_t>& storage()       { return mem_; }
    const std::map<uint64_t, uint8_t>& storage() const { return mem_; }

private:
    std::map<uint64_t, uint8_t> mem_;
};

}  // namespace flexnpu_sim::transport::axi

#endif  // FLEXNPU_TRANSPORT_AXI_SLAVE_AXI_MEMORY_BACKEND_H
