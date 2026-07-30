#ifndef AXI_MEMORY_MAP_H
#define AXI_MEMORY_MAP_H

#include <vector>
#include <initializer_list>
#include <string>

/**
 * Memory Region Descriptor
 */
template<typename ADDR_TYPE>
struct MemoryRegion {
    ADDR_TYPE base_addr;
    ADDR_TYPE addr_size;
    int slave_id;
    std::string name;

    MemoryRegion() : base_addr(0), addr_size(0), slave_id(-1), name("") {}

    MemoryRegion(ADDR_TYPE base, ADDR_TYPE size, int id, const std::string& n = "")
        : base_addr(base), addr_size(size), slave_id(id), name(n) {}

    bool contains(ADDR_TYPE addr) const {
        return (addr >= base_addr) && (addr < base_addr + addr_size);
    }
};

/**
 * Dynamic Memory Map (runtime configurable)
 */
template<typename ADDR_TYPE>
class DynamicMemoryMap {
public:
    using Region = MemoryRegion<ADDR_TYPE>;

    DynamicMemoryMap() = default;

    DynamicMemoryMap(std::initializer_list<Region> regions) : regions_(regions) {}

    void add_region(const Region& region) {
        regions_.push_back(region);
    }

    int find_slave_id(ADDR_TYPE addr) const {
        for (const auto& region : regions_) {
            if (region.contains(addr)) {
                return region.slave_id;
            }
        }
        return -1;  // Decode error
    }

    const Region& get_region(size_t idx) const {
        return regions_[idx];
    }

    size_t size() const { return regions_.size(); }

private:
    std::vector<Region> regions_;
};

#endif // AXI_MEMORY_MAP_H
