#ifndef AXI_EXCLUSIVE_MONITOR_H
#define AXI_EXCLUSIVE_MONITOR_H

#include "systemc/bus/axi/axi_config.h"
#include "axi_types.h"
#include <map>
#include <vector>

/**
 * AXI Exclusive Access Monitor
 *
 * Tracks exclusive read/write sequences and generates appropriate responses:
 * - EXOKAY: Exclusive access succeeded
 * - OKAY: Normal access or exclusive access failed
 */
template<typename CFG = AXI4_Default>
class AXIExclusiveMonitor {
public:
    using addr_t = typename CFG::addr_t;
    using id_t = typename CFG::id_t;

    struct ExclusiveReservation {
        int master_id;           // Which master made the reservation
        addr_t start_addr;       // Start of exclusive region
        addr_t end_addr;         // End of exclusive region (start + size)
        bool valid;              // Is this reservation still valid?

        ExclusiveReservation() : master_id(-1), start_addr(0), end_addr(0), valid(false) {}

        ExclusiveReservation(int mid, addr_t start, addr_t end)
            : master_id(mid), start_addr(start), end_addr(end), valid(true) {}
    };

    std::multimap<int, ExclusiveReservation> reservations;  // Multiple per master

    AXIExclusiveMonitor() = default;

    /**
     * Record exclusive read
     * Called when ARLOCK indicates exclusive access
     */
    void record_exclusive_read(int master_id, addr_t addr, unsigned int size_bytes) {
        addr_t start = addr;
        addr_t end = addr + size_bytes;

        // Insert new reservation (doesn't overwrite existing ones)
        reservations.insert({master_id, ExclusiveReservation(master_id, start, end)});

#ifdef VERBOSE
        std::cout << "[ExMon] Master " << master_id
                  << " exclusive read: 0x" << std::hex << addr << std::dec
                  << " (size=" << size_bytes << ")" << std::endl;
#endif
    }

    /**
     * Check and clear exclusive write
     * Returns true if exclusive access should succeed (EXOKAY)
     */
    bool check_exclusive_write(int master_id, addr_t addr, unsigned int size_bytes) {
        addr_t write_start = addr;
        addr_t write_end = addr + size_bytes;

        // Find all reservations for this master
        auto range = reservations.equal_range(master_id);
        for (auto it = range.first; it != range.second; ++it) {
            auto& reservation = it->second;
            if (!reservation.valid) continue;

            // Check if write address matches this reservation
            bool matches = (write_start >= reservation.start_addr) &&
                          (write_end <= reservation.end_addr);

            if (matches) {
                // Success - remove this reservation and return EXOKAY
#ifdef VERBOSE
                std::cout << "[ExMon] Master " << master_id
                          << " exclusive write SUCCESS: 0x" << std::hex << addr << std::dec
                          << " → EXOKAY" << std::endl;
#endif
                reservations.erase(it);
                return true;  // EXOKAY
            }
        }

        // No matching reservation found - return OKAY (failed)
#ifdef VERBOSE
        std::cout << "[ExMon] Master " << master_id
                  << " exclusive write FAILED: no matching reservation → OKAY" << std::endl;
#endif
        return false;  // OKAY
    }

    /**
     * Clear all exclusive reservations for a master
     * Called when another master accesses the same address
     */
    void clear_reservation(int master_id) {
        reservations.erase(master_id);
    }

    /**
     * Monitor normal (non-exclusive) access
     * Clears any overlapping exclusive reservations from ALL masters (including own)
     */
    void monitor_normal_access(int accessing_master, addr_t addr, unsigned int size_bytes) {
        addr_t access_start = addr;
        addr_t access_end = addr + size_bytes;

        // Iterate over all reservations and clear overlapping ones
        for (auto it = reservations.begin(); it != reservations.end(); ) {
            auto& res = it->second;
            if (!res.valid) {
                it = reservations.erase(it);
                continue;
            }

            // Check for overlap
            bool overlaps = !(access_end <= res.start_addr || access_start >= res.end_addr);

            if (overlaps) {
#ifdef VERBOSE
                std::cout << "[ExMon] Master " << it->first
                          << " exclusive reservation cleared by Master " << accessing_master
                          << " access at 0x" << std::hex << addr << std::dec << std::endl;
#endif
                it = reservations.erase(it);
            } else {
                ++it;
            }
        }
    }

    /**
     * Get response type based on lock type
     */
    AXIResponse get_read_response(int master_id, addr_t addr, unsigned int size_bytes,
                                   typename CFG::lock_t lock) {
        // For exclusive reads, always return OKAY (exclusive is only for writes)
        // The reservation is created, but response is OKAY
        return AXIResponse::OKAY;
    }

    /**
     * Get write response based on lock type and reservation
     */
    AXIResponse get_write_response(int master_id, addr_t addr, unsigned int size_bytes,
                                    typename CFG::lock_t lock) {
        if (lock == 1) {  // Exclusive write
            bool success = check_exclusive_write(master_id, addr, size_bytes);
            return success ? AXIResponse::EXOKAY : AXIResponse::OKAY;
        } else {
            // Normal write - monitor for clearing other reservations
            monitor_normal_access(master_id, addr, size_bytes);
            return AXIResponse::OKAY;
        }
    }
};

/**
 * Empty monitor (zero overhead when disabled)
 */
struct EmptyExclusiveMonitor {
    template<typename... Args> void record_exclusive_read(Args...) {}
    template<typename... Args> bool check_exclusive_write(Args...) { return false; }
    template<typename... Args> void clear_reservation(Args...) {}
    template<typename... Args> void monitor_normal_access(Args...) {}
    template<typename... Args>
    AXIResponse get_read_response(Args...) { return AXIResponse::OKAY; }
    template<typename... Args>
    AXIResponse get_write_response(Args...) { return AXIResponse::OKAY; }
};

#endif // AXI_EXCLUSIVE_MONITOR_H
