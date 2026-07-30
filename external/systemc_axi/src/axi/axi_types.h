#ifndef AXI_TYPES_H
#define AXI_TYPES_H

#include <systemc.h>

/**
 * AXI4 Burst Types
 */
enum class AXIBurstType : uint8_t {
    FIXED = 0,  // Fixed address burst
    INCR = 1,   // Incrementing address burst
    WRAP = 2,   // Wrapping address burst
    RESERVED = 3
};

/**
 * AXI4 Response Types
 */
enum class AXIResponse : uint8_t {
    OKAY = 0,    // Normal access success
    EXOKAY = 1,  // Exclusive access okay
    SLVERR = 2,  // Slave error
    DECERR = 3   // Decode error
};

/**
 * AXI Lock Types (AXI3 compatibility)
 */
enum class AXILockType : uint8_t {
    NORMAL = 0,     // Normal access
    EXCLUSIVE = 1,  // Exclusive access
    LOCKED = 2      // Locked access (AXI3 only)
};

/**
 * AXI4 Cache Attributes (simplified)
 */
enum class AXICacheType : uint8_t {
    NON_CACHEABLE_NON_BUFFERABLE = 0b0000,
    BUFFERABLE = 0b0001,
    CACHEABLE = 0b0010,
    CACHEABLE_BUFFERABLE = 0b0011
};

#endif // AXI_TYPES_H
