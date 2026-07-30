#ifndef FLEXNPU_TRANSPORT_AXI_AXI_CONFIG_H
#define FLEXNPU_TRANSPORT_AXI_AXI_CONFIG_H

// Pure C++ runtime knobs for AXI Master/Slave and legacy type aliases.
//
// The compile-time signal widths and channel types live in
// transport/axi/spec/axi_spec.h as axi::Spec<P, ID_W, ADDR_W, DATA_W, USER_W>.
// This header only carries runtime knobs (outstanding limits, delays, etc.)
// and backward-compatibility name aliases for the previous AXIConfig<> type.

#include "systemc/bus/axi/spec/axi_spec.h"

namespace axi {

// Runtime knobs that do not affect signal widths.
struct CommonConfig {
    unsigned max_outstanding_reads  = 8;
    unsigned max_outstanding_writes = 4;
    unsigned default_burst_len      = 16;   // beats
    bool     randomize_ready        = false;
    int      slave_read_delay_ns    = 10;
    int      slave_write_delay_ns   = 10;
    int      master_timeout_ns      = 1'000'000;
};

struct Axi3Config {
    bool allow_write_data_interleave = false; // AXI3 only; AXI4 forbids
};

struct Axi4Config {
    bool drive_qos    = false;
    bool drive_region = false;
};

} // namespace axi

// ============================================================================
// Backward-compatibility type aliases (global namespace)
//
// Consumers that still spell `AXIConfig<ID, ADDR, DATA>` or use `AXI4_Default`
// / `AXI3_Default` / `AXI_Legacy` resolve to equivalent axi::Spec<> instances.
// Spec<> already exposes id_t/addr_t/data_t/... and HAS_WID/HAS_QOS/HAS_REGION,
// so all uses that only read these traits compile unchanged.
//
// Full 9-parameter AXIConfig<> (explicit LEN_W/LOCK_W/ENABLE_WID/...) is no
// longer supported — Spec<> derives those from Protocol. Migrate call sites
// to pick Protocol::AXI3 or Protocol::AXI4 explicitly.
// ============================================================================
template<
    unsigned int ID_WIDTH   = 4,
    unsigned int ADDR_WIDTH = 32,
    unsigned int DATA_WIDTH = 32,
    unsigned int USER_WIDTH = 0
>
using AXIConfig = axi::Spec<axi::Protocol::AXI4, ID_WIDTH, ADDR_WIDTH, DATA_WIDTH, USER_WIDTH>;

using AXI4_Default = axi::Spec<axi::Protocol::AXI4, 4, 32, 32, 0>;
using AXI3_Default = axi::Spec<axi::Protocol::AXI3, 4, 32, 32, 0>;
using AXI_Legacy   = axi::Spec<axi::Protocol::AXI3, 2, 32, 32, 0>;

// Coarse slave-side pacing constants reused by legacy memory_slave wrappers.
struct AXITimingConfig {
    static constexpr int SLAVE_WRITE_DELAY = 10;
    static constexpr int SLAVE_READ_DELAY  = 10;
};

#endif // FLEXNPU_TRANSPORT_AXI_AXI_CONFIG_H
