#ifndef FLEXNPU_TRANSPORT_AXI_SPEC_AXI_PROTOCOL_H
#define FLEXNPU_TRANSPORT_AXI_SPEC_AXI_PROTOCOL_H

// Protocol enum and per-protocol traits for AXI3/AXI4.
// Traits carry width and feature flags that differ by protocol; signal/port
// widths that do not differ are defined in Spec<>.

namespace axi {

enum class Protocol {
    AXI3,
    AXI4,
};

template<Protocol P>
struct ProtocolTraits;

template<>
struct ProtocolTraits<Protocol::AXI3> {
    static constexpr Protocol PROTOCOL      = Protocol::AXI3;
    static constexpr unsigned LEN_W         = 4;
    static constexpr unsigned LOCK_W        = 2;
    static constexpr bool     HAS_WID       = true;
    static constexpr bool     HAS_QOS       = false;
    static constexpr bool     HAS_REGION    = false;
    static constexpr unsigned MAX_BURST_LEN = 16;
};

template<>
struct ProtocolTraits<Protocol::AXI4> {
    static constexpr Protocol PROTOCOL      = Protocol::AXI4;
    static constexpr unsigned LEN_W         = 8;
    static constexpr unsigned LOCK_W        = 1;
    static constexpr bool     HAS_WID       = false;
    static constexpr bool     HAS_QOS       = true;
    static constexpr bool     HAS_REGION    = true;
    static constexpr unsigned MAX_BURST_LEN = 256;
};

} // namespace axi

#endif // FLEXNPU_TRANSPORT_AXI_SPEC_AXI_PROTOCOL_H
