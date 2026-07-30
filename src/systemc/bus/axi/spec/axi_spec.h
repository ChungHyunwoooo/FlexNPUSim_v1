#ifndef FLEXNPU_TRANSPORT_AXI_SPEC_AXI_SPEC_H
#define FLEXNPU_TRANSPORT_AXI_SPEC_AXI_SPEC_H

// Spec<P, ID_W, ADDR_W, DATA_W, USER_W> — sole source of truth for signal
// widths and channel types. LEN/LOCK widths and HAS_* flags are inherited
// from ProtocolTraits<P>. All Master/Slave/Bus/Bind modules take Spec as
// their template parameter.

#include <systemc.h>
#include <type_traits>

#include "systemc/bus/axi/spec/axi_protocol.h"

namespace axi {

template<
    Protocol P,
    unsigned ID_W_   = 4,
    unsigned ADDR_W_ = 32,
    unsigned DATA_W_ = 128,
    unsigned USER_W_ = 0
>
struct Spec : ProtocolTraits<P> {
    using Traits = ProtocolTraits<P>;

    static constexpr unsigned ID_W   = ID_W_;
    static constexpr unsigned ADDR_W = ADDR_W_;
    static constexpr unsigned DATA_W = DATA_W_;
    static constexpr unsigned STRB_W = DATA_W_ / 8;
    static constexpr unsigned USER_W = USER_W_;

    static constexpr unsigned LEN_W    = Traits::LEN_W;
    static constexpr unsigned LOCK_W   = Traits::LOCK_W;
    static constexpr unsigned SIZE_W   = 3;
    static constexpr unsigned BURST_W  = 2;
    static constexpr unsigned CACHE_W  = 4;
    static constexpr unsigned PROT_W   = 3;
    static constexpr unsigned QOS_W    = 4;
    static constexpr unsigned REGION_W = 4;
    static constexpr unsigned RESP_W   = 2;

    static constexpr bool HAS_WID    = Traits::HAS_WID;
    static constexpr bool HAS_QOS    = Traits::HAS_QOS;
    static constexpr bool HAS_REGION = Traits::HAS_REGION;

    using id_t       = sc_uint<ID_W>;
    using addr_t     = sc_uint<ADDR_W>;
    using data_t     = std::conditional_t<(DATA_W <= 64), sc_uint<DATA_W>, sc_biguint<DATA_W>>;
    using strb_t     = std::conditional_t<(STRB_W <= 64), sc_uint<STRB_W>, sc_biguint<STRB_W>>;
    using len_t      = sc_uint<LEN_W>;
    using axi_size_t = sc_uint<SIZE_W>;
    using burst_t    = sc_uint<BURST_W>;
    using lock_t     = sc_uint<LOCK_W>;
    using cache_t    = sc_uint<CACHE_W>;
    using prot_t     = sc_uint<PROT_W>;
    using qos_t      = sc_uint<QOS_W>;
    using region_t   = sc_uint<REGION_W>;
    using resp_t     = sc_uint<RESP_W>;
};

using Axi3Spec_Default = Spec<Protocol::AXI3, 4, 32, 32, 0>;
using Axi4Spec_Default = Spec<Protocol::AXI4, 4, 32, 32, 0>;

} // namespace axi

#endif // FLEXNPU_TRANSPORT_AXI_SPEC_AXI_SPEC_H
