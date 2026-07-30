#ifndef FLEXNPU_TRANSPORT_AXI_AXI_SIGNAL_PORTS_H
#define FLEXNPU_TRANSPORT_AXI_AXI_SIGNAL_PORTS_H

#include "systemc/bus/axi/axi_config.h"

// Empty signal/port placeholders (used for AXI4 WID which does not exist).
struct EmptySignal {
    void write(int) {}
    int read() const { return 0; }
};

struct EmptyPort {};

/**
 * AXI Signal Container (Templated on Spec).
 *
 * CFG must expose id_t/addr_t/data_t/strb_t/len_t/axi_size_t/burst_t/
 * lock_t/cache_t/prot_t/qos_t/region_t/resp_t and the HAS_WID flag.
 * Both axi::Spec<...> and the legacy AXIConfig<> alias satisfy this.
 */
template<typename CFG = AXI4_Default>
class AXI_SIGNALS {
public:
    using id_type     = typename CFG::id_t;
    using addr_type   = typename CFG::addr_t;
    using data_type   = typename CFG::data_t;
    using strb_type   = typename CFG::strb_t;
    using len_type    = typename CFG::len_t;
    using size_type   = typename CFG::axi_size_t;
    using burst_type  = typename CFG::burst_t;
    using lock_type   = typename CFG::lock_t;
    using cache_type  = typename CFG::cache_t;
    using prot_type   = typename CFG::prot_t;
    using qos_type    = typename CFG::qos_t;
    using region_type = typename CFG::region_t;
    using resp_type   = typename CFG::resp_t;

    // AR channel
    sc_signal<id_type>     ARID;
    sc_signal<addr_type>   ARADDR;
    sc_signal<len_type>    ARLEN;
    sc_signal<size_type>   ARSIZE;
    sc_signal<burst_type>  ARBURST;
    sc_signal<lock_type>   ARLOCK;
    sc_signal<cache_type>  ARCACHE;
    sc_signal<prot_type>   ARPROT;
    sc_signal<qos_type>    ARQOS;
    sc_signal<region_type> ARREGION;
    sc_signal<bool>        ARVALID;
    sc_signal<bool>        ARREADY;

    // R channel
    sc_signal<id_type>     RID;
    sc_signal<data_type>   RDATA;
    sc_signal<resp_type>   RRESP;
    sc_signal<bool>        RLAST;
    sc_signal<bool>        RVALID;
    sc_signal<bool>        RREADY;

    // AW channel
    sc_signal<id_type>     AWID;
    sc_signal<addr_type>   AWADDR;
    sc_signal<len_type>    AWLEN;
    sc_signal<size_type>   AWSIZE;
    sc_signal<burst_type>  AWBURST;
    sc_signal<lock_type>   AWLOCK;
    sc_signal<cache_type>  AWCACHE;
    sc_signal<prot_type>   AWPROT;
    sc_signal<qos_type>    AWQOS;
    sc_signal<region_type> AWREGION;
    sc_signal<bool>        AWVALID;
    sc_signal<bool>        AWREADY;

    // W channel (conditional WID)
    typename std::conditional<CFG::HAS_WID, sc_signal<id_type>, EmptySignal>::type WID;
    sc_signal<data_type>   WDATA;
    sc_signal<strb_type>   WSTRB;
    sc_signal<bool>        WLAST;
    sc_signal<bool>        WVALID;
    sc_signal<bool>        WREADY;

    // B channel
    sc_signal<id_type>     BID;
    sc_signal<resp_type>   BRESP;
    sc_signal<bool>        BVALID;
    sc_signal<bool>        BREADY;

    AXI_SIGNALS() = default;
};

/**
 * Master-side port container.
 */
template<typename CFG = AXI4_Default>
class MASTER_PORTS {
public:
    using id_type     = typename CFG::id_t;
    using addr_type   = typename CFG::addr_t;
    using data_type   = typename CFG::data_t;
    using strb_type   = typename CFG::strb_t;
    using len_type    = typename CFG::len_t;
    using size_type   = typename CFG::axi_size_t;
    using burst_type  = typename CFG::burst_t;
    using lock_type   = typename CFG::lock_t;
    using cache_type  = typename CFG::cache_t;
    using prot_type   = typename CFG::prot_t;
    using qos_type    = typename CFG::qos_t;
    using region_type = typename CFG::region_t;
    using resp_type   = typename CFG::resp_t;

    // AR channel
    sc_out<id_type>     ARID;
    sc_out<addr_type>   ARADDR;
    sc_out<len_type>    ARLEN;
    sc_out<size_type>   ARSIZE;
    sc_out<burst_type>  ARBURST;
    sc_out<lock_type>   ARLOCK;
    sc_out<cache_type>  ARCACHE;
    sc_out<prot_type>   ARPROT;
    sc_out<qos_type>    ARQOS;
    sc_out<region_type> ARREGION;
    sc_out<bool>        ARVALID;
    sc_in<bool>         ARREADY;

    // R channel
    sc_in<id_type>     RID;
    sc_in<data_type>   RDATA;
    sc_in<resp_type>   RRESP;
    sc_in<bool>        RLAST;
    sc_in<bool>        RVALID;
    sc_out<bool>       RREADY;

    // AW channel
    sc_out<id_type>     AWID;
    sc_out<addr_type>   AWADDR;
    sc_out<len_type>    AWLEN;
    sc_out<size_type>   AWSIZE;
    sc_out<burst_type>  AWBURST;
    sc_out<lock_type>   AWLOCK;
    sc_out<cache_type>  AWCACHE;
    sc_out<prot_type>   AWPROT;
    sc_out<qos_type>    AWQOS;
    sc_out<region_type> AWREGION;
    sc_out<bool>        AWVALID;
    sc_in<bool>         AWREADY;

    // W channel (conditional WID)
    typename std::conditional<CFG::HAS_WID, sc_out<id_type>, EmptyPort>::type WID;
    sc_out<data_type>   WDATA;
    sc_out<strb_type>   WSTRB;
    sc_out<bool>        WLAST;
    sc_out<bool>        WVALID;
    sc_in<bool>         WREADY;

    // B channel
    sc_in<id_type>     BID;
    sc_in<resp_type>   BRESP;
    sc_in<bool>        BVALID;
    sc_out<bool>       BREADY;

    MASTER_PORTS() = default;
};

/**
 * Slave-side port container.
 */
template<typename CFG = AXI4_Default>
class SLAVE_PORTS {
public:
    using id_type     = typename CFG::id_t;
    using addr_type   = typename CFG::addr_t;
    using data_type   = typename CFG::data_t;
    using strb_type   = typename CFG::strb_t;
    using len_type    = typename CFG::len_t;
    using size_type   = typename CFG::axi_size_t;
    using burst_type  = typename CFG::burst_t;
    using lock_type   = typename CFG::lock_t;
    using cache_type  = typename CFG::cache_t;
    using prot_type   = typename CFG::prot_t;
    using qos_type    = typename CFG::qos_t;
    using region_type = typename CFG::region_t;
    using resp_type   = typename CFG::resp_t;

    // AR channel
    sc_in<id_type>      ARID;
    sc_in<addr_type>    ARADDR;
    sc_in<len_type>     ARLEN;
    sc_in<size_type>    ARSIZE;
    sc_in<burst_type>   ARBURST;
    sc_in<lock_type>    ARLOCK;
    sc_in<cache_type>   ARCACHE;
    sc_in<prot_type>    ARPROT;
    sc_in<qos_type>     ARQOS;
    sc_in<region_type>  ARREGION;
    sc_in<bool>         ARVALID;
    sc_out<bool>        ARREADY;

    // R channel
    sc_out<id_type>     RID;
    sc_out<data_type>   RDATA;
    sc_out<resp_type>   RRESP;
    sc_out<bool>        RLAST;
    sc_out<bool>        RVALID;
    sc_in<bool>         RREADY;

    // AW channel
    sc_in<id_type>      AWID;
    sc_in<addr_type>    AWADDR;
    sc_in<len_type>     AWLEN;
    sc_in<size_type>    AWSIZE;
    sc_in<burst_type>   AWBURST;
    sc_in<lock_type>    AWLOCK;
    sc_in<cache_type>   AWCACHE;
    sc_in<prot_type>    AWPROT;
    sc_in<qos_type>     AWQOS;
    sc_in<region_type>  AWREGION;
    sc_in<bool>         AWVALID;
    sc_out<bool>        AWREADY;

    // W channel (conditional WID)
    typename std::conditional<CFG::HAS_WID, sc_in<id_type>, EmptyPort>::type WID;
    sc_in<data_type>    WDATA;
    sc_in<strb_type>    WSTRB;
    sc_in<bool>         WLAST;
    sc_in<bool>         WVALID;
    sc_out<bool>        WREADY;

    // B channel
    sc_out<id_type>     BID;
    sc_out<resp_type>   BRESP;
    sc_out<bool>        BVALID;
    sc_in<bool>         BREADY;

    SLAVE_PORTS() = default;
};

/**
 * Bind master ports to signals.
 */
template<typename CFG>
void bind_port_signal(MASTER_PORTS<CFG>* ports, AXI_SIGNALS<CFG>& signals) {
    // AR
    ports->ARID(signals.ARID);
    ports->ARADDR(signals.ARADDR);
    ports->ARLEN(signals.ARLEN);
    ports->ARSIZE(signals.ARSIZE);
    ports->ARBURST(signals.ARBURST);
    ports->ARLOCK(signals.ARLOCK);
    ports->ARCACHE(signals.ARCACHE);
    ports->ARPROT(signals.ARPROT);
    ports->ARQOS(signals.ARQOS);
    ports->ARREGION(signals.ARREGION);
    ports->ARVALID(signals.ARVALID);
    ports->ARREADY(signals.ARREADY);

    // R
    ports->RID(signals.RID);
    ports->RDATA(signals.RDATA);
    ports->RRESP(signals.RRESP);
    ports->RLAST(signals.RLAST);
    ports->RVALID(signals.RVALID);
    ports->RREADY(signals.RREADY);

    // AW
    ports->AWID(signals.AWID);
    ports->AWADDR(signals.AWADDR);
    ports->AWLEN(signals.AWLEN);
    ports->AWSIZE(signals.AWSIZE);
    ports->AWBURST(signals.AWBURST);
    ports->AWLOCK(signals.AWLOCK);
    ports->AWCACHE(signals.AWCACHE);
    ports->AWPROT(signals.AWPROT);
    ports->AWQOS(signals.AWQOS);
    ports->AWREGION(signals.AWREGION);
    ports->AWVALID(signals.AWVALID);
    ports->AWREADY(signals.AWREADY);

    // W (conditional WID)
    if constexpr (CFG::HAS_WID) {
        ports->WID(signals.WID);
    }
    ports->WDATA(signals.WDATA);
    ports->WSTRB(signals.WSTRB);
    ports->WLAST(signals.WLAST);
    ports->WVALID(signals.WVALID);
    ports->WREADY(signals.WREADY);

    // B
    ports->BID(signals.BID);
    ports->BRESP(signals.BRESP);
    ports->BVALID(signals.BVALID);
    ports->BREADY(signals.BREADY);
}

/**
 * Bind slave ports to signals.
 */
template<typename CFG>
void bind_port_signal(SLAVE_PORTS<CFG>* ports, AXI_SIGNALS<CFG>& signals) {
    // AR
    ports->ARID(signals.ARID);
    ports->ARADDR(signals.ARADDR);
    ports->ARLEN(signals.ARLEN);
    ports->ARSIZE(signals.ARSIZE);
    ports->ARBURST(signals.ARBURST);
    ports->ARLOCK(signals.ARLOCK);
    ports->ARCACHE(signals.ARCACHE);
    ports->ARPROT(signals.ARPROT);
    ports->ARQOS(signals.ARQOS);
    ports->ARREGION(signals.ARREGION);
    ports->ARVALID(signals.ARVALID);
    ports->ARREADY(signals.ARREADY);

    // R
    ports->RID(signals.RID);
    ports->RDATA(signals.RDATA);
    ports->RRESP(signals.RRESP);
    ports->RLAST(signals.RLAST);
    ports->RVALID(signals.RVALID);
    ports->RREADY(signals.RREADY);

    // AW
    ports->AWID(signals.AWID);
    ports->AWADDR(signals.AWADDR);
    ports->AWLEN(signals.AWLEN);
    ports->AWSIZE(signals.AWSIZE);
    ports->AWBURST(signals.AWBURST);
    ports->AWLOCK(signals.AWLOCK);
    ports->AWCACHE(signals.AWCACHE);
    ports->AWPROT(signals.AWPROT);
    ports->AWQOS(signals.AWQOS);
    ports->AWREGION(signals.AWREGION);
    ports->AWVALID(signals.AWVALID);
    ports->AWREADY(signals.AWREADY);

    // W (conditional WID)
    if constexpr (CFG::HAS_WID) {
        ports->WID(signals.WID);
    }
    ports->WDATA(signals.WDATA);
    ports->WSTRB(signals.WSTRB);
    ports->WLAST(signals.WLAST);
    ports->WVALID(signals.WVALID);
    ports->WREADY(signals.WREADY);

    // B
    ports->BID(signals.BID);
    ports->BRESP(signals.BRESP);
    ports->BVALID(signals.BVALID);
    ports->BREADY(signals.BREADY);
}

#endif // FLEXNPU_TRANSPORT_AXI_AXI_SIGNAL_PORTS_H
