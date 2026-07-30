#ifndef AXI_TRANSACTION_H
#define AXI_TRANSACTION_H

#include "systemc/bus/axi/axi_config.h"

/**
 * AXI Transaction Descriptor
 */
template<typename CFG = AXI4_Default>
class AXITransaction {
public:
    using id_t = typename CFG::id_t;
    using addr_t = typename CFG::addr_t;
    using len_t = typename CFG::len_t;
    using axi_size_t = typename CFG::axi_size_t;
    using burst_t = typename CFG::burst_t;
    using lock_t = typename CFG::lock_t;
    using cache_t = typename CFG::cache_t;
    using prot_t = typename CFG::prot_t;
    using qos_t = typename CFG::qos_t;
    using region_t = typename CFG::region_t;
    using resp_t = typename CFG::resp_t;

    // Transaction attributes
    id_t ID;
    addr_t addr;
    axi_size_t size;
    burst_t type;
    len_t len;

    // New AXI4 attributes
    lock_t lock;
    cache_t cache;
    prot_t prot;
    qos_t qos;
    region_t region;
    resp_t resp;

    // Pipeline state flags
    bool bus_allow = false;
    bool ar_released = false;
    bool r_released = false;
    bool aw_released = false;
    bool w_released = false;
    bool b_released = false;

    // Exclusive write tracking
    bool is_exclusive_write = false;
    addr_t exclusive_addr = 0;
    unsigned int exclusive_size_bytes = 0;

    AXITransaction()
        : ID(0), addr(0), size(0), type(0), len(0),
          lock(0), cache(0), prot(0), qos(0), region(0), resp(0) {}
};

#endif // AXI_TRANSACTION_H
