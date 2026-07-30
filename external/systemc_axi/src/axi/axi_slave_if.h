#ifndef AXI_SLAVE_IF_H
#define AXI_SLAVE_IF_H

#include "systemc/bus/axi/axi_signal_ports.h"
#include "axi_types.h"
#include <map>
#include <queue>
#include <iostream>

/**
 * AXI Slave Interface Module (Templated)
 */
template<typename CFG = AXI4_Default, typename TIMING = AXITimingConfig>
SC_MODULE(AXI_slave_if) {
public:
    using id_t = typename CFG::id_t;
    using addr_t = typename CFG::addr_t;
    using data_t = typename CFG::data_t;
    using len_t = typename CFG::len_t;
    using axi_size_t = typename CFG::axi_size_t;
    using burst_t = typename CFG::burst_t;

    sc_in<bool> clk;
    SLAVE_PORTS<CFG> ports;

    // Transaction metadata
    id_t Rid, Wid;
    len_t Rlen, Wlen;
    data_t read_buffer, write_buffer;

    // Events
    sc_event read_reg_event, read_ready_event, get_read_data_event, set_read_data_event;
    sc_event write_reg_event, write_ready_event, set_write_data_event, write_resp_event;
    sc_event R_slave_event, W_slave_event, B_slave_event;

    // Address space
    addr_t baseAddr;
    unsigned int addrRange;

    // Pipeline queues
    std::queue<AXITransaction<CFG>*> readQueue;
    std::queue<AXITransaction<CFG>*> writeQueue;
    sc_event queue_not_empty_read, queue_not_empty_write;

    void sAR() {
        ports.ARREADY.write(0);
        wait();

        while (true) {
            do { wait(clk.posedge_event()); } while (ports.ARVALID.read() == 0);

            auto* activeRead = new AXITransaction<CFG>();
            activeRead->ID = ports.ARID.read();
            activeRead->addr = ports.ARADDR.read();
            activeRead->size = ports.ARSIZE.read();
            activeRead->type = ports.ARBURST.read();
            activeRead->len = ports.ARLEN.read();
            activeRead->lock = ports.ARLOCK.read();
            activeRead->cache = ports.ARCACHE.read();
            activeRead->prot = ports.ARPROT.read();
            activeRead->qos = ports.ARQOS.read();
            activeRead->region = ports.ARREGION.read();

            readQueue.push(activeRead);
            queue_not_empty_read.notify();
            // Set Rid/Rlen from the AR itself so that sR never sees stale
            // values from the previous transaction. Race: sAR notifies both
            // read_reg_event (handler) and R_slave_event (sR) in the same
            // delta; if sR runs first, it reads whatever Rlen the handler
            // last wrote — which might belong to the previous burst.
            Rid  = activeRead->ID;
            Rlen = activeRead->len;
#ifdef FLEXNPUSIM_AXI_TRACE
            std::fprintf(stderr, "[axi_sAR] AR received addr=0x%x len=%u id=%u qsize=%zu sim_ns=%llu\n",
                         (uint32_t)activeRead->addr.to_uint(),
                         (unsigned)activeRead->len.to_uint(),
                         (unsigned)activeRead->ID.to_uint(),
                         readQueue.size(),
                         (unsigned long long)sc_time_stamp().to_default_time_units());
#endif
            read_reg_event.notify();

            ports.ARREADY.write(1);
            wait(clk.posedge_event());

            R_slave_event.notify();
            ports.ARREADY.write(0);
            wait(clk.posedge_event());
            wait();
        }
    }

    void sR() {
        ports.RID.write(0);
        ports.RDATA.write(0);
        ports.RRESP.write(0);
        ports.RLAST.write(0);
        ports.RVALID.write(0);
        wait();

        while (true) {
            for (unsigned int i = 0; i < Rlen + 1; ++i) {
                bool rlast = (i == Rlen);

                ports.RID.write(Rid);
                ports.RDATA.write(read_buffer);
                ports.RRESP.write(static_cast<uint8_t>(AXIResponse::OKAY));
                ports.RVALID.write(1);
                ports.RLAST.write(rlast);

                do { wait(clk.posedge_event()); } while (ports.RREADY.read() == 0);

                get_read_data_event.notify();
                if (i != Rlen) {
                    // Deassert RVALID while waiting for next beat data.
                    // Without this, stale RVALID=1 with old RDATA propagates
                    // through the bus and the master consumes duplicate data.
                    ports.RVALID.write(0);
                    wait(set_read_data_event);
                }
            }

            ports.RLAST.write(0);
            ports.RVALID.write(0);
            wait();
        }
    }

    void sAW() {
        ports.AWREADY.write(0);
        wait();

        while (true) {
            while (ports.AWVALID.read() == 0) {
                wait(clk.posedge_event());
            }

            auto* activeWrite = new AXITransaction<CFG>();
            activeWrite->ID = ports.AWID.read();
            activeWrite->addr = ports.AWADDR.read();
            activeWrite->size = ports.AWSIZE.read();
            activeWrite->type = ports.AWBURST.read();
            activeWrite->len = ports.AWLEN.read();
            activeWrite->lock = ports.AWLOCK.read();
            activeWrite->cache = ports.AWCACHE.read();
            activeWrite->prot = ports.AWPROT.read();
            activeWrite->qos = ports.AWQOS.read();
            activeWrite->region = ports.AWREGION.read();

            writeQueue.push(activeWrite);
            queue_not_empty_write.notify();
            write_reg_event.notify();

            ports.AWREADY.write(1);
            wait(clk.posedge_event());

            ports.AWREADY.write(0);
            W_slave_event.notify();
            wait();
        }
    }

    void sW() {
        ports.WREADY.write(0);
        wait();

        while (true) {
            for (unsigned int i = 0; i < Wlen + 1; ++i) {
                ports.WREADY.write(1);

                do { wait(clk.posedge_event()); } while (ports.WVALID.read() == 0);

                // Read WID if present (AXI3)
                if constexpr (CFG::HAS_WID) {
                    Wid = ports.WID.read();
                }

                write_buffer = ports.WDATA.read();
                set_write_data_event.notify();

                ports.WREADY.write(0);
                if (i != Wlen) {
                    wait(write_resp_event);
                }
            }

            B_slave_event.notify();
            ports.WREADY.write(0);
            wait();
        }
    }

    void sB() {
        ports.BVALID.write(0);
        wait();

        while (true) {
            ports.BID.write(Wid);
            ports.BRESP.write(static_cast<uint8_t>(AXIResponse::OKAY));
            ports.BVALID.write(1);

            do { wait(clk.posedge_event()); } while (ports.BREADY.read() == 0);

            ports.BVALID.write(0);
            wait();
        }
    }

    // Burst address calculation
    addr_t burst_addr_change(addr_t current_addr, addr_t start_addr,
                             size_t bytes, burst_t type, len_t burst_len) {
        addr_t n_addr;
        addr_t wrap_boundary = (start_addr / (bytes * burst_len + 1)) * (bytes * burst_len + 1);
        addr_t address_n = wrap_boundary + (bytes * burst_len);

        switch (type.to_uint()) {
        case 0: // FIXED
            n_addr = start_addr;
            break;
        case 1: // INCR
            n_addr = current_addr + ((addr_t)1 << bytes);
            break;
        case 2: // WRAP
            if (current_addr + bytes == address_n) {
                n_addr = wrap_boundary;
            } else {
                n_addr = current_addr + bytes;
            }
            break;
        default:
            n_addr = current_addr;
            break;
        }
        return n_addr;
    }

    SC_HAS_PROCESS(AXI_slave_if);

    AXI_slave_if(sc_module_name _name, addr_t base_address, unsigned int address_range)
        : sc_module(_name), baseAddr(base_address), addrRange(address_range) {

        SC_THREAD(sAR);
        sensitive << ports.ARVALID;

        SC_THREAD(sR);
        sensitive << R_slave_event;

        SC_THREAD(sAW);
        sensitive << ports.AWVALID;

        SC_THREAD(sW);
        sensitive << W_slave_event;

        SC_THREAD(sB);
        sensitive << B_slave_event;
    }
};

#endif // AXI_SLAVE_IF_H
