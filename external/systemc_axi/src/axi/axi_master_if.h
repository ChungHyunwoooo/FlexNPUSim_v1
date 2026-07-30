#ifndef AXI_MASTER_IF_H
#define AXI_MASTER_IF_H

#include "systemc/bus/axi/axi_signal_ports.h"
#include "axi_types.h"
#include <map>
#include <vector>
#include <iostream>

/**
 * AXI Master Interface Module (Templated)
 */
template<typename CFG = AXI4_Default, typename TIMING = AXITimingConfig>
SC_MODULE(AXI_master_if) {
public:
    using id_t = typename CFG::id_t;
    using addr_t = typename CFG::addr_t;
    using data_t = typename CFG::data_t;
    using len_t = typename CFG::len_t;
    using axi_size_t = typename CFG::axi_size_t;
    using burst_t = typename CFG::burst_t;
    using resp_t = typename CFG::resp_t;

    sc_in<bool> clk{"Master_if_clk"};
    MASTER_PORTS<CFG> ports;

    // Buffers (AXI4 supports up to 256 beats)
    static constexpr std::size_t MAX_BURST_LEN = 256;
    data_t read_buffer[MAX_BURST_LEN];
    data_t write_buffer[MAX_BURST_LEN];

    // Events
    sc_event AR_master_event, R_master_event, read_transfer_done;
    sc_event AW_master_event, W_master_event, B_master_event, write_transfer_done;

    // Active transactions
    std::map<id_t, AXITransaction<CFG>*> activeWrites;
    std::map<id_t, AXITransaction<CFG>*> activeReads;

    void mAR() {
        wait();
        while (true) {
            auto it = activeReads.begin();
            for (; it != activeReads.end(); ++it) {
                ports.ARID.write(it->second->ID);
                ports.ARADDR.write(it->second->addr);
                ports.ARSIZE.write(it->second->size);
                ports.ARBURST.write(it->second->type);
                ports.ARLEN.write(it->second->len);
                ports.ARLOCK.write(it->second->lock);
                ports.ARCACHE.write(it->second->cache);
                ports.ARPROT.write(it->second->prot);
                ports.ARQOS.write(it->second->qos);
                ports.ARREGION.write(it->second->region);

                ports.ARVALID.write(1);
                do { wait(clk.posedge_event()); } while (ports.ARREADY.read() == 0);

                R_master_event.notify();
                ports.ARVALID.write(0);
            }
            wait();
        }
    }

    void mR() {
        wait();
        while (true) {
            auto it = activeReads.begin();
            if (it == activeReads.end()) {
                std::cout << "[" << sc_time_stamp() << "] Read error: activeReads empty" << std::endl;
                exit(0);
            }

            len_t len = it->second->len;
            uint32_t addr = (uint32_t)it->second->addr;
#ifdef FLEXNPUSIM_AXI_TRACE
            std::fprintf(stderr, "[axi_mR] beats=%u addr=0x%x sim_ns=%llu\n",
                         (unsigned)(len + 1), addr,
                         (unsigned long long)sc_time_stamp().to_default_time_units());
#endif
            for (unsigned int i = 0; i < len + 1; ++i) {
                ports.RREADY.write(1);
                uint64_t poll_start = sc_time_stamp().to_default_time_units();
                do { wait(clk.posedge_event()); } while (ports.RVALID.read() == 0);
                uint64_t poll_end = sc_time_stamp().to_default_time_units();
#ifdef FLEXNPUSIM_AXI_TRACE
                std::fprintf(stderr, "[axi_mR] beat %u got RVALID after %llu ns\n",
                             i, (unsigned long long)(poll_end - poll_start));
#else
                (void)poll_start; (void)poll_end;
#endif

                read_buffer[i] = ports.RDATA.read();
                it->second->resp = ports.RRESP.read();
                ports.RREADY.write(0);
            }
            read_transfer_done.notify();
            wait();
        }
    }

    void mAW() {
        ports.AWID.write(0);
        ports.AWADDR.write(0);
        ports.AWLEN.write(0);
        ports.AWSIZE.write(0);
        ports.AWBURST.write(0);
        ports.AWVALID.write(0);
        wait();

        while (true) {
            auto it = activeWrites.begin();
            for (; it != activeWrites.end(); ++it) {
                ports.AWID.write(it->second->ID);
                ports.AWADDR.write(it->second->addr);
                ports.AWSIZE.write(it->second->size);
                ports.AWBURST.write(it->second->type);
                ports.AWLEN.write(it->second->len);
                ports.AWLOCK.write(it->second->lock);
                ports.AWCACHE.write(it->second->cache);
                ports.AWPROT.write(it->second->prot);
                ports.AWQOS.write(it->second->qos);
                ports.AWREGION.write(it->second->region);
                ports.AWVALID.write(1);

                do { wait(clk.posedge_event()); } while (ports.AWREADY == 0);

                W_master_event.notify();
                ports.AWVALID.write(0);
                wait(clk.posedge_event());
            }
            wait();
        }
    }

    void mW() {
        ports.WVALID.write(0);
        wait();

        while (true) {
            auto it = activeWrites.begin();
            if (it == activeWrites.end()) {
                std::cout << "[" << sc_time_stamp() << "] Write error: activeWrites empty" << std::endl;
                exit(0);
            }

            for (; it != activeWrites.end(); ++it) {
                id_t master_id = it->second->ID;
                len_t len = it->second->len;

                for (unsigned int i = 0; i < len + 1; ++i) {
                    bool wlast = (i == len);

                    if constexpr (CFG::HAS_WID) {
                        ports.WID.write(master_id);
                    }
                    ports.WDATA.write(write_buffer[i]);
                    ports.WSTRB.write(~0);  // All bytes valid
                    ports.WLAST.write(wlast);
                    ports.WVALID.write(1);

                    do { wait(clk.posedge_event()); } while (ports.WREADY.read() == 0);
                }
                B_master_event.notify();
                ports.WVALID.write(0);
                ports.WLAST.write(0);
            }
            wait();
        }
    }

    void mB() {
        ports.BREADY.write(0);
        wait();

        while (true) {
            ports.BREADY.write(1);
            do { wait(clk.posedge_event()); } while (ports.BVALID == 0);

            // Wait an extra delta cycle to ensure bus has set BRESP
            wait(SC_ZERO_TIME);

            // Store response - match BID to find correct transaction
            id_t bid = ports.BID.read();
            auto it = activeWrites.find(bid);
            if (it != activeWrites.end()) {
                it->second->resp = ports.BRESP.read();
            }

            ports.BREADY.write(0);
            write_transfer_done.notify();
            wait();
        }
    }

    // Write API - returns response
    AXIResponse master_write(id_t ID, addr_t addr, data_t data,
                             size_t size, burst_t burst, len_t len) {
        auto* activeWrite = new AXITransaction<CFG>;
        activeWrite->ID = ID;
        activeWrite->addr = addr;
        activeWrite->size = size;
        activeWrite->type = burst;
        activeWrite->len = len;
        write_buffer[0] = data;

        activeWrites[ID] = activeWrite;
        AW_master_event.notify();
        wait(write_transfer_done);

        AXIResponse response = static_cast<AXIResponse>(activeWrite->resp.to_uint());
        delete activeWrite;
        activeWrites.erase(ID);
        return response;
    }

    // Write API - burst
    AXIResponse master_write(id_t ID, addr_t addr, data_t* data,
                             size_t size, burst_t burst, len_t len) {
        auto* activeWrite = new AXITransaction<CFG>;
        activeWrite->ID = ID;
        activeWrite->addr = addr;
        activeWrite->size = size;
        activeWrite->type = burst;
        activeWrite->len = len;

        for (unsigned int i = 0; i < len + 1; ++i) {
            write_buffer[i] = data[i];
        }

        activeWrites[ID] = activeWrite;
        AW_master_event.notify();
        wait(write_transfer_done);

        AXIResponse response = static_cast<AXIResponse>(activeWrite->resp.to_uint());
        delete activeWrite;
        activeWrites.erase(ID);
        return response;
    }

    // Read API - returns data and response
    std::pair<data_t*, AXIResponse> master_read(id_t ID, addr_t addr,
                                                 size_t size, burst_t burst, len_t len) {
        auto* activeRead = new AXITransaction<CFG>();
        activeRead->ID = ID;
        activeRead->addr = addr;
        activeRead->size = size;
        activeRead->type = burst;
        activeRead->len = len;

        activeReads[ID] = activeRead;
        AR_master_event.notify();
        wait(read_transfer_done);

        AXIResponse response = static_cast<AXIResponse>(activeRead->resp.to_uint());
        delete activeRead;
        activeReads.erase(ID);
        return {read_buffer, response};
    }

    // High-level convenience APIs

    // Single read transaction
    data_t read_single(addr_t addr) {
        auto result = master_read(0, addr, 2, 1, 0);  // ID=0, size=4bytes, INCR, len=0
        return result.first[0];
    }

    // Single write transaction
    void write_single(addr_t addr, data_t data) {
        master_write(0, addr, data, 2, 1, 0);  // ID=0, size=4bytes, INCR, len=0
    }

    // Single read with ID
    data_t read_single_with_id(addr_t addr, id_t id) {
        auto result = master_read(id, addr, 2, 1, 0);
        return result.first[0];
    }

    // Single write with ID
    void write_single_with_id(addr_t addr, data_t data, id_t id) {
        master_write(id, addr, data, 2, 1, 0);
    }

    // Burst write (pointer version)
    void write_burst(addr_t addr, data_t* data, burst_t burst_type) {
        // Assuming len is calculated from data array size, using default len=3 for now
        master_write(0, addr, data, 2, burst_type, 3);
    }

    // Burst write (vector version)
    void write_burst(addr_t addr, std::vector<data_t>& data, burst_t burst_type) {
        len_t len = data.size() > 0 ? data.size() - 1 : 0;
        for (std::size_t i = 0; i < data.size(); ++i) {
            write_buffer[i] = data[i];
        }
        master_write(0, addr, write_buffer, 2, burst_type, len);
    }

    // Burst read (pointer version)
    void read_burst(addr_t addr, len_t length, data_t* data, burst_t burst_type) {
        auto result = master_read(0, addr, 2, burst_type, length);
        for (unsigned int i = 0; i < length + 1; ++i) {
            data[i] = result.first[i];
        }
    }

    // Burst read (vector version)
    void read_burst(addr_t addr, len_t length, std::vector<data_t>& data, burst_t burst_type) {
        auto result = master_read(0, addr, 2, burst_type, length);
        data.clear();
        for (unsigned int i = 0; i < length + 1; ++i) {
            data.push_back(result.first[i]);
        }
    }

    // Burst write with size (pointer version)
    void write_burst_with_size(addr_t addr, data_t* data, burst_t burst_type, axi_size_t size) {
        master_write(0, addr, data, size.to_uint(), burst_type, 3);
    }

    // Burst write with size (vector version)
    void write_burst_with_size(addr_t addr, std::vector<data_t>& data, burst_t burst_type, unsigned int size) {
        len_t len = data.size() > 0 ? data.size() - 1 : 0;
        for (std::size_t i = 0; i < data.size(); ++i) {
            write_buffer[i] = data[i];
        }
        master_write(0, addr, write_buffer, size, burst_type, len);
    }

    // Burst read with size (vector version)
    void read_burst_with_size(addr_t addr, len_t length, std::vector<data_t>& data, burst_t burst_type, unsigned int size) {
        auto result = master_read(0, addr, size, burst_type, length);
        data.clear();
        for (unsigned int i = 0; i < length + 1; ++i) {
            data.push_back(result.first[i]);
        }
    }

    // Exclusive read
    data_t read_exclusive(addr_t addr) {
        auto* activeRead = new AXITransaction<CFG>();
        activeRead->ID = 0;
        activeRead->addr = addr;
        activeRead->size = 2;
        activeRead->type = 1;  // INCR
        activeRead->len = 0;
        activeRead->lock = 1;  // Exclusive access

        activeReads[0] = activeRead;
        AR_master_event.notify();
        wait(read_transfer_done);

        data_t result = read_buffer[0];
        delete activeRead;
        activeReads.erase(0);
        return result;
    }

    // Exclusive write
    resp_t write_exclusive(addr_t addr, data_t data) {
        auto* activeWrite = new AXITransaction<CFG>;
        activeWrite->ID = 0;
        activeWrite->addr = addr;
        activeWrite->size = 2;
        activeWrite->type = 1;  // INCR
        activeWrite->len = 0;
        activeWrite->lock = 1;  // Exclusive access
        write_buffer[0] = data;

        activeWrites[0] = activeWrite;
        AW_master_event.notify();
        wait(write_transfer_done);

        resp_t response = activeWrite->resp;
        delete activeWrite;
        activeWrites.erase(0);
        return response;
    }

    SC_CTOR(AXI_master_if) {
        SC_THREAD(mAR);
        sensitive << AR_master_event;

        SC_THREAD(mR);
        sensitive << R_master_event;

        SC_THREAD(mAW);
        sensitive << AW_master_event;

        SC_THREAD(mW);
        sensitive << W_master_event;

        SC_THREAD(mB);
        sensitive << B_master_event;
    }
};

#endif // AXI_MASTER_IF_H
