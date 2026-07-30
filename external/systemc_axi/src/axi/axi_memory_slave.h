#ifndef AXI_MEMORY_SLAVE_H
#define AXI_MEMORY_SLAVE_H

#include "axi_slave_if.h"
#include "axi_logger.h"
#include <vector>
#include <cstring>

/**
 * Simple Memory Slave (Wrapper around AXI_slave_if with actual memory storage)
 *
 * This module adds memory storage and read/write handling to the basic AXI slave interface.
 * Features:
 * - Byte-addressable memory storage
 * - WSTRB (write strobe) byte-enable support
 * - Burst address calculation (FIXED, INCR, WRAP)
 * - Configurable read/write latency
 */
template<typename CFG = AXI4_Default, typename TIMING = AXITimingConfig>
SC_MODULE(SimpleMemorySlave) {
public:
    using id_t = typename CFG::id_t;
    using addr_t = typename CFG::addr_t;
    using data_t = typename CFG::data_t;
    using len_t = typename CFG::len_t;
    using axi_size_t = typename CFG::axi_size_t;
    using burst_t = typename CFG::burst_t;
    using strb_t = typename CFG::strb_t;

    sc_in<bool> clk;
    SLAVE_PORTS<CFG> ports;

    // Memory storage
    std::vector<uint8_t> memory;
    addr_t baseAddr;
    unsigned int memorySize;

    // Internal slave interface
    AXI_slave_if<CFG, TIMING>* slave_if;

    // Configuration
    bool enable_logging;
    unsigned int read_latency;   // Additional cycles for read
    unsigned int write_latency;  // Additional cycles for write

    SC_HAS_PROCESS(SimpleMemorySlave);

    SimpleMemorySlave(sc_module_name name, addr_t base_address, unsigned int size,
                     bool log_enable = false, unsigned int rd_lat = 0, unsigned int wr_lat = 0)
        : sc_module(name), baseAddr(base_address), memorySize(size),
          enable_logging(log_enable), read_latency(rd_lat), write_latency(wr_lat) {

        // Allocate memory and initialize to zero
        memory.resize(memorySize, 0);

        // Create internal slave interface
        slave_if = new AXI_slave_if<CFG, TIMING>("internal_slave", base_address, size);

        // Connect clock
        slave_if->clk(clk);

        // Forward all port connections
        slave_if->ports.ARID(ports.ARID);
        slave_if->ports.ARADDR(ports.ARADDR);
        slave_if->ports.ARLEN(ports.ARLEN);
        slave_if->ports.ARSIZE(ports.ARSIZE);
        slave_if->ports.ARBURST(ports.ARBURST);
        slave_if->ports.ARLOCK(ports.ARLOCK);
        slave_if->ports.ARCACHE(ports.ARCACHE);
        slave_if->ports.ARPROT(ports.ARPROT);
        slave_if->ports.ARQOS(ports.ARQOS);
        slave_if->ports.ARREGION(ports.ARREGION);
        slave_if->ports.ARVALID(ports.ARVALID);
        slave_if->ports.ARREADY(ports.ARREADY);

        slave_if->ports.RID(ports.RID);
        slave_if->ports.RDATA(ports.RDATA);
        slave_if->ports.RRESP(ports.RRESP);
        slave_if->ports.RLAST(ports.RLAST);
        slave_if->ports.RVALID(ports.RVALID);
        slave_if->ports.RREADY(ports.RREADY);

        slave_if->ports.AWID(ports.AWID);
        slave_if->ports.AWADDR(ports.AWADDR);
        slave_if->ports.AWLEN(ports.AWLEN);
        slave_if->ports.AWSIZE(ports.AWSIZE);
        slave_if->ports.AWBURST(ports.AWBURST);
        slave_if->ports.AWLOCK(ports.AWLOCK);
        slave_if->ports.AWCACHE(ports.AWCACHE);
        slave_if->ports.AWPROT(ports.AWPROT);
        slave_if->ports.AWQOS(ports.AWQOS);
        slave_if->ports.AWREGION(ports.AWREGION);
        slave_if->ports.AWVALID(ports.AWVALID);
        slave_if->ports.AWREADY(ports.AWREADY);

        if constexpr (CFG::HAS_WID) {
            slave_if->ports.WID(ports.WID);
        }
        slave_if->ports.WDATA(ports.WDATA);
        slave_if->ports.WSTRB(ports.WSTRB);
        slave_if->ports.WLAST(ports.WLAST);
        slave_if->ports.WVALID(ports.WVALID);
        slave_if->ports.WREADY(ports.WREADY);

        slave_if->ports.BID(ports.BID);
        slave_if->ports.BRESP(ports.BRESP);
        slave_if->ports.BVALID(ports.BVALID);
        slave_if->ports.BREADY(ports.BREADY);

        // Start handler threads
        SC_THREAD(read_handler);
        sensitive << slave_if->read_reg_event;

        SC_THREAD(write_handler);
        sensitive << slave_if->set_write_data_event;

        if (enable_logging) {
            AXI_LOG_INFO("SimpleMemorySlave '{}' created: base=0x{:x}, size={} bytes",
                        std::string(name), base_address.to_uint(), size);
        }
    }

    ~SimpleMemorySlave() {
        delete slave_if;
    }

    /**
     * Read Handler Process
     * Responds to read requests from the slave interface
     */
    void read_handler() {
        while (true) {
            wait(slave_if->read_reg_event);

            if (slave_if->readQueue.empty()) continue;

            AXITransaction<CFG>* trans = slave_if->readQueue.front();
            slave_if->readQueue.pop();

            // Extract transaction parameters
            addr_t start_addr = trans->addr;
            axi_size_t size = trans->size;
            burst_t burst_type = trans->type;
            len_t burst_len = trans->len;
            id_t trans_id = trans->ID;

            unsigned int transfer_size = 1 << size.to_uint();  // Bytes per beat
            addr_t current_addr = start_addr;

            if (enable_logging) {
                AXI_LOG_DEBUG("Read: ID={}, addr=0x{:x}, len={}, size={}, burst={}",
                            trans_id.to_uint(), start_addr.to_uint(),
                            burst_len.to_uint(), transfer_size, burst_type.to_uint());
            }

            // Set transaction metadata for slave interface
            slave_if->Rid = trans_id;
            slave_if->Rlen = burst_len;

            // Read burst
            for (unsigned int beat = 0; beat <= burst_len.to_uint(); ++beat) {
                // Add read latency
                for (unsigned int lat = 0; lat < read_latency; ++lat) {
                    wait(clk.posedge_event());
                }

                // Read from memory
                data_t read_data = read_memory(current_addr, transfer_size);
                slave_if->read_buffer = read_data;

                // Notify slave interface to send data
                slave_if->set_read_data_event.notify();

                // Wait for data to be transferred
                wait(slave_if->get_read_data_event);

                // Calculate next address
                current_addr = slave_if->burst_addr_change(
                    current_addr, start_addr, size.to_uint(), burst_type, burst_len);
            }

            delete trans;
        }
    }

    /**
     * Write Handler Process
     * Responds to write requests from the slave interface
     * Pattern: wait for write_reg_event -> get transaction -> loop through beats
     */
    void write_handler() {
        while (true) {
            // Wait for new write transaction to start
            wait(slave_if->write_reg_event);

            if (slave_if->writeQueue.empty()) continue;

            // Get write transaction
            AXITransaction<CFG>* trans = slave_if->writeQueue.front();
            slave_if->writeQueue.pop();

            addr_t start_addr = trans->addr;
            axi_size_t size = trans->size;
            burst_t burst_type = trans->type;
            len_t burst_len = trans->len;
            id_t trans_id = trans->ID;

            unsigned int transfer_size = 1 << size.to_uint();
            addr_t current_addr = start_addr;

            if (enable_logging) {
                AXI_LOG_DEBUG("Write: ID={}, addr=0x{:x}, len={}, size={}, burst={}",
                            trans_id.to_uint(), start_addr.to_uint(),
                            burst_len.to_uint(), transfer_size, burst_type.to_uint());
            }

            // Set transaction metadata for slave interface
            slave_if->Wid = trans_id;
            slave_if->Wlen = burst_len;

            // Process all write beats
            for (unsigned int beat = 0; beat <= burst_len.to_uint(); ++beat) {
                // Wait for write data to arrive
                wait(slave_if->set_write_data_event);

                // Add write latency
                for (unsigned int lat = 0; lat < write_latency; ++lat) {
                    wait(clk.posedge_event());
                }

                // Get write data and strobe
                data_t write_data = slave_if->write_buffer;
                strb_t write_strb = ports.WSTRB.read();

                // Write to memory with byte-enable support
                write_memory(current_addr, write_data, write_strb, transfer_size);

                if (enable_logging) {
                    AXI_LOG_TRACE("  Write beat {}: addr=0x{:x}, data=0x{:x}, strb=0x{:x}",
                                 beat, current_addr.to_uint(), write_data.to_uint(), write_strb.to_uint());
                }

                // Calculate next address for next beat
                if (beat < burst_len.to_uint()) {
                    current_addr = slave_if->burst_addr_change(
                        current_addr, start_addr, size.to_uint(),
                        burst_type, burst_len);

                    // Notify slave interface we're ready for next beat
                    slave_if->write_resp_event.notify();
                }
            }

            // Transaction complete
            delete trans;
        }
    }

    /**
     * Read from memory with address bounds checking
     */
    data_t read_memory(addr_t addr, unsigned int num_bytes) {
        data_t result = 0;
        addr_t offset = addr - baseAddr;

        if (offset.to_uint() >= memorySize) {
            if (enable_logging) {
                AXI_LOG_WARN("Read out of bounds: addr=0x{:x}", addr.to_uint());
            }
            return 0;  // Return 0 for out-of-bounds reads
        }

        // Read bytes and assemble data (little-endian)
        for (unsigned int i = 0; i < num_bytes && (offset.to_uint() + i) < memorySize; ++i) {
            uint64_t byte_val = memory[offset.to_uint() + i];
            result |= (byte_val << (i * 8));
        }

        if (enable_logging) {
            AXI_LOG_TRACE("  Read  addr=0x{:x} data=0x{:x}", addr.to_uint(), result.to_uint());
        }

        return result;
    }

    /**
     * Write to memory with WSTRB byte-enable support
     * WSTRB bit[n] corresponds to data byte[n]
     */
    void write_memory(addr_t addr, data_t data, strb_t strb, unsigned int num_bytes) {
        addr_t offset = addr - baseAddr;

        if (offset.to_uint() >= memorySize) {
            if (enable_logging) {
                AXI_LOG_WARN("Write out of bounds: addr=0x{:x}", addr.to_uint());
            }
            return;
        }

        // Write bytes with strobe masking
        for (unsigned int i = 0; i < num_bytes && (offset.to_uint() + i) < memorySize; ++i) {
            if (strb.bit(i)) {  // Check if this byte is enabled
                uint8_t byte_val = (data.to_uint() >> (i * 8)) & 0xFF;
                memory[offset.to_uint() + i] = byte_val;
            }
        }

        if (enable_logging) {
            AXI_LOG_TRACE("  Write addr=0x{:x} data=0x{:x} strb=0x{:x}",
                         addr.to_uint(), data.to_uint(), strb.to_uint());
        }
    }

    /**
     * Direct memory access for testing/verification
     */
    uint32_t read_direct(addr_t addr) {
        addr_t offset = addr - baseAddr;
        if (offset.to_uint() + 3 >= memorySize) return 0;

        uint32_t result = 0;
        for (int i = 0; i < 4; ++i) {
            result |= (memory[offset.to_uint() + i] << (i * 8));
        }
        return result;
    }

    void write_direct(addr_t addr, uint32_t data) {
        addr_t offset = addr - baseAddr;
        if (offset.to_uint() + 3 >= memorySize) return;

        for (int i = 0; i < 4; ++i) {
            memory[offset.to_uint() + i] = (data >> (i * 8)) & 0xFF;
        }
    }

    /**
     * Fill memory with pattern (for testing)
     */
    void fill_memory(uint8_t pattern) {
        std::fill(memory.begin(), memory.end(), pattern);
        if (enable_logging) {
            AXI_LOG_DEBUG("Memory filled with pattern 0x{:02x}", pattern);
        }
    }

    /**
     * Dump memory contents (for debugging)
     */
    void dump_memory(addr_t start, unsigned int length) {
        addr_t offset = start - baseAddr;
        AXI_LOG_INFO("Memory dump: addr=0x{:x}, length={}", start.to_uint(), length);

        for (unsigned int i = 0; i < length && (offset.to_uint() + i) < memorySize; i += 16) {
            std::string line = fmt::format("  0x{:08x}:", (start.to_uint() + i));
            for (unsigned int j = 0; j < 16 && (i + j) < length; ++j) {
                if ((offset.to_uint() + i + j) < memorySize) {
                    line += fmt::format(" {:02x}", memory[offset.to_uint() + i + j]);
                }
            }
            AXI_LOG_INFO("{}", line);
        }
    }
};

#endif // AXI_MEMORY_SLAVE_H
