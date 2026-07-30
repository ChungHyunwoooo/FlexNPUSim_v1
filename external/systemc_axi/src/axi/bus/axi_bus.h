#ifndef AXI_BUS_H
#define AXI_BUS_H

#include "systemc/bus/axi/axi_signal_ports.h"
#include "../axi_memory_map.h"
#include "../profiler/axi_profiler.h"
#include "../axi_exclusive_monitor.h"
#include <map>
#include <deque>

/**
 * AXI Bus Arbiter (Templated with Profiler and Exclusive Access support)
 *
 * Features:
 * - Out-of-order completion: Different IDs can complete in any order
 * - In-order guarantee: Same ID completes in order (AXI4 rule)
 * - Exclusive access: ARLOCK/AWLOCK support with EXOKAY response
 */
template<
    typename CFG = AXI4_Default,
    typename MMAP = DynamicMemoryMap<typename CFG::addr_t>,
    bool ENABLE_PROFILING = false,
    bool ENABLE_EXCLUSIVE = true
>
class AXI_BUS : public sc_module {
public:
    using id_t = typename CFG::id_t;
    using addr_t = typename CFG::addr_t;

    sc_in<bool> clk;

    SLAVE_PORTS<CFG>* mPorts;   // Master-side ports
    MASTER_PORTS<CFG>* sPorts;  // Slave-side ports

    unsigned int master_num;
    unsigned int slave_num;

    MMAP memory_map;

    // Profiler (conditional, zero-overhead when disabled)
    typename std::conditional<ENABLE_PROFILING, AXIProfiler<CFG>, EmptyProfiler>::type profiler;

    // Exclusive Access Monitor (conditional)
    typename std::conditional<ENABLE_EXCLUSIVE, AXIExclusiveMonitor<CFG>, EmptyExclusiveMonitor>::type exclusive_monitor;

    struct MasterTransactionCount {
        unsigned int read = 0;
        unsigned int write = 0;
    };
    MasterTransactionCount* trans_counter;

    // Per-master, per-ID ordering tracker (for out-of-order completion)
    // Key: (master_id << 16) | transaction_id
    // Value: deque of transactions (FIFO per ID)
    std::map<uint32_t, std::deque<AXITransaction<CFG>*>> read_order_tracker;
    std::map<uint32_t, std::deque<AXITransaction<CFG>*>> write_order_tracker;

    // Transaction queues
    std::map<id_t, AXITransaction<CFG>*> readReq;
    std::map<id_t, AXITransaction<CFG>*> readData;
    std::map<id_t, AXITransaction<CFG>*> writeReq;
    std::map<id_t, AXITransaction<CFG>*> writeData;
    std::map<id_t, AXITransaction<CFG>*> writeB;

    SC_HAS_PROCESS(AXI_BUS);

    AXI_BUS(sc_module_name name_, int master_num, int slave_num, const MMAP& mmap = MMAP())
        : sc_module(name_), master_num(master_num), slave_num(slave_num), memory_map(mmap) {

        mPorts = new SLAVE_PORTS<CFG>[master_num];
        trans_counter = new MasterTransactionCount[master_num];
        sPorts = new MASTER_PORTS<CFG>[slave_num];

        SC_METHOD(arbitrate);
        sensitive << clk.pos();

        for (int i = 0; i < master_num; ++i) {
            // mPorts는 SLAVE_PORTS 타입 - sc_in 포트만 sensitive에 추가
            sensitive << mPorts[i].ARVALID;   // sc_in
            sensitive << mPorts[i].RREADY;    // sc_in (RVALID는 sc_out이므로 제외)
            sensitive << mPorts[i].AWVALID;   // sc_in
            sensitive << mPorts[i].WVALID;    // sc_in
            sensitive << mPorts[i].WLAST;     // sc_in
            sensitive << mPorts[i].WDATA;     // sc_in
            sensitive << mPorts[i].BREADY;    // sc_in
        }

        for (int i = 0; i < slave_num; ++i) {
            // sPorts는 MASTER_PORTS 타입 - sc_in 포트만 sensitive에 추가
            sensitive << sPorts[i].ARREADY;   // sc_in
            sensitive << sPorts[i].RID;       // sc_in
            sensitive << sPorts[i].RDATA;     // sc_in
            sensitive << sPorts[i].RRESP;     // sc_in
            sensitive << sPorts[i].RLAST;     // sc_in
            sensitive << sPorts[i].RVALID;    // sc_in
            sensitive << sPorts[i].AWREADY;   // sc_in
            sensitive << sPorts[i].WREADY;    // sc_in
            sensitive << sPorts[i].BID;       // sc_in
            sensitive << sPorts[i].BRESP;     // sc_in
            sensitive << sPorts[i].BVALID;    // sc_in
        }
    }

    ~AXI_BUS() {
        if constexpr (ENABLE_PROFILING) {
            profiler.print_statistics();
        }
        delete[] mPorts;
        delete[] trans_counter;
        delete[] sPorts;
    }

    void arbitrate() {
        // Detect new read transactions
        for (unsigned int i = 0; i < master_num; ++i) {
            if (mPorts[i].ARVALID.read() == 1) {
                id_t new_id = mPorts[i].ARID.read();
                addr_t new_addr = mPorts[i].ARADDR.read();

                if (!is_duplicate_transaction(i, new_id, new_addr, readReq) &&
                    !is_duplicate_transaction(i, new_id, new_addr, readData)) {

                    auto* trans = new AXITransaction<CFG>();
                    trans->ID = new_id;
                    trans->addr = new_addr;
                    trans->bus_allow = true;
                    readReq[i] = trans;
                    trans_counter[i].read++;
                }
            }

            // Detect new write transactions
            if (mPorts[i].AWVALID.read() == 1) {
                id_t new_id = mPorts[i].AWID.read();
                addr_t new_addr = mPorts[i].AWADDR.read();

                if (!is_duplicate_transaction(i, new_id, new_addr, writeReq) &&
                    !is_duplicate_transaction(i, new_id, new_addr, writeData) &&
                    !is_duplicate_transaction(i, new_id, new_addr, writeB)) {

                    auto* trans = new AXITransaction<CFG>();
                    trans->ID = new_id;
                    trans->addr = new_addr;
                    trans->bus_allow = true;
                    writeReq[i] = trans;
                    trans_counter[i].write++;
                }
            }
        }

        // Profiling: contention monitoring
        if constexpr (ENABLE_PROFILING) {
            for (unsigned int i = 0; i < master_num; ++i) {
                profiler.record_contention(
                    mPorts[i].ARVALID.read(), mPorts[i].ARREADY.read(),
                    mPorts[i].AWVALID.read(), mPorts[i].AWREADY.read(),
                    mPorts[i].WVALID.read(), mPorts[i].WREADY.read(),
                    mPorts[i].RVALID.read(), mPorts[i].RREADY.read(),
                    mPorts[i].BVALID.read(), mPorts[i].BREADY.read()
                );
            }
        }

        process_read_address();
        process_read_data();
        process_write_address();
        process_write_data();
        process_write_response();
    }

    bool is_duplicate_transaction(int port, id_t id, addr_t addr,
                                   std::map<id_t, AXITransaction<CFG>*>& queue) {
        if (queue.empty()) return false;

        auto it = queue.find(port);
        if (it != queue.end()) {
            if (it->second->ID == id && it->second->addr == addr) {
                return true;
            }
        }
        return false;
    }

    void arbitrate_conflicts(std::map<id_t, AXITransaction<CFG>*>& queue) {
        for (auto i = queue.begin(); i != queue.end(); ++i) {
            auto j = i;
            ++j;
            for (; j != queue.end(); ++j) {
                if (find_slave_id(i->second->addr) == find_slave_id(j->second->addr)) {
                    if (trans_counter[i->first].read <= trans_counter[j->first].read) {
                        j->second->bus_allow = false;
                    } else {
                        i->second->bus_allow = false;
                    }
                }
            }
        }
    }

    void process_read_address() {
        if (readReq.empty()) return;

        arbitrate_conflicts(readReq);

        for (auto it = readReq.begin(); it != readReq.end();) {
            if (!it->second->bus_allow) {
                it->second->bus_allow = true;
                ++it;
                continue;
            }

            id_t master_port = it->first;
            int slave_id = find_slave_id(it->second->addr);

            sync_master_n_slave_ar(master_port, slave_id);

            if (mPorts[master_port].ARVALID.read() == 1 &&
                mPorts[master_port].ARREADY.read() == 1 &&
                sPorts[slave_id].ARVALID.read() == 1) {
                it->second->ar_released = true;

                // Exclusive Access: Track exclusive read
                if constexpr (ENABLE_EXCLUSIVE) {
                    typename CFG::lock_t lock = mPorts[master_port].ARLOCK.read();
                    if (lock == 1) {  // Exclusive read
                        addr_t addr = mPorts[master_port].ARADDR.read();
                        typename CFG::axi_size_t size = mPorts[master_port].ARSIZE.read();
                        unsigned int size_bytes = 1 << size.to_uint();
                        exclusive_monitor.record_exclusive_read(master_port, addr, size_bytes);
                    }
                }

                // Profiling: AR handshake
                if constexpr (ENABLE_PROFILING) {
                    profiler.record_ar_handshake(
                        mPorts[master_port].ARID.read(),
                        mPorts[master_port].ARADDR.read(),
                        mPorts[master_port].ARLEN.read().to_uint()
                    );
                }

                // Out-of-order: Track ID-based order
                uint32_t order_key = (master_port << 16) | it->second->ID.to_uint();
                read_order_tracker[order_key].push_back(it->second);
            }

            if (it->second->ar_released && mPorts[master_port].ARVALID.read() == 0) {
                readData[master_port] = it->second;
                it = readReq.erase(it);
            } else {
                ++it;
            }
        }
    }

    void process_read_data() {
        if (readData.empty()) return;

        arbitrate_conflicts(readData);

        for (auto it = readData.begin(); it != readData.end();) {
            if (!it->second->bus_allow) {
                it->second->bus_allow = true;
                ++it;
                continue;
            }

            id_t master_port = it->first;
            int slave_id = find_slave_id(it->second->addr);

            sync_master_n_slave_r(master_port, slave_id);

            if (sPorts[slave_id].RVALID.read() == 1 &&
                mPorts[master_port].RLAST.read() == 1 &&
                mPorts[master_port].RVALID.read() == 1 &&
                sPorts[slave_id].RREADY.read() == 1) {

                // Out-of-order: Check if this is the oldest transaction with this ID
                uint32_t order_key = (master_port << 16) | it->second->ID.to_uint();
                auto& order_queue = read_order_tracker[order_key];

                if (!order_queue.empty() && order_queue.front() == it->second) {
                    // This is the oldest with this ID, can complete
                    it->second->r_released = true;
                    order_queue.pop_front();

                    // Profiling: R complete
                    if constexpr (ENABLE_PROFILING) {
                        profiler.record_r_complete(
                            mPorts[master_port].RID.read(),
                            mPorts[master_port].RRESP.read()
                        );
                    }
                }
            }

            if (it->second->r_released && mPorts[master_port].RVALID.read() == 0) {
                delete it->second;
                it = readData.erase(it);
            } else {
                ++it;
            }
        }
    }

    void process_write_address() {
        if (writeReq.empty()) return;

        arbitrate_conflicts(writeReq);

        for (auto it = writeReq.begin(); it != writeReq.end();) {
            if (!it->second->bus_allow) {
                it->second->bus_allow = true;
                ++it;
                continue;
            }

            id_t master_port = it->first;
            int slave_id = find_slave_id(it->second->addr);

            sync_master_n_slave_aw(master_port, slave_id);

            if (mPorts[master_port].AWVALID.read() == 1 &&
                mPorts[master_port].AWREADY.read() == 1 &&
                sPorts[slave_id].AWVALID.read() == 1) {
                it->second->aw_released = true;

                // Exclusive Access: Track exclusive write
                if constexpr (ENABLE_EXCLUSIVE) {
                    typename CFG::lock_t lock = mPorts[master_port].AWLOCK.read();
                    if (lock == 1) {  // Exclusive write
                        addr_t addr = mPorts[master_port].AWADDR.read();
                        typename CFG::axi_size_t size = mPorts[master_port].AWSIZE.read();
                        unsigned int size_bytes = 1 << size.to_uint();
                        // Store lock info in transaction for later use in B response
                        it->second->is_exclusive_write = true;
                        it->second->exclusive_addr = addr;
                        it->second->exclusive_size_bytes = size_bytes;
                    }
                }

                // Profiling: AW handshake
                if constexpr (ENABLE_PROFILING) {
                    profiler.record_aw_handshake(
                        mPorts[master_port].AWID.read(),
                        mPorts[master_port].AWADDR.read(),
                        mPorts[master_port].AWLEN.read().to_uint()
                    );
                }

                // Out-of-order: Track ID-based order
                uint32_t order_key = (master_port << 16) | it->second->ID.to_uint();
                write_order_tracker[order_key].push_back(it->second);
            }

            if (it->second->aw_released && mPorts[master_port].AWVALID.read() == 0) {
                writeData[master_port] = it->second;
                it = writeReq.erase(it);
            } else {
                ++it;
            }
        }
    }

    void process_write_data() {
        if (writeData.empty()) return;

        arbitrate_conflicts(writeData);

        for (auto it = writeData.begin(); it != writeData.end();) {
            if (!it->second->bus_allow) {
                it->second->bus_allow = true;
                ++it;
                continue;
            }

            id_t master_port = it->first;
            int slave_id = find_slave_id(it->second->addr);

            sync_master_n_slave_w(master_port, slave_id);

            if (mPorts[master_port].WVALID.read() == 1 &&
                mPorts[master_port].WLAST.read() == 1 &&
                sPorts[slave_id].WREADY.read() == 1) {
                it->second->w_released = true;
            }

            if (it->second->w_released && sPorts[slave_id].WVALID.read() == 0) {
                writeB[master_port] = it->second;
                it = writeData.erase(it);
            } else {
                ++it;
            }
        }
    }

    void process_write_response() {
        if (writeB.empty()) return;

        arbitrate_conflicts(writeB);

        for (auto it = writeB.begin(); it != writeB.end();) {
            if (!it->second->bus_allow) {
                it->second->bus_allow = true;
                ++it;
                continue;
            }

            id_t master_port = it->first;
            int slave_id = find_slave_id(it->second->addr);

            // Sync B channel signals (BID, BVALID, BREADY)
            mPorts[master_port].BID.write(sPorts[slave_id].BID.read());
            sPorts[slave_id].BREADY.write(mPorts[master_port].BREADY.read());
            mPorts[master_port].BVALID.write(sPorts[slave_id].BVALID.read());

            if (sPorts[slave_id].BVALID.read() == 1 &&
                mPorts[master_port].BREADY.read() == 1) {

                // Out-of-order: Check if this is the oldest transaction with this ID
                uint32_t order_key = (master_port << 16) | it->second->ID.to_uint();
                auto& order_queue = write_order_tracker[order_key];

                if (!order_queue.empty() && order_queue.front() == it->second) {
                    // This is the oldest with this ID, can complete
                    it->second->b_released = true;
                    order_queue.pop_front();

                    // Exclusive Access: Override BRESP if exclusive write, or monitor normal write
                    if constexpr (ENABLE_EXCLUSIVE) {
                        if (it->second->is_exclusive_write) {
                            typename CFG::lock_t lock = 1;  // Exclusive
                            AXIResponse resp = exclusive_monitor.get_write_response(
                                master_port,
                                it->second->exclusive_addr,
                                it->second->exclusive_size_bytes,
                                lock
                            );
                            // Override slave's BRESP with exclusive monitor's response
                            typename CFG::resp_t bresp = static_cast<uint8_t>(resp);
                            mPorts[master_port].BRESP.write(bresp);
                            // Also set transaction resp directly to avoid timing issues
                            it->second->resp = bresp;
                        } else {
                            // Normal write - notify exclusive monitor to clear overlapping reservations
                            typename CFG::lock_t lock = 0;  // Normal
                            addr_t addr = it->second->addr;
                            typename CFG::axi_size_t size = it->second->size;
                            unsigned int size_bytes = 1 << size.to_uint();
                            exclusive_monitor.get_write_response(master_port, addr, size_bytes, lock);
                            // For normal writes, use slave's BRESP
                            mPorts[master_port].BRESP.write(sPorts[slave_id].BRESP.read());
                        }
                    } else {
                        // No exclusive access support - just use slave's BRESP
                        mPorts[master_port].BRESP.write(sPorts[slave_id].BRESP.read());
                    }

                    // Profiling: B complete
                    if constexpr (ENABLE_PROFILING) {
                        profiler.record_b_complete(
                            mPorts[master_port].BID.read(),
                            mPorts[master_port].BRESP.read()
                        );
                    }
                }
            }

            if (it->second->b_released && mPorts[master_port].BVALID.read() == 0) {
                delete it->second;
                it = writeB.erase(it);
            } else {
                ++it;
            }
        }
    }

    void sync_master_n_slave_ar(int master, int slave) {
        sPorts[slave].ARID.write(mPorts[master].ARID.read());
        sPorts[slave].ARADDR.write(mPorts[master].ARADDR.read());
        sPorts[slave].ARLEN.write(mPorts[master].ARLEN.read());
        sPorts[slave].ARSIZE.write(mPorts[master].ARSIZE.read());
        sPorts[slave].ARBURST.write(mPorts[master].ARBURST.read());
        sPorts[slave].ARLOCK.write(mPorts[master].ARLOCK.read());
        sPorts[slave].ARCACHE.write(mPorts[master].ARCACHE.read());
        sPorts[slave].ARPROT.write(mPorts[master].ARPROT.read());
        sPorts[slave].ARQOS.write(mPorts[master].ARQOS.read());
        sPorts[slave].ARREGION.write(mPorts[master].ARREGION.read());
        sPorts[slave].ARVALID.write(mPorts[master].ARVALID.read());
        mPorts[master].ARREADY.write(sPorts[slave].ARREADY.read());
    }

    void sync_master_n_slave_r(int master, int slave) {
        sPorts[slave].RREADY.write(mPorts[master].RREADY.read());
        mPorts[master].RID.write(sPorts[slave].RID.read());
        mPorts[master].RDATA.write(sPorts[slave].RDATA.read());
        mPorts[master].RRESP.write(sPorts[slave].RRESP.read());
        mPorts[master].RLAST.write(sPorts[slave].RLAST.read());
        mPorts[master].RVALID.write(sPorts[slave].RVALID.read());
    }

    void sync_master_n_slave_aw(int master, int slave) {
        sPorts[slave].AWID.write(mPorts[master].AWID.read());
        sPorts[slave].AWADDR.write(mPorts[master].AWADDR.read());
        sPorts[slave].AWLEN.write(mPorts[master].AWLEN.read());
        sPorts[slave].AWSIZE.write(mPorts[master].AWSIZE.read());
        sPorts[slave].AWBURST.write(mPorts[master].AWBURST.read());
        sPorts[slave].AWLOCK.write(mPorts[master].AWLOCK.read());
        sPorts[slave].AWCACHE.write(mPorts[master].AWCACHE.read());
        sPorts[slave].AWPROT.write(mPorts[master].AWPROT.read());
        sPorts[slave].AWQOS.write(mPorts[master].AWQOS.read());
        sPorts[slave].AWREGION.write(mPorts[master].AWREGION.read());
        sPorts[slave].AWVALID.write(mPorts[master].AWVALID.read());
        mPorts[master].AWREADY.write(sPorts[slave].AWREADY.read());
    }

    void sync_master_n_slave_w(int master, int slave) {
        if constexpr (CFG::HAS_WID) {
            sPorts[slave].WID.write(mPorts[master].WID.read());
        }
        sPorts[slave].WDATA.write(mPorts[master].WDATA.read());
        sPorts[slave].WSTRB.write(mPorts[master].WSTRB.read());
        sPorts[slave].WLAST.write(mPorts[master].WLAST.read());
        sPorts[slave].WVALID.write(mPorts[master].WVALID.read());
        mPorts[master].WREADY.write(sPorts[slave].WREADY.read());
    }

    void sync_master_n_slave_b(int master, int slave) {
        mPorts[master].BID.write(sPorts[slave].BID.read());
        mPorts[master].BRESP.write(sPorts[slave].BRESP.read());
        sPorts[slave].BREADY.write(mPorts[master].BREADY.read());
        mPorts[master].BVALID.write(sPorts[slave].BVALID.read());
    }

    int find_slave_id(addr_t addr) {
        return memory_map.find_slave_id(addr);
    }
};

#endif // AXI_BUS_H
