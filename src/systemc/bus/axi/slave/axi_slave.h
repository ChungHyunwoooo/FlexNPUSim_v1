#ifndef FLEXNPU_TRANSPORT_AXI_SLAVE_AXI_SLAVE_H
#define FLEXNPU_TRANSPORT_AXI_SLAVE_AXI_SLAVE_H

// Signal-level AXI slave (R3).
//
// Design points that distinguish this from the legacy AXI_slave_if:
//   * 5 independent SC_THREADs — one per AR / AW / W / R / B channel, plus a
//     mem_tick SC_METHOD that advances the memory model once per cycle.
//   * Per-transaction state objects carried through sc_fifo queues. No
//     cross-transaction shared state.
//
// Memory timing is REAL hardware here — no latency formula:
//   * A memory model (MemPort) is driven like a DRAM controller: the slave
//     asks will_accept() before accepting an AR/AW (ready/valid backpressure),
//     submit()s each accepted transaction tagged with a token, tick()s the
//     model every cycle, and streams the response only once pop_completed()
//     reports that token done. Bank parallelism and backpressure live inside
//     the memory model; the shared read-data bus is serialized by the R
//     channel below (one beat per cycle), not by any formula.
//   * Responses stay in AR/AW-arrival order (same-ID FIFO — the DMA relies on
//     it for head-drain). A later burst that finishes first waits its turn.
//
// Design invariants (upstream must not violate):
//   * WREADY only asserts while this thread holds a live WriteOp (see w_thread).
//   * ARREADY/AWREADY are gated on the memory model's will_accept and on queue
//     space; an upstream that ignores a deasserted ready and pushes anyway
//     would block the acceptor thread.

#include <systemc.h>

#include <cstdint>
#include <deque>
#include <functional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "systemc/bus/axi/axi_config.h"         // CommonConfig
#include "systemc/bus/axi/axi_signal_ports.h"   // SLAVE_PORTS<Spec>
#include "systemc/bus/axi/spec/axi_spec.h"
#include "systemc/bus/axi/slave/axi_memory_backend.h"

namespace flexnpu_sim::transport::axi {

template <class Spec>
class Slave : public sc_core::sc_module {
public:
    using id_t    = typename Spec::id_t;
    using addr_t  = typename Spec::addr_t;
    using data_t  = typename Spec::data_t;
    using strb_t  = typename Spec::strb_t;
    using len_t   = typename Spec::len_t;
    using resp_t  = typename Spec::resp_t;

    static constexpr unsigned BEAT_BYTES = Spec::DATA_W / 8;

    sc_core::sc_in<bool> clk{"clk"};
    SLAVE_PORTS<Spec>    port;

    explicit Slave(sc_core::sc_module_name name,
                   MemoryBackend& backend,
                   const ::axi::CommonConfig& ccfg = ::axi::CommonConfig{})
        : sc_core::sc_module(name),
          backend_(backend),
          ccfg_(ccfg),
          rd_q_("rd_q", 256),
          aw_q_("aw_q", ccfg.max_outstanding_writes ? ccfg.max_outstanding_writes : 16),
          b_q_ ("b_q",  256)
    {
        SC_HAS_PROCESS(Slave);
        SC_THREAD(ar_thread);
        SC_THREAD(aw_thread);
        SC_THREAD(w_thread);
        SC_THREAD(r_thread);
        SC_THREAD(b_thread);
        SC_METHOD(mem_tick);
        sensitive << clk.pos();
        dont_initialize();
    }

    // Memory-model port — the real memory-controller interface. Left unset, the
    // slave behaves as a zero-latency memory (accept always, complete at once).
    struct MemPort {
        std::function<bool(uint64_t addr, bool is_write)>                 will_accept;
        std::function<void(uint64_t token, uint64_t addr,
                           uint32_t size, bool is_write)>                 submit;
        std::function<void()>                                            tick;
        std::function<bool(uint64_t& token)>                             pop_completed;
    };
    void set_mem_port(MemPort p) { mem_ = std::move(p); }

private:
    // Per-transaction state objects — no shared mutable scalars.
    struct ReadOp {
        unsigned id     = 0;
        uint64_t addr   = 0;
        unsigned stride = 1;
        uint64_t token  = 0;
        std::vector<std::vector<uint8_t>> beats;   // pre-fetched at AR arrival
    };
    struct WriteOp {
        unsigned id       = 0;
        uint64_t addr     = 0;
        unsigned beats    = 0;
        unsigned stride   = 1;
        unsigned received = 0;
    };
    struct RespB {
        unsigned id     = 0;
        uint64_t addr   = 0;
        unsigned beats  = 0;
        unsigned stride = 1;
        uint64_t token  = 0;
    };

    friend std::ostream& operator<<(std::ostream& os, const ReadOp&  o) {
        return os << "ReadOp(id="  << o.id << ",beats=" << o.beats.size() << ")";
    }
    friend std::ostream& operator<<(std::ostream& os, const WriteOp& o) {
        return os << "WriteOp(id=" << o.id << ",beats=" << o.beats
                  << ",received=" << o.received << ")";
    }
    friend std::ostream& operator<<(std::ostream& os, const RespB&   r) {
        return os << "RespB(id="   << r.id << ")";
    }

    // ---- Byte/data conversions -------------------------------------------
    static std::vector<uint8_t> data_to_bytes(const data_t& d, unsigned n) {
        std::vector<uint8_t> out(n);
        for (unsigned i = 0; i < n; ++i) {
            out[i] = static_cast<uint8_t>(
                d.range(i * 8 + 7, i * 8).to_uint() & 0xFFu);
        }
        return out;
    }
    static data_t bytes_to_data(const std::vector<uint8_t>& bytes) {
        data_t d = 0;
        for (unsigned i = 0; i < bytes.size(); ++i) {
            d.range(i * 8 + 7, i * 8) = bytes[i];
        }
        return d;
    }
    static std::vector<bool> strb_to_enable(const strb_t& s, unsigned n) {
        std::vector<bool> e(n);
        for (unsigned i = 0; i < n; ++i) {
            e[i] = (s.range(i, i).to_uint() & 1u) != 0u;
        }
        return e;
    }

    // ---- Memory-model helpers --------------------------------------------
    bool mem_will_accept(uint64_t addr, bool is_write) const {
        return !mem_.will_accept || mem_.will_accept(addr, is_write);
    }
    bool token_completed(uint64_t token) const {
        // No memory model -> zero latency, everything is instantly complete.
        return !mem_.submit || completed_.count(token) != 0;
    }

    // Advance the memory model once per cycle and collect completions.
    void mem_tick() {
        if (mem_.tick) mem_.tick();
        if (mem_.pop_completed) {
            uint64_t t;
            while (mem_.pop_completed(t)) completed_.insert(t);
        }
    }

    // ---- AR thread: gate on will_accept, pre-fetch beats, submit --------
    void ar_thread() {
        port.ARREADY.write(false);
        while (true) {
            sc_core::wait(clk.posedge_event());
            const bool ready = mem_will_accept(0, /*is_write=*/false)
                             && rd_q_.num_free() > 0;
            port.ARREADY.write(ready);
            if (ready && port.ARVALID.read()) {
                ReadOp op;
                op.id             = port.ARID.read().to_uint();
                const uint64_t a  = port.ARADDR.read().to_uint64();
                const unsigned n  = port.ARLEN.read().to_uint() + 1u;
                const unsigned st = 1u << port.ARSIZE.read().to_uint();
                op.addr           = a;
                op.stride         = st;
                op.token          = next_token_++;
                op.beats.reserve(n);
                for (unsigned i = 0; i < n; ++i) {
                    op.beats.push_back(backend_.read(a + i * st, st));
                }
                if (mem_.submit) mem_.submit(op.token, a, n * st, /*is_write=*/false);
                rd_q_.write(op);
            }
        }
    }

    // ---- AW thread: latch AW handshake, enqueue WriteOp -----------------
    void aw_thread() {
        port.AWREADY.write(false);
        while (true) {
            sc_core::wait(clk.posedge_event());
            const bool ready = aw_q_.num_free() > 0;
            port.AWREADY.write(ready);
            if (ready && port.AWVALID.read()) {
                WriteOp op;
                op.id     = port.AWID.read().to_uint();
                op.addr   = port.AWADDR.read().to_uint64();
                op.beats  = port.AWLEN.read().to_uint() + 1u;
                op.stride = 1u << port.AWSIZE.read().to_uint();
                aw_q_.write(op);
            }
        }
    }

    // ---- W thread: pair oldest AW with W beats, commit, submit ----------
    void w_thread() {
        port.WREADY.write(false);
        while (true) {
            WriteOp op = aw_q_.read();           // blocks until AW available
            port.WREADY.write(true);
            while (op.received < op.beats) {
                do {
                    sc_core::wait(clk.posedge_event());
                } while (!port.WVALID.read());

                const auto bytes  = data_to_bytes(port.WDATA.read(), op.stride);
                const auto enable = strb_to_enable(port.WSTRB.read(), op.stride);
                backend_.write(op.addr + op.received * op.stride, bytes, enable);
                ++op.received;
                if (port.WLAST.read()) break;
            }
            port.WREADY.write(false);
            // Submit the committed write to the memory model for timing.
            while (!mem_will_accept(op.addr, /*is_write=*/true)) {
                sc_core::wait(clk.posedge_event());
            }
            const uint64_t token = next_token_++;
            if (mem_.submit)
                mem_.submit(token, op.addr, op.beats * op.stride, /*is_write=*/true);
            b_q_.write(RespB{op.id, op.addr, op.beats, op.stride, token});
        }
    }

    // ---- R thread: stream read beats in AR-arrival order ---------------
    void r_thread() {
        port.RVALID.write(false);
        port.RLAST.write(false);
        while (true) {
            ReadOp op = rd_q_.read();            // blocks until AR submitted
            // Wait until the memory model reports this transaction complete.
            while (!token_completed(op.token)) {
                sc_core::wait(clk.posedge_event());
            }
            completed_.erase(op.token);
            const unsigned n = static_cast<unsigned>(op.beats.size());
            for (unsigned i = 0; i < n; ++i) {
                port.RID.write(id_t(op.id));
                port.RDATA.write(bytes_to_data(op.beats[i]));
                port.RRESP.write(resp_t(0));
                port.RLAST.write(i == n - 1u);
                port.RVALID.write(true);
                do {
                    sc_core::wait(clk.posedge_event());
                } while (!port.RREADY.read());
            }
            port.RVALID.write(false);
            port.RLAST.write(false);
        }
    }

    // ---- B thread: emit write responses in AW-arrival order ------------
    void b_thread() {
        port.BVALID.write(false);
        while (true) {
            RespB r = b_q_.read();               // blocks until WLAST seen
            while (!token_completed(r.token)) {
                sc_core::wait(clk.posedge_event());
            }
            completed_.erase(r.token);
            port.BID.write(id_t(r.id));
            port.BRESP.write(resp_t(0));
            port.BVALID.write(true);
            do {
                sc_core::wait(clk.posedge_event());
            } while (!port.BREADY.read());
            port.BVALID.write(false);
        }
    }

    MemoryBackend&      backend_;
    ::axi::CommonConfig ccfg_;
    MemPort             mem_;
    uint64_t            next_token_ = 0;
    std::unordered_set<uint64_t> completed_;

    sc_core::sc_fifo<ReadOp>  rd_q_;
    sc_core::sc_fifo<WriteOp> aw_q_;
    sc_core::sc_fifo<RespB>   b_q_;
};

}  // namespace flexnpu_sim::transport::axi

#endif  // FLEXNPU_TRANSPORT_AXI_SLAVE_AXI_SLAVE_H
