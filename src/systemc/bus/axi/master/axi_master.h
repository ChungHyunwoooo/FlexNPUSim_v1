#ifndef FLEXNPU_TRANSPORT_AXI_MASTER_AXI_MASTER_H
#define FLEXNPU_TRANSPORT_AXI_MASTER_AXI_MASTER_H

// Signal-level AXI master (R2).
//
// Design:
//   * 5 independent SC_THREADs — one per AR / R / AW / W / B channel.
//   * TicketPool allocates per-transaction slots; each slot carries the
//     request payload and receives responses. Same-ID ordering is enforced
//     by a per-ID FIFO of tickets consumed in R/B-arrival order (AXI4 rule).
//   * Per-slot TransferProfile is filled at issue / AR-AW handshake /
//     first-beat / last-beat / completion; its resp is read back through
//     read_profile() / write_profile().
//
// Public API — non-blocking and blocking variants both supported:
//     int tk = issue_read(addr, beats, size_bytes, id);
//     wait(read_done_event(tk));  // or poll read_done(tk)
//     auto beats = peek_read(tk); release_read(tk);
//
//     auto beats = read_burst(addr, beats, size_bytes, id);   // blocks
//     write_burst(addr, size_bytes, data, strb, id);          // blocks

#include <systemc.h>

#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include "systemc/bus/axi/axi_config.h"         // CommonConfig
#include "systemc/bus/axi/axi_signal_ports.h"   // MASTER_PORTS<Spec>
#include "systemc/bus/axi/master/axi_ticket_pool.h"
#include "systemc/bus/axi/spec/axi_spec.h"
#include "systemc/bus/common/transfer_profile.h"

namespace flexnpu_sim::transport::axi {

template <class Spec>
class Master : public sc_core::sc_module {
public:
    using id_t   = typename Spec::id_t;
    using addr_t = typename Spec::addr_t;
    using data_t = typename Spec::data_t;
    using strb_t = typename Spec::strb_t;
    using len_t  = typename Spec::len_t;
    using resp_t = typename Spec::resp_t;

    sc_core::sc_in<bool> clk{"clk"};
    MASTER_PORTS<Spec>   port;

    explicit Master(sc_core::sc_module_name name,
                    const ::axi::CommonConfig& ccfg = ::axi::CommonConfig{})
        : sc_core::sc_module(name),
          ccfg_(ccfg),
          read_tickets_(ccfg.max_outstanding_reads),
          write_tickets_(ccfg.max_outstanding_writes)
    {
        read_slots_.resize(read_tickets_.capacity());
        write_slots_.resize(write_tickets_.capacity());
        for (auto& s : read_slots_)  s = std::make_unique<ReadSlot>();
        for (auto& s : write_slots_) s = std::make_unique<WriteSlot>();

        SC_HAS_PROCESS(Master);
        SC_THREAD(ar_thread);
        SC_THREAD(r_thread);
        SC_THREAD(aw_thread);
        SC_THREAD(w_thread);
        SC_THREAD(b_thread);
    }

    // ---- Non-blocking issue ------------------------------------------------

    int issue_read(uint64_t addr, unsigned beats, unsigned size_bytes,
                   unsigned id = 0) {
        const int tk = read_tickets_.acquire();
        auto& s = *read_slots_[tk];
        s.reset();
        s.id             = id;
        s.addr           = addr;
        s.len            = (beats == 0 ? 0u : beats - 1u);
        s.size           = size_code(size_bytes);
        s.profile.is_read         = true;
        s.profile.ticket_slot     = static_cast<uint32_t>(tk);
        s.profile.addr            = addr;
        s.profile.beats           = beats;
        s.profile.beat_size_bytes = size_bytes;
        s.profile.issue_ns        = now_ns();
        s.data.reserve(beats);

        ar_queue_.push_back(tk);
        id_read_order_[id].push_back(tk);
        ar_event_.notify(sc_core::SC_ZERO_TIME);
        return tk;
    }

    int issue_write(uint64_t addr, unsigned size_bytes,
                    const std::vector<data_t>& data,
                    const std::vector<strb_t>& strb,
                    unsigned id = 0) {
        const int tk = write_tickets_.acquire();
        auto& s = *write_slots_[tk];
        s.reset();
        s.id   = id;
        s.addr = addr;
        s.len  = (data.empty() ? 0u
                               : static_cast<unsigned>(data.size()) - 1u);
        s.size = size_code(size_bytes);
        s.data = data;
        s.strb = strb;
        s.profile.is_read         = false;
        s.profile.ticket_slot     = static_cast<uint32_t>(tk);
        s.profile.addr            = addr;
        s.profile.beats           = static_cast<uint32_t>(data.size());
        s.profile.beat_size_bytes = size_bytes;
        s.profile.issue_ns        = now_ns();

        aw_queue_.push_back(tk);
        w_queue_.push_back(tk);
        id_write_order_[id].push_back(tk);
        aw_event_.notify(sc_core::SC_ZERO_TIME);
        w_event_.notify(sc_core::SC_ZERO_TIME);
        return tk;
    }

    // ---- Completion queries ------------------------------------------------

    bool               read_done      (int tk) const { return tk >= 0 && read_slots_[tk]->done; }
    sc_core::sc_event& read_done_event(int tk)       { return read_slots_[tk]->done_event; }
    std::vector<data_t> peek_read     (int tk) const { return read_slots_[tk]->data; }
    void               release_read   (int tk)       { read_tickets_.release(tk); }

    bool               write_done      (int tk) const { return tk >= 0 && write_slots_[tk]->done; }
    sc_core::sc_event& write_done_event(int tk)       { return write_slots_[tk]->done_event; }
    resp_t             write_resp      (int tk) const { return write_slots_[tk]->resp; }
    void               release_write   (int tk)       { write_tickets_.release(tk); }

    // ---- Blocking convenience ---------------------------------------------

    std::vector<data_t> read_burst(uint64_t addr, unsigned beats,
                                   unsigned size_bytes, unsigned id = 0) {
        const int tk = issue_read(addr, beats, size_bytes, id);
        while (!read_slots_[tk]->done) {
            sc_core::wait(read_slots_[tk]->done_event);
        }
        auto out = std::move(read_slots_[tk]->data);
        release_read(tk);
        return out;
    }

    void write_burst(uint64_t addr, unsigned size_bytes,
                     const std::vector<data_t>& data,
                     const std::vector<strb_t>& strb,
                     unsigned id = 0) {
        const int tk = issue_write(addr, size_bytes, data, strb, id);
        while (!write_slots_[tk]->done) {
            sc_core::wait(write_slots_[tk]->done_event);
        }
        release_write(tk);
    }

    // ---- Introspection -----------------------------------------------------

    unsigned outstanding_reads()  const { return read_tickets_.in_use();  }
    unsigned outstanding_writes() const { return write_tickets_.in_use(); }

    // Profile of last transaction that lived in @p tk (valid until release).
    const flexnpu_sim::transport::common::TransferProfile&
    read_profile (int tk) const { return read_slots_[tk]->profile; }
    const flexnpu_sim::transport::common::TransferProfile&
    write_profile(int tk) const { return write_slots_[tk]->profile; }

private:
    static uint64_t now_ns() {
        return static_cast<uint64_t>(
            sc_core::sc_time_stamp().to_default_time_units());
    }

    // AXI RRESP / BRESP encoding: 0=OKAY, 1=EXOKAY, 2=SLVERR, 3=DECERR.
    static flexnpu_sim::transport::TransportResp map_resp(resp_t r) {
        switch (r.to_uint()) {
            case 0: return flexnpu_sim::transport::TransportResp::Ok;
            case 1: return flexnpu_sim::transport::TransportResp::ExclusiveOk;
            case 2: return flexnpu_sim::transport::TransportResp::SlaveError;
            case 3: return flexnpu_sim::transport::TransportResp::DecodeError;
            default: return flexnpu_sim::transport::TransportResp::Other;
        }
    }
    static unsigned size_code(unsigned bytes) {
        unsigned b = (bytes == 0 ? 1u : bytes);
        unsigned code = 0;
        while (b > 1) { b >>= 1; ++code; }
        return code;
    }

    struct ReadSlot {
        unsigned id = 0;
        uint64_t addr = 0;
        unsigned len = 0;
        unsigned size = 0;
        std::vector<data_t> data;
        bool done = false;
        resp_t resp{0};
        sc_core::sc_event done_event;
        flexnpu_sim::transport::common::TransferProfile profile;
        void reset() {
            id = 0; addr = 0; len = 0; size = 0;
            data.clear(); done = false; resp = 0;
            profile = {};
        }
    };
    struct WriteSlot {
        unsigned id = 0;
        uint64_t addr = 0;
        unsigned len = 0;
        unsigned size = 0;
        std::vector<data_t> data;
        std::vector<strb_t> strb;
        bool done = false;
        resp_t resp{0};
        sc_core::sc_event done_event;
        flexnpu_sim::transport::common::TransferProfile profile;
        void reset() {
            id = 0; addr = 0; len = 0; size = 0;
            data.clear(); strb.clear(); done = false; resp = 0;
            profile = {};
        }
    };

    // ---- AR thread: drive read address --------------------------------
    void ar_thread() {
        port.ARVALID.write(false);
        while (true) {
            while (ar_queue_.empty()) sc_core::wait(ar_event_);
            const int tk = ar_queue_.front();
            ar_queue_.pop_front();
            auto& s = *read_slots_[tk];

            port.ARID   .write(id_t(s.id));
            port.ARADDR .write(addr_t(s.addr));
            port.ARLEN  .write(len_t(s.len));
            port.ARSIZE .write(typename Spec::axi_size_t(s.size));
            port.ARBURST.write(typename Spec::burst_t(1));  // INCR
            port.ARLOCK .write(typename Spec::lock_t(0));
            port.ARCACHE.write(typename Spec::cache_t(0));
            port.ARPROT .write(typename Spec::prot_t(0));
            if constexpr (Spec::HAS_QOS)    port.ARQOS.write(typename Spec::qos_t(0));
            if constexpr (Spec::HAS_REGION) port.ARREGION.write(typename Spec::region_t(0));
            port.ARVALID.write(true);

            do {
                sc_core::wait(clk.posedge_event());
            } while (!port.ARREADY.read());

            port.ARVALID.write(false);
            s.profile.ar_aw_ns = now_ns();
        }
    }

    // ---- R thread: accept read beats, route to ticket via per-ID FIFO --
    void r_thread() {
        port.RREADY.write(true);
        while (true) {
            sc_core::wait(clk.posedge_event());
            if (!port.RVALID.read()) continue;

            const unsigned rid = static_cast<unsigned>(port.RID.read().to_uint());
            auto it = id_read_order_.find(rid);
            if (it == id_read_order_.end() || it->second.empty()) {
                continue;
            }
            const int tk = it->second.front();
            auto& s = *read_slots_[tk];

            if (s.data.empty()) s.profile.first_beat_ns = now_ns();
            s.data.push_back(port.RDATA.read());
            s.resp = port.RRESP.read();

            if (port.RLAST.read()) {
                it->second.pop_front();
                s.profile.last_beat_ns = now_ns();
                s.profile.complete_ns  = s.profile.last_beat_ns;
                s.profile.resp =
                    (s.resp.to_uint() == 0u)
                        ? flexnpu_sim::transport::TransportResp::Ok
                        : map_resp(s.resp);
                s.done = true;
                s.done_event.notify(sc_core::SC_ZERO_TIME);
            }
        }
    }

    // ---- AW thread: drive write address -------------------------------
    void aw_thread() {
        port.AWVALID.write(false);
        while (true) {
            while (aw_queue_.empty()) sc_core::wait(aw_event_);
            const int tk = aw_queue_.front();
            aw_queue_.pop_front();
            auto& s = *write_slots_[tk];

            port.AWID   .write(id_t(s.id));
            port.AWADDR .write(addr_t(s.addr));
            port.AWLEN  .write(len_t(s.len));
            port.AWSIZE .write(typename Spec::axi_size_t(s.size));
            port.AWBURST.write(typename Spec::burst_t(1));  // INCR
            port.AWLOCK .write(typename Spec::lock_t(0));
            port.AWCACHE.write(typename Spec::cache_t(0));
            port.AWPROT .write(typename Spec::prot_t(0));
            if constexpr (Spec::HAS_QOS)    port.AWQOS.write(typename Spec::qos_t(0));
            if constexpr (Spec::HAS_REGION) port.AWREGION.write(typename Spec::region_t(0));
            port.AWVALID.write(true);

            do {
                sc_core::wait(clk.posedge_event());
            } while (!port.AWREADY.read());

            port.AWVALID.write(false);
            s.profile.ar_aw_ns = now_ns();
        }
    }

    // ---- W thread: stream write beats ---------------------------------
    void w_thread() {
        port.WVALID.write(false);
        port.WLAST .write(false);
        while (true) {
            while (w_queue_.empty()) sc_core::wait(w_event_);
            const int tk = w_queue_.front();
            w_queue_.pop_front();
            auto& s = *write_slots_[tk];

            for (unsigned i = 0; i <= s.len; ++i) {
                if constexpr (Spec::HAS_WID) port.WID.write(id_t(s.id));
                port.WDATA .write(s.data[i]);
                port.WSTRB .write(s.strb[i]);
                port.WLAST .write(i == s.len);
                port.WVALID.write(true);

                do {
                    sc_core::wait(clk.posedge_event());
                } while (!port.WREADY.read());

                if (i == 0)     s.profile.first_beat_ns = now_ns();
                if (i == s.len) s.profile.last_beat_ns  = now_ns();
            }
            port.WVALID.write(false);
            port.WLAST .write(false);
        }
    }

    // ---- B thread: accept write response ------------------------------
    void b_thread() {
        port.BREADY.write(true);
        while (true) {
            sc_core::wait(clk.posedge_event());
            if (!port.BVALID.read()) continue;

            const unsigned bid = static_cast<unsigned>(port.BID.read().to_uint());
            auto it = id_write_order_.find(bid);
            if (it == id_write_order_.end() || it->second.empty()) {
                continue;
            }
            const int tk = it->second.front();
            it->second.pop_front();
            auto& s = *write_slots_[tk];
            s.resp = port.BRESP.read();
            s.profile.complete_ns = now_ns();
            s.profile.resp = map_resp(s.resp);
            s.done = true;
            s.done_event.notify(sc_core::SC_ZERO_TIME);
        }
    }

    ::axi::CommonConfig ccfg_;
    TicketPool          read_tickets_;
    TicketPool          write_tickets_;
    std::vector<std::unique_ptr<ReadSlot>>  read_slots_;
    std::vector<std::unique_ptr<WriteSlot>> write_slots_;

    std::deque<int> ar_queue_;
    std::deque<int> aw_queue_;
    std::deque<int> w_queue_;

    std::unordered_map<unsigned, std::deque<int>> id_read_order_;
    std::unordered_map<unsigned, std::deque<int>> id_write_order_;

    sc_core::sc_event ar_event_;
    sc_core::sc_event aw_event_;
    sc_core::sc_event w_event_;
};

}  // namespace flexnpu_sim::transport::axi

#endif  // FLEXNPU_TRANSPORT_AXI_MASTER_AXI_MASTER_H
