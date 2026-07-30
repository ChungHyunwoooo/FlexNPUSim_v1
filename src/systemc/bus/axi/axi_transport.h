#ifndef FLEXNPU_TRANSPORT_AXI_AXI_TRANSPORT_H
#define FLEXNPU_TRANSPORT_AXI_AXI_TRANSPORT_H

// AxiTransport<Spec> — adapts R2 axi::Master<Spec> to the runtime-
// polymorphic MemoryMappedTransport interface. Callers (DmaEngine,
// Processor, NpuController) can hold a MemoryMappedTransport* and
// remain ignorant of AXI ID / beat width / strobe packing, while this
// adapter translates BurstReq into R2's ticket-based master API.
//
// Lifecycle:
//   * Construct with a reference to an already-elaborated Master and a
//     Capabilities descriptor built from the master's CommonConfig.
//   * issue_read / issue_write return a Ticket whose `id` is the R2
//     ticket slot (uint32). wait_* translates slot-back to the R2
//     accessor and releases the ticket.
//
// Data encoding:
//   * Bytes come in as std::vector<uint8_t>. Per-beat little-endian
//     packing into Spec::data_t; strb is '1 where present and inferred
//     from data.size() otherwise.
//   * ReadResult.data points into a per-ticket std::vector<uint8_t>
//     owned by this AxiTransport. The buffer stays valid until the
//     caller issues a new transfer on the same ticket slot.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "systemc/bus/axi/master/axi_master.h"
#include "systemc/bus/burst_req.h"
#include "systemc/bus/capabilities.h"
#include "systemc/bus/memory_mapped_transport.h"

namespace flexnpu_sim::transport::axi {

template <class Spec>
class AxiTransport : public flexnpu_sim::transport::MemoryMappedTransport {
public:
    using MasterT = Master<Spec>;
    using data_t  = typename Spec::data_t;
    using strb_t  = typename Spec::strb_t;

    static constexpr unsigned BEAT_BYTES = Spec::DATA_W / 8;

    AxiTransport(MasterT& m,
                 const flexnpu_sim::transport::Capabilities& caps)
        : master_(m), caps_(caps),
          read_buffers_ (caps.max_outstanding_reads),
          read_beat_bytes_(caps.max_outstanding_reads, BEAT_BYTES),
          write_buffers_(caps.max_outstanding_writes) {}

    // ---- MemoryMappedTransport --------------------------------------------

    flexnpu_sim::transport::Ticket
    issue_read(const flexnpu_sim::transport::BurstReq& req) override {
        const unsigned beats      = req.beats;
        const unsigned size_bytes = req.beat_size_bytes();
        const int slot = master_.issue_read(req.addr, beats, size_bytes,
                                            /*id=*/req.attr.qos);
        // Record the per-beat width so wait_read unpacks only the live bytes
        // (the signal type is sized to the compiled max, not this transfer).
        read_beat_bytes_[slot] = size_bytes;
        return make_ticket(slot);
    }

    flexnpu_sim::transport::Ticket
    issue_write(const flexnpu_sim::transport::BurstReq& req,
                const std::vector<uint8_t>& data,
                const std::vector<uint8_t>& strobe = {}) override {
        const unsigned beats      = req.beats;
        const unsigned size_bytes = req.beat_size_bytes();

        std::vector<data_t> beat_data(beats);
        std::vector<strb_t> beat_strb(beats);
        pack_beats(data, strobe, beats, size_bytes, beat_data, beat_strb);

        const int slot = master_.issue_write(req.addr, size_bytes,
                                             beat_data, beat_strb,
                                             /*id=*/req.attr.qos);
        return make_ticket(slot);
    }

    flexnpu_sim::transport::ReadResult
    wait_read(flexnpu_sim::transport::Ticket tk) override {
        const int slot = static_cast<int>(tk.id);
        while (!master_.read_done(slot)) {
            sc_core::wait(master_.read_done_event(slot));
        }
        const auto beats = master_.peek_read(slot);
        const unsigned beat_bytes = read_beat_bytes_[slot];
        auto& buf = read_buffers_[slot];
        buf.clear();
        buf.reserve(beats.size() * beat_bytes);
        for (const auto& bd : beats) {
            for (unsigned i = 0; i < beat_bytes; ++i) {
                buf.push_back(static_cast<uint8_t>(
                    bd.range(i * 8 + 7, i * 8).to_uint() & 0xFFu));
            }
        }
        flexnpu_sim::transport::ReadResult r;
        r.data  = buf.data();
        r.bytes = buf.size();
        r.resp  = master_.read_profile(slot).resp;
        master_.release_read(slot);
        return r;
    }

    flexnpu_sim::transport::WriteResult
    wait_write(flexnpu_sim::transport::Ticket tk) override {
        const int slot = static_cast<int>(tk.id);
        while (!master_.write_done(slot)) {
            sc_core::wait(master_.write_done_event(slot));
        }
        flexnpu_sim::transport::WriteResult r;
        r.resp = master_.write_profile(slot).resp;
        master_.release_write(slot);
        return r;
    }

    const flexnpu_sim::transport::Capabilities& caps() const override {
        return caps_;
    }

private:
    static flexnpu_sim::transport::Ticket make_ticket(int slot) {
        flexnpu_sim::transport::Ticket t;
        t.id = static_cast<uint32_t>(slot);
        return t;
    }

    // Little-endian pack of raw bytes into Spec::data_t per beat plus the
    // matching strobe mask. When the caller omits `strobe`, every byte
    // present in `data` is marked valid.
    static void pack_beats(const std::vector<uint8_t>& data,
                           const std::vector<uint8_t>& strobe,
                           unsigned beats, unsigned size_bytes,
                           std::vector<data_t>& out_data,
                           std::vector<strb_t>& out_strb) {
        const bool infer_strb = strobe.empty();
        for (unsigned b = 0; b < beats; ++b) {
            data_t d = 0;
            strb_t s = 0;
            for (unsigned i = 0; i < size_bytes; ++i) {
                const std::size_t idx = b * size_bytes + i;
                if (idx < data.size()) {
                    d.range(i * 8 + 7, i * 8) = data[idx];
                    const bool enable = infer_strb
                        ? true
                        : (idx < strobe.size() && strobe[idx] != 0);
                    if (enable) s.range(i, i) = 1;
                }
            }
            out_data[b] = d;
            out_strb[b] = s;
        }
    }

    MasterT&                                 master_;
    flexnpu_sim::transport::Capabilities     caps_;
    std::vector<std::vector<uint8_t>>        read_buffers_;
    std::vector<unsigned>                    read_beat_bytes_;
    std::vector<std::vector<uint8_t>>        write_buffers_;
};

}  // namespace flexnpu_sim::transport::axi

#endif  // FLEXNPU_TRANSPORT_AXI_AXI_TRANSPORT_H
