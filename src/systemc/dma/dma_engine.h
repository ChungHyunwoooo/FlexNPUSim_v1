/**
 * @file dma_engine.h
 * @brief DMA Engine SC_MODULE.
 *
 * Command channel driven by NpuController (start/busy/done + src/dst/size
 * signals). Burst plans come from DmaModel (4 KiB + max_burst_bytes
 * split, spec-compliant) and are dispatched through a runtime-
 * polymorphic MemoryMappedTransport — the transport decides AXI ID
 * assignment, beat packing, strobe handling. Legacy AXI_master_if path
 * was retired in R5.3f.
 *
 * Multi-outstanding (R5.4): bursts are issued in a sliding window sized
 * by transport caps().max_outstanding_{reads,writes}. The window is
 * drained head-first so GB byte ordering follows burst-issue order.
 * All issues use qos=0 → same AXI ID → the protocol guarantees
 * responses return in order, keeping head-drain safe.
 */

#pragma once

#include "common/debug_log.h"

#include <systemc.h>
#include "common/types.h"
#include "systemc/dma/model/dma_model.h"
#include "systemc/dma/model/dma_profile.h"
#include "systemc/bus/burst_req.h"
#include "systemc/bus/memory_mapped_transport.h"
#include <algorithm>
#include <cstring>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace flexnpu_sim {

SC_MODULE(DmaEngine) {
    // ---- Ports ----
    sc_in_clk clk;

    // Command interface driven by NpuController.
    sc_in<uint32_t>  src_addr;
    sc_in<uint32_t>  dst_addr;
    sc_in<uint32_t>  transfer_size;   // bytes
    sc_in<bool>      start;
    sc_out<bool>     busy;
    sc_out<bool>     done;
    sc_in<bool>      read_mode;       // true = Memory -> GB, false = GB -> Memory

    // Global-buffer view owned by NpuController.
    uint8_t* gb_data   = nullptr;
    uint32_t gb_offset = 0;
    uint32_t gb_size   = 0;

    // Aggregate byte counters — mirrors DmaModel stats, kept as plain
    // members so NPU-side timeline can read them without touching the
    // model.
    uint64_t total_read_bytes  = 0;
    uint64_t total_write_bytes = 0;

    SC_HAS_PROCESS(DmaEngine);

    // max_burst_beats = 0 leaves the DmaModel default in place
    // (overridden by attach_transport once caps are known).
    DmaEngine(sc_module_name name, uint32_t max_burst_beats = 0)
        : sc_module(name),
          model(DmaConfig(DEFAULT_BYTES_PER_BEAT,
                          max_burst_beats ? max_burst_beats : DEFAULT_MAX_BURST_BEATS))
    {
        SC_THREAD(transfer_thread);
    }

    // Attach the transport that will carry burst traffic. DmaModel is
    // rebuilt from the transport caps so the burst planner splits on the
    // actual beat width exposed by the bus. Required before sc_start;
    // transfer_thread aborts simulation if no transport is attached.
    void attach_transport(flexnpu_sim::transport::MemoryMappedTransport* t) {
        transport_ = t;
        if (!t) return;
        const auto& c = t->caps();
        const uint32_t beats = c.max_beats_per_burst
                                 ? c.max_beats_per_burst
                                 : DEFAULT_MAX_BURST_BEATS;
        const uint32_t bytes = c.max_burst_bytes
                                 ? (c.max_burst_bytes / beats)
                                 : DEFAULT_BYTES_PER_BEAT;
        model = DmaModel(DmaConfig(bytes, beats));
    }

    // DMA IP flow-control resources: the internal read/write FIFO depth and,
    // when the IP declares one, its explicit AXI issuing capability. These bound
    // the DMA's in-flight window (see read_window/write_window). fifo_bytes 0
    // leaves the default; max_issuing 0 = derive from the FIFO.
    void set_flow_control(uint32_t fifo_bytes,
                          uint32_t max_issuing_reads,
                          uint32_t max_issuing_writes) {
        if (fifo_bytes) fifo_bytes_ = fifo_bytes;
        max_issuing_reads_  = max_issuing_reads;
        max_issuing_writes_ = max_issuing_writes;
    }

    // ---- Pipelined read (continuous, no per-call pipeline drain) ----------
    //
    // A real DMA issues read address transactions across its whole descriptor
    // continuously — "chunks" are address ranges, not pipeline barriers. The
    // blocking start/done path (trigger_rdma) drains the in-flight window to
    // empty per call, re-paying the DRAM fill latency every transfer. These two
    // calls keep the window full ACROSS calls: pipe_read_issue() pushes a
    // transfer's bursts into the persistent in-flight pipeline (draining only
    // when the window is full), and pipe_read_flush() drains what remains (call
    // it once the whole layer's reads are issued, before reading elapsed time).
    //
    // Timing only: no GB copy. Used by the tiled executor, whose compute is
    // driven by operand counts and whose functional output is precomputed — it
    // never reads the streamed bytes back, so the DRAM traffic here models
    // memory latency/bandwidth without a destination write.
    void pipe_read_issue(uint32_t src, uint32_t size) {
        using namespace flexnpu_sim::transport;
        if (!transport_ || size == 0) return;
        const BurstPlan plan = model.submit_transfer(src, 0, size, DmaMode::Read);
        const std::size_t window = read_window();
        const uint8_t beat_log2 = transport_beat_log2();
        for (const auto& burst : plan.bursts) {
            while (rd_pipe_.size() >= window) pipe_drain_head();
            BurstReq req;
            req.addr           = burst.address;
            req.beats          = burst.beats;
            req.beat_size_log2 = burst.beat_bytes ? log2_of(burst.beat_bytes)
                                                  : beat_log2;
            req.type           = BurstType::Incr;
            rd_pipe_.push_back(transport_->issue_read(req));
        }
        model.complete_transfer();
    }
    void pipe_read_flush() {
        while (!rd_pipe_.empty()) pipe_drain_head();
    }

    // Pipelined write — the write mirror of pipe_read_issue/flush. Issues a
    // transfer's write bursts into a persistent in-flight pipeline (posted
    // writes) without draining per call, so consecutive tile writes keep the
    // pipeline full. Timing only: the payload is not read back on the tiled
    // path (the next layer's input read is timing-only), so a zero payload is
    // used rather than gathering the GB.
    void pipe_write_issue(uint32_t dst, uint32_t size) {
        using namespace flexnpu_sim::transport;
        if (!transport_ || size == 0) return;
        const BurstPlan plan = model.submit_transfer(0, dst, size, DmaMode::Write);
        const std::size_t window = write_window();
        const uint8_t beat_log2 = transport_beat_log2();
        for (const auto& burst : plan.bursts) {
            while (wr_pipe_.size() >= window) pipe_wdrain_head();
            BurstReq req;
            req.addr           = burst.address;
            req.beats          = burst.beats;
            req.beat_size_log2 = burst.beat_bytes ? log2_of(burst.beat_bytes)
                                                  : beat_log2;
            req.type           = BurstType::Incr;
            std::vector<uint8_t> payload(burst.bytes, 0);
            wr_pipe_.push_back(transport_->issue_write(req, payload));
        }
        model.complete_transfer();
    }
    void pipe_write_flush() {
        while (!wr_pipe_.empty()) pipe_wdrain_head();
    }

    void transfer_thread() {
        busy.write(false);
        done.write(false);

        while (true) {
            // Phase 1: level-check start so a pulse that arrives before
            // we return here is not missed.
            while (!start.read()) {
                wait(start.value_changed_event());
            }

            if (!transport_) {
                SC_REPORT_ERROR("DmaEngine",
                    "attach_transport must be called before the first DMA command");
                return;
            }

            busy.write(true);
            done.write(false);
            wait(clk.posedge_event());  // 1-cycle setup

            const uint32_t src     = src_addr.read();
            const uint32_t dst     = dst_addr.read();
            const uint32_t size    = transfer_size.read();
            const bool     is_read = read_mode.read();
            FLEXNPU_LOG(Trace, dma_eng,
                        "transfer_start mode=%s src=0x%x dst=0x%x size=%u",
                        is_read ? "R" : "W", src, dst, size);

            const DmaMode mode = is_read ? DmaMode::Read : DmaMode::Write;
            const BurstPlan plan = model.submit_transfer(src, dst, size, mode);
            FLEXNPU_LOG(Trace, dma_eng, "burst_plan bursts=%zu", plan.bursts.size());

            if (is_read) {
                execute_read_bursts(plan);
            } else {
                execute_write_bursts(plan);
            }
            FLEXNPU_LOG(Trace, dma_eng, "execute_bursts done");

            model.complete_transfer();
            total_read_bytes  = model.get_stats().total_read_bytes;
            total_write_bytes = model.get_stats().total_write_bytes;

            busy.write(false);
            done.write(true);

            // Phase 3-4: wait for controller to drop start, then drop done.
            while (start.read()) {
                wait(start.value_changed_event());
            }
            done.write(false);
        }
    }

private:
    static constexpr uint32_t DEFAULT_BYTES_PER_BEAT  = 8;
    static constexpr uint32_t DEFAULT_MAX_BURST_BEATS = 256;

    DmaModel                                        model;
    flexnpu_sim::transport::MemoryMappedTransport*  transport_          = nullptr;
    bool                                            gb_overflow_warned_ = false;
    uint32_t                                        fifo_bytes_         = 16u * 1024u;
    uint32_t                                        max_issuing_reads_  = 0;
    uint32_t                                        max_issuing_writes_ = 0;
    std::deque<flexnpu_sim::transport::Ticket>      rd_pipe_;  // persistent read pipeline
    std::deque<flexnpu_sim::transport::Ticket>      wr_pipe_;  // persistent write pipeline

    static uint8_t log2_of(uint32_t v) {
        uint8_t c = 0; while (v > 1) { v >>= 1; ++c; } return c;
    }

    uint8_t transport_beat_log2() const {
        const auto& c = transport_->caps();
        const uint32_t beats = c.max_beats_per_burst
                                 ? c.max_beats_per_burst
                                 : DEFAULT_MAX_BURST_BEATS;
        const uint32_t bytes = c.max_burst_bytes
                                 ? (c.max_burst_bytes / beats)
                                 : DEFAULT_BYTES_PER_BEAT;
        uint8_t log2 = 0;
        uint32_t v = bytes;
        while (v > 1) { v >>= 1; ++log2; }
        return log2;
    }

    void copy_bytes_to_gb(const uint8_t* src, uint32_t n, uint32_t& gb_pos) {
        if (!gb_data) return;
        for (uint32_t i = 0; i < n; ++i) {
            if (gb_size > 0 && gb_pos >= gb_size) {
                if (!gb_overflow_warned_) {
                    std::cerr << "[DMA] GB overflow (wrapping): gb_pos=" << gb_pos
                              << " gb_size=" << gb_size << "\n";
                    gb_overflow_warned_ = true;
                }
                gb_pos = gb_pos % gb_size;
            }
            gb_data[gb_pos++] = src[i];
        }
    }

    void gather_bytes_from_gb(std::vector<uint8_t>& out,
                              uint32_t n, uint32_t& gb_pos) {
        out.resize(n, 0);
        if (!gb_data) return;
        for (uint32_t i = 0; i < n; ++i) {
            if (gb_size > 0 && gb_pos >= gb_size) {
                if (!gb_overflow_warned_) {
                    std::cerr << "[DMA] GB overflow (wrapping): gb_pos=" << gb_pos
                              << " gb_size=" << gb_size << "\n";
                    gb_overflow_warned_ = true;
                }
                gb_pos = gb_pos % gb_size;
            }
            out[i] = gb_data[gb_pos++];
        }
    }

    // DmaModel::calculate_bursts emits spec-compliant sub-bursts already
    // (split at max_burst_bytes and the 4 KiB page boundary), so the
    // transport-side helpers issue each entry verbatim.
    //
    // Pipelining (R5.4): keep up to `window` bursts in flight. Before
    // issuing the next burst, if the window is full drain the head —
    // that is the oldest in-flight ticket, which under same-id AXI FIFO
    // ordering is also the first response to complete. Head-drain keeps
    // the GB copy order aligned with the burst-issue order.

    void execute_read_bursts(const BurstPlan& plan) {
        using namespace flexnpu_sim::transport;
        const uint8_t beat_log2 = transport_beat_log2();
        const std::size_t window = read_window();
        uint32_t gb_pos = gb_offset;

        std::deque<Ticket> in_flight;
        std::deque<uint32_t> expected_bytes;

        auto drain_head = [&]() {
            Ticket tk = in_flight.front(); in_flight.pop_front();
            const uint32_t want = expected_bytes.front(); expected_bytes.pop_front();
            auto res = transport_->wait_read(tk);
            if (res.data && res.bytes) {
                const uint32_t n = std::min<uint32_t>(
                    want, static_cast<uint32_t>(res.bytes));
                copy_bytes_to_gb(res.data, n, gb_pos);
            }
        };

        for (const auto& burst : plan.bursts) {
            if (in_flight.size() >= window) drain_head();
            BurstReq req;
            req.addr            = burst.address;
            req.beats           = burst.beats;
            req.beat_size_log2  = burst.beat_bytes ? log2_of(burst.beat_bytes)
                                                   : beat_log2;
            req.type            = BurstType::Incr;
            in_flight.push_back(transport_->issue_read(req));
            expected_bytes.push_back(burst.bytes);
        }
        while (!in_flight.empty()) drain_head();
    }

    void execute_write_bursts(const BurstPlan& plan) {
        using namespace flexnpu_sim::transport;
        const uint8_t beat_log2 = transport_beat_log2();
        const std::size_t window = write_window();
        uint32_t gb_pos = gb_offset;

        std::deque<Ticket> in_flight;

        auto drain_head = [&]() {
            Ticket tk = in_flight.front(); in_flight.pop_front();
            (void)transport_->wait_write(tk);
        };

        for (const auto& burst : plan.bursts) {
            // Payload gather reads the GB in burst-issue order; must happen
            // before issue so the data captured at issue-time matches the
            // intended slice even if the GB mutates later.
            std::vector<uint8_t> payload;
            gather_bytes_from_gb(payload, burst.bytes, gb_pos);

            if (in_flight.size() >= window) drain_head();

            BurstReq req;
            req.addr            = burst.address;
            req.beats           = burst.beats;
            req.beat_size_log2  = burst.beat_bytes ? log2_of(burst.beat_bytes)
                                                   : beat_log2;
            req.type            = BurstType::Incr;
            in_flight.push_back(transport_->issue_write(req, payload));
        }
        while (!in_flight.empty()) drain_head();
    }

    // In-flight window = the DMA IP's own resource: its internal read/write FIFO
    // bounds how many bursts stay outstanding (fifo_bytes / burst_bytes), capped
    // by an explicit AXI issuing capability when the IP declares one. It is then
    // hard-capped by the AXI master's outstanding-transaction capacity: the
    // master's ticket pool is sized to caps().max_outstanding_{reads,writes} and
    // acquire() BLOCKS when full, so a window wider than the pool would let the
    // pipe try to issue an (N+1)-th burst before it drains — acquire() blocks
    // inside issue, the pipe never drains to release a ticket, and the DMA
    // deadlocks. This bit whenever fifo_bytes/burst_bytes > max_outstanding
    // (e.g. a narrow bus halves burst_bytes, doubling the FIFO window past the
    // 4-deep write pool). Physically the DMA cannot have more transactions in
    // flight than the AXI interface tracks, so the cap is the real bound.
    std::size_t read_window() const {
        std::size_t w = outstanding_window(fifo_bytes_,
                                           transport_->caps().max_burst_bytes,
                                           max_issuing_reads_);
        const uint32_t axi = transport_->caps().max_outstanding_reads;
        if (axi && w > axi) w = axi;
        return w;
    }
    std::size_t write_window() const {
        std::size_t w = outstanding_window(fifo_bytes_,
                                           transport_->caps().max_burst_bytes,
                                           max_issuing_writes_);
        const uint32_t axi = transport_->caps().max_outstanding_writes;
        if (axi && w > axi) w = axi;
        return w;
    }

    // Drain the oldest in-flight pipelined read (advances sim time to its
    // completion). The persistent window keeps the DRAM pipeline full across
    // pipe_read_issue() calls, so latency is paid once, not per call.
    void pipe_drain_head() {
        flexnpu_sim::transport::Ticket tk = rd_pipe_.front();
        rd_pipe_.pop_front();
        (void)transport_->wait_read(tk);
    }
    void pipe_wdrain_head() {
        flexnpu_sim::transport::Ticket tk = wr_pipe_.front();
        wr_pipe_.pop_front();
        (void)transport_->wait_write(tk);
    }
};

}  // namespace flexnpu_sim
