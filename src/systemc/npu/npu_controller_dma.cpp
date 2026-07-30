/**
 * @file npu_controller_dma.cpp
 * @brief DMA orchestration (trigger_rdma/wdma, wdma_issue_thread)
 *
 * Split from npu_controller.cpp (behaviour-neutral); see
 * npu_controller.h for the class and npu_controller_internal.h for
 * the shared free helpers.
 */

#include "systemc/npu/npu_controller.h"
#include "systemc/npu/npu_controller_internal.h"
#include "system/packet_execution_policy.h"
#include "common/debug_log.h"
#include "compiler/dnn_image/tile_search.h"
#include "model/function/layer_function_runner.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "model/latency/latency_model_chain.h"
#include "systemc/memory/model/peak_bw_layer_model.h"

namespace flexnpu_sim {

// DMA Helpers — trigger_rdma / trigger_wdma (real AXI transfers)
// MemoryAxiSlave applies DRAM timing on the AXI response
// ============================================================================

// DMA call counters (g_* live in npu_controller_internal.h — shared with the
// read path; also exposed via FLEXNPU_LOG=debug for runtime stats).

void NpuController::trigger_rdma(uint32_t src, uint32_t size, uint32_t l2_off) {
    uint64_t t0 = now_ns();
    bool cache_hit = false;
    if (l2_off > gb_size_bytes() || size > (gb_size_bytes() - l2_off)) {
        std::ostringstream oss;
        oss << "GB RDMA overflow: off=" << l2_off
            << " size=" << size
            << " gb_size=" << gb_size_bytes();
        SC_REPORT_ERROR("NpuController", oss.str().c_str());
        sc_stop();
        return;
    }
    g_rdma_count++;
    g_rdma_bytes += size;
    FLEXNPU_LOG(Debug, dma,
                "RDMA #%lu enter layer=%u src=0x%x size=%u l2_off=%u sim_ns=%lu cum_bytes=%lu",
                g_rdma_count, current_layer_idx_, src, size, l2_off, now_ns(), g_rdma_bytes);
    if (suppress_dma_timing_) {
        // peak_bw tiled model: count the bytes, advance no sim time. The layer's
        // memory latency is applied once, in bulk, after the tile loop.
        perf_mem_reads += size; ++perf_read_txns;
        return;
    }
    if (global_buffer_.mode() == NpuGlobalBuffer::Mode::Cache) {
        // cache_load_range counts the hit/miss internally.
        if (global_buffer_.cache_load_range(src, size, l2_off)) {
            // Serve cache hit without DRAM transaction.
            wait(clk.posedge_event());
            cache_hit = true;
            add_timeline_event("rdma_cache_hit",
                               current_layer_idx_,
                               t0,
                               now_ns(),
                               size,
                               "cache_hit_serve");
            return;
        }
    }

    // 4-phase handshake — raise start (level), hold until done observed.
    uint64_t tx_setup_start = now_ns();
    rdma_ch_.engine.gb_offset = l2_off;
    rdma_ch_.src_addr.write(src);
    rdma_ch_.transfer_size.write(size);
    rdma_ch_.read_mode.write(true);
    rdma_ch_.start.write(true);
    uint64_t tx_setup_end = now_ns();
    add_timeline_event("rdma_tx",
                       current_layer_idx_,
                       tx_setup_start,
                       tx_setup_end,
                       0,
                       "setup_pulse");

    uint64_t tx_wait_start = now_ns();
    FLEXNPU_LOG(Trace, dma, "RDMA #%lu wait_begin cur_done=%d sim_ns=%lu",
                g_rdma_count, (int)rdma_ch_.done.read(), now_ns());
    if (!rdma_ch_.done.read()) {
        wait(rdma_ch_.done.posedge_event());
        wait(clk.posedge_event());  // 1-cycle sample latency (RTL parity)
    }
    FLEXNPU_LOG(Trace, dma, "RDMA #%lu wait_end sim_ns=%lu", g_rdma_count, now_ns());

    // Phase 3: lower start (ack), then wait for DMA to drop done (phase 4).
    rdma_ch_.start.write(false);
    if (rdma_ch_.done.read()) {
        wait(rdma_ch_.done.negedge_event());
    }
    uint64_t tx_wait_end = now_ns();
    add_timeline_event("rdma_tx",
                       current_layer_idx_,
                       tx_wait_start,
                       tx_wait_end,
                       0,
                       "wait_done");

    perf_mem_reads += size; ++perf_read_txns;
    add_timeline_event(cache_hit ? "rdma_cache_hit" : "rdma",
                       current_layer_idx_,
                       t0,
                       now_ns(),
                       size,
                       "burst_total");
    if (global_buffer_.mode() == NpuGlobalBuffer::Mode::Cache) {
        global_buffer_.cache_store_range(src, l2_off, size);
    } else {
        global_buffer_.scratchpad_mark_range(src, size);
    }
}

void NpuController::trigger_wdma(uint32_t dst, uint32_t size, uint32_t l2_off) {
    if (suppress_dma_timing_) {
        // peak_bw tiled model: count bytes, advance no sim time (bulk memory
        // term applied after the tile loop). Mirrors trigger_rdma.
        perf_mem_writes += size; ++perf_write_txns;
        return;
    }
    // Post to the single WDMA issue stage and block until it retires this
    // request. wdma_issue_thread is the sole writer of the wdma_* signals, so
    // the two requesters (tiled write-back in read_thread, fused drain in
    // write_thread) no longer share a multi-writer signal set — SC_ONE_WRITER
    // is restored. Serialization is preserved: the issue thread drives one
    // handshake at a time, in post order.
    WdmaReq req;
    req.dst = dst; req.size = size; req.l2_off = l2_off;
    wdma_queue_.push_back(&req);
    wdma_req_evt_.notify(SC_ZERO_TIME);
    while (!req.done) wait(wdma_retire_evt_);
}

void NpuController::wdma_issue_thread() {
    while (true) {
        while (wdma_queue_.empty()) wait(wdma_req_evt_);
        WdmaReq* req = wdma_queue_.front();
        wdma_queue_.pop_front();
        const uint32_t dst = req->dst, size = req->size, l2_off = req->l2_off;

        uint64_t t0 = now_ns();
        if (l2_off > gb_size_bytes() || size > (gb_size_bytes() - l2_off)) {
            std::ostringstream oss;
            oss << "GB WDMA overflow: off=" << l2_off
                << " size=" << size
                << " gb_size=" << gb_size_bytes();
            SC_REPORT_ERROR("NpuController", oss.str().c_str());
            sc_stop();
            req->done = true;
            wdma_retire_evt_.notify(SC_ZERO_TIME);
            continue;
        }
        g_wdma_count++;
        g_wdma_bytes += size;
        FLEXNPU_LOG(Debug, dma,
                    "WDMA #%lu enter layer=%u dst=0x%x size=%u l2_off=%u sim_ns=%lu cum_bytes=%lu",
                    g_wdma_count, current_layer_idx_, dst, size, l2_off, now_ns(), g_wdma_bytes);
        // 4-phase handshake — same pattern as trigger_rdma.
        uint64_t tx_setup_start = now_ns();
        wdma_ch_.engine.gb_offset = l2_off;
        wdma_ch_.dst_addr.write(dst);
        wdma_ch_.transfer_size.write(size);
        wdma_ch_.read_mode.write(false);
        wdma_ch_.start.write(true);
        uint64_t tx_setup_end = now_ns();
        add_timeline_event("wdma_tx",
                           current_layer_idx_,
                           tx_setup_start,
                           tx_setup_end,
                           0,
                           "setup_pulse");

        uint64_t tx_wait_start = now_ns();
        FLEXNPU_LOG(Trace, dma, "WDMA #%lu wait_begin cur_done=%d sim_ns=%lu",
                    g_wdma_count, (int)wdma_ch_.done.read(), now_ns());
        if (!wdma_ch_.done.read()) {
            wait(wdma_ch_.done.posedge_event());
            wait(clk.posedge_event());  // 1-cycle sample latency (RTL parity)
        }
        FLEXNPU_LOG(Trace, dma, "WDMA #%lu wait_end sim_ns=%lu", g_wdma_count, now_ns());

        wdma_ch_.start.write(false);
        if (wdma_ch_.done.read()) {
            wait(wdma_ch_.done.negedge_event());
        }
        uint64_t tx_wait_end = now_ns();
        add_timeline_event("wdma_tx",
                           current_layer_idx_,
                           tx_wait_start,
                           tx_wait_end,
                           0,
                           "wait_done");

        perf_mem_writes += size; ++perf_write_txns;
        add_timeline_event("wdma", current_layer_idx_, t0, now_ns(), size, "burst_total");
        if (global_buffer_.mode() == NpuGlobalBuffer::Mode::Cache) {
            global_buffer_.cache_store_range(dst, l2_off, size);
        } else {
            global_buffer_.scratchpad_mark_range(dst, size);
        }
        req->done = true;
        wdma_retire_evt_.notify(SC_ZERO_TIME);
    }
}


}  // namespace flexnpu_sim
