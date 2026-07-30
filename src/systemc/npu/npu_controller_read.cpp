/**
 * @file npu_controller_read.cpp
 * @brief Read path — tile-streaming DMA read (read_thread)
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

// ============================================================================
// Read Thread — Chunk-based DMA Read via trigger_rdma (M3, M5, M13)
// ============================================================================

// Stream the DNN image from DRAM (header, then body in GB-sized chunks) and
// parse it into `loader`. Returns false on a malformed/short image so the
// caller can idle. The previous per-packet reconstruction re-read each packet's
// start address twice (header then body), which the DMA/L2 path mishandled for
// deeper packets — this single contiguous pass replaced it.
bool NpuController::load_dnn_image() {
    // Assemble the DNN image address
    uint32_t dnn_addr_lo = regs[npu_reg::DNNSA / 4];
    uint32_t dnn_addr_hi = regs[npu_reg::DNNSA_MSB / 4];
    uint64_t dnn_base = (static_cast<uint64_t>(dnn_addr_hi) << 32) | dnn_addr_lo;

    // Read the DNN image header
    trigger_rdma(static_cast<uint32_t>(dnn_base),
                 sizeof(DnnImageHeader), 0);

    DnnImageHeader tmp_header;
    std::memcpy(&tmp_header, global_buffer_.data(), sizeof(DnnImageHeader));
    if (tmp_header.total_size < sizeof(DnnImageHeader) ||
        tmp_header.num_layers == 0) {
        return false;
    }

    // Load the full image body from DRAM in one contiguous streamed pass.
    std::vector<uint8_t> image_buf(tmp_header.total_size, 0);
    std::memcpy(image_buf.data(), &tmp_header, sizeof(DnnImageHeader));
    bool packet_read_ok = true;

    FLEXNPU_LOG(Debug, desc, "num_layers=%u total_size=%u header_sz=%lu",
                tmp_header.num_layers, tmp_header.total_size, sizeof(DnnImageHeader));
    const uint32_t body_off = sizeof(DnnImageHeader);
    const uint32_t body_len = (tmp_header.total_size > body_off)
                                  ? tmp_header.total_size - body_off : 0u;
    uint32_t bcopied = 0;
    while (bcopied < body_len) {
        const uint32_t chunk = std::min<uint32_t>(gb_size_bytes(), body_len - bcopied);
        trigger_rdma(static_cast<uint32_t>(dnn_base) + body_off + bcopied, chunk, 0);
        std::memcpy(image_buf.data() + body_off + bcopied, global_buffer_.data(), chunk);
        bcopied += chunk;
    }
    const uint32_t cursor = tmp_header.total_size;
    FLEXNPU_LOG(Debug, desc, "image load done cursor=%u total=%u",
                cursor, tmp_header.total_size);

    if (!packet_read_ok || cursor != tmp_header.total_size) {
        return false;
    }

    if (!loader.parse(image_buf.data(), image_buf.size())) {
        return false;
    }
    return true;
}

void NpuController::read_thread() {
    while (true) {
        wait(start_event);
        timeline_run_start_ns_ = now_ns();
        timeline_events_.clear();
        timeline_state_active_ = false;
        global_buffer_.reset_presence();
        retained_out_bytes_ = 0;  // retention ledger resets per run

        set_state(NpuState::LoadLayerDesc);
        update_status_reg();

        if (!load_dnn_image()) {
            set_state(NpuState::Idle);
            update_status_reg();
            continue;
        }

        // Per-layer processing
        total_layers_ = loader.num_layers();
        for (uint32_t layer_idx = 0; layer_idx < loader.num_layers(); ++layer_idx) {
            current_layer_idx_ = layer_idx;
            set_state(NpuState::DmaReadInput);
            update_status_reg();

            const auto& layer = loader.layer(layer_idx);
            const auto& meta = loader.function_meta(layer_idx);
            const PacketExecutionDecision decision =
                PacketExecutionPolicy::decide(layer, meta);

            // Descriptor sizes are in sizeof(float) units; the physical DMA
            // moves element_size_bytes per operand — the same conversion the
            // tiled path applies in lower_tile_schedule. Without it the full-sim
            // path over-charges DRAM traffic/timing by sizeof(float)/elem (2x at
            // INT16), which is exactly the GB-fit discontinuity vs the tiled path.
            const uint32_t elem = cfg_.hw.compute.element_size_bytes
                                      ? cfg_.hw.compute.element_size_bytes : 1u;
            uint32_t input_size  = static_cast<uint32_t>(
                static_cast<uint64_t>(decision.effective_input_size) * elem / sizeof(float));
            uint32_t weight_size = static_cast<uint32_t>(
                static_cast<uint64_t>(layer.weight.size) * elem / sizeof(float));
            uint32_t input_addr  = layer.input.address
                                 + cfg_.address_map.dram_base;
            uint32_t weight_addr = layer.weight.address
                                 + cfg_.address_map.dram_base;

            // Layer start timestamp (M16 + profiling)
            uint64_t layer_start_ns = now_ns();
            layer_prof_ = LayerProfile{};
            FLEXNPU_LOG(Info, read,
                        "layer %u/%u start type=%d input=%uB weight=%uB i1=0x%x i2=0x%x sim_ns=%lu",
                        layer_idx, total_layers_, (int)layer.type, input_size, weight_size,
                        meta.i1_connect, meta.i2_connect, layer_start_ns);

            // Reset layer state
            layer_input_loaded_ = false;
            layer_weight_loaded_ = false;
            layer_compute_done_ = false;
            gb_.reset_for_layer();

            // peak_bw memory model applies to BOTH execution paths: suppress
            // per-transfer DMA sim time here so the non-tiled dispatch path
            // (small acts/depthwise) also runs memory-fast; its bulk memory
            // term is applied at layer completion below. run_tiled_layer sets
            // and clears this itself for the tiled path.
            if (!layer_timing_)
                layer_timing_ = make_layer_timing_strategy(cfg_.dram, axi_bus_width_bytes_);
            suppress_dma_timing_ = layer_timing_->suppresses_dma();

            // Per-layer LM params (used by the @LAYER report line below).
            LmParams lp = loader.to_latency_params(layer_idx);

            // R9.5 tile executor: TileEnable layers — and any layer whose
            // operands exceed the GB — are run inline here, tile-by-tile with
            // GB-chunked DMA (no whole-tensor streaming / race). compute_thread
            // and write_thread skip these layers.
            if (layer_runs_tiled(layer)) {
                run_tiled_layer(layer, layer_idx, input_addr, weight_addr);
                const uint32_t pg = count_packet_group(layer_idx);
                if (pg > 1) layer_idx += pg - 1;
                continue;
            }

            // ============================================================
            // Fail-fast if layer doesn't fit in GB. Tile decomposition is
            // not done by the simulator — caller must provide a pre-tiled
            // network. This call may std::exit(1) with a clear message.
            check_layer_fits_or_die(layer);


            // ============================================================
            // EXISTING LAYER-AT-A-TIME CODE (unchanged below)
            // ============================================================

            // M3: Input + Weight DMA — chunk-level round-robin interleave.
            // Mirrors NVDLA CDMA streaming: a single DMA channel issues
            // alternating input/weight chunks so compute can start as soon
            // as ANY input AND ANY weight are present (instead of waiting
            // for the entire weight tensor to land after the entire input).
            const uint32_t stream_cap = input_stream_capacity_bytes();
            // Cap chunk size to one AXI burst — derived purely from the
            // configured AXI parameters. This mirrors how real DMA engines
            // split a software-issued descriptor into protocol-legal bursts.
            // Defaults in FlexNpuSimConfig::AxiConfig (max_burst_length=256,
            // data_width_bits=64) guarantee a non-zero result; if a user
            // explicitly configures both to zero, fall back to stream_cap
            // (single-shot DMA, equivalent to legacy behavior).
            const uint32_t axi_burst_bytes =
                cfg_.axi.max_burst_length *
                (cfg_.axi.data_width_bits / 8);
            uint32_t chunk_size = std::min(std::max(stream_cap / 2, 1u),
                                           std::max(input_size, weight_size));
            chunk_size = std::min(chunk_size, axi_burst_bytes);
            if (chunk_size == 0) chunk_size = stream_cap;
            uint32_t l2_stream_off = 0;

            if (decision.bypass_input_dma) {
                add_timeline_event("packet_bypass_input",
                                   layer_idx,
                                   now_ns(),
                                   now_ns() + 1,
                                   layer.input.size,
                                   "i1_connect_internal");
                gb_.input.loaded = layer.latency.fetch_operand0;
                layer_input_loaded_ = true;
                data_chunk_event.notify();

                // Weight DMA only (input bypassed)
                if (weight_size > 0) {
                    set_state(NpuState::DmaReadKernel);
                    update_status_reg();
                    uint32_t transferred = 0;
                    while (transferred < weight_size) {
                        if (l2_stream_off >= stream_cap) l2_stream_off = 0;
                        uint32_t space = stream_cap - l2_stream_off;
                        uint32_t this_chunk = std::min({chunk_size,
                                                        weight_size - transferred,
                                                        space});
                        if (this_chunk == 0) { l2_stream_off = 0; continue; }
                        if (cfg_.hw.buffers.global.backpressure_enabled) {
                            while (gb_.weight.full()) wait(clk.posedge_event());
                        }
                        trigger_rdma(weight_addr + transferred, this_chunk, l2_stream_off);
                        l2_stream_off += this_chunk;
                        transferred += this_chunk;
                        gb_.weight.loaded = static_cast<uint32_t>(
                            static_cast<uint64_t>(layer.latency.fetch_operand1)
                            * transferred / weight_size);
                        data_chunk_event.notify();
                    }
                }
            } else {
                // Round-robin input/weight DMA: alternate chunks so weight
                // and input become available to the compute thread roughly
                // in parallel. This is the NVDLA-style streaming pattern.
                set_state(NpuState::DmaReadInput);
                update_status_reg();
                uint32_t input_done = 0;
                uint32_t weight_done = 0;
                bool dma_input_turn = true;  // start with input
                while (input_done < input_size || weight_done < weight_size) {
                    bool can_input  = (input_done  < input_size);
                    bool can_weight = (weight_done < weight_size);
                    bool do_input = dma_input_turn ? can_input : !can_weight;
                    if (!can_input  && can_weight) do_input = false;
                    if (!can_weight && can_input)  do_input = true;

                    if (do_input) {
                        if (l2_stream_off >= stream_cap) l2_stream_off = 0;
                        uint32_t space = stream_cap - l2_stream_off;
                        uint32_t this_chunk = std::min({chunk_size,
                                                        input_size - input_done,
                                                        space});
                        if (this_chunk == 0) { l2_stream_off = 0; continue; }
                        if (cfg_.hw.buffers.global.backpressure_enabled) {
                            while (gb_.input.full()) wait(clk.posedge_event());
                        }
                        trigger_rdma(input_addr + input_done, this_chunk, l2_stream_off);
                        l2_stream_off += this_chunk;
                        input_done += this_chunk;
                        gb_.input.loaded = static_cast<uint32_t>(
                            static_cast<uint64_t>(layer.latency.fetch_operand0)
                            * input_done / input_size);
                        data_chunk_event.notify();
                    } else {
                        if (l2_stream_off >= stream_cap) l2_stream_off = 0;
                        uint32_t space = stream_cap - l2_stream_off;
                        uint32_t this_chunk = std::min({chunk_size,
                                                        weight_size - weight_done,
                                                        space});
                        if (this_chunk == 0) { l2_stream_off = 0; continue; }
                        if (cfg_.hw.buffers.global.backpressure_enabled) {
                            while (gb_.weight.full()) wait(clk.posedge_event());
                        }
                        // First weight chunk → switch state for visibility
                        if (weight_done == 0) {
                            set_state(NpuState::DmaReadKernel);
                            update_status_reg();
                        }
                        trigger_rdma(weight_addr + weight_done, this_chunk, l2_stream_off);
                        l2_stream_off += this_chunk;
                        weight_done += this_chunk;
                        gb_.weight.loaded = static_cast<uint32_t>(
                            static_cast<uint64_t>(layer.latency.fetch_operand1)
                            * weight_done / weight_size);
                        data_chunk_event.notify();
                    }
                    dma_input_turn = !dma_input_turn;
                }
                layer_input_loaded_ = true;
            }
            layer_weight_loaded_ = true;
            data_chunk_event.notify();

            // Record DMA read done
            layer_prof_.dma_read_ns = static_cast<uint64_t>(
                sc_time_stamp().to_seconds() * 1e9) - layer_start_ns;

            // Wait for compute to finish this layer
            if (!layer_compute_done_) {
                wait_with_timeline(compute_done_event,
                                   layer_idx,
                                   "read_wait_compute_done");
            }

            // M13: Input/Weight areas freed after compute
            gb_.input.reset();
            gb_.weight.reset();

            // Wait for write to complete (double buffer aware)
            if (!gb_.output.empty()) {
                if (!double_buffer_enabled_) {
                    wait_with_timeline(write_done_event,
                                       layer_idx,
                                       "read_wait_write_done_single");
                } else {
                    // Double buffer: check if next layer fits
                    if (layer_idx + 1 < loader.num_layers()) {
                        const auto& next = loader.layer(layer_idx + 1);
                        const auto& next_meta = loader.function_meta(layer_idx + 1);
                        const PacketExecutionDecision next_decision =
                            PacketExecutionPolicy::decide(next, next_meta);
                        uint32_t next_needed =
                            next_decision.effective_input_size + next.weight.size +
                            next_decision.effective_output_size;
                        uint32_t available = gb_size_bytes() - gb_.output.available();
                        if (next_needed > available) {
                            wait_with_timeline(write_done_event,
                                               layer_idx,
                                               "read_wait_write_done_db_capacity");
                        }
                    } else {
                        wait_with_timeline(write_done_event,
                                           layer_idx,
                                           "read_wait_write_done_last_layer");
                    }
                }
            }

            // M16: Layer cycles from wall-clock simulation time
            uint64_t layer_end_ns = now_ns();
            uint64_t layer_cycles = static_cast<uint64_t>(
                (layer_end_ns - layer_start_ns) / clk_period_ns_);
            perf_cycles += layer_cycles;

            // Per-layer profile output
            layer_prof_.total_ns = layer_end_ns - layer_start_ns;
            const char* ltype = "unknown";
            switch (layer.type) {
                case FuncType::Conv2D:          ltype = "conv2d"; break;
                case FuncType::DepthwiseConv2D: ltype = "dwconv"; break;
                case FuncType::FullyConnected:  ltype = "fc"; break;
                case FuncType::Pooling:         ltype = "pool"; break;
                case FuncType::Activation:      ltype = "act"; break;
                case FuncType::MatMul:          ltype = "matmul"; break;
                case FuncType::ElementWise:     ltype = "ewise"; break;
            }
            uint64_t overlap_ns = 0;
            uint64_t sum_phases = layer_prof_.dma_read_ns
                                + layer_prof_.compute_ns
                                + layer_prof_.write_ns;
            if (sum_phases > layer_prof_.total_ns)
                overlap_ns = sum_phases - layer_prof_.total_ns;

            if (layer_line_csv())
                std::cerr << "@LAYER," << layer_idx << "," << ltype
                          << "," << lp.fetch_operand0
                          << "," << lp.fetch_operand1
                          << "," << lp.write_output
                          << "," << lp.n_max
                          << "," << layer_prof_.dma_read_ns
                          << "," << layer_prof_.compute_ns
                          << "," << layer_prof_.write_ns
                          << "," << layer_prof_.total_ns
                          << "," << overlap_ns
                          << "," << layer_prof_.compute_cycles
                          << "," << layer_prof_.stall_cycles
                          << "," << layer_prof_.macs
                          << "," << (layer_prof_.compute_ns > layer_prof_.dma_read_ns
                                     ? "compute" : "memory")
                          << "\n";
            else
                std::cerr << " layer " << std::setw(2) << layer_idx << "  "
                          << std::left << std::setw(7) << ltype << std::right
                          << " done\n";

            // Skip downstream functions in pipeline group — their DMA is
            // bypassed and compute was handled by the pipeline chain.
            uint32_t pg = count_packet_group(layer_idx);
            // Non-tiled layers must also record their output for retention
            // (call with the group-start index; the helper resolves the tail).
            // Small activations run here, so without this the next tiled layer
            // finds a stale ledger and re-reads their output from DRAM.
            update_retention_ledger(layer_idx);
            if (pg > 1) {
                layer_idx += pg - 1;
            }
        }

        streams_done_ = true;
        data_chunk_event.notify();   // wake a compute waiting for data that
                                     // will never arrive (race-safe drain)
        maybe_finalize();
        FLEXNPU_LOG(Info, dma,
                    "DONE rdma_count=%lu rdma_bytes=%lu wdma_count=%lu wdma_bytes=%lu sim_ns=%lu",
                    g_rdma_count, g_rdma_bytes, g_wdma_count, g_wdma_bytes, now_ns());
        update_status_reg();
        flush_state_timeline();
        dump_timeline_trace();
        regs[npu_reg::NPUCR / 4] &= ~npu_reg::NPUCR_START;
    }
}


}  // namespace flexnpu_sim
