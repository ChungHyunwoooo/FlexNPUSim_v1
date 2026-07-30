/**
 * @file npu_controller_compute.cpp
 * @brief Compute path — LM-driven compute + completion (compute_thread)
 *
 * Split from npu_controller.cpp (behaviour-neutral); see
 * npu_controller.h for the class and npu_controller_internal.h for
 * the shared free helpers. compute_thread is decomposed into named
 * functional stages (precompute_layer_function, wait_gb_prefetch_gate,
 * run_layer_compute) so the thread body reads as an orchestrator.
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
// R10: Functional data path — pre-compute layer output bytes via function_model.
// Runs once per layer, produces bytes into layer_output_cache_ for write_thread
// consumption. Zero overhead when functional_ofm_enabled_ is false.
// ============================================================================

void NpuController::precompute_layer_function(uint32_t layer_idx,
                                              const LayerDescriptor& layer) {
    if (!(functional_ofm_enabled_ && layer.output.size > 0)) return;

    if (layer_output_cache_.size() < loader.num_layers()) {
        layer_output_cache_.resize(loader.num_layers());
    }
    const std::vector<uint8_t>* in_src = nullptr;
    if (layer_idx == 0) {
        in_src = &loader.input_fmap_bytes();
    } else if (layer_idx - 1 < layer_output_cache_.size()) {
        in_src = &layer_output_cache_[layer_idx - 1];
    }
    const uint32_t packet_index =
        loader.function_meta(layer_idx).packet_index;
    const auto& wt_src = loader.kernel_bytes(packet_index);
    std::vector<uint8_t> out_bytes;
    const uint8_t* in_ptr = (in_src && !in_src->empty()) ? in_src->data() : nullptr;
    const size_t   in_sz  = (in_src) ? in_src->size() : 0;
    const uint8_t* wt_ptr = (!wt_src.empty()) ? wt_src.data() : nullptr;
    const size_t   wt_sz  = wt_src.size();
    bool ok = run_layer_function(layer, in_ptr, in_sz, wt_ptr, wt_sz,
                                 out_bytes);
    if (!ok) {
        out_bytes.assign(layer.output.size, 0);
    }
    layer_output_cache_[layer_idx] = std::move(out_bytes);

    // R10: per-layer .bin dump (input/weight/output) for
    // reproducibility and CI diff. Gated by config.functional.bin_dump.
    if (functional_bin_dump_enabled_ && !functional_ofm_dir_.empty()) {
        auto dump = [&](const std::string& tag,
                        const uint8_t* p, size_t n) {
            if (!p || n == 0) return;
            const std::string path = functional_ofm_dir_ + "/layer_" +
                                     std::to_string(layer_idx) + "_" +
                                     tag + ".bin";
            std::ofstream ofs(path, std::ios::binary);
            if (ofs.is_open()) {
                ofs.write(reinterpret_cast<const char*>(p),
                          static_cast<std::streamsize>(n));
            }
        };
        dump("input",  in_ptr, in_sz);
        dump("weight", wt_ptr, wt_sz);
        const auto& out_ref = layer_output_cache_[layer_idx];
        dump("output", out_ref.data(), out_ref.size());
    }
}

// ============================================================================
// Step 1 (2026-04-28 2-stage redesign): GB-level prefetch gate.
// Wait until GB has accumulated >= prefetch_in/_weight before starting compute.
// Replaces the old threshold check inside apply_generation_rule (which compared
// against PE-side cumulative, conflating GB and PE buffer semantics).
// ============================================================================

void NpuController::wait_gb_prefetch_gate(uint32_t layer_idx,
                                          const LmParams& params) {
    const uint32_t thr_i = params.prefetch_in;
    const uint32_t thr_w = (params.fetch_operand1 == 0)
                               ? 0
                               : params.prefetch_wt;
    const uint64_t prefetch_start_ns = now_ns();
    while (!(gb_.input.loaded >= thr_i &&
             gb_.weight.loaded >= thr_w) &&
           !(layer_input_loaded_ && layer_weight_loaded_)) {
        wait(data_chunk_event);
    }
    if (timeline_enabled_) {
        const uint64_t prefetch_end_ns = now_ns();
        if (prefetch_end_ns > prefetch_start_ns) {
            add_timeline_event("prefetch_wait",
                               layer_idx,
                               prefetch_start_ns,
                               prefetch_end_ns,
                               0,
                               "gb_threshold");
        }
    }
}

// ============================================================================
// Run one layer's latency model (plus its multi-descriptor packet chain) to
// completion. Builds the LM(s), gates on the GB prefetch, then streams operands
// cycle-by-cycle. Returns total MACs; accumulates compute/stall cycle counts.
// ============================================================================

uint64_t NpuController::run_layer_compute(uint32_t layer_idx,
                                          const LayerDescriptor& layer,
                                          const LmParams& params,
                                          uint32_t pipeline_group_size,
                                          uint64_t& total_compute_cycles,
                                          uint64_t& total_stall_cycles) {
    const std::string layer_name = ufb_.layer_profiler_name(layer_idx, layer.type);

    if (!ufb_.lm) {
        LmHwParams main_hw = LmHwParams::from_config(
            cfg_, function_buffer_key(layer.type));
        ufb_.lm = std::make_unique<LmModel>(params,
                                                               main_hw,
                                                               layer_name);
    } else {
        ufb_.lm->reset(params);
        ufb_.lm->set_name(layer_name);
    }

    // ---- Pipeline detection: multi-descriptor packet ----
    std::vector<std::unique_ptr<LmModel>> pipeline_models;
    std::vector<double> pipeline_macs_per_write;
    std::vector<bool> pipeline_has_weight;
    // macs_per_write = (ops_per_output * final_output_count) / write_output
    // so that sum(n_t) over all cycles × macs_per_write = true total MACs.
    // Using ops_per_output directly was a ×passes_per_output inflation
    // (each writeback is a PARTIAL sum, not a completed output).
    auto derive_macs_per_write = [](const auto& layer) -> double {
        const uint64_t out_count = static_cast<uint64_t>(layer.output.height)
                                 * layer.output.width * layer.output.channels;
        const uint64_t tow = layer.latency.write_output;
        if (tow == 0 || out_count == 0) return static_cast<double>(layer.ops_per_output);
        return static_cast<double>(layer.ops_per_output) * static_cast<double>(out_count)
               / static_cast<double>(tow);
    };
    const double main_macs_per_write = derive_macs_per_write(layer);
    for (uint32_t d = 1; d < pipeline_group_size; d++) {
        LmParams p = loader.to_latency_params(layer_idx + d);
        const auto& pl = loader.layer(layer_idx + d);
        std::string pname = ufb_.layer_profiler_name(layer_idx + d, pl.type);
        LmHwParams stage_hw = LmHwParams::from_config(
            cfg_, function_buffer_key(pl.type));
        pipeline_models.push_back(
            std::make_unique<LmModel>(p, stage_hw, pname));
        pipeline_macs_per_write.push_back(derive_macs_per_write(pl));
        pipeline_has_weight.push_back(pl.weight.size > 0);
    }
    // M12: fetch_rate = n_max (PE-internal bandwidth)
    uint32_t fetch_rate = params.n_max;
    if (fetch_rate == 0) fetch_rate = 1;

    uint64_t compute_macs = 0;

    // Lambda: pipeline done when last model is done
    auto pipeline_done = [&]() -> bool {
        if (pipeline_models.empty()) return ufb_.lm->is_done();
        return pipeline_models.back()->is_done();
    };
    // Cycle-streaming chain (formalization: each stage's write-back
    // is offered to the next stage every cycle). LmChain preserves
    // un-accepted operands in a carry, so per-unit finite B^I on a
    // middle stage back-pressures instead of losing operands (the
    // old inline loop dropped the unaccepted remainder).
    std::vector<LmModel*> chain_stages;
    chain_stages.reserve(pipeline_models.size());
    for (auto& m : pipeline_models) chain_stages.push_back(m.get());
    LmChain stream_chain(std::move(chain_stages));
    std::vector<bool> pipeline_weight_fed(pipeline_models.size(), false);
    auto pipeline_chain = [&](uint32_t n_first) -> uint32_t {
        return stream_chain.step(
            n_first,
            [&](size_t d) -> uint32_t {
                if (pipeline_has_weight[d] && !pipeline_weight_fed[d]) {
                    pipeline_weight_fed[d] = true;
                    return loader
                        .layer(layer_idx + 1 + static_cast<uint32_t>(d))
                        .latency.fetch_operand1;
                }
                return 0;
            },
            [&](size_t d, const LmStepResult& r) {
                compute_macs += static_cast<uint64_t>(
                    static_cast<double>(r.outputs) *
                    pipeline_macs_per_write[d]);
            });
    };
    uint32_t consumed_input = 0;
    uint32_t consumed_weight = 0;
    gb_.output.reset();

    prof_compute_start_ns_ = now_ns();

    const uint64_t MAX_STALL = cfg_.hw.compute.max_compute_stall_cycles;
    uint64_t stall_count = 0;
    const uint64_t cycle_ns = std::max<uint64_t>(
        1, static_cast<uint64_t>(std::llround(clk_period_ns_)));

    wait_gb_prefetch_gate(layer_idx, params);

    // ============================================================
    // EXISTING LAYER-AT-A-TIME COMPUTE CODE (unchanged below)
    // ============================================================

    while (!pipeline_done()) {
        // M3: Check available operands (progressive arrival)
        uint32_t avail_i = gb_.input.loaded - consumed_input;
        uint32_t avail_k = gb_.weight.loaded - consumed_weight;
        bool all_loaded = layer_input_loaded_ && layer_weight_loaded_;

        // Stall: no data and DMA still in progress
        if (avail_i == 0 && avail_k == 0 && !all_loaded) {
            uint64_t stall_start = now_ns();
            wait(data_chunk_event);
            uint64_t stall_end = now_ns();
            total_stall_cycles += static_cast<uint64_t>(
                (stall_end - stall_start) / clk_period_ns_);
            if (timeline_enabled_) {
                const char* stall_reason = "stall_data_sync";
                const bool input_pending = !layer_input_loaded_;
                const bool weight_pending = !layer_weight_loaded_;
                if (input_pending && weight_pending) {
                    stall_reason = "stall_input_weight_data";
                } else if (input_pending) {
                    stall_reason = "stall_input_data";
                } else if (weight_pending) {
                    stall_reason = "stall_weight_data";
                }
                add_timeline_event("stall",
                                   layer_idx,
                                   stall_start,
                                   stall_end,
                                   0,
                                   stall_reason);
            }
            continue;
        }

        // Batch processing: consume available operands
        uint32_t batch_cycles = 0;
        uint32_t batch_i_used = 0;
        uint32_t batch_k_used = 0;
        const uint64_t batch_start_ns = now_ns();

        while (!pipeline_done()) {
            uint32_t f_i = std::min(avail_i - batch_i_used, fetch_rate);
            uint32_t f_k = std::min(avail_k - batch_k_used, fetch_rate);

            if (f_i == 0 && f_k == 0) {
                if (all_loaded) {
                    // Drain mode: pipeline flush
                    const uint64_t step_start_ns =
                        batch_start_ns + static_cast<uint64_t>(batch_cycles) * cycle_ns;
                    const uint64_t step_end_ns = step_start_ns + cycle_ns;
                    auto lm_r = ufb_.lm->update(0, 0);
                    uint32_t n_t = lm_r.outputs;
                    batch_cycles++;
                    compute_macs += static_cast<uint64_t>(
                        static_cast<double>(n_t) * main_macs_per_write);
                    uint32_t n_final = pipeline_chain(n_t);
                    gb_.output.feed(n_final);
                    if (timeline_enabled_) {
                        add_timeline_event("lm_step",
                                           layer_idx,
                                           step_start_ns,
                                           step_end_ns,
                                           0,
                                           "latency_drain",
                                           0,
                                           0,
                                           n_final,
                                           "latency_model");
                    }
                    if (n_final == 0) {
                        if (++stall_count > MAX_STALL) goto compute_exit;
                    } else {
                        stall_count = 0;
                    }
                    continue;
                }
                break;  // Need more DMA data
            }

            const uint64_t step_start_ns =
                batch_start_ns + static_cast<uint64_t>(batch_cycles) * cycle_ns;
            const uint64_t step_end_ns = step_start_ns + cycle_ns;
            auto lm_r = ufb_.lm->update(f_i, f_k);
            uint32_t n_t = lm_r.outputs;
            // Step 2 (2026-04-28 2-stage redesign): honor model's
            // actual consumption. If queue is full, consumed < offered
            // and the un-consumed amount stays in avail_* for next
            // cycle (do NOT advance batch_*_used silently).
            batch_i_used += lm_r.consumed_input;
            batch_k_used += lm_r.consumed_weight;
            batch_cycles++;
            compute_macs += static_cast<uint64_t>(
                static_cast<double>(n_t) * main_macs_per_write);
            uint32_t n_final = pipeline_chain(n_t);
            gb_.output.feed(n_final);
            if (timeline_enabled_) {
                add_timeline_event("lm_step",
                                   layer_idx,
                                   step_start_ns,
                                   step_end_ns,
                                   0,
                                   "latency_update",
                                   f_i,
                                   f_k,
                                   n_final,
                                   "latency_model");
            }
            // Bounded stall: if no operand consumed AND no output, the
            // PE buffer is full and compute is blocked. Without this
            // guard, partial-consumption retry can spin forever.
            if (lm_r.consumed_input == 0 &&
                lm_r.consumed_weight == 0 &&
                n_t == 0) {
                if (++stall_count > MAX_STALL) goto compute_exit;
            } else {
                stall_count = 0;
            }
        }

        consumed_input += batch_i_used;
        consumed_weight += batch_k_used;
        total_compute_cycles += batch_cycles;

        // M2: Advance simulation time for compute cycles
        if (batch_cycles > 0) {
            wait(sc_time(
                static_cast<double>(batch_cycles) * clk_period_ns_, SC_NS));
        }
    }
compute_exit:
    return compute_macs;
}

// ============================================================================
// Compute Thread — DMA/Compute Overlap (M1, M2, M3, M12, M16)
//
// Orchestrator: for each non-tiled layer it runs the functional stages above
// (precompute → LM run → prefetch/loop inside run_layer_compute), records
// profiling, then hands the output to write_thread and advances the pipeline.
// ============================================================================

void NpuController::compute_thread() {
    while (true) {
        wait(start_event);
        FLEXNPU_LOG(Debug, compute, "thread woken sim_ns=%lu", now_ns());
        wait_with_timeline(data_chunk_event, 0, "compute_wait_initial_data");

        compute_active_ = true;
        for (uint32_t layer_idx = 0; layer_idx < loader.num_layers(); ++layer_idx) {
            FLEXNPU_LOG(Debug, compute, "layer %u start sim_ns=%lu", layer_idx, now_ns());
            // M1: DmaAndCompute state — DMA overlapped with compute
            set_state(NpuState::DmaAndCompute);
            update_status_reg();

            LmParams params = loader.to_latency_params(layer_idx);
            const auto& layer = loader.layer(layer_idx);

            // Layers owned by the tile executor are skipped here — SAME
            // predicate as read_thread (do not wait for data that the tile
            // executor already consumed).
            if (layer_runs_tiled(layer)) {
                const uint32_t pg = count_packet_group(layer_idx);
                if (pg > 1) layer_idx += pg - 1;
                continue;
            }

            precompute_layer_function(layer_idx, layer);

            const uint32_t pipeline_group_size = count_packet_group(layer_idx);
            uint64_t total_compute_cycles = 0;
            uint64_t total_stall_cycles = 0;
            const uint64_t compute_macs = run_layer_compute(
                layer_idx, layer, params, pipeline_group_size,
                total_compute_cycles, total_stall_cycles);

            perf_macs += compute_macs;

            // Record compute profiling
            uint64_t compute_end_ns = now_ns();
            layer_prof_.compute_ns = compute_end_ns - prof_compute_start_ns_;
            layer_prof_.compute_cycles = total_compute_cycles;
            layer_prof_.stall_cycles = total_stall_cycles;
            layer_prof_.macs = compute_macs;
            add_timeline_event("compute",
                               layer_idx,
                               prof_compute_start_ns_,
                               compute_end_ns,
                               0,
                               "latency_model");

            // Signal compute done (for read_thread layer pipelining)
            layer_compute_done_ = true;
            compute_done_event.notify();

            // M4: Hand off to write_thread
            set_state(NpuState::DmaWriteOutput);
            update_status_reg();
            write_layer_idx_ = layer_idx;
            ++writes_pending_;
            output_ready_event.notify();

            // Wait for write to complete before next layer's compute (double buffer aware)
            if (!double_buffer_enabled_) {
                wait_with_timeline(write_done_event,
                                   layer_idx,
                                   "compute_wait_write_done_single");
                FLEXNPU_LOG(Trace, compute, "layer %u write_done_received(single) sim_ns=%lu",
                            layer_idx, now_ns());
            } else {
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
                                           "compute_wait_write_done_db_capacity");
                    }
                } else {
                    wait_with_timeline(write_done_event,
                                       layer_idx,
                                       "compute_wait_write_done_last_layer");
                }
            }

            gb_.input.reset();
            gb_.weight.reset();

            // Wait for the next layer's data — once read streaming has
            // finished, no new chunk will ever arrive (the trailing function
            // data of the last packet is already in the GB). Waiting after
            // streams_done_ would deadlock, so skip it.
            if (layer_idx + 1 < loader.num_layers() && !streams_done_) {
                wait_with_timeline(data_chunk_event,
                                   layer_idx + 1,
                                   "compute_wait_next_layer_data");
            }

            // ---- Pipeline fast-forward: sync subsequent descriptors ----
            // Downstream functions received their data through the pipeline
            // chain (conv output → act input), not from DMA.  No need to
            // wait on data_chunk_event — just signal completion and write.
            if (pipeline_group_size > 1) {
                for (uint32_t d = 1; d < pipeline_group_size; d++) {
                    uint32_t sub_idx = layer_idx + d;

                    set_state(NpuState::DmaAndCompute);
                    update_status_reg();

                    // Already computed in pipeline — signal done immediately
                    layer_compute_done_ = true;
                    compute_done_event.notify();

                    // Hand off to write_thread
                    set_state(NpuState::DmaWriteOutput);
                    update_status_reg();
                    write_layer_idx_ = sub_idx;
                    ++writes_pending_;
                    output_ready_event.notify();

                    // Wait for write (bypassed for internal consumers, actual for last)
                    wait_with_timeline(write_done_event,
                                       sub_idx,
                                       "compute_pipeline_sync_write");
                    FLEXNPU_LOG(Trace, compute, "sub %u write_done_received(pipe) sim_ns=%lu",
                                sub_idx, now_ns());

                    gb_.input.reset();
                    gb_.weight.reset();
                }
                layer_idx += pipeline_group_size - 1;
            }
        }

        compute_active_ = false;
        maybe_finalize();
        compute_tile_done.notify();
    }
}

// ============================================================================
// Execution-path ownership — ONE predicate decides which path runs a layer.
// read_thread's tile executor owns a layer iff this returns true; the fused
// compute/write pipeline owns it otherwise. Both threads MUST use this same
// predicate: the old asymmetry (read: TileEnable||exceeds_gb vs compute:
// TileEnable only) left compute waiting forever for data of a layer the
// tile executor had already run (the resnet18/mobilenet completion hang).
// ============================================================================

bool NpuController::layer_runs_tiled(const LayerDescriptor& layer) const {
    if (layer.flags & FuncFlag::TileEnable) return true;
    // A fused activation is handled by run_tiled_layer's zero-cost early return
    // (synchronous, no SC-thread handshake) — route it there so it does not run
    // on the non-tiled dispatch path, where skipping it would strand the
    // compute/write threads waiting on its data.
    if (layer.type == FuncType::Activation &&
        cfg_.hw.buffers.global.layer_fusion == "auto") return true;
    const uint64_t layer_bytes = static_cast<uint64_t>(layer.input.size)
                               + layer.weight.size + layer.output.size;
    return layer_bytes > static_cast<uint64_t>(gb_size_bytes());
}

// ============================================================================
// Completion ownership — the LAST finisher promotes AllDone
// ============================================================================

void NpuController::maybe_finalize() {
    if (!streams_done_ || compute_active_ || writes_pending_ > 0) return;
    if (state == NpuState::AllDone) return;
    set_state(NpuState::AllDone);
    update_status_reg();
}


}  // namespace flexnpu_sim
