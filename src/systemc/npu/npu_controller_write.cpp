/**
 * @file npu_controller_write.cpp
 * @brief Write path — output DMA write (write_thread)
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

static bool read_binary_file(const std::string& path, std::vector<uint8_t>& out) {
    out.clear();
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return false;
    ifs.seekg(0, std::ios::end);
    std::streamoff size = ifs.tellg();
    if (size < 0) return false;
    ifs.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    if (!out.empty()) {
        ifs.read(reinterpret_cast<char*>(out.data()), size);
        if (!ifs.good()) return false;
    }
    return true;
}
// ============================================================================
// Write Thread — Output DMA via trigger_wdma (M4)
// ============================================================================

void NpuController::write_thread() {
    while (true) {
        uint64_t wait_start_ns = now_ns();
        wait(output_ready_event);
        uint32_t layer_idx = write_layer_idx_;
        FLEXNPU_LOG(Debug, write, "layer %u trigger sim_ns=%lu (wait=%luns)",
                    layer_idx, now_ns(), now_ns() - wait_start_ns);
        add_timeline_event("wait",
                           layer_idx,
                           wait_start_ns,
                           now_ns(),
                           0,
                           "write_wait_output_ready");

        uint64_t write_start_ns = now_ns();

        const auto& layer = loader.layer(layer_idx);
        const auto& meta = loader.function_meta(layer_idx);
        const PacketExecutionDecision decision =
            PacketExecutionPolicy::decide(layer, meta);
        uint32_t output_addr = layer.output.address
                             + cfg_.address_map.dram_base;
        // Physical DMA bytes = operands * element_size_bytes (descriptor sizes
        // are sizeof(float) units) — matches the tiled path; see read_thread.
        const uint32_t elem = cfg_.hw.compute.element_size_bytes
                                  ? cfg_.hw.compute.element_size_bytes : 1u;
        uint32_t output_size = static_cast<uint32_t>(
            static_cast<uint64_t>(decision.effective_output_size) * elem / sizeof(float));

        if (decision.bypass_output_dma) {
            add_timeline_event("packet_bypass_output",
                               layer_idx,
                               write_start_ns,
                               write_start_ns + 1,
                               layer.output.size,
                               "consumed_by_internal_function");
        }

        std::vector<uint8_t> functional_ofm_bytes;
        bool use_functional_ofm = false;
        if (functional_ofm_enabled_ && output_size > 0) {
            // R10: prefer in-memory cache from compute_thread's function_model
            // forward. Falls back to file path (legacy) if cache empty.
            if (layer_idx < layer_output_cache_.size() &&
                !layer_output_cache_[layer_idx].empty()) {
                functional_ofm_bytes = layer_output_cache_[layer_idx];
                if (functional_ofm_bytes.size() < output_size) {
                    // R10.8: pout (partial-output) writeback bytes beyond the
                    // |O| final region. β_O=1 default routing keeps them in
                    // the output stream as zero dummy fill; traffic volume is
                    // preserved (Mode B supplies write_output).
                    functional_ofm_bytes.resize(output_size, 0);
                }
                use_functional_ofm = true;
            } else if (!functional_ofm_dir_.empty()) {
                std::string path = functional_ofm_dir_ + "/layer_" +
                                   std::to_string(layer_idx) + ".bin";
                if (read_binary_file(path, functional_ofm_bytes)) {
                    if (functional_ofm_bytes.size() < output_size) {
                        functional_ofm_bytes.resize(output_size, 0);
                    }
                    use_functional_ofm = true;
                } else {
                    std::cerr << "WARN: functional OFM file missing: " << path
                              << " (layer " << layer_idx << "), using zero-filled output\n";
                    functional_ofm_bytes.assign(output_size, 0);
                    use_functional_ofm = true;
                }
            } else {
                functional_ofm_bytes.assign(output_size, 0);
                use_functional_ofm = true;
            }
        }

        // Inter-layer retention: when the next packet group consumes this
        // output on-chip, the feature map never reaches DRAM (MIDAP's core
        // property). Suppress the physical write — symmetric with the tiled
        // path's ④ rollback and with peak_bw's suppressed writes. The
        // functional OFM already lives in layer_output_cache_, so downstream
        // compute is unaffected; only the spurious DRAM traffic/timing is cut.
        const bool retained_output = output_retained_next_layer(layer_idx);

        // Chunk-based output DMA via trigger_wdma
        uint32_t transferred = 0;
        const uint32_t output_base = output_region_base_offset();
        const uint32_t output_cap = output_region_capacity_bytes();
        FLEXNPU_LOG(Trace, write, "layer %u chunk_loop size=%u base=%u cap=%u",
                    layer_idx, output_size, output_base, output_cap);
        uint32_t l2_off = output_base;
        while (!retained_output && transferred < output_size) {
            if (l2_off >= output_base + output_cap) l2_off = output_base;
            uint32_t space = (output_base + output_cap) - l2_off;
            uint32_t chunk = std::min(output_cap, output_size - transferred);
            chunk = std::min(chunk, space);
            if (chunk == 0) {
                l2_off = output_base;
                continue;
            }
            if (use_functional_ofm && chunk > 0) {
                std::memcpy(global_buffer_.data() + l2_off,
                            functional_ofm_bytes.data() + transferred,
                            chunk);
            }
            trigger_wdma(output_addr + transferred, chunk, l2_off);
            l2_off += chunk;
            transferred += chunk;
        }

        layer_prof_.write_ns = now_ns() - write_start_ns;

        // M13: Free L2 output area
        gb_.output.reset();

        // Layer-end tail — compile-time derived overhead (per-packet +
        // per-atomic) absorbed by the controller before signaling complete.
        // Captures NVDLA-internal setup/CACC-drain/SDP-startup/ack cycles
        // that the descriptor (F^I..n_min) abstracts away.
        // Layer-end tail is pure HW-overhead bookkeeping (descriptor fetch,
        // subunit drain/pass cycles, IRQ). Nothing else interacts with the
        // NPU during that window, so we account for it by adding to
        // perf_cycles without advancing sim time — advancing sim time
        // through a huge tail wakes every clk.posedge SC_THREAD and blows
        // up wall-clock for no physical reason.
        const uint32_t tail_cycles = layer.latency.post_completion_cycles;
        if (tail_cycles > 0) {
            perf_cycles += tail_cycles;
            add_timeline_event("layer_tail",
                               layer_idx,
                               now_ns(),
                               now_ns(),
                               0,
                               "post_completion_cycles");
        }

        FLEXNPU_LOG(Trace, write, "layer %u tail_reached sim_ns=%lu", layer_idx, now_ns());
        if (writes_pending_ > 0) --writes_pending_;
        set_state(NpuState::LayerComplete);
        update_status_reg();

        write_done_event.notify();
        maybe_finalize();
        FLEXNPU_LOG(Trace, write, "layer %u done_notified sim_ns=%lu", layer_idx, now_ns());
    }
}

}  // namespace flexnpu_sim
