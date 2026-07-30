/**
 * @file npu_controller_tile.cpp
 * @brief Tile scheduling + execution (schedule, run_tiled_layer, retention)
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

uint32_t NpuController::count_packet_group(uint32_t start_idx) const {
    if (start_idx >= loader.num_layers()) return 0;
    uint32_t pid = loader.function_meta(start_idx).packet_layer_id;
    uint32_t count = 1;
    while (start_idx + count < loader.num_layers() &&
           loader.function_meta(start_idx + count).packet_layer_id == pid) {
        count++;
    }
    return count;
}

// Inter-layer retention ledger (D3): record the buffer the NEXT layer will
// read, so its input DMA can be served from the GB instead of DRAM. Two facts
// make the source subtle, and both paths (tiled here + non-tiled dispatch) must
// agree or containment silently fails:
//   (1) The next layer reads the packet group's LAST function output (a fused
//       conv+act ends at the act), not this representative layer's output.
//   (2) A layer whose whole footprint fits the GB runs on the non-tiled path,
//       which used to skip this update entirely — leaving the ledger a buffer
//       behind and miscounting the next layer's input as a DRAM read.
// Guard: retain only when the tail output actually fits the GB.
void NpuController::update_retention_ledger(uint32_t start_layer_idx) {
    if (cfg_.hw.buffers.global.layer_fusion != "auto") {
        retained_out_bytes_ = 0;
        return;
    }
    const uint32_t pg = count_packet_group(start_layer_idx);
    const LayerDescriptor& tail =
        loader.layer(start_layer_idx + (pg > 1 ? pg - 1 : 0));
    const uint64_t elem = std::max<uint32_t>(1, cfg_.hw.compute.element_size_bytes);
    if (tail.output.size / sizeof(float) * elem <=
            static_cast<uint64_t>(gb_size_kb()) * 1024) {
        retained_out_addr_  = tail.output.address;
        retained_out_bytes_ = tail.output.size;  // descriptor units
    } else {
        retained_out_bytes_ = 0;
    }
}

// Producer-side mirror of build_layer_schedule's consumer retention hit: does
// the next packet group's input fall entirely within this layer's output, with
// that output small enough to stay in the GB? When both hold, the feature map
// is handed to the next layer on-chip (MIDAP's inter-layer retention) and the
// DRAM output write never happens — the non-tiled write_thread skips it, the
// same traffic the tiled path rolls back in ④.
bool NpuController::output_retained_next_layer(uint32_t layer_idx) const {
    if (cfg_.hw.buffers.global.layer_fusion != "auto") return false;
    const uint32_t pg = count_packet_group(layer_idx);
    const LayerDescriptor& tail =
        loader.layer(layer_idx + (pg > 1 ? pg - 1 : 0));
    const uint64_t elem = std::max<uint32_t>(1, cfg_.hw.compute.element_size_bytes);
    // Output must fit the GB to be retainable on-chip (same guard as the ledger).
    if (tail.output.size / sizeof(float) * elem >
            static_cast<uint64_t>(gb_size_kb()) * 1024)
        return false;
    // The consumer is the first layer of the next packet group. A final layer
    // (no consumer) writes its output to DRAM for real.
    const uint32_t consumer_idx = layer_idx + pg;
    if (consumer_idx >= loader.num_layers()) return false;
    const LayerDescriptor& consumer = loader.layer(consumer_idx);
    const uint64_t in_addr  = consumer.input.address;
    const uint64_t in_bytes = consumer.input.size;   // descriptor units, as tail
    return in_addr >= tail.output.address &&
           in_addr + in_bytes <= tail.output.address + tail.output.size;
}

// ============================================================================
// Tile Execution Helpers
// ============================================================================

// Build the runtime tile transaction schedule for a layer: the compiler's
// tiling decision (compute_tile_schedule) lowered to dataflow-baked
// transactions. Each transaction is one output tile whose footprint fits the
// GB, so the tile executor never streams a whole oversized tensor.
// Compression scales boundary BYTES only (a compressed stream moves fewer bytes
// across the GB boundary); operand counts — the compute side — are unchanged.
// ratio > 1.0 = compressed (Nullhop sparse feature maps), 1.0 = no-op (default).
// Input zero-skip (skip_permille) then removes skipped input elements from the
// boundary too, composing multiplicatively. Applied before spill so psum
// round-trips see the same output scaling.
void NpuController::apply_boundary_byte_scaling(std::vector<TileTxn>& txns,
                                                const LayerDescriptor& layer) const {
    const auto& gbc = cfg_.hw.buffers.global;
    auto scale = [](uint32_t bytes, double ratio) -> uint32_t {
        if (ratio <= 1.0 || bytes == 0) return bytes;
        return static_cast<uint32_t>(
            std::max<double>(1.0, std::llround(bytes / ratio)));
    };
    if (gbc.input_compression_ratio  > 1.0 ||
        gbc.weight_compression_ratio > 1.0 ||
        gbc.output_compression_ratio > 1.0) {
        for (auto& t : txns) {
            t.input_fetch_bytes  = scale(t.input_fetch_bytes,
                                         gbc.input_compression_ratio);
            t.weight_fetch_bytes = scale(t.weight_fetch_bytes,
                                         gbc.weight_compression_ratio);
            t.output_write_bytes = scale(t.output_write_bytes,
                                         gbc.output_compression_ratio);
        }
    }
    // Zero-skip (per-layer, CSV preprocess_type=ZeroSkipping + skip_permille):
    // skipped input elements never cross the boundary. Composes multiplicatively
    // with input compression above.
    const uint32_t skip_pm = std::min<uint32_t>(layer.preprocess_param0, 1000u);
    if (skip_pm > 0) {
        const uint64_t keep_pm = 1000u - skip_pm;
        const std::string& smodel = cfg_.hw.compute.sparsity_model;
        // "off" (legacy): only the input DMA shrinks. "remove"/"lm"/"overhead"
        // also remove the skipped operands from F^W_LM (weight fetch) and from
        // the compute F^O (fewer MAC passes) — the ideal benefit. The "lm"/
        // "overhead" mechanism cost is added later, in run_tiled_layer.
        const bool remove_all = (smodel == "remove" || smodel == "lm" ||
                                 smodel == "overhead");
        auto keep = [keep_pm](uint32_t v) -> uint32_t {
            return static_cast<uint32_t>(std::max<uint64_t>(
                1, (static_cast<uint64_t>(v) * keep_pm) / 1000));
        };
        for (auto& t : txns) {
            if (t.input_fetch_bytes > 0) t.input_fetch_bytes = keep(t.input_fetch_bytes);
            if (remove_all) {
                if (t.weight_fetch_bytes > 0)
                    t.weight_fetch_bytes = keep(t.weight_fetch_bytes);
                if (t.output_write_operands > 0)   // compute F^O (not the write)
                    t.output_write_operands = keep(t.output_write_operands);
            }
        }
    }
}

std::vector<TileTxn> NpuController::build_layer_schedule(
    const LayerDescriptor& layer) const {
    const uint32_t elem = cfg_.hw.compute.element_size_bytes
                              ? cfg_.hw.compute.element_size_bytes : 1u;
    const uint32_t out_h = std::max(1u, layer.output.height);
    const uint32_t out_w = std::max(1u, layer.output.width);
    const uint32_t out_c = std::max(1u, layer.output.channels);
    const uint32_t in_c  = std::max(1u, layer.input.channels);
    const uint32_t kh = std::max(1u, layer.weight.height);
    const uint32_t kw = std::max(1u, layer.weight.width);

    // Output-tile size. Default (RTL/compiler-parity): the compiler-baked tile
    // from the descriptor. With spill modeling on, the tile is instead DERIVED
    // from the runtime GB capacity: reserve half the buffer to hold the partial
    // output tile resident (psum) and half to double-buffer the streamed
    // input+weight. A smaller GB forces a smaller tile → more spatial tiles →
    // the stationary-agnostic operand refetch below scales up accordingly.
    uint32_t th = layer.tile.t_r ? layer.tile.t_r : out_h;
    uint32_t tw = layer.tile.t_c ? layer.tile.t_c : out_w;
    if (cfg_.hw.buffers.global.model_spill) {
        // GB-aware tile from the selected compiler strategy (compiler.strategy).
        // Only reached under opt-in spill modeling — the default path keeps the
        // descriptor tile, so this cannot change the anchors.
        compiler::TileSearchInput ts;
        ts.out_h = out_h; ts.out_w = out_w; ts.out_c = out_c;
        ts.in_c = in_c;
        ts.kernel_h = kh; ts.kernel_w = kw;
        ts.stride = std::max(1u, layer.stride);
        ts.elem_bytes = elem;
        ts.gb_bytes = gb_size_bytes();
        ts.resident_cap_bytes = gb_size_bytes();
        ts.dataflow = (cfg_.hw.dataflow == "IS") ? Dataflow::IS
                    : (cfg_.hw.dataflow == "OS") ? Dataflow::OS
                    : Dataflow::WS;
        compiler::TilingStrategy strat = compiler::TilingStrategy::Dataflow;
        compiler::tiling_strategy_from_string(cfg_.compiler.strategy, strat);
        try {
            const compiler::TileSearchResult r =
                compiler::search_output_tile(ts, strat);
            th = r.t_h; tw = r.t_w;
        } catch (const std::exception& e) {
            std::cerr << "[compiler] " << e.what()
                      << " — using descriptor tile\n";
        }
    }

    // Fixed GB partition shapes the per-region tile budgets (0 = legacy
    // 40/40/20 split of the whole capacity).
    const auto& gb_part = cfg_.hw.buffers.global;
    const bool fixed_part = (gb_part.partition_mode == "fixed");
    TileSchedule sched = compute_tile_schedule(
        "rt", layer.layer_id, out_h, out_w, out_c, in_c, kh, kw,
        std::max(1u, layer.stride), layer.padding,
        gb_size_kb(),
        th, tw,
        fixed_part ? gb_part.input_partition_kb  : 0,
        fixed_part ? gb_part.weight_partition_kb : 0,
        fixed_part ? gb_part.output_partition_kb : 0,
        layer.type == FuncType::DepthwiseConv2D);

    OperandResidency residency;
    // Weight GB-residency: explicit config wins; "dataflow" (default) falls
    // back to the dataflow inference (WS/OS resident, IS streamed). Decoupling
    // lets an input-stationary mapping still hold weights resident (MIDAP's
    // FILTER_LOAD=ONE), which the dataflow tag alone cannot express. The
    // capacity check below still demotes a resident weight that does not fit.
    const std::string& wstat = cfg_.hw.buffers.global.weight_gb_stationary;
    residency.weight_resident =
        (wstat == "resident") ? true :
        (wstat == "streamed") ? false :
        (cfg_.hw.dataflow != "IS" && cfg_.hw.dataflow != "OS");
    residency.input_resident  = (cfg_.hw.dataflow == "IS");

    const uint32_t out_elems = out_h * out_w * out_c;
    const uint32_t passes = out_elems
        ? std::max(1u, layer.latency.write_output / out_elems) : 1u;
    // Operand (element) count of the whole input, matching weight_total's units
    // (fetch_operand1 is already an element count). layer.input.size is in
    // sizeof(float) units like the schedule footprints, so recover elements by
    // dividing by sizeof(float) — dividing by `elem` over-counted the resident
    // (IS) input by sizeof(float)/elem (4x at INT8), inflating DRAM reads and
    // the capacity-residency check. Physical bytes are × elem downstream.
    const uint32_t input_total  = layer.input.size / sizeof(float);
    const uint32_t weight_total = layer.latency.fetch_operand1;

    // Capacity-aware residency (hierarchy v1, D2): a declared-resident
    // operand that cannot physically coexist with the per-tile working set
    // in the GB is demoted to per-tile streaming — 2.3MB of weights do not
    // "reside" in a 128KB buffer. Working set = resident bytes + the largest
    // tile's (input footprint + output tile), in element bytes (the
    // schedule's footprints are in sizeof(float) units).
    uint64_t ws_stream_in = 0;   // per-tile (input + output) — WS case
    uint64_t ws_stream_wt = 0;   // per-tile (weight + output) — IS case
    const uint64_t gb_bytes_total = static_cast<uint64_t>(gb_size_kb()) * 1024;
    {
        constexpr uint32_t kSched = sizeof(float);
        for (const auto& td : sched.tiles) {
            const uint64_t in_b  = td.total_input_fetch_bytes  / kSched * elem;
            const uint64_t wt_b  = td.total_kernel_fetch_bytes / kSched * elem;
            const uint64_t out_b = static_cast<uint64_t>(td.output_height) *
                                   td.output_width * out_c * elem;
            ws_stream_in = std::max(ws_stream_in, in_b + out_b);
            ws_stream_wt = std::max(ws_stream_wt, wt_b + out_b);
        }
        const uint64_t gb_bytes = static_cast<uint64_t>(gb_size_kb()) * 1024;
        const uint64_t wt_bytes = static_cast<uint64_t>(weight_total) * elem;
        const uint64_t in_bytes = static_cast<uint64_t>(input_total) * elem;
        if (residency.weight_resident && wt_bytes + ws_stream_in > gb_bytes) {
            residency.weight_resident = false;
            std::cerr << "[residency] layer " << layer.layer_id << ": weights ("
                      << (wt_bytes >> 10) << " KB) + working set ("
                      << (ws_stream_in >> 10) << " KB) exceed GB ("
                      << (gb_bytes >> 10) << " KB) — weights streamed per tile\n";
        }
        if (residency.input_resident && in_bytes + ws_stream_wt > gb_bytes) {
            residency.input_resident = false;
            std::cerr << "[residency] layer " << layer.layer_id << ": input ("
                      << (in_bytes >> 10) << " KB) + working set ("
                      << (ws_stream_wt >> 10) << " KB) exceed GB — "
                         "input streamed per tile\n";
        }
    }

    std::vector<TileTxn> txns =
        lower_tile_schedule(sched, residency, input_total, weight_total, elem, passes);

    // Compression + input zero-skip scale the GB-boundary bytes only.
    apply_boundary_byte_scaling(txns, layer);

    // Inter-layer forwarding (hierarchy v1, D3, opt-in): when the retained
    // previous output covers this layer's whole input and can coexist with
    // the layer's working set in the GB, the input is served on-chip — the
    // DMA bytes vanish (path I3 becomes a GB hit) while operand counts (the
    // compute-side feed) are unchanged. All-or-nothing in v1.
    last_input_forwarded_bytes_ = 0;  // reset per layer; set on a retention hit below
    if (cfg_.hw.buffers.global.layer_fusion == "auto" &&
        retained_out_bytes_ > 0) {
        const uint64_t in_addr  = layer.input.address;
        // Descriptor sizes are in sizeof(float) units; convert to physical
        // bytes for the capacity check (containment uses descriptor units on
        // both sides, so it is unit-consistent as-is).
        const uint64_t in_bytes = layer.input.size;
        const uint64_t in_phys  = layer.input.size / sizeof(float) * elem;
        const uint64_t resident_wt =
            residency.weight_resident
                ? static_cast<uint64_t>(weight_total) * elem : 0;
        const bool contained =
            in_addr >= retained_out_addr_ &&
            in_addr + in_bytes <= retained_out_addr_ + retained_out_bytes_;
        const bool fits =
            in_phys + resident_wt + ws_stream_wt <= gb_bytes_total;
        if (std::getenv("FLEXNPUSIM_EMIT_TRACE"))
            std::cerr << "[retention] layer=" << layer.layer_id
                      << " in=0x" << std::hex << in_addr << "+" << in_bytes
                      << " retained=0x" << retained_out_addr_ << "+"
                      << retained_out_bytes_ << std::dec
                      << " contained=" << contained << " fits=" << fits << "\n";
        if (contained && fits) {
            uint64_t fwd = 0;
            for (auto& t : txns) { fwd += t.input_fetch_bytes;
                                   t.input_fetch_bytes = 0; }
            // Hand the forwarded size to run_tiled_layer so it can roll back
            // the producer's matching DRAM write (④): the buffer stayed in GB,
            // so neither the read here nor that write reached DRAM.
            last_input_forwarded_bytes_ = fwd;
            std::cerr << "[retention] layer " << layer.layer_id
                      << ": input served from GB (" << (fwd >> 10)
                      << " KB forwarded, no DRAM round-trip)\n";
        }
    }

    // psum spill: partial outputs live in the accumulator when output_location
    // is a dedicated store (accumulator/pe_local), otherwise in the GB. When a
    // tile's output exceeds that store and the reduction is split (num_passes>1),
    // the partial sum cannot stay resident across passes and round-trips DRAM:
    // (P-1) writes + (P-1) re-reads of the tile output on top of the final write.
    if (cfg_.hw.buffers.global.model_spill) {
        const std::string& loc = cfg_.hw.buffers.global.output_location;
        const uint64_t psum_store =
            (loc == "accumulator" || loc == "pe_local")
                ? static_cast<uint64_t>(cfg_.hw.buffers.accumulator.capacity_kb) * 1024
                : static_cast<uint64_t>(gb_size_bytes());
        for (size_t k = 0; k < txns.size() && k < sched.execution_order.size(); ++k) {
            const uint32_t tid = sched.execution_order[k];
            if (tid >= sched.tiles.size()) continue;
            const auto& td = sched.tiles[tid];
            const uint64_t out_bytes = static_cast<uint64_t>(td.output_height) *
                                       td.output_width * out_c * elem;
            if (td.num_passes > 1 && out_bytes > psum_store) {
                const uint64_t rt = static_cast<uint64_t>(td.num_passes - 1) * out_bytes;
                txns[k].psum_spill_write_bytes = static_cast<uint32_t>(rt);
                txns[k].psum_spill_read_bytes  = static_cast<uint32_t>(rt);
            }
        }
    }
    return txns;
}

uint64_t NpuController::tile_compute_cycles(const TileTxn& t, const LmParams& lp,
                                            const LmHwParams& feed_hw,
                                            const LayerDescriptor& layer,
                                            uint32_t layer_idx,
                                            uint64_t& tile_emit_ops) const {
    // Compute this tile as a mini-layer, analytically. The tile's operands are
    // resident (DMA'd by the caller), so the datapath issues at n_max/cycle once
    // the pipeline fills: T = ceil(F^O_tile / n_max) + l. This replaces a
    // per-issue-step loop that iterated F^O/n_max times (millions for deep
    // layers) just to arrive at the same closed-form value.
    const uint32_t nmax = std::max(1u, lp.n_max);
    // Compute-bound reconstruction: the datapath issues at n_max/cycle
    // (II = 1, systolic hop = 1 cycle) once the pipeline fills.
    const uint64_t base_pass_cycles =
        (static_cast<uint64_t>(t.output_write_operands) + nmax - 1) / nmax;
    const uint64_t compute_term = base_pass_cycles + lp.issue_latency;
    // Feed term: cycles to stream this tile's operands through the
    // configured GB→PE ports. Unlimited (0) ports contribute nothing.
    uint64_t feed_term = 0;
    auto feed = [](uint32_t ops, uint32_t port) -> uint64_t {
        return port ? (static_cast<uint64_t>(ops) + port - 1) / port : 0;
    };
    feed_term = std::max(feed_term,
                         feed(t.input_fetch_operands,  feed_hw.max_input_ops_per_cycle));
    feed_term = std::max(feed_term,
                         feed(t.weight_fetch_operands, feed_hw.max_weight_ops_per_cycle));
    feed_term = std::max(feed_term,
                         feed(t.output_write_operands, feed_hw.max_output_ops_per_cycle));
    const uint64_t tile_cycles = std::max(compute_term, feed_term);

    // Output-emitting operations for avg_output_ratio. The enable
    // structure decides how supply starvation manifests:
    //  - lockstep (n_min == n_max): every emission is full
    //    width; starvation stalls emissions, it never narrows them →
    //    emit slots = ⌈F^O_tile/n_max⌉ regardless of feed.
    //  - independent enable (n_min < n_max): a feed-bound tile spreads
    //    its issues across the feed window at reduced width, floored at
    //    n_min (below n_min the issue gate waits instead) →
    //    emit slots = min(max(ideal, feed), ⌈F^O_tile/n_min⌉).
    // No invented constants: only feed_term, n_min, n_max.
    static const bool emit_trace = (std::getenv("FLEXNPUSIM_EMIT_TRACE") != nullptr);
    const uint32_t nmin = std::max(1u, lp.n_min);
    // Lockstep override: the k/g/s derivation yields n_min = g, which
    // cannot express lockstep enable (e.g. NVDLA CACC where the true
    // n_min = n_max). hw.compute.enable_gran supplies the enable-structure
    // summary the derivation lacks; the CSV column `enable_gran`
    // overrides it per layer (layer.latency.enable_gran = (mode<<24)|N).
    // All three modes reduce to one emit-width floor: lockstep → n_max
    // (starvation stalls, never narrows), independent → n_min,
    // grouped:N → N clamped to [1, n_max].
    const std::string& eg = cfg_.hw.compute.enable_gran;
    const bool lockstep_global =
        (eg == "lockstep") || (eg != "independent" && nmin == nmax);
    uint32_t floor_w = lockstep_global ? nmax : nmin;
    const uint32_t enable_gran = layer.latency.enable_gran;
    switch (enable_gran >> 24) {
        case 1: floor_w = nmax; break;
        case 2: floor_w = nmin; break;
        case 3: floor_w = std::min(
                    std::max(enable_gran & 0xFFFFFFu, 1u),
                    static_cast<uint32_t>(nmax)); break;
        default: break;
    }
    const uint64_t ideal_slots = base_pass_cycles;
    tile_emit_ops = ideal_slots;
    if (feed_term > ideal_slots && floor_w < nmax) {
        const uint64_t floor_cap =
            (static_cast<uint64_t>(t.output_write_operands) + floor_w - 1) / floor_w;
        tile_emit_ops = std::min(feed_term, floor_cap);
    }
    if (emit_trace)
        std::cerr << "[emit] layer=" << layer_idx
                  << " in_ops=" << t.input_fetch_operands
                  << " wt_ops=" << t.weight_fetch_operands
                  << " out_ops=" << t.output_write_operands
                  << " nmin=" << nmin << " nmax=" << nmax
                  << " ideal=" << ideal_slots
                  << " feed=" << feed_term
                  << " emit=" << tile_emit_ops
                  << " floor=" << floor_w
                  << (floor_w == nmax ? " lockstep" : "") << "\n";
    return tile_cycles;
}

void NpuController::run_tiled_layer(const LayerDescriptor& layer, uint32_t layer_idx,
                                   uint32_t input_addr, uint32_t weight_addr) {
    // Fused activation epilogue: under layer fusion, ReLU/unary element-wise
    // activations are applied in the producer conv's output drain (a free
    // epilogue, as in MIDAP), not as a separate layer. Zero cycles, zero
    // traffic; just record the output for the next layer's retention.
    if (layer.type == FuncType::Activation &&
        cfg_.hw.buffers.global.layer_fusion == "auto") {
        suppress_dma_timing_ = false;
        update_retention_ledger(layer_idx);
        return;
    }
    const uint32_t output_addr = layer.output.address + cfg_.address_map.dram_base;
    const std::vector<TileTxn> schedule = build_layer_schedule(layer);
    // When this layer's input was served from the retained GB output, the
    // producer's matching DRAM output-write was equally unnecessary — the
    // buffer never left the chip. Roll that write back on both the global
    // counter and the producer's per-layer record, symmetric with the read
    // build_layer_schedule already suppressed. Without this, retention removed
    // the read but still charged the write, inflating DRAM writes ~14×.
    if (last_input_forwarded_bytes_ > 0) {
        const uint64_t wb = last_input_forwarded_bytes_;
        perf_mem_writes -= std::min<uint64_t>(perf_mem_writes, wb);
        if (!layer_records.empty()) {
            auto& prod = layer_records.back();
            prod.mem_wr_bytes -= std::min<uint64_t>(prod.mem_wr_bytes, wb);
            prod.out_wr_bytes -= std::min<uint64_t>(prod.out_wr_bytes, wb);
        }
    }
    const LmParams lp = loader.to_latency_params(layer_idx);

    const uint64_t layer_start_ns = now_ns();
    // The timing strategy decides whether per-transfer DMA sim timing is
    // suppressed (peak_bw) or accrued (sim); the memory term is applied below.
    if (!layer_timing_)
        layer_timing_ = make_layer_timing_strategy(cfg_.dram, axi_bus_width_bytes_);
    suppress_dma_timing_ = layer_timing_->suppresses_dma();
    uint64_t compute_cycles = 0;         // overlap pool (unserialized tiles)
    uint64_t compute_report_cycles = 0;  // + serialized tiles, for reporting
    uint64_t layer_rd_bytes = 0, layer_wr_bytes = 0;  // per-layer report sums
    uint64_t layer_out_wr = 0, layer_psum_wr = 0, layer_psum_rd = 0;
    // Output-emitting operations: issue slots in which the PE array emits
    // ≥1 output (analytic model: ⌈F^O_tile/n_max⌉ per tile, the ragged final
    // slot included). Denominator of avg_output_ratio below.
    uint64_t emit_ops = 0;
    // Feed-bound roofline (P2-4b): GB→PE port widths (operands/cycle, 0 =
    // unlimited) bound how fast a tile's operands reach the datapath. The
    // tile term becomes max(compute, feed) — PE-side ports become effective
    // in the main path without re-introducing a per-cycle model.
    const LmHwParams feed_hw =
        LmHwParams::from_config(cfg_, function_buffer_key(layer.type));

    for (const auto& t : schedule) {
        // --- DMA-load this tile's operands. Split into GB-sized chunks so a
        //     large tile footprint never overflows the L2 streaming buffer. ---
        auto stream_read = [&](uint32_t addr, uint32_t bytes) {
            if (bytes == 0) return;
            if (suppress_dma_timing_) {
                // peak_bw: count bytes, advance no sim time (bulk term after loop).
                perf_mem_reads += bytes; ++perf_read_txns;
            } else {
                // Continuous DMA: issue this transfer's bursts into the persistent
                // in-flight pipeline WITHOUT draining, so consecutive tile reads
                // keep the DRAM pipeline full — latency is paid once per layer, not
                // per transfer. Drained by pipe_read_flush() after the tile loop.
                rdma_ch_.engine.pipe_read_issue(addr, bytes);
                perf_mem_reads += bytes; ++perf_read_txns;
            }
        };
        if (t.input_fetch_bytes > 0)  stream_read(input_addr + t.input_addr_offset,
                                                  t.input_fetch_bytes);
        if (t.weight_fetch_bytes > 0) stream_read(weight_addr, t.weight_fetch_bytes);
        // psum spill re-reads: partial output loaded back from DRAM each pass
        // when it cannot stay resident (0 unless model_spill exposes it).
        if (t.psum_spill_read_bytes > 0)
            stream_read(output_addr + t.output_addr_offset, t.psum_spill_read_bytes);
        // NOTE: perf_mem_reads is already incremented inside trigger_rdma (the
        // stream_read calls above) — re-adding here double-counted every read
        // (input+weight+psum) in the tiled path only. layer_rd_bytes is the
        // separate per-layer report counter and IS accumulated here.
        layer_rd_bytes += t.input_fetch_bytes + t.weight_fetch_bytes
                        + t.psum_spill_read_bytes;
        layer_psum_rd  += t.psum_spill_read_bytes;

        // --- Compute this tile as a mini-layer (analytic timing). ---
        uint64_t tile_emit_ops = 0;
        const uint64_t tile_cycles =
            tile_compute_cycles(t, lp, feed_hw, layer, layer_idx, tile_emit_ops);
        emit_ops += tile_emit_ops;
        // Tile-boundary overlap bound: with unlimited PE buffers (default 0)
        // every tile's compute overlaps the streamed DMA and the layer time
        // is resolved once after the loop (max of memory vs compute). A
        // finite input/weight queue smaller than this tile's streamed
        // operands cannot hold the next tile during compute — this tile's
        // load and compute serialize. Conservative bound; no partial-overlap
        // factor is invented.
        const bool queue_serial =
            (feed_hw.input_buf_capacity != 0 &&
             t.input_fetch_operands  > feed_hw.input_buf_capacity) ||
            (feed_hw.weight_buf_capacity != 0 &&
             t.weight_fetch_operands > feed_hw.weight_buf_capacity);
        if (queue_serial) {
            // Advance time now: this tile's compute cannot hide under the
            // next tile's DMA. Reported compute still includes it (the
            // datapath did run) — only the overlap pool excludes it.
            wait(static_cast<double>(tile_cycles) * clk_period_ns_, SC_NS);
            compute_report_cycles += tile_cycles;
        } else {
            // Accumulate compute; do NOT advance time per tile. The layer
            // time is the larger of memory vs compute+overhead, resolved by
            // a single overlap wait after the loop.
            compute_cycles += tile_cycles;
        }

        // --- Physical write-back (M1): final outputs and psum spills
        //     traverse the same AXI/DRAM path as reads, so DRAM write timing
        //     and R/W turnaround (MC knobs) are exercised. Concurrency with
        //     write_thread is serialized inside trigger_wdma. Byte counting
        //     happens inside trigger_wdma (perf_mem_writes) — no double add.
        auto stream_write = [&](uint32_t addr, uint32_t bytes) {
            if (bytes == 0) return;
            if (suppress_dma_timing_) {
                perf_mem_writes += bytes; ++perf_write_txns;
            } else {
                // Continuous write pipeline (mirror of the read path): issue the
                // transfer's write bursts without draining, flushed at layer end.
                wdma_ch_.engine.pipe_write_issue(addr, bytes);
                perf_mem_writes += bytes; ++perf_write_txns;
            }
        };
        if (t.psum_spill_write_bytes > 0)
            stream_write(output_addr + t.output_addr_offset,
                         t.psum_spill_write_bytes);
        if (t.output_write_bytes > 0)
            stream_write(output_addr + t.output_addr_offset,
                         t.output_write_bytes);
        layer_wr_bytes  += t.output_write_bytes + t.psum_spill_write_bytes;
        layer_out_wr    += t.output_write_bytes;
        layer_psum_wr   += t.psum_spill_write_bytes;
    }

    // Cycle-streaming chain (LmChain closed form). A multi-function packet's
    // trailing stages — NVDLA CACC→SDP→PDP, or conv→act→pool — pipeline behind
    // the producer with a same-cycle hand-off (latency_model_chain.h). Each trailing stage
    // therefore adds its pipeline fill l_k always, and raises the packet time to
    // its own steady throughput S_k = ⌈F^O_k/n_max_k⌉ (II = 1) only when it is
    // the bottleneck (max, not sum — the LmChain "chained < sequential" contract).
    // Single-function packets (pg=1) skip this loop → unchanged. This is the
    // principled replacement for treating a trailing function as free.
    {
        const uint32_t pg = count_packet_group(layer_idx);
        for (uint32_t k = 1; k < pg; ++k) {
            const LmParams lpk = loader.to_latency_params(layer_idx + k);
            const uint64_t nmk = std::max(1u, lpk.n_max);
            const uint64_t sk  = (static_cast<uint64_t>(lpk.write_output)
                                   + nmk - 1) / nmk;
            compute_cycles += lpk.issue_latency;             // stage pipeline fill
            compute_cycles = std::max(compute_cycles, sk);   // bottleneck stage
        }
    }

    // Sparsity-mechanism cost for zero-skipped layers (NullHop). The operand
    // removal ("remove"/"lm"/"overhead") already shrank F^I/F^W/F^O above;
    // here the mechanism's own cost is added on top of that ideal benefit:
    //   "lm"       = a zero-detect/compaction stage that SCANS the DENSE input
    //                stream every cycle at sparsity_detect_width — chain-
    //                overlapped, so it adds its pipeline fill and only bottle-
    //                necks the layer when the scan is slower than the MAC;
    //   "overhead" = a lump per-layer latency (mechanism cost as a constant).
    // "remove" (ideal) and "off" add nothing here.
    if (layer.preprocess_param0 > 0) {
        const std::string& sm = cfg_.hw.compute.sparsity_model;
        if (sm == "lm") {
            const uint64_t dw = cfg_.hw.compute.sparsity_detect_width
                ? cfg_.hw.compute.sparsity_detect_width
                : std::max(1u, cfg_.hw.compute.ops_per_pass *
                               cfg_.hw.compute.num_output_lanes);
            const uint64_t s_detect =
                (static_cast<uint64_t>(lp.fetch_operand0) + dw - 1) / dw;
            compute_cycles += lp.issue_latency;                    // detect fill
            compute_cycles = std::max(compute_cycles, s_detect);   // scan-bound
        } else if (sm == "overhead") {
            compute_cycles += cfg_.hw.compute.sparsity_overhead_cyc;
        }
    }

    suppress_dma_timing_ = false;  // end of this layer's suppressed DMA window

    // Drain the continuous read+write pipelines: all tile reads/writes were
    // issued without draining, so the DRAM pipeline stayed full across the
    // layer. Draining now advances sim time to the last burst's completion — a
    // single fill latency plus the streamed bandwidth, the real-hardware behavior.
    rdma_ch_.engine.pipe_read_flush();
    wdma_ch_.engine.pipe_write_flush();

    // Overlap resolution: the layer occupies max(memory, compute+overhead).
    // In "sim" mode the per-tile DMA already advanced sim time by the memory
    // cost (dma_elapsed_cyc); wait only the compute excess not hidden under it.
    const uint64_t dma_elapsed_cyc = static_cast<uint64_t>(
        (now_ns() - layer_start_ns) / std::max(1.0, clk_period_ns_));
    // The strategy charges the layer memory term: sim = accrued DMA time,
    // peak_bw = traffic at the bus bandwidth plus one read latency.
    const uint64_t mem_cyc =
        layer_timing_->mem_cycles(layer_rd_bytes, layer_wr_bytes, dma_elapsed_cyc);
    const uint64_t layer_time = std::max(mem_cyc, compute_cycles);
    if (layer_time > dma_elapsed_cyc)
        wait(static_cast<double>(layer_time - dma_elapsed_cyc)
                 * clk_period_ns_, SC_NS);

    const uint64_t layer_macs = static_cast<uint64_t>(layer.ops_per_output)
        * layer.output.height * layer.output.width * layer.output.channels;
    perf_macs += layer_macs;
    const uint64_t layer_end_ns = now_ns();
    perf_cycles += static_cast<uint64_t>(
        (layer_end_ns - layer_start_ns) / std::max(1.0, clk_period_ns_));

    // max_output_ratio = PE utilization = fraction of peak issue width
    // achieved = F^O / (layer_cycles · n_max). Low when the layer is memory-
    // bound (DMA cycles dominate), high when compute-bound.
    const uint64_t layer_cycles = static_cast<uint64_t>(
        (layer_end_ns - layer_start_ns) / std::max(1.0, clk_period_ns_));
    const uint32_t nmax_r = std::max(1u, lp.n_max);
    double max_output_ratio = 0.0;
    if (layer_cycles > 0)
        max_output_ratio = static_cast<double>(lp.write_output) /
                           (static_cast<double>(layer_cycles) * nmax_r);
    if (max_output_ratio > 1.0) max_output_ratio = 1.0;

    // avg_output_ratio = mean number of outputs the PE array produced per
    // output-emitting operation = F^O / (output-emitting operations).
    // ∈ [1, n_max]; n_max means every emitting operation filled the full
    // issue width.
    double avg_output_ratio = 0.0;
    if (emit_ops > 0)
        avg_output_ratio = static_cast<double>(lp.write_output) /
                           static_cast<double>(emit_ops);

    const char* ltype = (layer.type == FuncType::Conv2D)          ? "conv2d"
                      : (layer.type == FuncType::DepthwiseConv2D)  ? "dwconv"
                      : (layer.type == FuncType::FullyConnected)   ? "fc"
                      : (layer.type == FuncType::Pooling)          ? "pool"
                      : (layer.type == FuncType::Activation)       ? "act" : "other";

    // psum-store (OFMAP-SRAM) writes as a function of psum residency: a resident
    // accumulator (output-stationary, or a dedicated accumulator/pe_local store)
    // holds the output across the k-way reduction and writes it once; otherwise
    // every pass-completion (F^O) round-trips the store.
    const uint64_t out_elems_layer =
        static_cast<uint64_t>(std::max(1u, layer.output.height)) *
        std::max(1u, layer.output.width) * std::max(1u, layer.output.channels);
    const std::string& psum_loc = cfg_.hw.buffers.global.output_location;
    const bool psum_resident = (cfg_.hw.dataflow == "OS") ||
        psum_loc == "accumulator" || psum_loc == "pe_local";
    const uint64_t psum_sram_wr =
        psum_resident ? out_elems_layer : lp.write_output;

    if (layer_line_csv())
        std::cerr << "@LAYER," << layer_idx << "," << ltype
                  << "," << lp.fetch_operand0
                  << "," << lp.fetch_operand1
                  << "," << lp.write_output
                  << "," << lp.n_max
                  << ",0,0,0," << (layer_end_ns - layer_start_ns)
                  << ",0," << (compute_cycles + compute_report_cycles)
                  << ",0," << layer_macs << ",tiled,"
                  << max_output_ratio << "," << avg_output_ratio
                  << "," << psum_sram_wr << "\n";
    else
        std::cerr << " layer " << std::setw(2) << layer_idx << "  "
                  << std::left << std::setw(7) << ltype << std::right
                  << " done\n";

    LayerPerfRecord rec;
    rec.layer_id         = layer_idx;
    rec.type             = ltype;
    rec.total_cycles     = layer_cycles;
    rec.compute_cycles   = compute_cycles + compute_report_cycles;
    rec.memory_cycles    = mem_cyc;
    rec.macs             = layer_macs;
    rec.f_o              = lp.write_output;
    rec.mem_rd_bytes     = layer_rd_bytes;
    rec.mem_wr_bytes     = layer_wr_bytes;
    rec.out_wr_bytes     = layer_out_wr;
    rec.psum_wr_bytes    = layer_psum_wr;
    rec.psum_rd_bytes    = layer_psum_rd;
    rec.psum_sram_wr     = psum_sram_wr;
    rec.n_max            = nmax_r;
    rec.max_output_ratio = max_output_ratio;
    rec.avg_output_ratio = avg_output_ratio;
    layer_records.push_back(rec);

    // Retention ledger update (D3): this layer's output becomes the next
    // layer's forwarding candidate. Shared with the non-tiled path so both
    // record it consistently — see update_retention_ledger.
    update_retention_ledger(layer_idx);
}

bool NpuController::check_layer_fits_or_die(const LayerDescriptor& layer) const {
    // Simulator does NOT auto-decide tiling. Tile schedule is the user's /
    // compiler's responsibility. We only check that the (already-tiled)
    // layer fits in the configured GB. If it doesn't, we fail fast so the
    // user can either provide a tile schedule or increase GB capacity.
    if (cfg_.hw.buffers.global.capacity_kb == 0) {
        // GB capacity unconfigured: assume unlimited (legacy default).
        return false;
    }

    // R9: if the descriptor carries a TileShape with tile_enable, the caller
    // has asserted that per-tile footprint fits; skip whole-layer check.
    // Actual tile-aware DMA iteration arrives in R9.5 (controller tile loop +
    // GB partition management). Until then, the aggregate bytes in the
    // descriptor already reflect tile-induced refetch traffic (Mode B
    // derivation), so simulated cycle counts remain meaningful.
    if (layer.flags & FuncFlag::TileEnable) {
        return true;  // tile mode — skip whole-layer fit check
    }

    uint64_t layer_bytes = static_cast<uint64_t>(layer.input.size)
                         + layer.weight.size
                         + layer.output.size;
    uint64_t gb_bytes = static_cast<uint64_t>(
        cfg_.hw.buffers.global.capacity_kb) * 1024;

    if (layer_bytes > gb_bytes) {
        std::cerr << "[FATAL] Layer " << layer.layer_id
                  << " requires " << (layer_bytes / 1024) << " KB "
                  << "(input=" << (layer.input.size / 1024) << "KB"
                  << " weight=" << (layer.weight.size / 1024) << "KB"
                  << " output=" << (layer.output.size / 1024) << "KB) "
                  << "but Global Buffer is "
                  << (gb_bytes / 1024) << " KB. "
                  << "Tile decomposition is the compiler's responsibility — "
                  << "either reduce layer dimensions, increase "
                  << "global_buffer.total_capacity_kb, or provide a tile "
                  << "schedule (set TileShape columns in the network CSV to "
                  << "enable flags.TileEnable)." << std::endl;
        sc_core::sc_stop();
        std::exit(1);
    }
    return false;  // tile mode is not auto-activated
}


}  // namespace flexnpu_sim
