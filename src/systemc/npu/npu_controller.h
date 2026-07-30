/**
 * @file npu_controller.h
 * @brief NPU Controller SC_MODULE
 *
 * FSM + 13 registers + L2 Global Buffer + Read/Compute/Write pipeline.
 * Serves CPU register reads/writes as an AXI slave.
 * Drives the two DMA engines (rDMAC + wDMAC) through pointers.
 *
 * v2.0: fast_dma removed — every DMA goes through trigger_rdma/wdma as a
 *       real AXI transfer. MemoryAxiSlave applies DRAM timing on the AXI
 *       response.
 *
 * 5 SC_THREADs:
 *   slave_read_thread  — CPU register reads
 *   slave_write_thread — CPU register writes (START trigger)
 *   read_thread        — per-tile DMA reads (drives rDMAC)
 *   compute_thread     — L2 polling → latency model update
 *   write_thread       — per-tile DMA writes (drives wDMAC)
 */

#pragma once

#include <systemc.h>
#include "axi.h"
#include "systemc/npu/model/gb_state.h"
#include "systemc/npu/model/npu_global_buffer.h"
#include "common/tile_descriptor.h"
#include "common/types.h"
#include "systemc/dma/dma_engine.h"
#include "systemc/npu/dma_channel.h"
#include "systemc/npu/ufb.h"
#include "compiler/dnn_image/dnn_image_loader.h"
#include "common/flexnpu_config.h"
#include "model/latency/latency_model.h"
#include "systemc/memory/model/layer_timing_strategy.h"
#include <algorithm>
#include <deque>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace flexnpu_sim {

// NPU register bus: 32-bit addr, 32-bit data
using NpuRegConfig = AXI4_Default;

SC_MODULE(NpuController) {
    // ========================================================================
    // Constants
    // ========================================================================
    static constexpr uint32_t MAX_BURST_BEATS  = 256;  // AXI4 max burst length

    // ========================================================================
    // Ports
    // ========================================================================
    sc_in_clk clk;

    // rDMAC / wDMAC live in the Npu container (siblings). The controller DRIVES
    // them through the channels' command signals: rdma via trigger_rdma
    // (read_thread) — SC_ONE_WRITER; wdma via wdma_issue_thread — SC_ONE_WRITER
    // (requesters post via trigger_wdma). *_busy/*_done are driven by the engines.
    DmaChannel& rdma_ch_;
    DmaChannel& wdma_ch_;

    // ========================================================================
    // Internal: AXI slave interface
    // ========================================================================
    AXI_slave_if<NpuRegConfig>* slave_if;

    // ========================================================================
    // Global Buffer (byte storage + scratchpad/cache presence model)
    // ========================================================================
    static constexpr uint32_t GB_SIZE_DEFAULT = 256 * 1024;  // 256KB default
    /// On-chip SRAM. Owned by the Npu container and shared in by reference
    /// (Phase B: GB is the controller's sibling, not its property).
    NpuGlobalBuffer& global_buffer_;
    config::FlexNpuSimConfig cfg_;

    /// Canonical GB size accessors.
    uint32_t gb_size_bytes() const { return global_buffer_.size(); }
    uint32_t gb_size_kb()    const { return global_buffer_.size() / 1024; }

    // GB operand-level state (input/weight/output regions)
    GbState gb_;

    // R10: per-layer output byte cache for chaining (layer N+1 input
    // sourced from layer N output when functional path is active).
    std::vector<std::vector<uint8_t>> layer_output_cache_;

    // ========================================================================
    // Registers (v2.0 Table 13, 13 in total)
    // ========================================================================
    uint32_t regs[npu_reg::REG_COUNT];
    uint32_t npu_base_ = memory_map::ADDR_NPU;  ///< MMIO register base (config-driven).

    // ========================================================================
    // FSM & Model
    // ========================================================================
    NpuState state;
    DnnImageLoader loader;
    /// Compute datapath (PE-array timing model), owned by the Npu container and
    /// driven by the controller's GB→PE feed loop (run_layer_compute).
    Ufb& ufb_;

    // ========================================================================
    // Performance Counters
    // ========================================================================
    uint64_t perf_cycles     = 0;
    uint64_t perf_macs       = 0;
    uint64_t perf_mem_reads  = 0;
    uint64_t perf_mem_writes = 0;
    // DMA transfer (transaction) counts, co-incremented with the byte counters
    // above — one per trigger_rdma/trigger_wdma call. Report derives average
    // bytes/transaction (DMA chunk size) from these. Profiling stat, not an
    // MMIO-visible HW counter.
    uint64_t perf_read_txns  = 0;
    uint64_t perf_write_txns = 0;

    // Per-layer record for report generation (-report). One entry per
    // executed layer, mirroring the @LAYER trace fields.
    struct LayerPerfRecord {
        uint32_t    layer_id      = 0;
        std::string type;
        uint64_t    total_cycles  = 0;   ///< wall cycles for the layer
        uint64_t    compute_cycles= 0;   ///< compute term (serialized incl.)
        uint64_t    memory_cycles = 0;   ///< roofline memory term (DMA elapsed
                                         ///< at overlap resolution; serialized
                                         ///< tile waits included)
        uint64_t    macs          = 0;
        uint64_t    f_o           = 0;   ///< pass completions (F^O)
        uint64_t    mem_rd_bytes  = 0;   ///< input+weight+psum re-reads
        uint64_t    mem_wr_bytes  = 0;   ///< output writes + psum spills
        // Output-stream decomposition (paper terms: F^O = accumulator-side
        // pass completions; boundary output write W^O = final stores + spill):
        uint64_t    out_wr_bytes  = 0;   ///< final output stores only
        uint64_t    psum_wr_bytes = 0;   ///< partial-output spill writes
        uint64_t    psum_rd_bytes = 0;   ///< partial-output reloads
        // psum-store (OFMAP-SRAM) write traffic as a function of psum residency
        // (psum_home): resident accumulator (output-stationary, or a dedicated
        // accumulator/pe_local store) writes the output once; a non-resident
        // psum round-trips the store every pass-completion (= F^O). Matches
        // SCALE-Sim's SRAM OFMAP writes (OS -> output-once, WS/IS -> F^O).
        uint64_t    psum_sram_wr  = 0;
        uint32_t    n_max         = 1;
        double      max_output_ratio = 0.0;  ///< F^O/(cycles·n_max)
        double      avg_output_ratio = 0.0;  ///< F^O/emit_ops
    };
    std::vector<LayerPerfRecord> layer_records;

    // Layer progress (packed into NPUSR bits[8:23] for driver watchdog)
    uint32_t current_layer_idx_ = 0;
    uint32_t total_layers_      = 0;

    // ========================================================================
    // Per-layer Profiling (DSE analysis)
    // ========================================================================
    struct LayerProfile {
        uint64_t dma_read_ns  = 0;   // time spent in DMA reads
        uint64_t compute_ns   = 0;   // time spent in compute (stalls included)
        uint64_t write_ns     = 0;   // time spent in DMA writes
        uint64_t total_ns     = 0;   // whole-layer time
        uint64_t compute_cycles = 0; // pure compute cycle count
        uint64_t stall_cycles = 0;   // cycles stalled waiting on DMA
        uint64_t macs         = 0;   // MAC operation count
    };
    LayerProfile layer_prof_;
    uint64_t prof_compute_start_ns_ = 0;  // recorded by compute_thread

    // ========================================================================
    // AXI Bus Configuration (Step 2: M7)
    // ========================================================================
    uint32_t axi_bus_width_bytes_ = 8;  // default 64-bit AXI bus

    // ========================================================================
    // Double Buffer Configuration
    // ========================================================================
    bool double_buffer_enabled_ = false;
    // Set true only while run_tiled_layer streams a layer's tiles under the
    // "peak_bw" memory-time model: trigger_rdma/trigger_wdma then count bytes
    // but advance no sim time, and run_tiled_layer applies one bus-bandwidth-
    // bound memory term for the whole layer instead. The tiled executor is
    // synchronous, so no concurrent DMA observes this flag.
    bool suppress_dma_timing_ = false;
    // Config-selected layer memory-timing strategy (sim vs peak_bw); replaces
    // the scattered `tiled_dma_timing == "peak_bw"` branches.
    std::unique_ptr<LayerTimingStrategy> layer_timing_;
    double clk_period_ns_ = 10.0;  // Default 100 MHz

    // Optional functional OFM write-back source
    bool functional_ofm_enabled_ = false;
    std::string functional_ofm_dir_;
    bool functional_bin_dump_enabled_ = false;

    // (GB mode + scratchpad/cache presence now live in global_buffer_.)

    // ========================================================================
    // Synchronization
    // ========================================================================
    sc_event start_event;
    sc_event read_tile_done;
    sc_event compute_tile_done;
    // WDMA issue stage — the SINGLE writer of the wdma_* command signals.
    // Both requesters (physical write-back in run_tiled_layer/read_thread and
    // the fused drain in write_thread) post a descriptor and block;
    // wdma_issue_thread is the only process that drives the handshake, so
    // SC_ONE_WRITER holds (no SC_MANY_WRITERS relaxation, no cooperative mutex).
    struct WdmaReq { uint32_t dst = 0, size = 0, l2_off = 0; bool done = false; };
    std::deque<WdmaReq*> wdma_queue_;
    sc_event             wdma_req_evt_;     // requester → issue thread
    sc_event             wdma_retire_evt_;  // issue thread → requesters
    // Inter-layer retention ledger (hierarchy v1, D3): the previous tiled
    // layer's output range, eligible to serve the next layer's input from
    // the GB without a DRAM round-trip (opt-in: layer_fusion=auto).
    uint64_t retained_out_addr_  = 0;
    uint64_t retained_out_bytes_ = 0;
    // Bytes the current layer's input DMA served from the retained GB output
    // (set by build_layer_schedule when it forwards a read). run_tiled_layer
    // rolls the producer's matching DRAM output-write back by the same amount,
    // so a retained activation never counts against DRAM on either side.
    mutable uint64_t last_input_forwarded_bytes_ = 0;

    // DMA/Compute overlap events (Step 5)
    sc_event data_chunk_event;      // read_thread → compute_thread: new data arrived
    sc_event output_ready_event;    // compute_thread → write_thread: output produced
    sc_event write_done_event;      // write_thread → compute_thread: output DMA done
    sc_event compute_done_event;    // compute_thread → read_thread: compute done (Step 7)

    // Layer state tracking (Step 5)
    bool layer_input_loaded_   = false;  // current layer's input fully loaded
    bool layer_weight_loaded_  = false;  // current layer's weights fully loaded
    bool layer_compute_done_   = false;  // current layer's compute finished
    uint32_t write_layer_idx_  = 0;      // layer index handed to write_thread

    // ========================================================================
    // Constructor
    // ========================================================================
    SC_HAS_PROCESS(NpuController);

    NpuController(sc_module_name name, NpuGlobalBuffer& gb,
                  DmaChannel& rdma_ch, DmaChannel& wdma_ch, Ufb& ufb,
                  uint32_t npu_base = memory_map::ADDR_NPU);
    ~NpuController();

    /**
     * @brief Bind the AXI slave ports to signals
     */
    void bind(AXI_SIGNALS<NpuRegConfig>& signals);

    /**
     * @brief Set double buffer mode
     */
    void set_double_buffer(bool enable) { double_buffer_enabled_ = enable; }

    /**
     * @brief Set AXI bus width (Step 2: M7)
     * @param bits AXI data bus width in bits (32, 64, 128, 256)
     */
    void set_axi_bus_width(uint32_t bits) { axi_bus_width_bytes_ = bits / 8; }

    /**
     * @brief Set NPU clock period in nanoseconds
     */
    void set_clock_period_ns(double ns) { clk_period_ns_ = (ns > 0.0) ? ns : 10.0; }

    /**
     * @brief Set NPU system configuration (compute array, buffers, dma, etc.)
     */
    void set_system_config(const config::FlexNpuSimConfig& cfg) {
        cfg_ = cfg;
        const uint32_t elem = cfg.hw.compute.element_size_bytes;
        gb_.configure(
            cfg.hw.buffers.global.capacity_kb,
            cfg.hw.buffers.global.input_partition_kb,
            cfg.hw.buffers.global.weight_partition_kb,
            cfg.hw.buffers.global.output_partition_kb,
            elem,
            gb_partition_mode_from_string(cfg.hw.buffers.global.partition_mode),
            gb_output_location_from_string(cfg.hw.buffers.global.output_location),
            cfg.hw.buffers.global.input_compression_ratio,
            cfg.hw.buffers.global.weight_compression_ratio,
            cfg.hw.buffers.global.output_compression_ratio);
        if (cfg.functional.enabled) {
            functional_ofm_enabled_ = true;
            if (functional_ofm_dir_.empty()) functional_ofm_dir_ = cfg.functional.ofm_dir;
            functional_bin_dump_enabled_ = cfg.functional.bin_dump;
        }
    }

    /**
     * @brief Enable/disable functional OFM write-back path.
     *
     * When enabled, write_thread loads per-layer OFM bytes from files
     * in functional_ofm_dir_ and writes those bytes to DRAM via WDMA.
     */
    void set_functional_ofm_mode(bool enable) { functional_ofm_enabled_ = enable; }

    /**
     * @brief Set per-layer OFM file directory.
     * File naming convention: layer_<layer_idx>.bin
     */
    void set_functional_ofm_dir(const std::string& dir) { functional_ofm_dir_ = dir; }

    /**
     * @brief Select the global-buffer mode (scratchpad/cache).
     */
    void set_gb_mode(NpuGlobalBuffer::Mode mode);

    /**
     * @brief Set the global-buffer cache line size in bytes.
     */
    void set_gb_cache_line_size(uint32_t bytes);

    /**
     * @brief Enable timeline trace output. Empty path disables tracing.
     */
    void set_timeline_output(const std::string& path);

private:
    // ========================================================================
    // SC_THREADs
    // ========================================================================

    /** AXI slave: serves CPU register reads */
    void slave_read_thread();

    /** AXI slave: serves CPU register writes (START trigger) */
    void slave_write_thread();

    /** Per-layer DMA reads (drives rDMAC) */
    void read_thread();

    /** L2 polling → latency model update */
    void compute_thread();

    // compute_thread functional stages — one named step per phase, so
    // compute_thread reads as an orchestrator (see npu_controller_compute.cpp).
    /** R10: pre-compute this layer's output bytes via the function model. */
    void precompute_layer_function(uint32_t layer_idx, const LayerDescriptor& layer);
    /** GB prefetch gate: block until enough operands have accumulated. */
    void wait_gb_prefetch_gate(uint32_t layer_idx, const LmParams& params);
    /** Run the layer's LM (+ packet chain) to completion; returns total MACs
     *  and accumulates compute/stall cycle counts. Drives ufb_.lm. */
    uint64_t run_layer_compute(uint32_t layer_idx, const LayerDescriptor& layer,
                               const LmParams& params, uint32_t pipeline_group_size,
                               uint64_t& total_compute_cycles,
                               uint64_t& total_stall_cycles);

    /** Per-layer DMA writes (drives wDMAC) */
    void write_thread();

    /** WDMA issue stage — sole driver of wdma_* command signals (SC_ONE_WRITER) */
    void wdma_issue_thread();

    // ========================================================================
    // Helpers
    // ========================================================================

    /** Register read (offset → value) */
    uint32_t reg_read(uint32_t offset) const;

    /** Register write (offset, value) — fires start_event when START is written */
    void reg_write(uint32_t offset, uint32_t value);

    /** Refresh NPUSR */
    void update_status_reg();

    /** Stream the DNN image (header + body) from DRAM and parse it into
     *  `loader`. Returns false on a malformed/short image (caller idles). */
    bool load_dnn_image();

    /** Trigger a DMA read and wait for done (real AXI transfer) */
    void trigger_rdma(uint32_t src, uint32_t size, uint32_t l2_off);

    /** Trigger a DMA write and wait for done (real AXI transfer) */
    void trigger_wdma(uint32_t dst, uint32_t size, uint32_t l2_off);

    uint32_t input_stream_capacity_bytes() const;
    uint32_t output_region_base_offset() const;
    uint32_t output_region_capacity_bytes() const;
    void dump_timeline_trace() const;
    uint64_t now_ns() const;
    const char* state_to_string(NpuState s) const;
    void add_timeline_event(const std::string& type,
                            uint32_t layer,
                            uint64_t start_ns,
                            uint64_t end_ns,
                            uint64_t bytes = 0,
                            const std::string& detail = "",
                            uint32_t lm_input_i = 0,
                            uint32_t lm_input_w = 0,
                            uint32_t lm_output = 0,
                            const std::string& subject = "");
    std::string current_subject() const;
    void flush_state_timeline();
    void set_state(NpuState s);

    // Completion ownership (topology stabilization): AllDone is promoted by
    // the LAST finisher — read streaming done AND compute loop idle AND no
    // pending write handoffs. Previously read_thread declared AllDone while
    // a write handoff was still in flight; write_thread then dragged the FSM
    // back to LayerComplete and the CPU driver polled NPUSR forever.
    void maybe_finalize();
    /// Single execution-path ownership predicate (tiled vs fused) — shared
    /// by read_thread and compute_thread. See the definition for why.
    bool layer_runs_tiled(const LayerDescriptor& layer) const;
    bool     streams_done_   = false;
    bool     compute_active_ = false;
    uint32_t writes_pending_ = 0;
    void wait_with_timeline(sc_event& ev, uint32_t layer, const char* detail);

    /** Count consecutive descriptors sharing the same packet_layer_id */
    uint32_t count_packet_group(uint32_t start_idx) const;

    /** Fail-fast if layer doesn't fit in GB. Always returns false (tile
     *  mode is not auto-activated; tile schedule is the user's
     *  responsibility). */
    bool check_layer_fits_or_die(const LayerDescriptor& layer) const;

    // Runtime tile schedule (compiler tiling decision → dataflow-baked txns).
    std::vector<TileTxn> build_layer_schedule(const LayerDescriptor& layer) const;
    // Scale a schedule's GB-boundary bytes for compression + input zero-skip
    // (operand/compute counts unchanged). Mutates txns in place.
    void apply_boundary_byte_scaling(std::vector<TileTxn>& txns,
                                     const LayerDescriptor& layer) const;
    // Execute one TileEnable layer tile-by-tile inline (read_thread driver):
    // per tile DMA → compute (local latency model) → writeback. Returns total
    // MACs; updates perf counters + layer_prof_. Race-free (each tile fits GB).
    void run_tiled_layer(const LayerDescriptor& layer, uint32_t layer_idx,
                         uint32_t input_addr, uint32_t weight_addr);
    // Analytic per-tile timing: compute-bound vs feed-bound cycles for one tile
    // (max of the two), plus its output-emitting op count (tile_emit_ops, for
    // avg_output_ratio). Pure — no DMA, no sim-time advance.
    uint64_t tile_compute_cycles(const TileTxn& t, const LmParams& lp,
                                 const LmHwParams& feed_hw,
                                 const LayerDescriptor& layer, uint32_t layer_idx,
                                 uint64_t& tile_emit_ops) const;
    // Record the packet group's tail output as the retention forwarding
    // candidate. Called from both execution paths (tiled + non-tiled dispatch)
    // so the ledger never lags a buffer behind. start_layer_idx = group start.
    void update_retention_ledger(uint32_t start_layer_idx);

    // True when this layer's output is consumed on-chip by the next packet
    // group (inter-layer retention) and so never reaches DRAM. The producer-
    // side mirror of the consumer's containment test in build_layer_schedule:
    // the non-tiled write path uses it to suppress the spurious output write,
    // symmetric with the tiled path's ④ write rollback (see run_tiled_layer).
    bool output_retained_next_layer(uint32_t layer_idx) const;

    struct TimelineEvent {
        std::string type;   // rdma, rdma_cache_hit, wdma, compute, wait, stall, state
        std::string detail; // event subtype/reason
        std::string subject; // read_thread, compute_thread, write_thread, latency_model, ...
        uint32_t layer = 0;
        uint64_t start_ns = 0;
        uint64_t end_ns = 0;
        uint64_t bytes = 0;
        uint32_t lm_input_i = 0;
        uint32_t lm_input_w = 0;
        uint32_t lm_output = 0;
    };
    bool timeline_enabled_ = false;
    std::string timeline_output_path_;
    std::vector<TimelineEvent> timeline_events_;
    bool timeline_state_active_ = false;
    NpuState timeline_state_ = NpuState::Idle;
    uint32_t timeline_state_layer_ = 0;
    uint64_t timeline_state_start_ns_ = 0;
    uint64_t timeline_run_start_ns_ = 0;
};

} // namespace flexnpu_sim
