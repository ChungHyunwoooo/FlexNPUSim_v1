/**
 * @file flexnpu_config.h
 * @brief Unified user-facing configuration root (R0c).
 *
 * Single source of truth for all simulator settings. Loaded from one JSON
 * file (JSONC: comments allowed) via load_flexnpu_config() in
 * src/common/config_loader.cpp.
 *
 * R6 made this the canonical config for the main simulation path
 * (NpuController, LmHwParams, flexnpusim_system). Since R2 it is the
 * sole config: the legacy SimulatorConfig/NpuSystemConfig family and the
 * flexnpusim_run CLI were removed. All code uses FlexNpuSimConfig.
 */

#pragma once

#include "common/types.h"               // PeConfig, PeType, Dataflow
#include "systemc/bus/capabilities.h"  // ModelingMode

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace flexnpu_sim::config {

// ============================================================================
// Hardware — system configure (no accelerator-specific subunit names)
// ============================================================================
//
// Schema is vendor-agnostic. `compute` describes the compute array as a
// generic unit (systolic / adder-tree / spatial) with ops_per_pass, parallel
// passes, output lanes. Subunit-specific overhead (NVDLA CACC/SDP/PDP, TPU
// activation stage, etc.) is modeled by writing each subunit as its own
// function record in the network CSV; the controller chains them (latency_model_chain.h).

struct ComputeCfg {
    std::string unit_type           = "systolic";  ///< systolic | adder_tree | spatial
    uint32_t    ops_per_pass        = 16;
    uint32_t    parallel_passes     = 1;
    uint32_t    num_output_lanes    = 16;
    uint32_t    element_size_bytes  = 2;
    uint32_t    local_psum_buffer_kb = 16;
    // Tile-loading strategy: prefetch_in scaling. 1.0 = preload (full
    // first-pass data before compute), <1.0 = streaming (partial data triggers
    // compute). 0 = unset -> treated as 1.0 (preload). Shipped hw specs set
    // this explicitly (model/hw/npu/*.json).
    double      threshold_ratio      = 0.0;
    // Zero-skipping sparsity preprocessing gate: "on"/"off"; "auto" (legacy
    // default) behaves as "off". Shipped hw specs set this explicitly.
    std::string zero_skipping        = "auto";
    // How activation sparsity (preprocess_param0 = skip_permille) is modeled —
    // three fidelity levels for the same physical zero-skipping (NullHop):
    //   "remove"   = skipped operands removed from F^I_LM, F^W_LM and compute
    //                F^O (the ideal benefit: fewer fetches, fewer MAC passes);
    //   "lm"       = "remove" PLUS a sparsity-handling stage in the packet
    //                chain (zero-detect/compaction unit) that scans the DENSE
    //                stream — its pipeline latency overlaps the MAC (chain), so
    //                net speedup is less than ideal;
    //   "overhead" = "remove" PLUS a lump latency per layer (the mechanism
    //                cost as a constant, pessimistic bound);
    //   "off"      = legacy: only input-fetch bytes scale (memory only).
    std::string sparsity_model       = "off";
    // Sparsity-mechanism knobs (used by "lm"/"overhead"): detect width
    // (dense operands scanned per cycle) and the lump overhead cycles.
    uint32_t    sparsity_detect_width = 0;   // 0 = ops_per_pass·num_output_lanes
    uint32_t    sparsity_overhead_cyc = 0;
    // Issue-enable structure (enable-granularity knob). "lockstep" = all
    // accumulators enable together (NVDLA CACC): starvation stalls emissions
    // at full width. "independent" = per-accumulator enable: starvation
    // narrows the emission width down to n_min. "auto" = infer from the
    // derived extrema (lockstep iff n_min == n_max) — the k/g/s derivation
    // cannot express lockstep (it yields n_min = g), so lockstep hardware
    // should set this explicitly.
    std::string enable_gran   = "auto";
    // Deadlock-watchdog: bail the compute loop after this many consecutive
    // no-progress cycles. A safety backstop (default 10000), now config-driven
    // so no behaviour-gating constant remains hardcoded in the controller.
    uint32_t    max_compute_stall_cycles = 10000;
};

struct GlobalBufferCfg {
    uint32_t    capacity_kb              = 512;
    std::string partition_mode           = "fixed";     ///< fixed (default) | shared | flexible
    uint32_t    input_partition_kb       = 0;           ///< 0 = uncapped.
    uint32_t    weight_partition_kb      = 0;
    uint32_t    output_partition_kb      = 0;
    uint32_t    read_port_bytes_per_cycle  = 0;         ///< 0 = unlimited.
    uint32_t    write_port_bytes_per_cycle = 0;
    bool        backpressure_enabled     = true;
    bool        weight_reuse_across_tiles = true;
    // Opt-in spill modeling (tile executor only). When true, the runtime derives
    // the output-tile size from this GB capacity instead of the compiler-baked
    // tile, so a too-small buffer forces spatial re-tiling (weight/input refetch)
    // and, when partial outputs cannot stay resident, psum round-trips to DRAM.
    // Default false → tile executor uses the descriptor tile (byte-identical to
    // the RTL-validated / compiler path).
    bool        model_spill              = false;
    std::string double_buffer            = "none";      ///< none | ping_pong | group
    // Partial-output (psum) placement so it is not double-counted against the
    // GB pool: gb (Nullhop sep SRAM) | accumulator (NVDLA CACC) | pe_local (MIDAP OS).
    std::string output_location          = "gb";
    // Effective capacity = physical × ratio. >1 for sparse compressed feature
    // maps (Nullhop). 1.0 = no compression.
    double      input_compression_ratio  = 1.0;
    double      weight_compression_ratio = 1.0;
    double      output_compression_ratio = 1.0;
    // GB access model: "" = scratchpad (default). "cache" enables the L2
    // cache implementation (hit/miss tracked per l2_cache_line_bytes line).
    std::string l2_mode                  = "";
    uint32_t    l2_cache_line_bytes      = 0;   ///< 0 = default 64
    // Layer fusion (keeps the previous layer's output resident in the GB so
    // the next layer reads it on-chip instead of a DRAM round-trip; Alwani et
    // al., Fused-Layer CNN Accelerators, MICRO 2016). "auto" = fuse when the
    // output fits the GB; "none" = always spill to DRAM.
    std::string layer_fusion             = "none";
    // Weight residency at the GB level (WMEM-like), stated explicitly rather
    // than inferred from the compute dataflow. MIDAP is input-stationary at
    // the feature level yet keeps weights resident (FILTER_LOAD=ONE), a
    // combination the dataflow tag alone cannot express. "dataflow" (default)
    // = derive from the dataflow (WS/OS resident, IS streamed); "resident" =
    // weight loaded once and reused across tiles (demoted to per-tile stream
    // only when it exceeds the weight partition); "streamed" = per-tile fetch.
    std::string weight_gb_stationary     = "dataflow";
};

struct AccumulatorCfg {
    uint32_t    capacity_kb              = 0;
    uint32_t    port_bytes_per_cycle     = 0;
};

// PE-internal operand buffers fed from GB. Capacity bounds how much of
// (F^I, F^W) the latency model can hold at once, which forces backpressure
// on DMA when compute consumption is slower than GB delivery (and stall on
// compute when buffers are empty during tile transitions).
struct PeBufferCfg {
    uint32_t    input_buffer_capacity_kb  = 0;   ///< 0 = unlimited.
    uint32_t    weight_buffer_capacity_kb = 0;   ///< 0 = unlimited.
};

struct BuffersCfg {
    GlobalBufferCfg global{};
    AccumulatorCfg  accumulator{};
    // Default PE-internal buffer caps (all latency-model units).
    PeBufferCfg     pe_buffers{};
    // Per-unit overrides keyed by function type ("conv2d", "activation",
    // ...). A latency model corresponds to one function descriptor — the
    // RTL units they abstract may have DIFFERENT internal buffer sizes,
    // so one uniform value cannot describe them (user review, 2026-07-10).
    std::map<std::string, PeBufferCfg> pe_buffers_per_function{};
};

struct DmaCfg {
    // profile expands into the fields below (resolve_profile); individual keys
    // then override per field. "generic" = the historical defaults here.
    std::string profile          = "generic";
    uint32_t    read_channels    = 1;
    uint32_t    write_channels   = 1;
    uint32_t    buffer_size_kb   = 16;    ///< Internal FIFO depth (→ outstanding window).
    uint32_t    max_burst_length = 8;
    uint32_t    max_issuing_reads  = 0;   ///< AXI read issuing capability (0 = derive).
    uint32_t    max_issuing_writes = 0;
};

struct ComputeInterconnectCfg {
    std::string topology             = "direct";  ///< direct | mesh | tree
    bool        multicast            = false;
    uint32_t    bandwidth_bytes_per_cycle = 0;    ///< 0 = unlimited.
};

struct HwConfig {
    std::string             preset_file;   ///< Optional: external HW JSON path.
    std::string             preset_name;   ///< e.g. "nvdla-large".
    double                  clock_cycle_ns = 10.0;
    std::string             dataflow       = "WS";  ///< WS | IS | OS

    ComputeCfg              compute{};
    BuffersCfg              buffers{};
    DmaCfg                  dma{};
    ComputeInterconnectCfg  compute_interconnect{};
};

// ============================================================================
// Network workload (CSV preferred; JSON listing fallback)
// ============================================================================

struct NetworkConfig {
    std::string csv;             ///< layer descriptor CSV (preferred).
    std::string json;            ///< Optional alternative: layer list as JSON.
    uint32_t    input_h          = 224;
    uint32_t    input_w          = 224;
    uint32_t    input_channels   = 3;
    std::string scenario_label;  ///< For logging.
};

// ============================================================================
// AXI / Transport (R5)
//
// Multi-outstanding + protocol switch + modeling mode fields.
// ============================================================================

struct AxiConfig {
    std::string protocol            = "AXI4";   ///< "AXI3" or "AXI4".
    uint32_t    data_width_bits     = 64;
    uint32_t    addr_width_bits     = 32;
    uint32_t    id_width_bits       = 4;
    uint32_t    max_burst_length    = 256;      ///< Beats — AXI3=16, AXI4=256.
    uint32_t    max_outstanding_reads  = 8;
    uint32_t    max_outstanding_writes = 4;
    transport::ModelingMode modeling_mode = transport::ModelingMode::Full;

    // Bus arbitration (R4, picked up by topology if not overridden there).
    std::string default_arbit_policy = "RoundRobin";
};

// ============================================================================
// Topology (R4b) — externalized N×M crossbar wiring
// ============================================================================

struct MasterCfg {
    std::string name;
    uint32_t    port_bits             = 64;
    uint32_t    max_outstanding_reads  = 8;
    uint32_t    max_outstanding_writes = 0;
    uint8_t     qos          = 0;
    uint8_t     weight       = 1;
    uint8_t     priority     = 0;
    uint8_t     default_qos  = 0;
};

struct SlaveCfg {
    std::string name;
    uint32_t    port_bits             = 128;
    uint64_t    addr_base             = 0;
    uint64_t    addr_size             = 0;
    uint32_t    base_read_latency_cyc  = 0;
    uint32_t    base_write_latency_cyc = 0;
};

struct LinkCfg {
    std::string master;
    std::string slave;
    uint32_t    bridge_extra_cyc      = 0;
    std::string width_converter;       ///< "64to128", etc., or empty.
};

struct InterconnectCfg {
    std::string type                       = "axi_crossbar";
    std::string arbit_policy               = "RoundRobin";  ///< 5 enum names.
    uint32_t    extra_latency_per_hop_cyc  = 0;
    bool        preemption                 = false;
};

struct TopologyConfig {
    std::string                topology_file;  ///< Optional: $include path.
    std::vector<MasterCfg>     masters;
    std::vector<SlaveCfg>      slaves;
    std::vector<LinkCfg>       links;
    InterconnectCfg            interconnect{};
};

// ============================================================================
// DRAM — the external memory. `tier` selects fidelity: the internal tiers
// (ideal/bandwidth/bank) take their params from this JSON; `cycle` hands an ini
// to DRAMSim3. The DRAM model owns only memory timing (tiling is the controller's).
// ============================================================================
struct DramConfig {
    std::string tier = "cycle";        ///< ideal | bandwidth | bank | cycle

    // ideal / bandwidth / bank — base access latency (NPU cycles). 0 = none.
    uint32_t read_latency_cyc  = 100;
    uint32_t write_latency_cyc = 30;

    // bandwidth — shared-bus bandwidth (bytes / NPU cycle; 0 = unlimited) and
    // whether writes consume it (MIDAP INCLUDE_DRAM_WRITE=False -> false).
    uint32_t bandwidth_bytes_per_cycle = 0;
    bool     include_writes            = true;

    // bank — row-buffer (line) locality.
    uint32_t banks        = 8;
    uint32_t line_bytes   = 8192;
    uint32_t row_hit_cyc  = 20;
    uint32_t row_miss_cyc = 45;

    // cycle — DRAMSim3.
    std::string ini;          ///< DRAMSim3 ini path
    std::string output_dir;   ///< optional DRAMSim3 stats dir

    // Controller roofline path (kept until the controller task moves the
    // whole-layer memory term onto the AXI/bandwidth-tier path). "sim" charges
    // memory through the DMA/AXI transactions; "peak_bw" uses the analytical
    // roofline (read_latency_cyc + bytes / bandwidth_bytes_per_cycle, honoring
    // include_writes).
    std::string tiled_dma_timing = "sim";
};

// ============================================================================
// Functional path (R10)
// ============================================================================

struct FunctionalConfig {
    bool        enabled        = false;
    std::string ofm_dir        = "build/ofm_cache";
    uint32_t    seed           = 42;
    bool        bin_dump       = false;   ///< Dump per-layer input/weight/output .bin files
};

// ============================================================================
// Sim infrastructure
// ============================================================================

struct SimConfig {
    std::string output_dir         = "out/run";
    std::string timeline_csv;
    std::string output_csv;
    bool        timeline_enabled   = false;
    bool        profiling_enabled  = true;
};

// ============================================================================
// Compiler — how the per-layer descriptor (F^I/F^W/F^O, l, n, threshold) is
// obtained. `type=manual` uses the descriptor given directly in the DNN image
// (Mode 1); `type=compile` derives it from hw.npu_mapping + the network
// (Mode 2), and `strategy` selects the tiling/scheduling policy — see
// compiler::TilingStrategy (tile_search.h) and docs/research/. Strings (not
// enums) here keep this section decoupled from the compiler layer; the consumer
// resolves them via compiler::tiling_strategy_from_string.
// ============================================================================

struct CompilerCfg {
    std::string type     = "compile";   ///< manual | compile
    std::string strategy = "dataflow";  ///< compile only: dataflow | min_tiles | min_traffic | min_latency
};

// ============================================================================
// Logging
// ============================================================================

struct LogConfig {
    std::string default_level = "info";
    /// Per-channel overrides — channel name → level string.
    std::map<std::string, std::string> channels;
};

// ============================================================================
// DSE sweep (optional)
// ============================================================================

struct DseSweepConfig {
    std::string base_config;                    ///< Base file the sweep extends.
    std::string expand        = "cartesian";    ///< cartesian | zip
    std::string output_dir    = "out/sweep";
    uint32_t    parallelism   = 1;
    /// Sweep axes: dotted-path key → list of values (any JSON scalar).
    /// Stored as raw JSON strings; expanded by scripts/dse_sweep.py.
    std::map<std::string, std::string> axes;
};

// ============================================================================
// System address map (MMIO) — where components sit in the address space.
// Defaults match the built-in memory_map constants (types.h). The NPU register
// map (offsets within npu_base) is a fixed HW spec (npu_reg in types.h).
// ============================================================================

// System memory map — the loader/linker's placement regions and the MMIO
// aperture. Bases and sizes are hardware facts; defaults mirror memory_map::
// (types.h). The loader places tensors as image-relative offsets resolved
// against dram_base, and asserts the placed footprint against dram_size (a
// linker region-overflow check); the MMIO aperture (npu_base) holds no data.
struct AddressMapCfg {
    uint32_t npu_base       = 0x40000000;  ///< NPU control-register base (MMIO).
    uint32_t npu_size       = 0x00010000;  ///< MMIO aperture size (64 KiB).
    uint32_t sram_base      = 0x20000000;  ///< on-chip SRAM/GB base.
    uint32_t sram_size      = 0x00100000;  ///< on-chip SRAM/GB size (1 MiB).
    uint32_t dram_base      = 0x80000000;  ///< main DRAM base.
    uint64_t dram_size      = 0x80000000;  ///< main DRAM size (2 GiB).
    uint32_t dram_ctrl_base = 0x50000000;  ///< DRAM controller base.
};

// ============================================================================
// Processor (host CPU) — latency model for the NPU-driver firmware.
// The firmware runs natively on the SystemC host; these costs turn its
// instruction mix (per-class counts from disassembly) and memory accesses
// into cycles. See src/systemc/processor/processor.h (software_delay/mem_access).
// ============================================================================

struct ProcessorCfg {
    // Per-instruction-class cost (cycles) for software_delay(jmp,mul,div,gen).
    uint32_t cost_jump    = 5;   ///< branch/jump
    uint32_t cost_mul     = 4;   ///< integer multiply
    uint32_t cost_div     = 10;  ///< integer divide
    uint32_t cost_generic = 1;   ///< everything else (ALU/mov/etc.)
    // Cache model for mem_access(): a hit/miss-ratio penalty.
    uint32_t cache_hit_latency  = 1;    ///< cycles on an L1 hit
    uint32_t cache_miss_latency = 100;  ///< cycles on a miss (DRAM round-trip)
    double   cache_miss_ratio   = 1.0;  ///< fraction of accesses that miss
                                        ///< (device MMIO registers: always 1.0)
};

// ============================================================================
// Root
// ============================================================================

struct FlexNpuSimConfig {
    HwConfig         hw{};
    NetworkConfig    network{};
    AxiConfig        axi{};
    TopologyConfig   topology{};
    DramConfig       dram{};
    AddressMapCfg    address_map{};
    ProcessorCfg     processor{};
    FunctionalConfig functional{};
    SimConfig        sim{};
    CompilerCfg      compiler{};
    LogConfig        logging{};
    DseSweepConfig   dse{};

    /// Source path the config was loaded from (for diagnostics).
    std::string      source_path;
};

// ============================================================================
// Config -> core-type bridges. The hw JSON is the single source of hardware
// truth (the C++ preset factories are gone); these map its compute block onto
// the core types consumed by the compiler and the latency model.
// ============================================================================

inline PeConfig pe_config_from(const ComputeCfg& c) {
    PeConfig pe;
    pe.pe_type               = (c.unit_type == "adder_tree") ? PeType::AdderTree
                                                             : PeType::Systolic;
    pe.ops_per_pass          = c.ops_per_pass ? c.ops_per_pass : 1;
    pe.parallel_passes       = c.parallel_passes ? c.parallel_passes : 1;
    pe.num_output_lanes      = c.num_output_lanes ? c.num_output_lanes : 1;
    pe.partial_sum_buffer_kb = c.local_psum_buffer_kb;
    pe.element_size_bytes    = c.element_size_bytes;
    return pe;
}

inline Dataflow dataflow_from(const std::string& s) {
    if (s == "IS") return Dataflow::IS;
    if (s == "OS") return Dataflow::OS;
    return Dataflow::WS;
}

/// threshold_ratio with the unset(0) -> preload(1.0) rule applied.
inline double effective_threshold_ratio(const ComputeCfg& c) {
    return c.threshold_ratio > 0.0 ? c.threshold_ratio : 1.0;
}

}  // namespace flexnpu_sim::config
