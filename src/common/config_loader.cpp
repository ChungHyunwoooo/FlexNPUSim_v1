/**
 * @file config_loader.cpp
 * @brief JSONC → FlexNpuSimConfig parser (R0c).
 *
 * Each section parser is small + defensive. Missing optional fields fall
 * back to struct defaults defined in flexnpu_config.h. Required fields
 * (currently: network.csv OR network.json) throw with a clear message.
 *
 * Schema reference: see model/hw/npu/nvdla-large.json (annotated example TBD).
 */

#include "common/config_loader.h"
#include "common/json.h"
#include "common/debug_log.h"
#include "systemc/dma/model/dma_profile.h"  // DmaParams, resolve_profile

#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace flexnpu_sim::config {

namespace {

using flexnpu_sim::json::Json;

// ---------------------------------------------------------------------------
// Small generic helpers.
// ---------------------------------------------------------------------------

template <typename T>
T get_or(const Json& parent, const char* key, T fallback) {
    if (!parent.is_object() || !parent.contains(key)) return fallback;
    try { return parent.at(key).get<T>(); }
    catch (const Json::type_error&) { return fallback; }
}

const Json& sub(const Json& parent, const char* key) {
    static const Json empty = Json::object();
    if (parent.is_object() && parent.contains(key)) {
        return parent.at(key);
    }
    return empty;
}

/// Typo guard: a key the parser does not recognize is reported instead of
/// silently vanishing (an unknown key means the user believes a knob exists
/// that the tool never reads).
void warn_unknown_keys(const Json& j,
                       std::initializer_list<const char*> allowed,
                       const char* block) {
    if (!j.is_object()) return;
    for (auto it = j.begin(); it != j.end(); ++it) {
        bool known = false;
        for (const char* k : allowed)
            if (it.key() == k) { known = true; break; }
        if (!known) {
            // The config file is loaded more than once per run (early preset
            // resolution + section extraction); report each typo once.
            static std::set<std::string> reported;
            std::string msg = std::string(block) + "." + it.key();
            if (reported.insert(msg).second)
                std::cerr << "[config] warning: unknown key '" << msg
                          << "' — ignored (typo?)\n";
        }
    }
}

void parse_string_map(const Json& parent, const char* key,
                      std::map<std::string, std::string>& out) {
    if (!parent.is_object() || !parent.contains(key)) return;
    const Json& m = parent.at(key);
    if (!m.is_object()) return;
    for (auto it = m.begin(); it != m.end(); ++it) {
        out.emplace(it.key(), it.value().is_string()
                                  ? it.value().get<std::string>()
                                  : it.value().dump());
    }
}

// ---------------------------------------------------------------------------
// Section parsers
// ---------------------------------------------------------------------------

ComputeCfg parse_compute(const Json& j) {
    warn_unknown_keys(j, {"unit_type","ops_per_pass","parallel_passes","num_output_lanes","element_size_bytes","local_psum_buffer_kb","max_compute_stall_cycles","threshold_ratio","zero_skipping","enable_granularity","sparsity_model","sparsity_detect_width","sparsity_overhead_cyc"}, "hw.compute");
    ComputeCfg c;
    c.unit_type             = get_or<std::string>(j, "unit_type",             c.unit_type);
    c.ops_per_pass          = get_or<uint32_t>   (j, "ops_per_pass",          c.ops_per_pass);
    c.parallel_passes       = get_or<uint32_t>   (j, "parallel_passes",       c.parallel_passes);
    c.num_output_lanes      = get_or<uint32_t>   (j, "num_output_lanes",      c.num_output_lanes);
    c.element_size_bytes    = get_or<uint32_t>   (j, "element_size_bytes",    c.element_size_bytes);
    c.local_psum_buffer_kb  = get_or<uint32_t>   (j, "local_psum_buffer_kb",  c.local_psum_buffer_kb);
    c.max_compute_stall_cycles = get_or<uint32_t>(j, "max_compute_stall_cycles", c.max_compute_stall_cycles);
    c.threshold_ratio       = get_or<double>     (j, "threshold_ratio",       c.threshold_ratio);
    c.zero_skipping         = get_or<std::string>(j, "zero_skipping",         c.zero_skipping);
    c.enable_gran    = get_or<std::string>(j, "enable_granularity",    c.enable_gran);
    c.sparsity_model        = get_or<std::string>(j, "sparsity_model",        c.sparsity_model);
    c.sparsity_detect_width = get_or<uint32_t>   (j, "sparsity_detect_width", c.sparsity_detect_width);
    c.sparsity_overhead_cyc = get_or<uint32_t>   (j, "sparsity_overhead_cyc", c.sparsity_overhead_cyc);
    return c;
}

GlobalBufferCfg parse_global_buffer(const Json& j) {
    warn_unknown_keys(j, {"capacity_kb","partition_mode","input_partition_kb","weight_partition_kb","output_partition_kb","read_port_bytes_per_cycle","write_port_bytes_per_cycle","backpressure_enabled","weight_reuse_across_tiles","model_spill","double_buffer","output_location","input_compression_ratio","weight_compression_ratio","output_compression_ratio","l2_mode","l2_cache_line_bytes","layer_fusion","inter_layer_retention","weight_gb_stationary"}, "hw.global_buffer");
    GlobalBufferCfg c;
    c.capacity_kb                = get_or<uint32_t>   (j, "capacity_kb",                c.capacity_kb);
    c.partition_mode             = get_or<std::string>(j, "partition_mode",             c.partition_mode);
    c.input_partition_kb         = get_or<uint32_t>   (j, "input_partition_kb",         c.input_partition_kb);
    c.weight_partition_kb        = get_or<uint32_t>   (j, "weight_partition_kb",        c.weight_partition_kb);
    c.output_partition_kb        = get_or<uint32_t>   (j, "output_partition_kb",        c.output_partition_kb);
    c.read_port_bytes_per_cycle  = get_or<uint32_t>   (j, "read_port_bytes_per_cycle",  c.read_port_bytes_per_cycle);
    c.write_port_bytes_per_cycle = get_or<uint32_t>   (j, "write_port_bytes_per_cycle", c.write_port_bytes_per_cycle);
    c.backpressure_enabled       = get_or<bool>       (j, "backpressure_enabled",       c.backpressure_enabled);
    c.weight_reuse_across_tiles  = get_or<bool>       (j, "weight_reuse_across_tiles",  c.weight_reuse_across_tiles);
    c.model_spill                = get_or<bool>       (j, "model_spill",                c.model_spill);
    c.double_buffer              = get_or<std::string>(j, "double_buffer",              c.double_buffer);
    c.output_location            = get_or<std::string>(j, "output_location",            c.output_location);
    c.input_compression_ratio    = get_or<double>     (j, "input_compression_ratio",    c.input_compression_ratio);
    c.weight_compression_ratio   = get_or<double>     (j, "weight_compression_ratio",   c.weight_compression_ratio);
    c.output_compression_ratio   = get_or<double>     (j, "output_compression_ratio",   c.output_compression_ratio);
    c.l2_mode                    = get_or<std::string>(j, "l2_mode",                    c.l2_mode);
    c.l2_cache_line_bytes        = get_or<uint32_t>   (j, "l2_cache_line_bytes",        c.l2_cache_line_bytes);
    // "inter_layer_retention" is the old key for this field; accept it as a
    // backward-compatible alias, then let the new "layer_fusion" key win when
    // both are present (new absent -> the old-key value is kept).
    c.layer_fusion               = get_or<std::string>(j, "inter_layer_retention",      c.layer_fusion);
    c.layer_fusion               = get_or<std::string>(j, "layer_fusion",               c.layer_fusion);
    c.weight_gb_stationary       = get_or<std::string>(j, "weight_gb_stationary",       c.weight_gb_stationary);
    return c;
}

// buffer.{input,weight,output} capacity is given as {capacity, unit} where
// unit is "kb" (default) or "operands". Normalized to KB (the struct's unit);
// 0 = unlimited (unit-agnostic). operands→kb via element size (ceil) — sub-KB
// operand precision is not preserved (no shipped spec needs it).
uint32_t buffer_cap_kb(const Json& b, uint32_t elem) {
    if (!b.is_object()) return 0;
    const uint32_t cap = get_or<uint32_t>(b, "capacity", 0u);
    if (cap == 0) return 0;
    const std::string unit = get_or<std::string>(b, "unit", std::string("kb"));
    if (unit == "operands") {
        const uint32_t e = elem ? elem : 1;
        return (cap * e + 1023) / 1024;
    }
    return cap;  // kb
}

// New-schema `buffer` block: input/weight → PE-internal operand buffers,
// output → accumulator. `per_function` overrides inherit unset fields from
// the defaults (one latency model ↔ one function descriptor).
void parse_buffer_block(const Json& j, BuffersCfg& b, uint32_t elem) {
    b.pe_buffers.input_buffer_capacity_kb  = buffer_cap_kb(sub(j, "input"),  elem);
    b.pe_buffers.weight_buffer_capacity_kb = buffer_cap_kb(sub(j, "weight"), elem);
    b.accumulator.capacity_kb              = buffer_cap_kb(sub(j, "output"), elem);
    warn_unknown_keys(j, {"input","weight","output","per_function"}, "hw.buffer");

    const Json& pf = sub(j, "per_function");
    if (!pf.is_object()) return;
    for (auto it = pf.begin(); it != pf.end(); ++it) {
        const Json& fn = it.value();
        if (!fn.is_object()) continue;
        PeBufferCfg p = b.pe_buffers;  // inherit defaults; only present keys override
        if (fn.contains("input"))
            p.input_buffer_capacity_kb  = buffer_cap_kb(fn.at("input"),  elem);
        if (fn.contains("weight"))
            p.weight_buffer_capacity_kb = buffer_cap_kb(fn.at("weight"), elem);
        b.pe_buffers_per_function[it.key()] = p;
    }
}

DmaCfg parse_dma(const Json& j) {
    warn_unknown_keys(j, {"profile","read_channels","write_channels","buffer_size_kb",
                          "max_burst_length","max_issuing_reads","max_issuing_writes"},
                      "hw.dma");
    DmaCfg c;
    // profile sets the per-field defaults; explicit JSON keys override each one.
    c.profile = get_or<std::string>(j, "profile", c.profile);
    const DmaParams p = resolve_profile(c.profile);
    c.read_channels      = get_or<uint32_t>(j, "read_channels",      p.read_channels);
    c.write_channels     = get_or<uint32_t>(j, "write_channels",     p.write_channels);
    c.buffer_size_kb     = get_or<uint32_t>(j, "buffer_size_kb",     p.fifo_bytes / 1024u);
    c.max_burst_length   = get_or<uint32_t>(j, "max_burst_length",   p.max_burst_beats);
    c.max_issuing_reads  = get_or<uint32_t>(j, "max_issuing_reads",  p.max_issuing_reads);
    c.max_issuing_writes = get_or<uint32_t>(j, "max_issuing_writes", p.max_issuing_writes);
    return c;
}

ComputeInterconnectCfg parse_compute_interconnect(const Json& j) {
    ComputeInterconnectCfg c;
    c.topology                 = get_or<std::string>(j, "topology",                 c.topology);
    c.multicast                = get_or<bool>       (j, "multicast",                c.multicast);
    c.bandwidth_bytes_per_cycle = get_or<uint32_t>  (j, "bandwidth_bytes_per_cycle", c.bandwidth_bytes_per_cycle);
    return c;
}

HwConfig parse_hw(const Json& j) {
    warn_unknown_keys(j, {"preset_file","preset_name","clock_cycle_ns","dataflow","npu_mapping","buffer","fetch_bitwidth","write_bitwidth","global_buffer","dma","compute_interconnect"}, "hw");
    HwConfig c;
    c.preset_file   = get_or<std::string>(j, "preset_file",    c.preset_file);
    c.preset_name   = get_or<std::string>(j, "preset_name",    c.preset_name);
    c.clock_cycle_ns = get_or<double>    (j, "clock_cycle_ns", c.clock_cycle_ns);
    c.dataflow      = get_or<std::string>(j, "dataflow",       c.dataflow);

    // Mode 2 structural source (physical HW → compiler-derived descriptor).
    c.compute        = parse_compute      (sub(j, "npu_mapping"));
    // GB/L2 pool.
    c.buffers.global = parse_global_buffer(sub(j, "global_buffer"));
    // LM-internal operand buffers (input/weight/output + per-function).
    parse_buffer_block(sub(j, "buffer"), c.buffers, c.compute.element_size_bytes);

    // GB↔PE port width (bits/cycle → bytes/cycle; 0 = unlimited).
    const uint32_t fetch_bits = get_or<uint32_t>(j, "fetch_bitwidth", 0u);
    const uint32_t write_bits = get_or<uint32_t>(j, "write_bitwidth", 0u);
    if (fetch_bits > 0) c.buffers.global.read_port_bytes_per_cycle  = fetch_bits / 8;
    if (write_bits > 0) c.buffers.global.write_port_bytes_per_cycle = write_bits / 8;

    c.dma                  = parse_dma                  (sub(j, "dma"));
    c.compute_interconnect = parse_compute_interconnect (sub(j, "compute_interconnect"));
    return c;
}

NetworkConfig parse_network(const Json& j, bool require_source) {
    NetworkConfig c;
    c.csv             = get_or<std::string>(j, "csv",             c.csv);
    c.json            = get_or<std::string>(j, "json",            c.json);
    c.input_h         = get_or<uint32_t>   (j, "input_h",         c.input_h);
    c.input_w         = get_or<uint32_t>   (j, "input_w",         c.input_w);
    c.input_channels  = get_or<uint32_t>   (j, "input_channels",  c.input_channels);
    c.scenario_label  = get_or<std::string>(j, "scenario_label",  c.scenario_label);
    if (require_source && c.csv.empty() && c.json.empty()) {
        throw std::runtime_error(
            "network: at least one of {csv, json} must be specified");
    }
    return c;
}

AxiConfig parse_axi(const Json& j) {
    warn_unknown_keys(j, {"protocol","data_width_bits","addr_width_bits","id_width_bits","max_burst_length","max_outstanding_reads","max_outstanding_writes","modeling_mode","default_arbit_policy"}, "axi");
    AxiConfig c;
    c.protocol               = get_or<std::string>(j, "protocol",               c.protocol);
    c.data_width_bits        = get_or<uint32_t>   (j, "data_width_bits",        c.data_width_bits);
    c.addr_width_bits        = get_or<uint32_t>   (j, "addr_width_bits",        c.addr_width_bits);
    c.id_width_bits          = get_or<uint32_t>   (j, "id_width_bits",          c.id_width_bits);
    c.max_burst_length       = get_or<uint32_t>   (j, "max_burst_length",       c.max_burst_length);
    c.max_outstanding_reads  = get_or<uint32_t>   (j, "max_outstanding_reads",  c.max_outstanding_reads);
    c.max_outstanding_writes = get_or<uint32_t>   (j, "max_outstanding_writes", c.max_outstanding_writes);
    c.default_arbit_policy   = get_or<std::string>(j, "default_arbit_policy",   c.default_arbit_policy);

    const std::string mode = get_or<std::string>(j, "modeling_mode", std::string("Full"));
    c.modeling_mode = (mode == "LatencyOnly")
                          ? transport::ModelingMode::LatencyOnly
                          : transport::ModelingMode::Full;
    return c;
}

MasterCfg parse_master(const Json& j) {
    MasterCfg c;
    c.name                   = get_or<std::string>(j, "name",                   c.name);
    c.port_bits              = get_or<uint32_t>   (j, "port_bits",              c.port_bits);
    c.max_outstanding_reads  = get_or<uint32_t>   (j, "max_outstanding_reads",  c.max_outstanding_reads);
    c.max_outstanding_writes = get_or<uint32_t>   (j, "max_outstanding_writes", c.max_outstanding_writes);
    c.qos                    = get_or<int>        (j, "qos",                    c.qos);
    c.weight                 = get_or<int>        (j, "weight",                 c.weight);
    c.priority               = get_or<int>        (j, "priority",               c.priority);
    c.default_qos            = get_or<int>        (j, "default_qos",            c.default_qos);
    return c;
}

SlaveCfg parse_slave(const Json& j) {
    SlaveCfg c;
    c.name                   = get_or<std::string>(j, "name",                   c.name);
    c.port_bits              = get_or<uint32_t>   (j, "port_bits",              c.port_bits);
    c.addr_base              = get_or<uint64_t>   (j, "addr_base",              c.addr_base);
    c.addr_size              = get_or<uint64_t>   (j, "addr_size",              c.addr_size);
    c.base_read_latency_cyc  = get_or<uint32_t>   (j, "base_read_latency_cyc",  c.base_read_latency_cyc);
    c.base_write_latency_cyc = get_or<uint32_t>   (j, "base_write_latency_cyc", c.base_write_latency_cyc);
    return c;
}

LinkCfg parse_link(const Json& j) {
    LinkCfg c;
    c.master           = get_or<std::string>(j, "master",           c.master);
    c.slave            = get_or<std::string>(j, "slave",            c.slave);
    c.bridge_extra_cyc = get_or<uint32_t>   (j, "bridge_extra_cyc", c.bridge_extra_cyc);
    c.width_converter  = get_or<std::string>(j, "width_converter",  c.width_converter);
    return c;
}

InterconnectCfg parse_interconnect(const Json& j) {
    InterconnectCfg c;
    c.type                      = get_or<std::string>(j, "type",                      c.type);
    c.arbit_policy              = get_or<std::string>(j, "arbit_policy",              c.arbit_policy);
    c.extra_latency_per_hop_cyc = get_or<uint32_t>   (j, "extra_latency_per_hop_cyc", c.extra_latency_per_hop_cyc);
    c.preemption                = get_or<bool>       (j, "preemption",                c.preemption);
    return c;
}

TopologyConfig parse_topology(const Json& j) {
    TopologyConfig c;
    c.topology_file = get_or<std::string>(j, "file", c.topology_file);

    if (j.contains("masters") && j.at("masters").is_array()) {
        for (const auto& m : j.at("masters")) c.masters.push_back(parse_master(m));
    }
    if (j.contains("slaves") && j.at("slaves").is_array()) {
        for (const auto& s : j.at("slaves")) c.slaves.push_back(parse_slave(s));
    }
    if (j.contains("links") && j.at("links").is_array()) {
        for (const auto& l : j.at("links")) c.links.push_back(parse_link(l));
    }
    c.interconnect = parse_interconnect(sub(j, "interconnect"));
    return c;
}

DramConfig parse_dram(const Json& j) {
    warn_unknown_keys(j, {"tier","read_latency_cyc","write_latency_cyc",
        "bandwidth_bytes_per_cycle","include_writes","banks","line_bytes",
        "row_hit_cyc","row_miss_cyc","ini","output_dir","tiled_dma_timing"}, "dram");
    DramConfig c;
    c.tier               = get_or<std::string>(j, "tier",               c.tier);
    c.read_latency_cyc   = get_or<uint32_t>   (j, "read_latency_cyc",   c.read_latency_cyc);
    c.write_latency_cyc  = get_or<uint32_t>   (j, "write_latency_cyc",  c.write_latency_cyc);
    c.bandwidth_bytes_per_cycle = get_or<uint32_t>(j, "bandwidth_bytes_per_cycle", c.bandwidth_bytes_per_cycle);
    c.include_writes     = get_or<bool>       (j, "include_writes",     c.include_writes);
    c.banks              = get_or<uint32_t>   (j, "banks",              c.banks);
    c.line_bytes         = get_or<uint32_t>   (j, "line_bytes",         c.line_bytes);
    c.row_hit_cyc        = get_or<uint32_t>   (j, "row_hit_cyc",        c.row_hit_cyc);
    c.row_miss_cyc       = get_or<uint32_t>   (j, "row_miss_cyc",       c.row_miss_cyc);
    c.ini                = get_or<std::string>(j, "ini",                c.ini);
    c.output_dir         = get_or<std::string>(j, "output_dir",         c.output_dir);
    c.tiled_dma_timing   = get_or<std::string>(j, "tiled_dma_timing",   c.tiled_dma_timing);
    return c;
}

FunctionalConfig parse_functional(const Json& j) {
    FunctionalConfig c;
    c.enabled  = get_or<bool>       (j, "enabled",  c.enabled);
    c.ofm_dir  = get_or<std::string>(j, "ofm_dir",  c.ofm_dir);
    c.seed     = get_or<uint32_t>   (j, "seed",     c.seed);
    c.bin_dump = get_or<bool>       (j, "bin_dump", c.bin_dump);
    return c;
}

SimConfig parse_sim(const Json& j) {
    SimConfig c;
    c.output_dir         = get_or<std::string>(j, "output_dir",         c.output_dir);
    c.timeline_csv       = get_or<std::string>(j, "timeline_csv",       c.timeline_csv);
    c.output_csv         = get_or<std::string>(j, "output_csv",         c.output_csv);
    c.timeline_enabled   = get_or<bool>       (j, "timeline_enabled",   c.timeline_enabled);
    c.profiling_enabled  = get_or<bool>       (j, "profiling_enabled",  c.profiling_enabled);
    return c;
}

CompilerCfg parse_compiler(const Json& j) {
    warn_unknown_keys(j, {"type","strategy"}, "compiler");
    CompilerCfg c;
    c.type     = get_or<std::string>(j, "type",     c.type);
    c.strategy = get_or<std::string>(j, "strategy", c.strategy);
    // Strategy is stored as a string and resolved by the compiler consumer
    // (compiler::tiling_strategy_from_string) so this section stays decoupled
    // from the compiler layer. Only the small type enum is checked here.
    if (c.type != "manual" && c.type != "compile") {
        std::cerr << "[config] warning: compiler.type '" << c.type
                  << "' unknown (expected manual|compile); using 'compile'\n";
        c.type = "compile";
    }
    return c;
}

LogConfig parse_logging(const Json& j) {
    LogConfig c;
    c.default_level = get_or<std::string>(j, "default", c.default_level);
    parse_string_map(j, "channels", c.channels);
    return c;
}

DseSweepConfig parse_dse(const Json& j) {
    DseSweepConfig c;
    c.base_config = get_or<std::string>(j, "base_config", c.base_config);
    c.expand      = get_or<std::string>(j, "expand",      c.expand);
    c.output_dir  = get_or<std::string>(j, "output_dir",  c.output_dir);
    c.parallelism = get_or<uint32_t>   (j, "parallelism", c.parallelism);
    if (j.contains("axes") && j.at("axes").is_object()) {
        for (auto it = j.at("axes").begin(); it != j.at("axes").end(); ++it) {
            // Store the value list as raw JSON text — sweep runner expands.
            c.axes.emplace(it.key(), it.value().dump());
        }
    }
    return c;
}

// ---------------------------------------------------------------------------
// Cross-section validation (top-level invariants)
// ---------------------------------------------------------------------------

void validate(const FlexNpuSimConfig& cfg) {
    // Topology address range overlap check.
    for (size_t i = 0; i < cfg.topology.slaves.size(); ++i) {
        const auto& a = cfg.topology.slaves[i];
        for (size_t j = i + 1; j < cfg.topology.slaves.size(); ++j) {
            const auto& b = cfg.topology.slaves[j];
            const uint64_t a_end = a.addr_base + a.addr_size;
            const uint64_t b_end = b.addr_base + b.addr_size;
            const bool overlap = !(a_end <= b.addr_base || b_end <= a.addr_base);
            if (overlap) {
                throw std::runtime_error(
                    "topology: address range overlap between slaves '" +
                    a.name + "' and '" + b.name + "'");
            }
        }
    }
    // Link references.
    for (const auto& l : cfg.topology.links) {
        bool master_ok = false, slave_ok = false;
        for (const auto& m : cfg.topology.masters) if (m.name == l.master) master_ok = true;
        for (const auto& s : cfg.topology.slaves)  if (s.name == l.slave)  slave_ok = true;
        if (!master_ok)
            throw std::runtime_error("topology: link references unknown master '" + l.master + "'");
        if (!slave_ok)
            throw std::runtime_error("topology: link references unknown slave '" + l.slave + "'");
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry
// ---------------------------------------------------------------------------

// Address fields accept a JSON number or a "0x..."/decimal string.
static uint32_t get_addr(const Json& j, const char* key, uint32_t def) {
    if (!j.contains(key)) return def;
    const auto& v = j.at(key);
    if (v.is_number_unsigned() || v.is_number_integer()) return v.get<uint32_t>();
    if (v.is_string()) {
        try { return static_cast<uint32_t>(std::stoull(v.get<std::string>(), nullptr, 0)); }
        catch (...) { return def; }
    }
    return def;
}

AddressMapCfg parse_address_map(const Json& j) {
    AddressMapCfg c;
    c.npu_base       = get_addr(j, "npu_base",       c.npu_base);
    c.npu_size       = get_or<uint32_t>(j, "npu_size", c.npu_size);
    c.sram_base      = get_addr(j, "sram_base",      c.sram_base);
    c.dram_base      = get_addr(j, "dram_base",      c.dram_base);
    c.dram_ctrl_base = get_addr(j, "dram_ctrl_base", c.dram_ctrl_base);
    // Sizes accept readable *_kb / *_mb keys; fall back to struct defaults.
    if (j.is_object() && j.contains("sram_size_kb"))
        c.sram_size = get_or<uint32_t>(j, "sram_size_kb", 0) * 1024u;
    if (j.is_object() && j.contains("dram_size_mb"))
        c.dram_size = static_cast<uint64_t>(get_or<uint32_t>(j, "dram_size_mb", 0)) * 1024u * 1024u;
    return c;
}

ProcessorCfg parse_processor(const Json& j) {
    warn_unknown_keys(j, {"cost_jump","cost_mul","cost_div","cost_generic",
                          "cache_hit_latency","cache_miss_latency","cache_miss_ratio"},
                      "processor");
    ProcessorCfg c;
    c.cost_jump          = get_or<uint32_t>(j, "cost_jump",          c.cost_jump);
    c.cost_mul           = get_or<uint32_t>(j, "cost_mul",           c.cost_mul);
    c.cost_div           = get_or<uint32_t>(j, "cost_div",           c.cost_div);
    c.cost_generic       = get_or<uint32_t>(j, "cost_generic",       c.cost_generic);
    c.cache_hit_latency  = get_or<uint32_t>(j, "cache_hit_latency",  c.cache_hit_latency);
    c.cache_miss_latency = get_or<uint32_t>(j, "cache_miss_latency", c.cache_miss_latency);
    c.cache_miss_ratio   = get_or<double>  (j, "cache_miss_ratio",   c.cache_miss_ratio);
    return c;
}

FlexNpuSimConfig load_flexnpu_config(const std::filesystem::path& path,
                                      bool require_network_source) {
    Json root = json::load_or_throw(path);

    FlexNpuSimConfig cfg;
    cfg.source_path = path.string();
    cfg.hw          = parse_hw         (sub(root, "hw"));
    cfg.network     = parse_network    (sub(root, "network"), require_network_source);
    cfg.axi         = parse_axi        (sub(root, "axi"));
    cfg.topology    = parse_topology   (sub(root, "topology"));
    cfg.dram        = parse_dram       (sub(root, "dram"));
    cfg.address_map = parse_address_map(sub(root, "address_map"));
    cfg.processor   = parse_processor  (sub(root, "processor"));
    cfg.functional  = parse_functional (sub(root, "functional"));
    cfg.sim         = parse_sim        (sub(root, "sim"));
    cfg.compiler    = parse_compiler   (sub(root, "compiler"));
    cfg.logging     = parse_logging    (sub(root, "logging"));
    cfg.dse         = parse_dse        (sub(root, "dse"));

    validate(cfg);

    FLEXNPU_LOG(Info, topology,
                "loaded %s: masters=%zu slaves=%zu links=%zu arbit=%s "
                "dram_tier=%s",
                cfg.source_path.c_str(),
                cfg.topology.masters.size(),
                cfg.topology.slaves.size(),
                cfg.topology.links.size(),
                cfg.topology.interconnect.arbit_policy.c_str(),
                cfg.dram.tier.c_str());
    return cfg;
}

}  // namespace flexnpu_sim::config
