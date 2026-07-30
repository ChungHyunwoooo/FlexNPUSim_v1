/**
 * @file flexnpusim_system.cpp
 * @brief DSE System Simulation: parameterized NPU + DRAM BankModel
 *
 * Full SystemC simulation with cycle-accurate DRAM (Tier 3: BankModel).
 * Accepts JSON + CSV inputs:
 *   - fixed HW/system config in JSON
 *   - layer/function descriptors in CSV
 * For zero-skipping NPUs (Nullhop), applies sparsity preprocessing to fetch_operand0.
 *
 * Usage:
 *   ./flexnpusim -hw_conf model/hw/npu/<design>.json \
 *                    -network model/sw/<class>/<design>/<network>.csv \
 *                    -o result.csv
 *
 * Output (CSV to stdout; the -o file additionally carries a memory_timing tail):
 *   npu,network,bus_bits,l2_kb,dram,resolution,dataflow,double_buf,threshold,
 *   cycles,macs,mem_rd,mem_wr,completed,fps,util_pct,dram_bw_gbps,bus_util_pct
 */

#include <systemc.h>
#include "system/flexnpusim_system.h"
#include "system/scenario_spec.h"
#include "system/performance_report.h"
#include "axi.h"
#include "systemc/npu/npu_controller.h"
#include "systemc/npu/npu.h"
#include "systemc/processor/processor.h"
#include "systemc/dma/dma_engine.h"
#include "common/types.h"
#include "compiler/frontend/loaders/network_csv.h"
#include "common/flexnpu_config.h"
#include "common/config_loader.h"
#include "compiler/dnn_image/dnn_image_format.h"
#include "compiler/dnn_image/dnn_image_packet.h"
#include "compiler/dnn_image/dnn_image_generator.h"
#include "compiler/dnn_image/dnn_image_loader.h"
#include "model/function/sparsity_preprocess.h"
#include "systemc/memory/wrapper/memory_axi_slave_v2.h"
#include "systemc/bus/wrapper/interconnect_axi_bus_v2.h"
#include "systemc/bus/axi/axi_signal_ports.h"
#include "systemc/bus/axi/axi_transport.h"
#include "systemc/bus/axi/bus/axi_arbiter.h"
#include "systemc/bus/axi/master/axi_master.h"
#include "systemc/bus/axi/spec/axi_spec.h"
#include "systemc/bus/capabilities.h"
#include "systemc/memory/dram/dram_timing.h"
#include "systemc/memory/dram/factory.h"
#ifdef HAVE_OPENCV
#include "model/function/ofm_file_generator.h"
#endif
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cctype>
#include <map>
#include <limits>
#include <filesystem>

using namespace flexnpu_sim;

// Data-bus AXI Spec — protocol/ID/addr widths fixed; data width follows the
// same compile-time knob used by the legacy DmaAxiConfig so each
// flexnpusim_system<W> binary carries a matching SpecT.
using DataSpecT = axi::Spec<axi::Protocol::AXI4, 4, 32,
                            FLEXNPUSIM_DMA_AXI_DATA_W, 0>;

// Compile-time master-count dispatcher for the data bus. R4 Bus requires
// NumM as a template parameter, so flexnpusim_system switches on
// total_masters and instantiates the matching specialization. The bus's
// lifetime is owned by an sc_module base pointer; no methods are called
// on it after binding.
template <unsigned NumM>
static std::unique_ptr<sc_core::sc_module> build_data_bus_n(
    sc_clock& clk, uint64_t base, uint64_t size,
    const std::vector<std::unique_ptr<AXI_SIGNALS<DataSpecT>>>& m_sigs,
    AXI_SIGNALS<DataSpecT>& s_sig,
    transport::axi::ArbitPolicy policy,
    const std::vector<transport::axi::MasterArbitCfg>& master_cfgs)
{
    using WrapperT = InterconnectAxiBusV2<DataSpecT, NumM, 1>;
    std::vector<transport::axi::MasterArbitCfg> mcfgs(NumM);
    for (unsigned i = 0; i < NumM && i < master_cfgs.size(); ++i)
        mcfgs[i] = master_cfgs[i];
    std::array<typename WrapperT::BusT::SlaveRange, 1> ranges{};
    ranges[0].base = base;
    ranges[0].size = size;
    auto bus = std::make_unique<WrapperT>(
        "data_bus", policy, mcfgs, ranges);
    bus->clk(clk);
    for (unsigned i = 0; i < NumM; ++i) bus->bind_master(i, *m_sigs[i]);
    bus->bind_slave(0, s_sig);
    return bus;
}

// ============================================================================
// CLI helpers
// ============================================================================

static bool split_design_network_from_csv(const std::string& path,
                                          std::string& design,
                                          std::string& network) {
    size_t p = path.find_last_of("/\\");
    std::string base = (p == std::string::npos) ? path : path.substr(p + 1);
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) base = base.substr(0, dot);
    size_t sep = base.find('_');
    if (sep == std::string::npos || sep == 0 || sep + 1 >= base.size()) return false;
    design = base.substr(0, sep);
    network = base.substr(sep + 1);
    return true;
}

static transport::axi::ArbitPolicy parse_bus_arbit(const std::string& s) {
    using transport::axi::ArbitPolicy;
    if (s == "WeightedRR"    || s == "weighted_rr")    return ArbitPolicy::WeightedRR;
    if (s == "FixedPriority" || s == "fixed_priority") return ArbitPolicy::FixedPriority;
    if (s == "PriorityRR"    || s == "priority_rr")    return ArbitPolicy::PriorityRR;
    if (s == "QoSAware"      || s == "qos_aware")      return ArbitPolicy::QoSAware;
    return ArbitPolicy::RoundRobin;  // default / "RoundRobin"
}

static std::string trim_copy(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) b++;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) e--;
    return s.substr(b, e - b);
}

static void ensure_parent_dir(const std::string& file_path) {
    if (file_path.empty()) return;
    std::error_code ec;
    std::filesystem::path p(file_path);
    if (!p.has_parent_path()) return;
    std::filesystem::create_directories(p.parent_path(), ec);
}

// ============================================================================
// sc_main
// ============================================================================

int flexnpu_sim::system::run_flexnpusim_system(int argc, char* argv[]) {
    auto print_usage = [&](const char* prog) {
        std::cerr << "Usage:\n"
                  << "  " << prog
                  << " -hw_conf <hw.json> -network <model/sw/<class>/<design>/<network>.csv> "
                  << "[-scenario <design:network>] [-model_root <models|model>] "
                  << "-o <result.csv> [-timeline <trace.csv>] "
                  << "[-report <report.txt>]\n";
    };

    std::string npu_name = "custom";
    std::string net_name = "custom-csv";
    std::string report_path;
    uint32_t bus_bits = 64;
    uint32_t l2_kb = 256;
    std::string dram_type = "DDR4-2400";
    uint32_t resolution = 32;
    int double_buf_arg = -1;  // -1 = config decides
    double sys_clock_ns = 10.0;
    std::string dnn_input_json;
    std::string layer_spec_csv;
    std::string output_csv_path;
    std::string timeline_csv_path;
    std::string scenario_name;
    std::string model_root_hint;
    bool functional_ofm_enabled = false;
    std::string functional_ofm_dir;
    std::string l2_mode = "scratchpad";
    uint32_t l2_cache_line_bytes = 64;

    if (argc <= 1) {
        print_usage(argv[0]);
        return 1;
    }
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "-hw_conf" && i + 1 < argc) {
            dnn_input_json = argv[++i];
        } else if (a == "-network" && i + 1 < argc) {
            layer_spec_csv = argv[++i];
        } else if (a == "-o" && i + 1 < argc) {
            output_csv_path = argv[++i];
        } else if (a == "-timeline" && i + 1 < argc) {
            timeline_csv_path = argv[++i];
        } else if (a == "-scenario" && i + 1 < argc) {
            scenario_name = argv[++i];
        } else if (a == "-model_root" && i + 1 < argc) {
            model_root_hint = argv[++i];
        } else if (a == "-report" && i + 1 < argc) {
            report_path = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete option: " << a << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!scenario_name.empty() && (dnn_input_json.empty() || layer_spec_csv.empty())) {
        std::string scenario_err;
        std::string scenario_npu;
        if (!resolve_scenario_to_paths(scenario_name, model_root_hint,
                                       dnn_input_json, layer_spec_csv,
                                       scenario_npu, scenario_err)) {
            std::cerr << "ERROR: " << scenario_err << "\n";
            return 1;
        }
        if (!scenario_npu.empty()) {
            npu_name = scenario_npu;
        }
    }

    if (dnn_input_json.empty() || layer_spec_csv.empty()) {
        std::cerr << "ERROR: -hw_conf/-network or -scenario are required.\n";
        print_usage(argv[0]);
        return 1;
    }

    // Derive <design_name>_<network_name>.csv tokens when possible.
    std::string design;
    std::string network_tag;
    if (split_design_network_from_csv(layer_spec_csv, design, network_tag)) {
        npu_name = design;
        net_name = network_tag;
    }

    if (sys_clock_ns <= 0.0) sys_clock_ns = 10.0;

    // Load the unified hw JSON once, early. Two things must happen before the
    // code below: (1) hw.preset_name must select the preset struct — it used
    // to be read only after get_preset, so JSON preset_name never took effect
    // (historical: JSON preset_name once lost to a hardcoded default); (2) compute
    // overrides (threshold_ratio, zero_skipping) must land before image
    // generation and sparsity preprocessing. CLI -npu keeps priority, same
    // rule as -scenario/design above. Legacy dnn-profile JSONs fail the flex
    // parse and keep defaults.
    bool flex_early_ok = false;
    flexnpu_sim::config::FlexNpuSimConfig flex_early;
    if (!dnn_input_json.empty()) {
        try {
            flex_early = flexnpu_sim::config::load_flexnpu_config(
                dnn_input_json, /*require_network_source=*/false);
            flex_early_ok = true;
        } catch (const std::exception&) {
            // Not a flex-schema JSON; compat handles it below.
        }
    }
    if (flex_early_ok && !flex_early.hw.preset_name.empty())
        npu_name = flex_early.hw.preset_name;
    // GB access-mode override (makes the existing cache implementation
    // reachable): hw.buffers.global.l2_mode = "cache" | "scratchpad".
    if (flex_early_ok && !flex_early.hw.buffers.global.l2_mode.empty())
        l2_mode = flex_early.hw.buffers.global.l2_mode;
    if (flex_early_ok && flex_early.hw.buffers.global.l2_cache_line_bytes > 0)
        l2_cache_line_bytes = flex_early.hw.buffers.global.l2_cache_line_bytes;
    // AXI data-bus width is a runtime knob (the signal is compiled at a fixed
    // maximum; beat accounting derives from this value). AXI4 defines the data
    // width as a power of two in [8, 1024] bits — no wider bus is meaningful.
    if (flex_early_ok && flex_early.axi.data_width_bits != 64) {
        const uint32_t w = flex_early.axi.data_width_bits;
        const bool pow2 = w >= 8 && (w & (w - 1)) == 0;
        if (!pow2 || w > 1024) {
            std::cerr << "[config] error: axi.data_width_bits=" << w
                      << " is not a valid AXI4 data width "
                         "(power of two in [8, 1024])\n";
            return 1;
        }
        if (w > FLEXNPUSIM_DMA_AXI_DATA_W) {
            std::cerr << "[config] error: axi.data_width_bits=" << w
                      << " exceeds the compiled maximum "
                      << FLEXNPUSIM_DMA_AXI_DATA_W
                      << "b; rebuild with -DFLEXNPUSIM_AXI_MAX_DATA_W=" << w
                      << " (<= 1024)\n";
            return 1;
        }
        bus_bits = w;
    }

    // Config honesty warnings (P0): fields the runtime silently ignores must
    // not pretend to work. One line each, stderr, non-fatal.
    if (flex_early_ok) {
        const auto& fx = flex_early;
        // axi.data_width_bits is a runtime knob now (wired into bus_bits
        // above); a width exceeding the compiled max already errored out.
        if (!fx.hw.buffers.global.weight_reuse_across_tiles)
            std::cerr << "[config] warning: weight_reuse_across_tiles=false is "
                         "not modeled — tile-to-tile weight retention follows "
                         "the dataflow (WS keeps weights resident)\n";
        if (fx.hw.buffers.accumulator.port_bytes_per_cycle != 0)
            std::cerr << "[config] warning: accumulator.port_bytes_per_cycle "
                         "is not modeled (accumulator capacity is)\n";
    }

    // The hw JSON is the single source of hardware truth: the C++ preset
    // factories are gone, so a valid FlexNPUSim config is mandatory.
    if (!flex_early_ok) {
        std::cerr << "ERROR: -hw_conf must be a valid FlexNPUSim config JSON: "
                  << dnn_input_json << "\n";
        return 1;
    }
    if (!flex_early.network.scenario_label.empty())
        net_name = flex_early.network.scenario_label;
    if (flex_early.hw.buffers.global.capacity_kb > 0)
        l2_kb = flex_early.hw.buffers.global.capacity_kb;
    if (!flex_early.dram.tier.empty())
        dram_type = flex_early.dram.tier;
    if (flex_early.hw.clock_cycle_ns > 0.0)
        sys_clock_ns = flex_early.hw.clock_cycle_ns;
    if (flex_early.hw.buffers.global.double_buffer != "none")
        double_buf_arg = 1;
    if (std::getenv("FLEXNPUSIM_VERBOSE"))
        std::cerr << "Loaded hw config: " << dnn_input_json << "\n";

    const PeConfig pe = flexnpu_sim::config::pe_config_from(flex_early.hw.compute);
    const Dataflow pe_dataflow = flexnpu_sim::config::dataflow_from(flex_early.hw.dataflow);
    const double threshold_ratio =
        flexnpu_sim::config::effective_threshold_ratio(flex_early.hw.compute);
    const bool zero_skipping = (flex_early.hw.compute.zero_skipping == "on");

    bool db_enabled = (double_buf_arg >= 0) ? (double_buf_arg != 0) : false;
    std::string df_str = (pe_dataflow == Dataflow::WS) ? "WS" :
                         (pe_dataflow == Dataflow::IS) ? "IS" : "OS";

    // Demo-clean banner: one config card. Diagnostic detail lines throughout
    // this function are demoted behind FLEXNPUSIM_VERBOSE=1 (not deleted).
    const bool verbose_log = (std::getenv("FLEXNPUSIM_VERBOSE") != nullptr);
    const char* RULE =
        "─────────────────────────────────────────────────────────────────";
    std::cerr << RULE << "\n"
              << " FlexNPUSim  ·  " << npu_name << "  ·  " << net_name << "\n"
              << RULE << "\n"
              << " NPU       dataflow=" << df_str
              << "   pe(k,g,lanes)=(" << pe.ops_per_pass << ","
              << pe.parallel_passes << "," << pe.num_output_lanes
              << ")   GB=" << l2_kb << "KB"
              << "   psum=" << pe.partial_sum_buffer_kb << "KB\n"
              << " Memory    AXI=" << bus_bits << "b   DRAM=" << dram_type << "\n"
              << " Network   " << (layer_spec_csv.empty() ? "-" : layer_spec_csv)
              << "\n";
    if (verbose_log)
        std::cerr << " Extra     db=" << db_enabled
                  << "   functional_ofm=" << (functional_ofm_enabled ? "on" : "off")
                  << "   thresh=" << threshold_ratio
                  << "   l2_mode=" << l2_mode
                  << "   l2_line=" << l2_cache_line_bytes
                  << "   input_gen=" << resolution << "x" << resolution
                  << "   dnn_profile=" << (dnn_input_json.empty() ? "-" : dnn_input_json)
                  << "   clk=" << sys_clock_ns << "ns\n";

    // ---- 1. Generate DNN image ----
    NetworkConfig net_cfg;
    net_cfg.input_height = resolution; net_cfg.input_width = resolution;
    net_cfg.input_channels = 3; net_cfg.num_classes = 10;

    frontend::UnifiedCsvModel csv_model;
    std::string csv_err;
    if (!frontend::generate_network_from_layer_spec_csv(layer_spec_csv, flex_early, csv_model, csv_err) ||
        csv_model.image.empty()) {
        std::cerr << "ERROR: failed to generate DNN image from layer_spec_csv: "
                  << layer_spec_csv;
        if (!csv_err.empty()) {
            std::cerr << " (" << csv_err << ")";
        }
        std::cerr << "\n";
        return 1;
    }
    DnnImage image = std::move(csv_model.image);
    net_cfg = csv_model.net_cfg;
    std::cerr << " Layers    " << csv_model.num_layers << " functions · "
              << csv_model.num_layer_packets << " layer packets\n"
              << RULE << "\n";
    if (verbose_log)
        std::cerr << "DNN image: " << image.size() << " bytes\n";

    // ---- 2. Sparsity preprocessing (zero-skipping NPUs) ----
    if (zero_skipping) {
        std::cerr << "Applying sparsity preprocessing (zero-skipping)...\n";
        apply_sparsity(image, net_cfg);
    }

#ifdef HAVE_OPENCV
    if (functional_ofm_enabled) {
        if (functional_ofm_dir.empty()) {
            functional_ofm_dir = "build/ofm_cache";
        }
        OfmFileGenConfig ofm_cfg;
        ofm_cfg.input_height = net_cfg.input_height;
        ofm_cfg.input_width = net_cfg.input_width;
        ofm_cfg.input_channels = net_cfg.input_channels;
        ofm_cfg.seed = 42;

        std::string ofm_err;
        std::cerr << "Generating functional OFM files: " << functional_ofm_dir << "\n";
        if (!generate_ofm_files(image, functional_ofm_dir, ofm_cfg, &ofm_err)) {
            std::cerr << "ERROR: failed functional OFM generation: " << ofm_err << "\n";
            return 1;
        }
    }
#else
    if (functional_ofm_enabled) {
        std::cerr << "ERROR: functional OFM mode requires OpenCV build (HAVE_OPENCV)\n";
        return 1;
    }
#endif

    // ---- 3. DRAM timing model — pulled from unified hw JSON via load_flexnpu_config
    uint32_t cfg_outstanding_reads  = 0;
    uint32_t cfg_outstanding_writes = 0;
    uint32_t cfg_read_channels  = 1;
    uint32_t cfg_write_channels = 1;
    uint32_t cfg_max_burst_beats = 0;  // 0 → DmaEngine uses compiled default
    uint32_t cfg_gb_read_port_bytes  = 0;
    uint32_t cfg_gb_write_port_bytes = 0;
    uint32_t cfg_pe_input_buf_kb     = 0;
    uint32_t cfg_pe_weight_buf_kb    = 0;
    uint32_t cfg_npu_base  = memory_map::ADDR_NPU;
    uint32_t cfg_dram_base = memory_map::ADDR_DRAM;
    bool        cfg_model_spill     = false;
    std::string cfg_output_location = "gb";
    uint32_t    cfg_accumulator_kb  = 0;
    if (!dnn_input_json.empty()) {
        try {
            auto flex = flexnpu_sim::config::load_flexnpu_config(dnn_input_json, /*require_network_source=*/false);
            cfg_outstanding_reads  = flex.axi.max_outstanding_reads;
            cfg_outstanding_writes = flex.axi.max_outstanding_writes;
            if (flex.hw.dma.read_channels  > 0) cfg_read_channels  = flex.hw.dma.read_channels;
            if (flex.hw.dma.write_channels > 0) cfg_write_channels = flex.hw.dma.write_channels;
            cfg_max_burst_beats = flex.hw.dma.max_burst_length;
            cfg_gb_read_port_bytes  = flex.hw.buffers.global.read_port_bytes_per_cycle;
            cfg_gb_write_port_bytes = flex.hw.buffers.global.write_port_bytes_per_cycle;
            cfg_pe_input_buf_kb     = flex.hw.buffers.pe_buffers.input_buffer_capacity_kb;
            cfg_pe_weight_buf_kb    = flex.hw.buffers.pe_buffers.weight_buffer_capacity_kb;
            cfg_model_spill         = flex.hw.buffers.global.model_spill;
            cfg_output_location     = flex.hw.buffers.global.output_location;
            cfg_accumulator_kb      = flex.hw.buffers.accumulator.capacity_kb;
            cfg_npu_base  = flex.address_map.npu_base;
            cfg_dram_base = flex.address_map.dram_base;
        } catch (const std::exception&) {
            // flex_early_ok already gated above; keep defaults defensively.
        }
    }
    // The DRAM tier is selected by config (ideal | bandwidth | bank | cycle);
    // the internal tiers take their params from JSON, `cycle` hands an ini to
    // DRAMSim3. sys_clock_ns keeps the internal tiers' cycle latencies unchanged
    // through the AXI slave's DRAM->sys conversion.
    std::unique_ptr<DramTiming> dram_timing =
        make_dram_timing(flex_early.dram, sys_clock_ns);
    if (verbose_log)
        std::cerr << "Loaded DRAM tier: " << dram_timing->name() << "\n";

    // ---- 4. SystemC wiring ----
    sc_clock clk("clk", sys_clock_ns, SC_NS);

    AXI_SIGNALS<NpuRegConfig> reg_bus;

    // NPU Controller (L2 size from CLI)
    Npu npu("npu", static_cast<uint32_t>(l2_kb) * 1024, cfg_npu_base);
    npu.clk(clk);
    npu.bind(reg_bus);
    npu.set_axi_bus_width(bus_bits);  // M7: AXI bus width from CLI
    npu.set_clock_period_ns(sys_clock_ns);
    npu.set_functional_ofm_mode(functional_ofm_enabled);
    npu.set_functional_ofm_dir(functional_ofm_dir);
    npu.set_gb_cache_line_size(l2_cache_line_bytes);
    ensure_parent_dir(timeline_csv_path);
    npu.set_timeline_output(timeline_csv_path);
    if (l2_mode == "cache") {
        npu.set_gb_mode(NpuGlobalBuffer::Mode::Cache);
    } else {
        npu.set_gb_mode(NpuGlobalBuffer::Mode::Scratchpad);
    }

    npu.set_double_buffer(db_enabled);

    // Build the canonical FlexNpuSimConfig: the loaded HW JSON is the BASE
    // (every parsed field rides along — no more piecemeal field-by-field
    // copying, which silently dropped compression/partition/enable knobs
    // three separate times this release), then CLI-derived
    // values override on top.
    flexnpu_sim::config::FlexNpuSimConfig cfg =
        flex_early_ok ? flex_early : flexnpu_sim::config::FlexNpuSimConfig{};
    cfg.hw.preset_name = npu_name;
    cfg.hw.clock_cycle_ns = sys_clock_ns;
    cfg.hw.dataflow = df_str;
    if (cfg.hw.compute.ops_per_pass == 0)     cfg.hw.compute.ops_per_pass = 1;
    if (cfg.hw.compute.parallel_passes == 0)  cfg.hw.compute.parallel_passes = 1;
    if (cfg.hw.compute.num_output_lanes == 0) cfg.hw.compute.num_output_lanes = 1;
    cfg.hw.buffers.global.capacity_kb   = l2_kb;
    cfg.hw.buffers.global.model_spill      = cfg_model_spill;
    cfg.hw.buffers.global.output_location  = cfg_output_location;
    cfg.hw.buffers.accumulator.capacity_kb = cfg_accumulator_kb;
    cfg.hw.buffers.global.read_port_bytes_per_cycle  = cfg_gb_read_port_bytes;
    cfg.hw.buffers.global.write_port_bytes_per_cycle = cfg_gb_write_port_bytes;
    // (compression/partition/enable_gran/pe geometry now ride along
    // from the flex-based cfg above.)
    cfg.hw.buffers.pe_buffers.input_buffer_capacity_kb  = cfg_pe_input_buf_kb;
    cfg.hw.buffers.pe_buffers.weight_buffer_capacity_kb = cfg_pe_weight_buf_kb;
    cfg.axi.data_width_bits             = bus_bits;
    if (cfg_outstanding_reads  > 0) cfg.axi.max_outstanding_reads  = cfg_outstanding_reads;
    if (cfg_outstanding_writes > 0) cfg.axi.max_outstanding_writes = cfg_outstanding_writes;
    cfg.address_map.npu_base  = cfg_npu_base;
    cfg.address_map.dram_base = cfg_dram_base;
    npu.set_system_config(cfg);

    // DMA engines — outstanding depth from hw.json via load_flexnpu_config above.
    axi::CommonConfig axi_ccfg;
    if (cfg_outstanding_reads  > 0) axi_ccfg.max_outstanding_reads  = cfg_outstanding_reads;
    if (cfg_outstanding_writes > 0) axi_ccfg.max_outstanding_writes = cfg_outstanding_writes;

    // Multi-channel DMA: channel 0 of each kind (rDMAC/wDMAC) now lives INSIDE
    // the NPU, which owns and drives it (see NpuController — its command path is
    // internal, single-writer). Channels 1..N-1 remain top-level idle fillers on
    // dummy command signals (Phase 2 enables parallel dispatch). Burst traffic
    // flows through the MemoryMappedTransport attached below; channel 0's two
    // transports are handed to the NPU via npu.attach_dma_transports().

    // Idle channel dummy signals: keep alive for SystemC elaboration lifetime.
    struct DmaIdleSignals {
        sc_signal<uint32_t> src_addr, dst_addr, transfer_size;
        sc_signal<bool>     start, busy, done, read_mode;
    };
    std::vector<std::unique_ptr<DmaEngine>>      rdma_extra_engines;
    std::vector<std::unique_ptr<DmaEngine>>      wdma_extra_engines;
    std::vector<std::unique_ptr<DmaIdleSignals>> rdma_extra_idle;
    std::vector<std::unique_ptr<DmaIdleSignals>> wdma_extra_idle;

    auto spawn_idle = [&](const std::string& name,
                           std::vector<std::unique_ptr<DmaEngine>>& engines,
                           std::vector<std::unique_ptr<DmaIdleSignals>>& idles) {
        auto idle = std::make_unique<DmaIdleSignals>();
        auto eng = std::make_unique<DmaEngine>(name.c_str(), cfg_max_burst_beats);
        eng->clk(clk);
        eng->src_addr(idle->src_addr);
        eng->dst_addr(idle->dst_addr);
        eng->transfer_size(idle->transfer_size);
        eng->start(idle->start);
        eng->busy(idle->busy);
        eng->done(idle->done);
        eng->read_mode(idle->read_mode);
        engines.push_back(std::move(eng));
        idles.push_back(std::move(idle));
    };

    for (uint32_t i = 1; i < cfg_read_channels; i++) {
        spawn_idle("rdmac_" + std::to_string(i),
                   rdma_extra_engines, rdma_extra_idle);
    }
    for (uint32_t i = 1; i < cfg_write_channels; i++) {
        spawn_idle("wdmac_" + std::to_string(i),
                   wdma_extra_engines, wdma_extra_idle);
    }

    // NPU <-> DMA command signals are now INTERNAL to the NPU (channel 0
    // rDMAC/wDMAC are the NPU's own children; the WDMA multi-writer signal set
    // and its cooperative mutex are gone — a single internal issue stage drives
    // wdma_* under SC_ONE_WRITER). The system assembly only supplies the
    // channel-0 data-path transports, done via attach_dma_transports() below.

    // Memory (R5 V2): R3 axi::Slave + DenseRamBackend + DRAM timing hook.
    constexpr uint32_t MEM_SIZE        = 64 * 1024 * 1024;  // 64 MB backing
    constexpr uint64_t DRAM_ROUTE_SPAN = 0x80000000ull;     // 2 GB @ 0x8000_0000
    auto dram_base = cfg_dram_base;

    MemoryAxiSlaveV2<DataSpecT> memory(
        "memory", dram_base, MEM_SIZE, std::move(dram_timing), axi_ccfg);
    memory.clk(clk);

    // Per-master V2 AXI stack: Master<DataSpecT> + AxiTransport + its own
    // AXI_SIGNALS<DataSpecT> bundle. One entry per DMA channel, read-first
    // ordering (channels 0..N_r-1 reads, N_r..N_r+N_w-1 writes) to match
    // the legacy bind order and the NPU controller's channel mapping.
    const uint32_t total_masters = cfg_read_channels + cfg_write_channels;
    std::vector<std::unique_ptr<transport::axi::Master<DataSpecT>>>    v2_masters;
    std::vector<std::unique_ptr<transport::axi::AxiTransport<DataSpecT>>> v2_transports;
    std::vector<std::unique_ptr<AXI_SIGNALS<DataSpecT>>>                v2_m_sigs;
    AXI_SIGNALS<DataSpecT>                                              v2_s_sig;

    auto make_v2_master = [&](const std::string& tag) {
        auto m = std::make_unique<transport::axi::Master<DataSpecT>>(
            tag.c_str(), axi_ccfg);
        m->clk(clk);
        auto sig = std::make_unique<AXI_SIGNALS<DataSpecT>>();
        bind_port_signal(&m->port, *sig);

        transport::Capabilities caps;
        caps.max_outstanding_reads  = axi_ccfg.max_outstanding_reads;
        caps.max_outstanding_writes = axi_ccfg.max_outstanding_writes;
        caps.max_beats_per_burst    = DataSpecT::MAX_BURST_LEN;
        // Runtime AXI width: beat bytes = bus_bits/8, independent of the
        // compiled signal width (the signal is sized to the compiled max).
        caps.max_burst_bytes        = DataSpecT::MAX_BURST_LEN * (bus_bits / 8);
        caps.addr_alignment_bytes   = 1;
        caps.boundary_bytes         = 4096;
        auto t = std::make_unique<transport::axi::AxiTransport<DataSpecT>>(*m, caps);
        v2_masters.push_back(std::move(m));
        v2_transports.push_back(std::move(t));
        v2_m_sigs.push_back(std::move(sig));
    };

    make_v2_master("v2_rdmac_0");
    for (uint32_t i = 1; i < cfg_read_channels; ++i) {
        make_v2_master("v2_rdmac_" + std::to_string(i));
    }
    make_v2_master("v2_wdmac_0");
    for (uint32_t i = 1; i < cfg_write_channels; ++i) {
        make_v2_master("v2_wdmac_" + std::to_string(i));
    }

    // Attach transports to DmaEngines. Order matches make_v2_master calls
    // above: reads [0..N_r-1], writes [N_r..N_r+N_w-1]. Channel 0 read/write go
    // to the NPU's internal engines; channels 1..N-1 to the idle fillers.
    npu.attach_dma_transports(v2_transports[0].get(),
                              v2_transports[cfg_read_channels].get());
    // DMA IP flow-control: the engine's in-flight window comes from its own FIFO
    // depth / issuing capability (not the AXI outstanding window).
    npu.set_dma_flow_control(cfg.hw.dma.buffer_size_kb * 1024u,
                             cfg.hw.dma.max_issuing_reads,
                             cfg.hw.dma.max_issuing_writes);
    for (uint32_t i = 0; i < rdma_extra_engines.size(); ++i) {
        rdma_extra_engines[i]->attach_transport(v2_transports[1 + i].get());
    }
    for (uint32_t i = 0; i < wdma_extra_engines.size(); ++i) {
        wdma_extra_engines[i]->attach_transport(
            v2_transports[cfg_read_channels + 1 + i].get());
    }

    // Slave-side signal bundle wires the V2 bus to the V2 memory slave.
    bind_port_signal(&memory.port(), v2_s_sig);

    // Bus arbitration from config: axi.default_arbit_policy selects among
    // the arbiter's 5 policies (default RoundRobin = previous hard-coded
    // behavior); per-master qos/weight/priority come from topology.masters
    // (by order) when declared.
    transport::axi::ArbitPolicy bus_policy = transport::axi::ArbitPolicy::RoundRobin;
    std::vector<transport::axi::MasterArbitCfg> bus_mcfgs;
    if (flex_early_ok) {
        bus_policy = parse_bus_arbit(flex_early.axi.default_arbit_policy);
        for (const auto& m : flex_early.topology.masters) {
            transport::axi::MasterArbitCfg mc;
            mc.qos = m.qos; mc.weight = m.weight;
            mc.priority = m.priority; mc.default_qos = m.default_qos;
            bus_mcfgs.push_back(mc);
        }
    }

    // R4 Bus<DataSpecT, NumM, 1> via compile-time dispatcher. Supported
    // master counts cover the DSE sweep: 2 (1+1), 3 (2+1), 6 (4+2), 12 (8+4).
    std::unique_ptr<sc_core::sc_module> data_bus_owner;
    switch (total_masters) {
        case 1:  data_bus_owner = build_data_bus_n<1>(clk, dram_base, DRAM_ROUTE_SPAN, v2_m_sigs, v2_s_sig, bus_policy, bus_mcfgs); break;
        case 2:  data_bus_owner = build_data_bus_n<2>(clk, dram_base, DRAM_ROUTE_SPAN, v2_m_sigs, v2_s_sig, bus_policy, bus_mcfgs); break;
        case 3:  data_bus_owner = build_data_bus_n<3>(clk, dram_base, DRAM_ROUTE_SPAN, v2_m_sigs, v2_s_sig, bus_policy, bus_mcfgs); break;
        case 4:  data_bus_owner = build_data_bus_n<4>(clk, dram_base, DRAM_ROUTE_SPAN, v2_m_sigs, v2_s_sig, bus_policy, bus_mcfgs); break;
        case 5:  data_bus_owner = build_data_bus_n<5>(clk, dram_base, DRAM_ROUTE_SPAN, v2_m_sigs, v2_s_sig, bus_policy, bus_mcfgs); break;
        case 6:  data_bus_owner = build_data_bus_n<6>(clk, dram_base, DRAM_ROUTE_SPAN, v2_m_sigs, v2_s_sig, bus_policy, bus_mcfgs); break;
        case 8:  data_bus_owner = build_data_bus_n<8>(clk, dram_base, DRAM_ROUTE_SPAN, v2_m_sigs, v2_s_sig, bus_policy, bus_mcfgs); break;
        case 10: data_bus_owner = build_data_bus_n<10>(clk, dram_base, DRAM_ROUTE_SPAN, v2_m_sigs, v2_s_sig, bus_policy, bus_mcfgs); break;
        case 12: data_bus_owner = build_data_bus_n<12>(clk, dram_base, DRAM_ROUTE_SPAN, v2_m_sigs, v2_s_sig, bus_policy, bus_mcfgs); break;
        case 16: data_bus_owner = build_data_bus_n<16>(clk, dram_base, DRAM_ROUTE_SPAN, v2_m_sigs, v2_s_sig, bus_policy, bus_mcfgs); break;
        default:
            throw std::runtime_error("flexnpusim_system: unsupported total_masters="
                                     + std::to_string(total_masters));
    }

    // CPU driver
    Processor cpu("cpu");
    cpu.npu_base  = cfg_npu_base;
    cpu.dram_base = cfg_dram_base;
    cpu.configure(cfg.processor);   // latency model params from hw JSON
    cpu.clk(clk);
    cpu.bind(reg_bus);

    // ---- 5. Preload memory ----
    std::memset(memory.memory().data(), 0xAA,
                std::min(static_cast<size_t>(MEM_SIZE), memory.memory().size()));
    std::memcpy(memory.memory().data(), image.raw(),
                std::min(image.size(), static_cast<size_t>(MEM_SIZE)));

    // ---- 6. Run ----
    if (verbose_log)
        std::cerr << "Starting simulation...\n";
    else
        // Silence the kernel's "Simulation stopped by user." info line.
        sc_core::sc_report_handler::set_actions(
            "/OSCI/SystemC", sc_core::SC_INFO, sc_core::SC_DO_NOTHING);
    // Run until the Processor stops the simulation (normal completion or watchdog).
    sc_start();

    // ---- 7. Performance counters (read inside SC_THREAD, stored in cpu) ----
    // Plain digits right after each label are load-bearing: demo/run.py
    // parses "(Cycles|MACs|Mem reads|Mem writes):\s*(\d+)".
    std::cerr << "\n" << RULE << "\n"
              << " RESULTS   " << npu_name << " · " << net_name
              << "   [" << (cpu.completed ? "COMPLETED" : "INCOMPLETE") << "]\n"
              << RULE << "\n" << std::fixed
              << "  Cycles:     " << cpu.cnt0
              << "   (" << std::setprecision(2) << cpu.cnt0 * sys_clock_ns / 1e6
              << " ms @ clk=" << std::setprecision(0) << sys_clock_ns << "ns)\n"
              << "  MACs:       " << cpu.cnt1
              << "   (" << std::setprecision(2) << cpu.cnt1 / 1e9 << " GMAC)\n"
              << "  Mem reads:  " << cpu.cnt2 << " bytes   ("
              << std::setprecision(1) << cpu.cnt2 / 1e6 << " MB)\n"
              << "  Mem writes: " << cpu.cnt3 << " bytes   ("
              << std::setprecision(1) << cpu.cnt3 / 1e6 << " MB)\n"
              << "  DRAM model: " << memory.timing_name() << "\n"
              << "  Completed:  " << (cpu.completed ? "YES" : "NO") << "\n"
              << RULE << "\n";
    std::cerr << std::defaultfloat;

    // ---- 7b. EDA-style report (-report <file>) ----
    if (!report_path.empty()) {
        ensure_parent_dir(report_path);
        std::ofstream rp(report_path);
        if (!rp) {
            std::cerr << "ERROR: cannot open report file: " << report_path << "\n";
        } else {
            write_performance_report(
                rp, npu.controller().layer_records, cfg, npu_name, net_name, df_str,
                memory.timing_name(), l2_kb, bus_bits, sys_clock_ns,
                cpu.cnt0, cpu.cnt1, cpu.cnt2, cpu.cnt3,
                npu.controller().perf_read_txns, npu.controller().perf_write_txns, cpu.completed);
            std::cerr << "Report written: " << report_path << "\n";
        }
    }

    // ---- 8. CSV output (stdout + optional file) ----
    // Base columns through `completed` (cnt0..3 = cycles, macs, mem_rd, mem_wr).
    std::ostringstream csv_line;
    csv_line << npu_name << "," << net_name << ","
             << bus_bits << "," << l2_kb << "," << dram_type << ","
             << resolution << ","
             << df_str << "," << (db_enabled ? 1 : 0) << ","
             << threshold_ratio << ","
             << cpu.cnt0 << "," << cpu.cnt1 << "," << cpu.cnt2 << "," << cpu.cnt3 << ","
             << (cpu.completed ? 1 : 0);

    // Run-level comparison vector (derived; same definitions as the -report):
    // inference rate, PE utilization, achieved DRAM bandwidth. Appended so a
    // sweep can be ranked without opening each per-run report.
    const uint64_t peak_mac = std::max<uint64_t>(1,
        static_cast<uint64_t>(pe.ops_per_pass) * pe.num_output_lanes *
        std::max(1u, pe.parallel_passes));
    const double fps = cpu.cnt0
        ? 1.0e9 / (static_cast<double>(cpu.cnt0) * sys_clock_ns) : 0.0;
    const double util_pct = cpu.cnt0
        ? 100.0 * cpu.cnt1 / (static_cast<double>(cpu.cnt0) * peak_mac) : 0.0;
    const double dram_bw_gbps = cpu.cnt0
        ? static_cast<double>(cpu.cnt2 + cpu.cnt3) /
              (static_cast<double>(cpu.cnt0) * sys_clock_ns) : 0.0;  // B/ns = GB/s
    // bus_util = achieved (rd+wr) over one-beat-per-cycle peak; the tier-agnostic
    // "how much of the data bus did we use" number.
    const double peak_bus_bw = sys_clock_ns > 0 ? (bus_bits / 8.0) / sys_clock_ns : 0.0;
    const double bus_util_pct = peak_bus_bw > 0 ? 100.0 * dram_bw_gbps / peak_bus_bw : 0.0;
    std::ostringstream metrics;
    metrics << std::fixed << std::setprecision(2)
            << "," << fps << "," << util_pct << "," << dram_bw_gbps << "," << bus_util_pct;

    std::cout << csv_line.str() << metrics.str() << "\n";

    if (!output_csv_path.empty()) {
        ensure_parent_dir(output_csv_path);
        // Keep appended rows aligned to whatever schema an existing file already
        // declares: the metrics vector was added later, so detect it by column
        // name and emit only what the header carries.
        bool write_header = true;
        bool include_metrics_cols = true;
        {
            std::ifstream ifs(output_csv_path);
            if (ifs.good()) {
                std::string first;
                std::getline(ifs, first);
                const std::string header = trim_copy(first);
                if (!header.empty()) {
                    write_header = false;
                    include_metrics_cols =
                        (header.find("dram_bw_gbps") != std::string::npos);
                }
            }
        }
        std::ofstream ofs(output_csv_path, std::ios::app);
        if (!ofs.is_open()) {
            std::cerr << "WARN: cannot open output CSV file: " << output_csv_path << "\n";
        } else {
            if (write_header) {
                ofs << "npu,network,bus_bits,l2_kb,dram,resolution,dataflow,double_buf,threshold,"
                       "cycles,macs,mem_rd,mem_wr,completed,"
                       "fps,util_pct,dram_bw_gbps,bus_util_pct\n";
            }
            std::string row = csv_line.str();
            if (include_metrics_cols) row += metrics.str();
            ofs << row << "\n";
        }
    }

    return cpu.completed ? 0 : 1;
}
