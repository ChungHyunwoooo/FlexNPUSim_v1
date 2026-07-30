/**
 * @file test_config.cpp
 * @brief Contract tests for common/config_loader + flexnpu_config.
 *
 * The fixture below is a full-block sample of the config schema, written by
 * the test itself (the retired examples/ tree used to play this role) — it
 * doubles as a schema reference and the seed for regenerating examples/.
 */

#include "common/config_loader.h"

#include <unistd.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace flexnpu_sim::config;

static const char* kFixture = R"JSON(
// FlexNPUSim full-block config sample (JSONC).
{
  "hw": {
    "preset_name": "nvdla-large",
    "dataflow": "WS",
    "npu_mapping": { "ops_per_pass": 16 },
    "buffer": {
      "input":  { "capacity": 2, "unit": "kb" },
      "weight": { "capacity": 4, "unit": "kb" },
      "per_function": {
        "activation": { "input": { "capacity": 1, "unit": "kb" } }
      }
    },
    "global_buffer": { "capacity_kb": 512 }
  },
  "network": {
    "csv": "model/sw/cnn/nvdla-large/tiny2layer.csv",
    "input_channels": 3
  },
  "axi": {
    "protocol": "AXI4",
    "max_outstanding_reads": 8,
    "max_outstanding_writes": 4
  },
  "dram": {
    "tier": "bandwidth",
    "read_latency_cyc": 100,
    "bandwidth_bytes_per_cycle": 26,
    "include_writes": false
  },
  "topology": {
    "masters": [
      { "name": "rdma_input",  "port_bits": 64, "max_outstanding_reads": 8, "weight": 2, "priority": 1 },
      { "name": "wdma_output", "port_bits": 64, "max_outstanding_writes": 4, "weight": 1, "priority": 2 },
      { "name": "cpu",         "port_bits": 32, "max_outstanding_reads": 1, "max_outstanding_writes": 1, "priority": 0 }
    ],
    "slaves": [
      { "name": "dram",    "port_bits": 64, "addr_base": 2147483648, "addr_size": 1073741824 },
      { "name": "npu_reg", "port_bits": 32, "addr_base": 1073741824, "addr_size": 65536 }
    ],
    "links": [
      { "master": "rdma_input",  "slave": "dram" },
      { "master": "wdma_output", "slave": "dram" },
      { "master": "cpu",         "slave": "npu_reg" },
      { "master": "cpu",         "slave": "dram" }
    ],
    "interconnect": { "type": "axi_crossbar", "arbit_policy": "RoundRobin" }
  },
  "sim": { "output_dir": "out/test_config" },
  "compiler": { "type": "manual", "strategy": "min_traffic" }
}
)JSON";

static void test_full_block_parse(const fs::path& file) {
    auto cfg = load_flexnpu_config(file);

    assert(cfg.hw.preset_name == "nvdla-large");
    assert(cfg.hw.dataflow == "WS");
    assert(cfg.hw.compute.ops_per_pass == 16);
    assert(cfg.hw.buffers.global.capacity_kb == 512);
    // per-unit PE buffers: defaults + function-type override inheriting
    // the unset field (LM<->function descriptor granularity, 2026-07-10)
    assert(cfg.hw.buffers.pe_buffers.input_buffer_capacity_kb == 2);
    assert(cfg.hw.buffers.pe_buffers.weight_buffer_capacity_kb == 4);
    const auto& pf = cfg.hw.buffers.pe_buffers_per_function;
    assert(pf.count("activation") == 1);
    assert(pf.at("activation").input_buffer_capacity_kb == 1);
    assert(pf.at("activation").weight_buffer_capacity_kb == 4 &&
           "override inherits unset fields from the default");

    assert(cfg.network.csv.find("tiny2layer.csv") != std::string::npos);
    assert(cfg.network.input_channels == 3);

    assert(cfg.axi.protocol == "AXI4");
    assert(cfg.axi.max_outstanding_reads == 8);
    assert(cfg.axi.max_outstanding_writes == 4);

    assert(cfg.dram.tier == "bandwidth");
    assert(cfg.dram.read_latency_cyc == 100);
    assert(cfg.dram.bandwidth_bytes_per_cycle == 26);
    assert(cfg.dram.include_writes == false);

    assert(cfg.topology.masters.size() == 3);
    assert(cfg.topology.slaves.size() == 2);
    assert(cfg.topology.links.size() == 4);
    assert(cfg.topology.interconnect.arbit_policy == "RoundRobin");

    assert(cfg.compiler.type == "manual");
    assert(cfg.compiler.strategy == "min_traffic");

    assert(!cfg.source_path.empty());
}

static void test_hw_only_mode(const fs::path& dir) {
    // require_network_source=false accepts a workload-less HW spec.
    const fs::path p = dir / "hw_only.json";
    std::ofstream(p) << R"({ "hw": { "preset_name": "midap" } })";
    auto cfg = load_flexnpu_config(p, /*require_network_source=*/false);
    assert(cfg.hw.preset_name == "midap");

    bool threw = false;
    try {
        (void)load_flexnpu_config(p);  // default: workload required
    } catch (const std::exception&) {
        threw = true;
    }
    assert(threw && "missing network source must throw by default");
}

int main() {
    const fs::path dir = fs::temp_directory_path() /
                         ("flexnpu_config_test_" + std::to_string(::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir);

    const fs::path file = dir / "full.json";
    std::ofstream(file) << kFixture;

    test_full_block_parse(file);
    test_hw_only_mode(dir);

    fs::remove_all(dir);
    std::cout << "test_config: all contracts hold\n";
    return 0;
}
