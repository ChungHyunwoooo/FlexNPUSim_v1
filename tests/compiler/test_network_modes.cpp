/**
 * @file test_network_modes.cpp
 * @brief The compiler's two documented input modes, end to end:
 *        network CSV -> dnn image -> loader -> latency descriptor.
 *
 * Contracts:
 *  1. Mode 1 (auto) — a k/g/s row derives the descriptor: n_max = g*s,
 *     n_min = g (the k/g/s derivation rule)
 *  2. Mode 2 (oracle) — explicit f_i/f_w/f_o/l_pass/n_max/n_min columns
 *     are adopted verbatim, overriding the derivation
 *  3. the emitted image carries the format magic/version and a corrupted
 *     magic is rejected by the loader
 *  4. the compute structure enters the derivation ONLY as a parameter
 *     (the formalization rule): pe_type=systolic -> l = k, adder_tree ->
 *     l = 1 + ceil(log2 k); the runtime model consumes l, never the
 *     structure itself
 */

#include "common/flexnpu_config.h"
#include "compiler/dnn_image/dnn_image_format.h"
#include "compiler/dnn_image/dnn_image_loader.h"
#include "compiler/frontend/loaders/network_csv.h"

#include <unistd.h>

#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace flexnpu_sim;

static const char* kHeader =
    "layer_id,function_id,layer_name,layer_type,function_type,"
    "i1_connect,i2_connect,input_h,input_w,input_c,kernel_h,kernel_w,"
    "kernel_c,kernel_count,stride,padding,dilation,tile_h,tile_w,pe_type,"
    "k,g,s,psum_kb,preprocess_type,preprocess_param0,preprocess_param1,"
    "input_addr,kernel_addr,output_addr";

static const char* kRowCommon =
    "0,0,conv0,conv2d,conv2d,4294967295,4294967295,8,8,3,3,3,3,8,1,1,1,"
    "4,4,systolic,4,2,8,0,none,0,0,4096,8192,12288";

static config::FlexNpuSimConfig test_cfg() {
    config::FlexNpuSimConfig cfg;
    cfg.hw.dataflow = "WS";
    cfg.hw.compute.ops_per_pass = 16;
    cfg.hw.compute.num_output_lanes = 16;
    cfg.hw.buffers.global.capacity_kb = 256;
    cfg.hw.compute.threshold_ratio = 1.0;
    return cfg;
}

static frontend::UnifiedCsvModel load(const fs::path& file) {
    frontend::UnifiedCsvModel out;
    std::string err;
    const bool ok =
        frontend::generate_network_from_layer_spec_csv(file.string(),
                                                       test_cfg(), out, err);
    assert(ok && "csv must load");
    return out;
}

int main() {
    const fs::path dir = fs::temp_directory_path() /
                         ("flexnpu_modes_" + std::to_string(::getpid()));
    fs::create_directories(dir);

    // -- 1. Mode 1 (auto): k/g/s derivation -------------------------------
    std::ofstream(dir / "auto.csv") << kHeader << "\n" << kRowCommon << "\n";
    auto auto_model = load(dir / "auto.csv");
    assert(auto_model.num_layers == 1);

    DnnImageLoader l2;
    assert(l2.parse(auto_model.image.data.data(), auto_model.image.data.size()));
    const auto p2 = l2.to_latency_params(0);
    assert(p2.n_max == 2 * 8 && "n_max = g*s");
    assert(p2.n_min == 2 && "n_min = g");
    assert(p2.fetch_operand0 > 0 && p2.fetch_operand1 > 0);

    // -- 2. Mode 2 (oracle): verbatim adoption -----------------------------
    std::ofstream(dir / "oracle.csv")
        << kHeader << ",f_i,f_w,f_o,l_pass,n_max,n_min\n"
        << kRowCommon << ",111,222,333,7,12,3\n";
    auto oracle_model = load(dir / "oracle.csv");

    DnnImageLoader l1;
    assert(l1.parse(oracle_model.image.data.data(), oracle_model.image.data.size()));
    const auto p1 = l1.to_latency_params(0);
    assert(p1.fetch_operand0 == 111);
    assert(p1.fetch_operand1 == 222);
    assert(p1.write_output == 333);
    assert(p1.issue_latency == 7);
    assert(p1.n_max == 12);
    assert(p1.n_min == 3);

    // -- 3. format magic: corrupt image is rejected ------------------------
    auto bytes = auto_model.image.data;   // copy
    assert(bytes.size() > 4);
    std::memset(bytes.data(), 0x00, 4);   // clobber DNN_IMAGE_MAGIC
    DnnImageLoader bad;
    assert(!bad.parse(bytes.data(), bytes.size()) &&
           "corrupted magic must be rejected");

    // -- 4. unit type is a derivation parameter only ------------------------
    // Same row, k=4: systolic l = k = 4; adder_tree l = 1 + ceil(log2 4) = 3.
    // (kRowCommon carries pe_type=systolic, k=4.)
    assert(p2.issue_latency == 4 && "systolic: l = k");
    std::string tree_row(kRowCommon);
    const auto pos = tree_row.find("systolic");
    tree_row.replace(pos, 8, "adder_tree");
    std::ofstream(dir / "tree.csv") << kHeader << "\n" << tree_row << "\n";
    auto tree_model = load(dir / "tree.csv");
    DnnImageLoader l4;
    assert(l4.parse(tree_model.image.data.data(), tree_model.image.data.size()));
    assert(l4.to_latency_params(0).issue_latency == 3 &&
           "adder_tree: l = 1 + ceil(log2 k)");

    fs::remove_all(dir);
    std::cout << "test_network_modes: all contracts hold\n";
    return 0;
}
