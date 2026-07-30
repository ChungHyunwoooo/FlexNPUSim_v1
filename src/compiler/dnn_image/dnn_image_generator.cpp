/**
 * @file dnn_image_generator.cpp
 * @brief DNN Image Generator Implementation
 */

#include "compiler/dnn_image/dnn_image_generator.h"
#include "common/tile_descriptor.h"
#include "common/types.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace flexnpu_sim {

void DnnImageGenerator::set_layer_pe_override(FuncType type, const PeConfig& pe) {
    for (auto& ov : layer_pe_overrides_) {
        if (ov.type == type) {
            ov.pe = pe;
            return;
        }
    }
    layer_pe_overrides_.push_back({type, pe});
}

void DnnImageGenerator::set_layer_pe_override(uint32_t layer_index, const PeConfig& pe) {
    for (auto& ov : layer_index_pe_overrides_) {
        if (ov.layer_index == layer_index) {
            ov.pe = pe;
            return;
        }
    }
    layer_index_pe_overrides_.push_back({layer_index, pe});
}

uint32_t DnnImageGenerator::estimate_per_output_ops(const LayerSpec& spec) {
    switch (spec.type) {
        case FuncType::Conv2D:
            return spec.kernel_height * spec.kernel_width * spec.input_channels;
        case FuncType::DepthwiseConv2D:
            return spec.kernel_height * spec.kernel_width;
        case FuncType::FullyConnected:
            return spec.input_channels;
        case FuncType::Pooling:
            return spec.kernel_height * spec.kernel_width;
        case FuncType::Activation:
            return 1;
        case FuncType::MatMul:
            return spec.kernel_channels; // d_k
        case FuncType::ElementWise:
            return 1;
        // Normalization / reduction reduce over the feature (channel) axis:
        // n(o) = input_channels (softmax sum, layernorm mean+var, reduce).
        case FuncType::Softmax:
        case FuncType::LayerNorm:
        case FuncType::Reduction:
            return spec.input_channels;
        case FuncType::Gather:  // index-select copy
            return 1;
        default:
            return 1;
    }
}

PeConfig DnnImageGenerator::sanitize_pe_config(PeConfig pe) {
    if (pe.ops_per_pass == 0) pe.ops_per_pass = 1;
    if (pe.parallel_passes == 0) pe.parallel_passes = 1;
    if (pe.num_output_lanes == 0) pe.num_output_lanes = 1;
    return pe;
}

PeConfig DnnImageGenerator::resolve_pe_config(const LayerSpec& spec, uint32_t layer_index) const {
    // Highest priority: explicit per-layer-index override.
    for (const auto& ov : layer_index_pe_overrides_) {
        if (ov.layer_index == layer_index) {
            return sanitize_pe_config(ov.pe);
        }
    }

    // Next priority: explicit per-layer-type override.
    for (const auto& ov : layer_pe_overrides_) {
        if (ov.type == spec.type) {
            return sanitize_pe_config(ov.pe);
        }
    }

    PeConfig base = sanitize_pe_config(pe_config_);
    if (pe_grouping_mode_ == PeGroupingMode::Fixed) {
        return base;
    }

    // AdaptiveByOps: keep total multiplier budget (k*G) from user input,
    // then derive layer-effective logical grouping from ops_per_output.
    const uint32_t ops_per_output = estimate_per_output_ops(spec);
    const uint32_t total_pe = std::max(1u, base.ops_per_pass * base.total_execution_units());
    const uint32_t target_k = std::max(1u, std::min(base.ops_per_pass, ops_per_output));

    uint32_t ops_eff = 1;
    for (uint32_t cand = 1; cand <= target_k; ++cand) {
        if ((total_pe % cand) == 0) {
            ops_eff = cand;
        }
    }
    if (ops_eff == 0) ops_eff = 1;

    uint32_t total_units_eff = std::max(1u, total_pe / ops_eff);
    uint32_t passes_need = std::max(1u, (ops_per_output + ops_eff - 1) / ops_eff);
    uint32_t passes_eff = std::min(base.parallel_passes, passes_need);
    passes_eff = std::max(1u, std::min(passes_eff, total_units_eff));
    uint32_t lanes_eff = std::max(1u, total_units_eff / passes_eff);

    PeConfig eff = base;
    eff.ops_per_pass = ops_eff;
    eff.parallel_passes = passes_eff;
    eff.num_output_lanes = lanes_eff;
    return sanitize_pe_config(eff);
}

// ============================================================================
// apply_latency_params: dispatcher based on configure() state
// ============================================================================

void DnnImageGenerator::apply_latency_params(LayerSpec& spec) const {
    if (configured_) {
        uint32_t layer_index = layer_sequence_index_++;
        PeConfig effective_pe = resolve_pe_config(spec, layer_index);
        set_tiled_latency_params(spec, 0, 0, l2_buffer_kb_,
                                  dataflow_, effective_pe);
    } else {
        set_default_latency_params(spec);
    }
}

// ============================================================================
// Single Conv2D
// ============================================================================

DnnImage DnnImageGenerator::generate_single_conv2d(
    uint32_t in_h, uint32_t in_w, uint32_t in_c, uint32_t out_c,
    uint32_t kernel_size, uint32_t stride, uint32_t padding) {
    layer_sequence_index_ = 0;

    DnnImageCompiler compiler;

    LayerSpec spec;
    spec.name = "conv2d_single";
    spec.type = FuncType::Conv2D;
    spec.stride = stride;
    spec.padding = padding;
    spec.dilation = 1;

    spec.input_height = in_h;
    spec.input_width = in_w;
    spec.input_channels = in_c;

    spec.kernel_height = kernel_size;
    spec.kernel_width = kernel_size;
    spec.kernel_channels = in_c;
    spec.kernel_count = out_c;

    spec.input_address = 0;
    spec.weight_address = tensor_size(in_h, in_w, in_c);
    spec.output_address = spec.weight_address +
                          (kernel_size * kernel_size * in_c * out_c * sizeof(float));

    apply_latency_params(spec);

    compiler.add_layer(spec);
    return compiler.compile();
}

// ============================================================================
// Test Network (3 layers)
// ============================================================================

DnnImage DnnImageGenerator::generate_test_network(const NetworkConfig& config) {
    layer_sequence_index_ = 0;
    DnnImageCompiler compiler;
    uint32_t current_addr = 0;

    // Layer 1: Conv2D (3x3, 32 filters)
    LayerSpec conv1;
    conv1.name = "conv1";
    conv1.type = FuncType::Conv2D;
    conv1.stride = 1;
    conv1.padding = 1;
    conv1.dilation = 1;

    conv1.input_height = config.input_height;
    conv1.input_width = config.input_width;
    conv1.input_channels = config.input_channels;

    conv1.kernel_height = 3;
    conv1.kernel_width = 3;
    conv1.kernel_channels = config.input_channels;
    conv1.kernel_count = 32;

    conv1.input_address = current_addr;
    current_addr += tensor_size(conv1.input_height, conv1.input_width, conv1.input_channels);

    conv1.weight_address = current_addr;
    current_addr += 3 * 3 * config.input_channels * 32 * sizeof(float);

    uint32_t conv1_out_h = (config.input_height + 2 * conv1.padding - 3) / conv1.stride + 1;
    uint32_t conv1_out_w = (config.input_width + 2 * conv1.padding - 3) / conv1.stride + 1;
    conv1.output_address = current_addr;
    current_addr += tensor_size(conv1_out_h, conv1_out_w, 32);

    apply_latency_params(conv1);
    compiler.add_layer(conv1);

    // Layer 2: MaxPooling (2x2, stride 2)
    LayerSpec pool1;
    pool1.name = "pool1";
    pool1.type = FuncType::Pooling;
    pool1.stride = 2;
    pool1.padding = 0;
    pool1.dilation = 1;

    pool1.input_height = conv1_out_h;
    pool1.input_width = conv1_out_w;
    pool1.input_channels = 32;

    pool1.kernel_height = 2;
    pool1.kernel_width = 2;
    pool1.kernel_channels = 32;
    pool1.kernel_count = 32;

    pool1.input_address = conv1.output_address;
    pool1.weight_address = 0; // Pooling doesn't need weights

    uint32_t pool1_out_h = (conv1_out_h - 2) / 2 + 1;
    uint32_t pool1_out_w = (conv1_out_w - 2) / 2 + 1;
    pool1.output_address = current_addr;
    current_addr += tensor_size(pool1_out_h, pool1_out_w, 32);

    apply_latency_params(pool1);
    compiler.add_layer(pool1);

    // Layer 3: Fully Connected (flatten to num_classes)
    LayerSpec fc1;
    fc1.name = "fc1";
    fc1.type = FuncType::FullyConnected;
    fc1.stride = 1;
    fc1.padding = 0;
    fc1.dilation = 1;

    fc1.input_height = pool1_out_h;
    fc1.input_width = pool1_out_w;
    fc1.input_channels = 32;

    fc1.kernel_height = 1;
    fc1.kernel_width = 1;
    fc1.kernel_channels = pool1_out_h * pool1_out_w * 32;
    fc1.kernel_count = config.num_classes;

    fc1.input_address = pool1.output_address;
    fc1.weight_address = current_addr;
    current_addr += fc1.kernel_channels * fc1.kernel_count * sizeof(float);

    fc1.output_address = current_addr;
    current_addr += tensor_size(1, 1, config.num_classes);

    apply_latency_params(fc1);
    compiler.add_layer(fc1);

    return compiler.compile();
}

// ============================================================================
// VGG-16 (21 layers: 5 blocks + 3 FC)
// ============================================================================

DnnImage DnnImageGenerator::generate_vgg16(const NetworkConfig& config) {
    layer_sequence_index_ = 0;
    DnnImageCompiler compiler;
    uint32_t current_addr = config.base_dram_addr;

    uint32_t in_h = config.input_height;
    uint32_t in_w = config.input_width;
    uint32_t in_c = config.input_channels;

    // Block 1: 2×Conv(64) + Pool  = 3 layers
    add_vgg_block(compiler, current_addr, in_h, in_w, in_c, 64, 2);
    // Block 2: 2×Conv(128) + Pool = 3 layers
    add_vgg_block(compiler, current_addr, in_h, in_w, in_c, 128, 2);
    // Block 3: 3×Conv(256) + Pool = 4 layers
    add_vgg_block(compiler, current_addr, in_h, in_w, in_c, 256, 3);
    // Block 4: 3×Conv(512) + Pool = 4 layers
    add_vgg_block(compiler, current_addr, in_h, in_w, in_c, 512, 3);
    // Block 5: 3×Conv(512) + Pool = 4 layers
    add_vgg_block(compiler, current_addr, in_h, in_w, in_c, 512, 3);

    // FC1: flatten → 4096
    {
        uint32_t flat_size = in_h * in_w * in_c;
        LayerSpec fc;
        fc.name = "fc1";
        fc.type = FuncType::FullyConnected;
        fc.stride = 1; fc.padding = 0; fc.dilation = 1;
        fc.input_height = in_h; fc.input_width = in_w; fc.input_channels = in_c;
        fc.kernel_height = 1; fc.kernel_width = 1;
        fc.kernel_channels = flat_size; fc.kernel_count = 4096;
        fc.input_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);
        fc.weight_address = current_addr;
        current_addr += flat_size * 4096 * sizeof(float);
        fc.output_address = current_addr;
        current_addr += tensor_size(1, 1, 4096);
        apply_latency_params(fc);
        compiler.add_layer(fc);
        in_h = 1; in_w = 1; in_c = 4096;
    }

    // FC2: 4096 → 4096
    {
        LayerSpec fc;
        fc.name = "fc2";
        fc.type = FuncType::FullyConnected;
        fc.stride = 1; fc.padding = 0; fc.dilation = 1;
        fc.input_height = 1; fc.input_width = 1; fc.input_channels = 4096;
        fc.kernel_height = 1; fc.kernel_width = 1;
        fc.kernel_channels = 4096; fc.kernel_count = 4096;
        fc.input_address = current_addr;
        current_addr += tensor_size(1, 1, 4096);
        fc.weight_address = current_addr;
        current_addr += 4096 * 4096 * sizeof(float);
        fc.output_address = current_addr;
        current_addr += tensor_size(1, 1, 4096);
        apply_latency_params(fc);
        compiler.add_layer(fc);
    }

    // FC3: 4096 → num_classes
    {
        LayerSpec fc;
        fc.name = "fc3";
        fc.type = FuncType::FullyConnected;
        fc.stride = 1; fc.padding = 0; fc.dilation = 1;
        fc.input_height = 1; fc.input_width = 1; fc.input_channels = 4096;
        fc.kernel_height = 1; fc.kernel_width = 1;
        fc.kernel_channels = 4096; fc.kernel_count = config.num_classes;
        fc.input_address = current_addr;
        current_addr += tensor_size(1, 1, 4096);
        fc.weight_address = current_addr;
        current_addr += 4096 * config.num_classes * sizeof(float);
        fc.output_address = current_addr;
        current_addr += tensor_size(1, 1, config.num_classes);
        apply_latency_params(fc);
        compiler.add_layer(fc);
    }

    return compiler.compile();
}

DnnImage DnnImageGenerator::generate_vgg19(const NetworkConfig& config) {
    layer_sequence_index_ = 0;
    // Stub - similar to VGG16 but with more layers per block
    return generate_vgg16(config);
}

// ============================================================================
// MobileNet-V1 (29 layers: 1 conv_stem + 13×2 DW-Sep + avgpool + fc)
// ============================================================================

DnnImage DnnImageGenerator::generate_mobilenet_v1(const NetworkConfig& config) {
    layer_sequence_index_ = 0;
    DnnImageCompiler compiler;
    uint32_t current_addr = config.base_dram_addr;

    uint32_t in_h = config.input_height;
    uint32_t in_w = config.input_width;
    uint32_t in_c = config.input_channels;

    // conv_stem: Conv 3×3, 32, stride=2, padding=1
    {
        LayerSpec conv;
        conv.name = "conv_stem";
        conv.type = FuncType::Conv2D;
        conv.stride = 2; conv.padding = 1; conv.dilation = 1;
        conv.input_height = in_h; conv.input_width = in_w; conv.input_channels = in_c;
        conv.kernel_height = 3; conv.kernel_width = 3;
        conv.kernel_channels = in_c; conv.kernel_count = 32;
        conv.input_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);
        conv.weight_address = current_addr;
        current_addr += 3 * 3 * in_c * 32 * sizeof(float);
        in_h = (in_h + 2 * 1 - 3) / 2 + 1;
        in_w = (in_w + 2 * 1 - 3) / 2 + 1;
        in_c = 32;
        conv.output_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);
        apply_latency_params(conv);
        compiler.add_layer(conv);
    }

    // 13 Depthwise Separable blocks (2 layers each = 26 layers)
    struct DwsConfig { uint32_t out_c; uint32_t stride; };
    DwsConfig blocks[] = {
        {64,   1},  {128,  2},  {128,  1},  {256,  2},
        {256,  1},  {512,  2},  {512,  1},  {512,  1},
        {512,  1},  {512,  1},  {512,  1},  {1024, 2},
        {1024, 1},
    };

    for (const auto& blk : blocks) {
        add_depthwise_separable_block(compiler, current_addr,
                                       in_h, in_w, in_c,
                                       blk.out_c, blk.stride);
    }

    // avgpool (global)
    {
        LayerSpec pool;
        pool.name = "avgpool";
        pool.type = FuncType::Pooling;
        pool.stride = 1; pool.padding = 0; pool.dilation = 1;
        pool.input_height = in_h; pool.input_width = in_w; pool.input_channels = in_c;
        pool.kernel_height = in_h; pool.kernel_width = in_w;
        pool.kernel_channels = 0; pool.kernel_count = 0;
        pool.input_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);
        pool.weight_address = 0;
        in_h = 1; in_w = 1;
        pool.output_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);
        apply_latency_params(pool);
        compiler.add_layer(pool);
    }

    // fc
    {
        LayerSpec fc;
        fc.name = "fc";
        fc.type = FuncType::FullyConnected;
        fc.stride = 1; fc.padding = 0; fc.dilation = 1;
        fc.input_height = 1; fc.input_width = 1; fc.input_channels = in_c;
        fc.kernel_height = 1; fc.kernel_width = 1;
        fc.kernel_channels = in_c; fc.kernel_count = config.num_classes;
        fc.input_address = current_addr;
        current_addr += tensor_size(1, 1, in_c);
        fc.weight_address = current_addr;
        current_addr += in_c * config.num_classes * sizeof(float);
        fc.output_address = current_addr;
        current_addr += tensor_size(1, 1, config.num_classes);
        apply_latency_params(fc);
        compiler.add_layer(fc);
    }

    return compiler.compile();
}

// ============================================================================
// MobileNet-V2 (54 layers)
// ============================================================================

DnnImage DnnImageGenerator::generate_mobilenet_v2(const NetworkConfig& config) {
    layer_sequence_index_ = 0;
    DnnImageCompiler compiler;
    uint32_t current_addr = config.base_dram_addr;

    uint32_t in_h = config.input_height;
    uint32_t in_w = config.input_width;
    uint32_t in_c = config.input_channels;

    // 1. Initial Conv2D 3x3, stride=2, padding=1 (3 -> 32, 224 -> 112)
    {
        LayerSpec conv;
        conv.name = "conv_stem";
        conv.type = FuncType::Conv2D;
        conv.stride = 2;
        conv.padding = 1;
        conv.dilation = 1;

        conv.input_height = in_h;
        conv.input_width = in_w;
        conv.input_channels = in_c;

        conv.kernel_height = 3;
        conv.kernel_width = 3;
        conv.kernel_channels = in_c;
        conv.kernel_count = 32;

        conv.input_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);
        conv.weight_address = current_addr;
        current_addr += 3 * 3 * in_c * 32 * sizeof(float);

        in_h = (in_h + 2 * 1 - 3) / 2 + 1;
        in_w = (in_w + 2 * 1 - 3) / 2 + 1;
        in_c = 32;

        conv.output_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);

        apply_latency_params(conv);
        compiler.add_layer(conv);
    }

    // 2-8. MobileNetV2 inverted residual blocks
    struct InvResConfig { uint32_t t; uint32_t c; uint32_t n; uint32_t s; };
    InvResConfig blocks[] = {
        {1,  16,  1, 1},
        {6,  24,  2, 2},
        {6,  32,  3, 2},
        {6,  64,  4, 2},
        {6,  96,  3, 1},
        {6, 160,  3, 2},
        {6, 320,  1, 1},
    };

    for (const auto& blk : blocks) {
        for (uint32_t i = 0; i < blk.n; ++i) {
            uint32_t s = (i == 0) ? blk.s : 1;
            add_inverted_residual_block(compiler, current_addr,
                                        in_h, in_w, in_c,
                                        blk.c, blk.t, s);
        }
    }

    // 9. Final Conv1x1 (320 -> 1280)
    {
        LayerSpec conv;
        conv.name = "conv_head";
        conv.type = FuncType::Conv2D;
        conv.stride = 1;
        conv.padding = 0;
        conv.dilation = 1;

        conv.input_height = in_h;
        conv.input_width = in_w;
        conv.input_channels = in_c;

        conv.kernel_height = 1;
        conv.kernel_width = 1;
        conv.kernel_channels = in_c;
        conv.kernel_count = 1280;

        conv.input_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);
        conv.weight_address = current_addr;
        current_addr += 1 * 1 * in_c * 1280 * sizeof(float);
        conv.output_address = current_addr;
        current_addr += tensor_size(in_h, in_w, 1280);

        in_c = 1280;

        apply_latency_params(conv);
        compiler.add_layer(conv);
    }

    // 10. AvgPool (global average pooling)
    {
        LayerSpec pool;
        pool.name = "avgpool";
        pool.type = FuncType::Pooling;
        pool.stride = 1;
        pool.padding = 0;
        pool.dilation = 1;

        pool.input_height = in_h;
        pool.input_width = in_w;
        pool.input_channels = in_c;

        pool.kernel_height = in_h;
        pool.kernel_width = in_w;
        pool.kernel_channels = 0;
        pool.kernel_count = 0;

        pool.input_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);
        pool.weight_address = 0;

        in_h = 1;
        in_w = 1;

        pool.output_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);

        apply_latency_params(pool);
        compiler.add_layer(pool);
    }

    // 11. Fully Connected (1280 -> num_classes)
    {
        LayerSpec fc;
        fc.name = "fc";
        fc.type = FuncType::FullyConnected;
        fc.stride = 1;
        fc.padding = 0;
        fc.dilation = 1;

        fc.input_height = 1;
        fc.input_width = 1;
        fc.input_channels = in_c;

        fc.kernel_height = 1;
        fc.kernel_width = 1;
        fc.kernel_channels = in_c;
        fc.kernel_count = config.num_classes;

        fc.input_address = current_addr;
        current_addr += tensor_size(1, 1, in_c);
        fc.weight_address = current_addr;
        current_addr += in_c * config.num_classes * sizeof(float);
        fc.output_address = current_addr;
        current_addr += tensor_size(1, 1, config.num_classes);

        apply_latency_params(fc);
        compiler.add_layer(fc);
    }

    return compiler.compile();
}

// ============================================================================
// ResNet-18 (23 layers)
// ============================================================================

DnnImage DnnImageGenerator::generate_resnet18(const NetworkConfig& config) {
    layer_sequence_index_ = 0;
    DnnImageCompiler compiler;
    uint32_t current_addr = config.base_dram_addr;

    uint32_t in_h = config.input_height;
    uint32_t in_w = config.input_width;
    uint32_t in_c = config.input_channels;

    // conv1: Conv 7×7, 64, stride=2, padding=3
    {
        LayerSpec conv;
        conv.name = "conv1";
        conv.type = FuncType::Conv2D;
        conv.stride = 2; conv.padding = 3; conv.dilation = 1;
        conv.input_height = in_h; conv.input_width = in_w; conv.input_channels = in_c;
        conv.kernel_height = 7; conv.kernel_width = 7;
        conv.kernel_channels = in_c; conv.kernel_count = 64;
        conv.input_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);
        conv.weight_address = current_addr;
        current_addr += 7 * 7 * in_c * 64 * sizeof(float);
        in_h = (in_h + 2 * 3 - 7) / 2 + 1;
        in_w = (in_w + 2 * 3 - 7) / 2 + 1;
        in_c = 64;
        conv.output_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);
        apply_latency_params(conv);
        compiler.add_layer(conv);
    }

    // maxpool: Pool 3×3, stride=2, padding=1
    {
        LayerSpec pool;
        pool.name = "maxpool";
        pool.type = FuncType::Pooling;
        pool.stride = 2; pool.padding = 1; pool.dilation = 1;
        pool.input_height = in_h; pool.input_width = in_w; pool.input_channels = in_c;
        pool.kernel_height = 3; pool.kernel_width = 3;
        pool.kernel_channels = 0; pool.kernel_count = 0;
        pool.input_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);
        pool.weight_address = 0;
        in_h = (in_h + 2 * 1 - 3) / 2 + 1;
        in_w = (in_w + 2 * 1 - 3) / 2 + 1;
        pool.output_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);
        apply_latency_params(pool);
        compiler.add_layer(pool);
    }

    // layer1: 2× BasicBlock(64), no downsample (4 layers)
    add_residual_block(compiler, current_addr, in_h, in_w, in_c, 64, false);
    add_residual_block(compiler, current_addr, in_h, in_w, in_c, 64, false);

    // layer2: 2× BasicBlock(128), first downsample (5 layers)
    add_residual_block(compiler, current_addr, in_h, in_w, in_c, 128, true);
    add_residual_block(compiler, current_addr, in_h, in_w, in_c, 128, false);

    // layer3: 2× BasicBlock(256), first downsample (5 layers)
    add_residual_block(compiler, current_addr, in_h, in_w, in_c, 256, true);
    add_residual_block(compiler, current_addr, in_h, in_w, in_c, 256, false);

    // layer4: 2× BasicBlock(512), first downsample (5 layers)
    add_residual_block(compiler, current_addr, in_h, in_w, in_c, 512, true);
    add_residual_block(compiler, current_addr, in_h, in_w, in_c, 512, false);

    // avgpool (global)
    {
        LayerSpec pool;
        pool.name = "avgpool";
        pool.type = FuncType::Pooling;
        pool.stride = 1; pool.padding = 0; pool.dilation = 1;
        pool.input_height = in_h; pool.input_width = in_w; pool.input_channels = in_c;
        pool.kernel_height = in_h; pool.kernel_width = in_w;
        pool.kernel_channels = 0; pool.kernel_count = 0;
        pool.input_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);
        pool.weight_address = 0;
        in_h = 1; in_w = 1;
        pool.output_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);
        apply_latency_params(pool);
        compiler.add_layer(pool);
    }

    // fc
    {
        LayerSpec fc;
        fc.name = "fc";
        fc.type = FuncType::FullyConnected;
        fc.stride = 1; fc.padding = 0; fc.dilation = 1;
        fc.input_height = 1; fc.input_width = 1; fc.input_channels = in_c;
        fc.kernel_height = 1; fc.kernel_width = 1;
        fc.kernel_channels = in_c; fc.kernel_count = config.num_classes;
        fc.input_address = current_addr;
        current_addr += tensor_size(1, 1, in_c);
        fc.weight_address = current_addr;
        current_addr += in_c * config.num_classes * sizeof(float);
        fc.output_address = current_addr;
        current_addr += tensor_size(1, 1, config.num_classes);
        apply_latency_params(fc);
        compiler.add_layer(fc);
    }

    return compiler.compile();
}

DnnImage DnnImageGenerator::generate_resnet50(const NetworkConfig& config) {
    layer_sequence_index_ = 0;
    // Stub - bottleneck residual blocks
    return generate_resnet18(config);
}

// ============================================================================
// Helper: VGG Block (N conv3x3 + pool2x2)
// ============================================================================

void DnnImageGenerator::add_vgg_block(
    DnnImageCompiler& compiler,
    uint32_t& current_addr,
    uint32_t& in_h, uint32_t& in_w,
    uint32_t& in_c, uint32_t out_c, uint32_t num_conv_layers) {

    uint32_t h = in_h;
    uint32_t w = in_w;
    uint32_t c = in_c;

    // Add num_conv_layers Conv2D layers
    for (uint32_t i = 0; i < num_conv_layers; ++i) {
        LayerSpec conv;
        conv.name = "vgg_conv";
        conv.type = FuncType::Conv2D;
        conv.stride = 1;
        conv.padding = 1;
        conv.dilation = 1;

        conv.input_height = h;
        conv.input_width = w;
        conv.input_channels = c;

        conv.kernel_height = 3;
        conv.kernel_width = 3;
        conv.kernel_channels = c;
        conv.kernel_count = out_c;

        conv.input_address = current_addr;
        current_addr += tensor_size(h, w, c);

        conv.weight_address = current_addr;
        current_addr += 3 * 3 * c * out_c * sizeof(float);

        conv.output_address = current_addr;
        current_addr += tensor_size(h, w, out_c);

        apply_latency_params(conv);
        compiler.add_layer(conv);

        c = out_c; // Update channels for next layer
    }

    // Add pooling layer
    LayerSpec pool;
    pool.name = "vgg_pool";
    pool.type = FuncType::Pooling;
    pool.stride = 2;
    pool.padding = 0;
    pool.dilation = 1;

    pool.input_height = h;
    pool.input_width = w;
    pool.input_channels = out_c;

    pool.kernel_height = 2;
    pool.kernel_width = 2;
    pool.kernel_channels = out_c;
    pool.kernel_count = out_c;

    pool.input_address = current_addr - tensor_size(h, w, out_c);
    pool.weight_address = 0;

    h = (h - 2) / 2 + 1;
    w = (w - 2) / 2 + 1;

    pool.output_address = current_addr;
    current_addr += tensor_size(h, w, out_c);

    apply_latency_params(pool);
    compiler.add_layer(pool);

    // Update caller's variables
    in_h = h;
    in_w = w;
    in_c = out_c;
}

// ============================================================================
// Helper: Depthwise Separable Block (DW 3x3 + PW 1x1)
// ============================================================================

void DnnImageGenerator::add_depthwise_separable_block(
    DnnImageCompiler& compiler,
    uint32_t& current_addr,
    uint32_t& in_h, uint32_t& in_w, uint32_t& in_c,
    uint32_t out_channels, uint32_t stride) {

    uint32_t dw_out_h = (in_h + 2 * 1 - 3) / stride + 1;
    uint32_t dw_out_w = (in_w + 2 * 1 - 3) / stride + 1;

    // 1. Depthwise Conv 3x3
    LayerSpec dw;
    dw.name = "dw_conv3x3";
    dw.type = FuncType::DepthwiseConv2D;
    dw.stride = stride;
    dw.padding = 1;
    dw.dilation = 1;

    dw.input_height = in_h;
    dw.input_width = in_w;
    dw.input_channels = in_c;

    dw.kernel_height = 3;
    dw.kernel_width = 3;
    dw.kernel_channels = in_c;
    dw.kernel_count = 1;

    dw.input_address = current_addr;
    current_addr += tensor_size(in_h, in_w, in_c);
    dw.weight_address = current_addr;
    current_addr += 3 * 3 * in_c * sizeof(float);
    dw.output_address = current_addr;
    current_addr += tensor_size(dw_out_h, dw_out_w, in_c);

    apply_latency_params(dw);
    compiler.add_layer(dw);

    // 2. Pointwise Conv 1x1
    LayerSpec pw;
    pw.name = "pw_conv1x1";
    pw.type = FuncType::Conv2D;
    pw.stride = 1;
    pw.padding = 0;
    pw.dilation = 1;

    pw.input_height = dw_out_h;
    pw.input_width = dw_out_w;
    pw.input_channels = in_c;

    pw.kernel_height = 1;
    pw.kernel_width = 1;
    pw.kernel_channels = in_c;
    pw.kernel_count = out_channels;

    pw.input_address = current_addr;
    current_addr += tensor_size(dw_out_h, dw_out_w, in_c);
    pw.weight_address = current_addr;
    current_addr += 1 * 1 * in_c * out_channels * sizeof(float);
    pw.output_address = current_addr;
    current_addr += tensor_size(dw_out_h, dw_out_w, out_channels);

    apply_latency_params(pw);
    compiler.add_layer(pw);

    // Update caller's variables
    in_h = dw_out_h;
    in_w = dw_out_w;
    in_c = out_channels;
}

// ============================================================================
// Helper: Inverted Residual Block (MobileNetV2)
// ============================================================================

void DnnImageGenerator::add_inverted_residual_block(
    DnnImageCompiler& compiler,
    uint32_t& current_addr,
    uint32_t& in_h, uint32_t& in_w, uint32_t& in_c,
    uint32_t out_c, uint32_t expansion_factor, uint32_t stride) {

    uint32_t expanded_c = in_c * expansion_factor;

    // 1. Expand Conv1x1 (only when expansion_factor > 1)
    if (expansion_factor > 1) {
        LayerSpec expand;
        expand.name = "invres_expand_1x1";
        expand.type = FuncType::Conv2D;
        expand.stride = 1;
        expand.padding = 0;
        expand.dilation = 1;

        expand.input_height = in_h;
        expand.input_width = in_w;
        expand.input_channels = in_c;

        expand.kernel_height = 1;
        expand.kernel_width = 1;
        expand.kernel_channels = in_c;
        expand.kernel_count = expanded_c;

        expand.input_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);
        expand.weight_address = current_addr;
        current_addr += 1 * 1 * in_c * expanded_c * sizeof(float);
        expand.output_address = current_addr;
        current_addr += tensor_size(in_h, in_w, expanded_c);

        apply_latency_params(expand);
        compiler.add_layer(expand);
    }

    // 2. Depthwise Conv 3x3
    uint32_t dw_out_h = (in_h + 2 * 1 - 3) / stride + 1;
    uint32_t dw_out_w = (in_w + 2 * 1 - 3) / stride + 1;

    LayerSpec dw;
    dw.name = "invres_dw3x3";
    dw.type = FuncType::DepthwiseConv2D;
    dw.stride = stride;
    dw.padding = 1;
    dw.dilation = 1;

    dw.input_height = in_h;
    dw.input_width = in_w;
    dw.input_channels = expanded_c;

    dw.kernel_height = 3;
    dw.kernel_width = 3;
    dw.kernel_channels = expanded_c;
    dw.kernel_count = 1;

    dw.input_address = current_addr;
    current_addr += tensor_size(in_h, in_w, expanded_c);
    dw.weight_address = current_addr;
    current_addr += 3 * 3 * expanded_c * sizeof(float);
    dw.output_address = current_addr;
    current_addr += tensor_size(dw_out_h, dw_out_w, expanded_c);

    apply_latency_params(dw);
    compiler.add_layer(dw);

    // Update spatial dimensions after stride
    in_h = dw_out_h;
    in_w = dw_out_w;

    // 3. Project Conv1x1
    LayerSpec project;
    project.name = "invres_project_1x1";
    project.type = FuncType::Conv2D;
    project.stride = 1;
    project.padding = 0;
    project.dilation = 1;

    project.input_height = in_h;
    project.input_width = in_w;
    project.input_channels = expanded_c;

    project.kernel_height = 1;
    project.kernel_width = 1;
    project.kernel_channels = expanded_c;
    project.kernel_count = out_c;

    project.input_address = current_addr;
    current_addr += tensor_size(in_h, in_w, expanded_c);
    project.weight_address = current_addr;
    current_addr += 1 * 1 * expanded_c * out_c * sizeof(float);
    project.output_address = current_addr;
    current_addr += tensor_size(in_h, in_w, out_c);

    apply_latency_params(project);
    compiler.add_layer(project);

    // Update channels
    in_c = out_c;
}

// ============================================================================
// Helper: Residual Block (ResNet BasicBlock: 2× Conv3x3 + opt shortcut)
// ============================================================================

void DnnImageGenerator::add_residual_block(
    DnnImageCompiler& compiler,
    uint32_t& current_addr,
    uint32_t& in_h, uint32_t& in_w,
    uint32_t& in_c, uint32_t out_c, bool downsample) {

    uint32_t stride = downsample ? 2 : 1;

    // Optional shortcut Conv1x1 (when downsample or channel change)
    if (downsample || in_c != out_c) {
        LayerSpec shortcut;
        shortcut.name = "res_shortcut_1x1";
        shortcut.type = FuncType::Conv2D;
        shortcut.stride = stride;
        shortcut.padding = 0;
        shortcut.dilation = 1;

        shortcut.input_height = in_h;
        shortcut.input_width = in_w;
        shortcut.input_channels = in_c;

        shortcut.kernel_height = 1;
        shortcut.kernel_width = 1;
        shortcut.kernel_channels = in_c;
        shortcut.kernel_count = out_c;

        shortcut.input_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);
        shortcut.weight_address = current_addr;
        current_addr += 1 * 1 * in_c * out_c * sizeof(float);

        uint32_t sh_h = (in_h + 2 * 0 - 1) / stride + 1;
        uint32_t sh_w = (in_w + 2 * 0 - 1) / stride + 1;

        shortcut.output_address = current_addr;
        current_addr += tensor_size(sh_h, sh_w, out_c);

        apply_latency_params(shortcut);
        compiler.add_layer(shortcut);
    }

    // Conv3x3 #1
    {
        uint32_t out_h = (in_h + 2 * 1 - 3) / stride + 1;
        uint32_t out_w = (in_w + 2 * 1 - 3) / stride + 1;

        LayerSpec conv1;
        conv1.name = "res_conv3x3_1";
        conv1.type = FuncType::Conv2D;
        conv1.stride = stride;
        conv1.padding = 1;
        conv1.dilation = 1;

        conv1.input_height = in_h;
        conv1.input_width = in_w;
        conv1.input_channels = in_c;

        conv1.kernel_height = 3;
        conv1.kernel_width = 3;
        conv1.kernel_channels = in_c;
        conv1.kernel_count = out_c;

        conv1.input_address = current_addr;
        current_addr += tensor_size(in_h, in_w, in_c);
        conv1.weight_address = current_addr;
        current_addr += 3 * 3 * in_c * out_c * sizeof(float);

        conv1.output_address = current_addr;
        current_addr += tensor_size(out_h, out_w, out_c);

        apply_latency_params(conv1);
        compiler.add_layer(conv1);

        in_h = out_h;
        in_w = out_w;
    }

    // Conv3x3 #2 (stride=1)
    {
        LayerSpec conv2;
        conv2.name = "res_conv3x3_2";
        conv2.type = FuncType::Conv2D;
        conv2.stride = 1;
        conv2.padding = 1;
        conv2.dilation = 1;

        conv2.input_height = in_h;
        conv2.input_width = in_w;
        conv2.input_channels = out_c;

        conv2.kernel_height = 3;
        conv2.kernel_width = 3;
        conv2.kernel_channels = out_c;
        conv2.kernel_count = out_c;

        conv2.input_address = current_addr;
        current_addr += tensor_size(in_h, in_w, out_c);
        conv2.weight_address = current_addr;
        current_addr += 3 * 3 * out_c * out_c * sizeof(float);

        conv2.output_address = current_addr;
        current_addr += tensor_size(in_h, in_w, out_c);

        apply_latency_params(conv2);
        compiler.add_layer(conv2);
    }

    in_c = out_c;
}

// ============================================================================
// Utility
// ============================================================================

uint32_t DnnImageGenerator::tensor_size(uint32_t h, uint32_t w, uint32_t c) const {
    return h * w * c * sizeof(float);
}

// ============================================================================
// set_tiled_latency_params (Mode A: tile-based derivation)
// ============================================================================

void DnnImageGenerator::set_tiled_latency_params(LayerSpec& spec,
                                                   uint32_t tile_height,
                                                   uint32_t tile_width,
                                                   uint32_t l2_buffer_kb,
                                                   Dataflow dataflow,
                                                   PeConfig pe_config) const {
    pe_config = sanitize_pe_config(pe_config);

    // Calculate output dimensions
    uint32_t out_h, out_w, out_c;

    if (spec.type == FuncType::Conv2D || spec.type == FuncType::DepthwiseConv2D || spec.type == FuncType::Pooling) {
        uint32_t effective_kernel_h = spec.kernel_height + (spec.kernel_height - 1) * (spec.dilation - 1);
        uint32_t effective_kernel_w = spec.kernel_width + (spec.kernel_width - 1) * (spec.dilation - 1);

        out_h = (spec.input_height + 2 * spec.padding - effective_kernel_h) / spec.stride + 1;
        out_w = (spec.input_width + 2 * spec.padding - effective_kernel_w) / spec.stride + 1;
        out_c = (spec.type == FuncType::DepthwiseConv2D || spec.type == FuncType::Pooling)
                ? spec.input_channels : spec.kernel_count;
    } else if (spec.type == FuncType::FullyConnected) {
        out_h = 1;
        out_w = 1;
        out_c = spec.kernel_count;
    } else if (spec.type == FuncType::MatMul) {
        // MatMul: output is (input_height × kernel_count), reduction dim = kernel_channels
        // QKt: out = (L × L), n = d_k;  AV: out = (L × d_k), n = L
        out_h = spec.input_height;
        out_w = spec.input_width;
        out_c = spec.kernel_count > 0 ? spec.kernel_count : spec.input_channels;
    } else {
        // Activation, ElementWise, or unknown — fall back to default mode
        set_default_latency_params(spec, pe_config);
        return;
    }

    // Per-output ops from layer type
    uint32_t ops_per_output = estimate_per_output_ops(spec);
    spec.ops_per_output = ops_per_output;

    // passes_per_output = ceil(ops_per_output / ops_per_pass)
    uint32_t passes_per_output = (ops_per_output + pe_config.ops_per_pass - 1) / pe_config.ops_per_pass;

    // write_output = passes_per_output * output_element_count
    spec.write_output = passes_per_output * out_h * out_w * out_c;

    // Dataflow-aware base traffic
    if (spec.type == FuncType::MatMul) {
        // MatMul: no tile schedule needed. Both operands loaded once.
        // I = first operand (e.g., query rows), W = second operand (e.g., key rows)
        spec.fetch_operand0 = spec.input_height * spec.input_width * spec.input_channels;
        spec.fetch_operand1 = (spec.kernel_channels > 0 && spec.kernel_count > 0)
            ? spec.input_height * spec.kernel_channels * spec.kernel_count
            : spec.fetch_operand0;  // symmetric fallback
    } else {
        // Conv/DW/FC/Pool: compute tile schedule (Mode A)
        TileSchedule schedule = compute_tile_schedule(
            spec.name,
            0,  // layer_id (not critical for latency calc)
            out_h, out_w, out_c,
            spec.input_channels,
            spec.kernel_height,
            spec.kernel_width,
            spec.stride,
            spec.padding,
            l2_buffer_kb,
            tile_height,
            tile_width,
            0, 0, 0,  // budgets: legacy split (unchanged default)
            spec.type == FuncType::DepthwiseConv2D
        );

        if (dataflow == Dataflow::WS) {
            // Weight Stationary: kernel loaded once (unique), input refetched per tile
            spec.fetch_operand1 = spec.kernel_height * spec.kernel_width *
                                          spec.kernel_channels * spec.kernel_count;
            spec.fetch_operand0 = static_cast<uint32_t>(
                schedule.total_layer_input_bytes / sizeof(float));
        } else if (dataflow == Dataflow::IS) {
            // Input Stationary: input resident, read once — but at the array-feed
            // (im2col) size P·Q·C·R·S, NOT the halo-compacted physical size
            // H·W·C. The resident-operand feed is its im2col footprint; weight
            // (WS case below) has no halo so its im2col == physical, which is why
            // only the input was inconsistent. This unifies the two: a resident
            // operand always counts its im2col feed.
            spec.fetch_operand0 = static_cast<uint32_t>(
                static_cast<uint64_t>(out_h) * out_w * spec.input_channels *
                spec.kernel_height * spec.kernel_width);
            spec.fetch_operand1 = static_cast<uint32_t>(
                schedule.total_layer_kernel_bytes / sizeof(float));
        } else { // OS
            // Output Stationary: both refetched per tile
            spec.fetch_operand0 = static_cast<uint32_t>(
                schedule.total_layer_input_bytes / sizeof(float));
            spec.fetch_operand1 = static_cast<uint32_t>(
                schedule.total_layer_kernel_bytes / sizeof(float));
        }

    }

    // Derive spill_ratio: partial-sum writeback fraction
    // spill_ratio = n_wb / (passes_per_output - parallel_passes), where n_wb depends on local psum capacity vs concurrent outputs
    uint64_t N_O = static_cast<uint64_t>(out_h) * out_w * out_c;
    float delta = 0.0f;
    if (passes_per_output > pe_config.parallel_passes) {
        uint32_t psum_capacity_elems = (pe_config.partial_sum_buffer_kb * 1024) / sizeof(float);
        // Number of concurrent outputs in one tile
        uint32_t concurrent = static_cast<uint32_t>(
            std::min(N_O, static_cast<uint64_t>(pe_config.n_max())));
        if (psum_capacity_elems == 0) {
            delta = 1.0f;  // no local storage: full writeback
        } else if (concurrent > psum_capacity_elems) {
            // Excess outputs must writeback each pass transition
            delta = 1.0f - static_cast<float>(psum_capacity_elems) / concurrent;
        }
        // else: all concurrent outputs fit, spill_ratio=0
    }

    // Writeback-reload traffic
    // fetch_operand0 += spill_ratio * beta_input * (passes_per_output - parallel_passes) * output_element_count (and similarly for weight)
    if (delta > 0.0f && passes_per_output > pe_config.parallel_passes) {
        uint64_t wb_total = static_cast<uint64_t>(passes_per_output - pe_config.parallel_passes) * N_O;
        uint64_t wb_scaled = static_cast<uint64_t>(static_cast<double>(wb_total) * delta);
        if (dataflow == Dataflow::WS) {
            spec.fetch_operand0 += static_cast<uint32_t>(
                std::min(wb_scaled, static_cast<uint64_t>(UINT32_MAX)));
        } else if (dataflow == Dataflow::IS) {
            spec.fetch_operand1 += static_cast<uint32_t>(
                std::min(wb_scaled, static_cast<uint64_t>(UINT32_MAX)));
        } else if (dataflow == Dataflow::OS) {
            // OS: outputs pinned, both inputs and weights resent
            spec.fetch_operand0 += static_cast<uint32_t>(
                std::min(wb_scaled, static_cast<uint64_t>(UINT32_MAX)));
            spec.fetch_operand1 += static_cast<uint32_t>(
                std::min(wb_scaled, static_cast<uint64_t>(UINT32_MAX)));
        }
    }

    // Per-layer-type threshold defaults
    switch (spec.type) {
        case FuncType::Conv2D:
            spec.prefetch_in  = spec.kernel_height * spec.kernel_width * spec.input_channels;
            spec.prefetch_wt = spec.kernel_height * spec.kernel_width * spec.input_channels;
            break;
        case FuncType::DepthwiseConv2D:
            spec.prefetch_in  = spec.kernel_height * spec.kernel_width;
            spec.prefetch_wt = spec.kernel_height * spec.kernel_width;
            break;
        case FuncType::FullyConnected:
            spec.prefetch_in  = spec.input_channels;
            spec.prefetch_wt = spec.input_channels;
            break;
        case FuncType::Pooling:
            spec.prefetch_in  = spec.kernel_height * spec.kernel_width;
            spec.prefetch_wt = 0;
            break;
        case FuncType::MatMul:
            spec.prefetch_in  = spec.kernel_channels;
            spec.prefetch_wt = spec.kernel_channels;
            break;
        default:
            spec.prefetch_in  = 1;
            spec.prefetch_wt = 1;
            break;
    }
    if (spec.prefetch_in == 0) spec.prefetch_in = 1;
    if (spec.prefetch_wt == 0 && spec.fetch_operand1 > 0) spec.prefetch_wt = 1;

    // n_max = parallel_passes * num_output_lanes;
    // n_min = parallel_passes.
    spec.n_max = pe_config.n_max();
    spec.n_min = pe_config.n_min();

    // Pipeline depth from PE type
    if (pe_config.pe_type == PeType::Systolic) {
        spec.issue_latency = pe_config.ops_per_pass;
    } else { // AdderTree: issue_latency = 1 + ceil(log2(ops_per_pass))
        uint32_t log2k = 0;
        uint32_t val = pe_config.ops_per_pass;
        while (val > 1) { val = (val + 1) / 2; log2k++; }
        spec.issue_latency = 1 + log2k;
    }
    // Layer-end overhead is a descriptor field fed by the CSV oracle
    // (post_cycles column); the generator path leaves it at zero.
    spec.post_completion_cycles = 0;
}

// ============================================================================
// set_default_latency_params (non-tiled fallback)
// ============================================================================

void DnnImageGenerator::set_default_latency_params(LayerSpec& spec,
                                                    const PeConfig& pe_config) const {
    PeConfig pe = sanitize_pe_config(pe_config);

    // Calculate output dimensions
    uint32_t out_h, out_w, out_c;

    if (spec.type == FuncType::Conv2D || spec.type == FuncType::DepthwiseConv2D || spec.type == FuncType::Pooling) {
        uint32_t effective_kernel_h = spec.kernel_height + (spec.kernel_height - 1) * (spec.dilation - 1);
        uint32_t effective_kernel_w = spec.kernel_width + (spec.kernel_width - 1) * (spec.dilation - 1);

        out_h = (spec.input_height + 2 * spec.padding - effective_kernel_h) / spec.stride + 1;
        out_w = (spec.input_width + 2 * spec.padding - effective_kernel_w) / spec.stride + 1;
        out_c = (spec.type == FuncType::DepthwiseConv2D || spec.type == FuncType::Pooling)
                ? spec.input_channels : spec.kernel_count;
    } else if (spec.type == FuncType::FullyConnected) {
        out_h = 1;
        out_w = 1;
        out_c = spec.kernel_count;
    } else {
        out_h = spec.input_height;
        out_w = spec.input_width;
        out_c = spec.input_channels;
    }

    // Per-output ops from layer type
    uint32_t ops_per_output = estimate_per_output_ops(spec);
    spec.ops_per_output = ops_per_output;

    // passes_per_output = ceil(ops_per_output / ops_per_pass)
    uint32_t passes_per_output = (ops_per_output + pe.ops_per_pass - 1) / pe.ops_per_pass;

    // write_output = passes_per_output * output_element_count
    spec.write_output = passes_per_output * out_h * out_w * out_c;

    // Input/kernel operands (default: no tiling, full operand count)
    if (spec.type == FuncType::Conv2D || spec.type == FuncType::DepthwiseConv2D) {
        spec.fetch_operand0 = spec.input_height * spec.input_width * spec.input_channels;
        spec.fetch_operand1 = spec.kernel_height * spec.kernel_width *
                                     spec.kernel_channels * spec.kernel_count;
    } else if (spec.type == FuncType::FullyConnected) {
        spec.fetch_operand0 = spec.input_height * spec.input_width * spec.input_channels;
        spec.fetch_operand1 = spec.kernel_channels * spec.kernel_count;
    } else {
        spec.fetch_operand0 = spec.input_height * spec.input_width * spec.input_channels;
        spec.fetch_operand1 = 0; // No kernel for pooling/activation/elementwise
    }

    // Derive spill_ratio and writeback-reload traffic
    uint64_t N_O = static_cast<uint64_t>(out_h) * out_w * out_c;
    if (passes_per_output > pe.parallel_passes) {
        uint32_t psum_cap = (pe.partial_sum_buffer_kb * 1024) / sizeof(float);
        uint32_t concurrent = static_cast<uint32_t>(
            std::min(N_O, static_cast<uint64_t>(pe.n_max())));
        float delta = 0.0f;
        if (psum_cap == 0) {
            delta = 1.0f;
        } else if (concurrent > psum_cap) {
            delta = 1.0f - static_cast<float>(psum_cap) / concurrent;
        }
        if (delta > 0.0f) {
            uint64_t wb_total = static_cast<uint64_t>(passes_per_output - pe.parallel_passes) * N_O;
            uint64_t wb_scaled = static_cast<uint64_t>(static_cast<double>(wb_total) * delta);
            Dataflow df = configured_ ? dataflow_ : Dataflow::WS;
            if (df == Dataflow::WS) {
                spec.fetch_operand0 += static_cast<uint32_t>(
                    std::min(wb_scaled, static_cast<uint64_t>(UINT32_MAX)));
            } else if (df == Dataflow::IS) {
                spec.fetch_operand1 += static_cast<uint32_t>(
                    std::min(wb_scaled, static_cast<uint64_t>(UINT32_MAX)));
            } else if (df == Dataflow::OS) {
                spec.fetch_operand0 += static_cast<uint32_t>(
                    std::min(wb_scaled, static_cast<uint64_t>(UINT32_MAX)));
                spec.fetch_operand1 += static_cast<uint32_t>(
                    std::min(wb_scaled, static_cast<uint64_t>(UINT32_MAX)));
            }
        }
    }

    // Per-layer-type threshold defaults
    switch (spec.type) {
        case FuncType::Conv2D:
            spec.prefetch_in  = spec.kernel_height * spec.kernel_width * spec.input_channels;
            spec.prefetch_wt = spec.kernel_height * spec.kernel_width * spec.input_channels;
            break;
        case FuncType::DepthwiseConv2D:
            spec.prefetch_in  = spec.kernel_height * spec.kernel_width;
            spec.prefetch_wt = spec.kernel_height * spec.kernel_width;
            break;
        case FuncType::FullyConnected:
            spec.prefetch_in  = spec.input_channels;
            spec.prefetch_wt = spec.input_channels;
            break;
        case FuncType::Pooling:
            spec.prefetch_in  = spec.kernel_height * spec.kernel_width;
            spec.prefetch_wt = 0;
            break;
        case FuncType::MatMul:
            spec.prefetch_in  = spec.kernel_channels;
            spec.prefetch_wt = spec.kernel_channels;
            break;
        default:
            spec.prefetch_in  = 1;
            spec.prefetch_wt = 1;
            break;
    }
    if (spec.prefetch_in == 0) spec.prefetch_in = 1;
    if (spec.prefetch_wt == 0 && spec.fetch_operand1 > 0) spec.prefetch_wt = 1;

    // n_max = parallel_passes * num_output_lanes;
    // n_min = parallel_passes.
    spec.n_max = pe.n_max();
    spec.n_min = pe.n_min();

    // Pipeline depth from PE type
    if (pe.pe_type == PeType::Systolic) {
        spec.issue_latency = pe.ops_per_pass;
    } else { // AdderTree: issue_latency = 1 + ceil(log2(ops_per_pass))
        uint32_t log2k = 0;
        uint32_t val = pe.ops_per_pass;
        while (val > 1) { val = (val + 1) / 2; log2k++; }
        spec.issue_latency = 1 + log2k;
    }
    spec.post_completion_cycles = 0;
}

} // namespace flexnpu_sim
