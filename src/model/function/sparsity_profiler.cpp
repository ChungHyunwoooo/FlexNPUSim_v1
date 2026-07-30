/**
 * @file sparsity_profiler.cpp
 * @brief Per-layer activation sparsity profiler implementation
 *
 * Uses the function model (OpenCV DNN backend) for Conv2D and
 * DepthwiseConv2D forward passes. Pooling and FC are implemented
 * directly to avoid weight format complications.
 */

#include "model/function/sparsity_profiler.h"
#include "model/function/function_model_if.h"
#include "compiler/dnn_image/dnn_image_loader.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

namespace flexnpu_sim {

// ============================================================================
// Utilities
// ============================================================================

double SparsityProfiler::compute_sparsity(const Tensor& t) {
    if (t.data.empty()) return 0.0;
    uint32_t zeros = 0;
    for (float v : t.data) {
        if (v == 0.0f) ++zeros;
    }
    return static_cast<double>(zeros) / t.data.size();
}

uint32_t SparsityProfiler::adjust_fi(uint32_t original_fi, double input_sparsity) {
    if (input_sparsity <= 0.0) return original_fi;
    if (input_sparsity >= 1.0) return 0;
    return static_cast<uint32_t>(
        std::round(original_fi * (1.0 - input_sparsity)));
}

// ============================================================================
// Tensor generation helpers
// ============================================================================

/// He initialization: N(0, sqrt(2/fan_in))
static Tensor he_init_tensor(uint32_t batch, uint32_t channels,
                              uint32_t height, uint32_t width,
                              unsigned seed) {
    Tensor t;
    t.batch    = batch;
    t.channels = channels;
    t.height   = height;
    t.width    = width;
    t.data.resize(t.size());

    uint32_t fan_in = channels * height * width;
    float stddev = std::sqrt(2.0f / std::max(fan_in, 1u));

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, stddev);
    for (auto& v : t.data) v = dist(rng);

    return t;
}

/// Random input in [0, 1] (simulates normalized image pixels)
static Tensor random_input_tensor(uint32_t height, uint32_t width,
                                   uint32_t channels, unsigned seed) {
    Tensor t;
    t.batch    = 1;
    t.channels = channels;
    t.height   = height;
    t.width    = width;
    t.data.resize(t.size());

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : t.data) v = dist(rng);

    return t;
}

/// ReLU in-place (avoids OpenCV overhead for activation-only pass)
static void apply_relu_inplace(Tensor& t) {
    for (auto& v : t.data) {
        if (v < 0.0f) v = 0.0f;
    }
}

static void apply_zero_skipping_inplace(Tensor& t, double keep_ratio, unsigned seed) {
    keep_ratio = std::max(0.0, std::min(1.0, keep_ratio));
    if (keep_ratio >= 1.0) return;
    if (keep_ratio <= 0.0) {
        std::fill(t.data.begin(), t.data.end(), 0.0f);
        return;
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (auto& v : t.data) {
        if (dist(rng) > keep_ratio) v = 0.0f;
    }
}

// ============================================================================
// Simple pooling (no OpenCV, handles max/global-avg)
// ============================================================================

/// Max pooling for non-negative inputs (post-ReLU).
/// For non-negative inputs, max and avg pooling produce identical
/// zero/non-zero patterns: output is 0 iff all inputs in the window are 0.
static Tensor simple_pool(const Tensor& input,
                           uint32_t kh, uint32_t kw,
                           uint32_t stride, uint32_t pad) {
    uint32_t out_h, out_w;

    // Global pooling: kernel covers entire spatial extent
    if (kh >= input.height && kw >= input.width) {
        out_h = 1;
        out_w = 1;
    } else {
        out_h = (input.height + 2 * pad - kh) / stride + 1;
        out_w = (input.width  + 2 * pad - kw) / stride + 1;
    }

    Tensor result;
    result.batch    = 1;
    result.channels = input.channels;
    result.height   = out_h;
    result.width    = out_w;
    result.data.resize(result.size(), 0.0f);

    for (uint32_t c = 0; c < result.channels; ++c) {
        for (uint32_t oh = 0; oh < out_h; ++oh) {
            for (uint32_t ow = 0; ow < out_w; ++ow) {
                float val = 0.0f;
                uint32_t count = 0;
                uint32_t sh = (kh >= input.height) ? 0 : oh * stride;
                uint32_t sw = (kw >= input.width)  ? 0 : ow * stride;
                uint32_t eh = (kh >= input.height) ? input.height : kh;
                uint32_t ew = (kw >= input.width)  ? input.width  : kw;

                for (uint32_t ki = 0; ki < eh; ++ki) {
                    for (uint32_t kj = 0; kj < ew; ++kj) {
                        int ih = static_cast<int>(sh + ki) - static_cast<int>(pad);
                        int iw = static_cast<int>(sw + kj) - static_cast<int>(pad);
                        if (ih >= 0 && ih < static_cast<int>(input.height) &&
                            iw >= 0 && iw < static_cast<int>(input.width)) {
                            val = std::max(val, input.at(
                                static_cast<uint32_t>(ih),
                                static_cast<uint32_t>(iw), c));
                            ++count;
                        }
                    }
                }
                result.at(oh, ow, c) = val;
            }
        }
    }
    return result;
}

// ============================================================================
// Simple FC (no OpenCV, generates weights on-the-fly)
// ============================================================================

/// Fully connected layer: output[j] = sum_i(input[i] * W[j,i])
/// Generates He-init weights on-the-fly to avoid large allocations.
static Tensor simple_fc(const Tensor& input, uint32_t out_features,
                         unsigned seed) {
    uint32_t in_features = input.height * input.width * input.channels;

    float stddev = std::sqrt(2.0f / std::max(in_features, 1u));
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, stddev);

    Tensor result;
    result.batch    = 1;
    result.channels = out_features;
    result.height   = 1;
    result.width    = 1;
    result.data.resize(out_features, 0.0f);

    for (uint32_t j = 0; j < out_features; ++j) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < in_features; ++i) {
            sum += input.data[i] * dist(rng);
        }
        result.data[j] = sum;
    }
    return result;
}

// ============================================================================
// SparsityProfiler::profile
// ============================================================================

std::vector<SparsityProfiler::LayerSparsity>
SparsityProfiler::profile(const DnnImage& image,
                            uint32_t input_h, uint32_t input_w,
                            uint32_t input_c, unsigned seed) {
    DnnImageLoader loader;
    if (!loader.parse(image.data.data(), image.data.size())) {
        return {};
    }

    Tensor current = random_input_tensor(input_h, input_w, input_c, seed);
    std::vector<LayerSparsity> results;

    for (uint32_t i = 0; i < loader.num_layers(); ++i) {
        const auto& ld = loader.layer(i);

        switch (ld.type) {
        case FuncType::Conv2D: {
            // Weight: [out_c, in_c, kH, kW]
            uint32_t out_c = ld.output.channels;
            uint32_t in_c  = current.channels;
            Tensor weight = he_init_tensor(
                out_c, in_c, ld.weight.height, ld.weight.width,
                seed + i * 1000 + 1);

            Conv2dConfig cfg;
            cfg.stride   = ld.stride;
            cfg.padding  = ld.padding;
            cfg.dilation = 1;
            cfg.groups   = 1;
            Conv2dModel conv(cfg);
            current = conv.forward(current, weight);
            apply_relu_inplace(current);
            break;
        }
        case FuncType::DepthwiseConv2D: {
            // Weight: [channels, 1, kH, kW]
            uint32_t channels = current.channels;
            Tensor weight = he_init_tensor(
                channels, 1, ld.weight.height, ld.weight.width,
                seed + i * 1000 + 1);

            DepthwiseConv2dConfig cfg;
            cfg.stride   = ld.stride;
            cfg.padding  = ld.padding;
            cfg.dilation = 1;
            DepthwiseConv2dModel dw(cfg);
            current = dw.forward(current, weight);
            apply_relu_inplace(current);
            break;
        }
        case FuncType::Pooling: {
            current = simple_pool(current,
                                   ld.weight.height, ld.weight.width,
                                   ld.stride, ld.padding);
            break;
        }
        case FuncType::FullyConnected: {
            uint32_t out_features = ld.output.channels;
            current = simple_fc(current, out_features, seed + i * 1000 + 1);
            apply_relu_inplace(current);
            break;
        }
        case FuncType::Activation: {
            // Activation subtype is not encoded in LayerDescriptor.
            // Use ReLU as a stable default for sparsity estimation.
            apply_relu_inplace(current);
            break;
        }
        default:
            // ElementWise, MatMul: pass through.
            // Function metadata is packed in packet_meta:
            // [31:24]=FuncType, [23:16]=PreprocessType, [15:0]=function_id.
            uint32_t meta = ld.packet_meta;
            FuncType fn_type =
                static_cast<FuncType>((meta >> 24) & 0xFFu);
            PreprocessType pre_type =
                static_cast<PreprocessType>((meta >> 16) & 0xFFu);
            if (fn_type == FuncType::Preprocess &&
                pre_type == PreprocessType::ZeroSkipping) {
                double keep_ratio =
                    (ld.latency.fetch_operand0 > 0)
                        ? static_cast<double>(ld.latency.write_output)
                          / static_cast<double>(ld.latency.fetch_operand0)
                        : 1.0;
                apply_zero_skipping_inplace(current, keep_ratio, seed + i * 777 + 7);
            }
            break;
        }

        LayerSparsity ls;
        ls.layer_id        = i;
        ls.type            = ld.type;
        ls.output_sparsity = compute_sparsity(current);
        results.push_back(ls);
    }

    return results;
}

} // namespace flexnpu_sim
