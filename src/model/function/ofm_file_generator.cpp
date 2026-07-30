/**
 * @file ofm_file_generator.cpp
 * @brief Functional OFM precompute and per-layer file dump.
 */

#include "model/function/ofm_file_generator.h"
#include "compiler/dnn_image/dnn_image_loader.h"
#include "model/function/function_model_if.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

namespace flexnpu_sim {
namespace fs = std::filesystem;

static void set_err(std::string* err, const std::string& msg) {
    if (err) *err = msg;
}

static Tensor random_input_tensor(uint32_t h, uint32_t w, uint32_t c, uint32_t seed) {
    Tensor t;
    t.batch = 1;
    t.channels = c;
    t.height = h;
    t.width = w;
    t.data.resize(t.size());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : t.data) v = dist(rng);
    return t;
}

static Tensor he_weight_tensor(uint32_t out_c,
                               uint32_t in_c,
                               uint32_t k_h,
                               uint32_t k_w,
                               uint32_t seed) {
    Tensor w;
    w.batch = out_c;
    w.channels = in_c;
    w.height = k_h;
    w.width = k_w;
    w.data.resize(w.size());
    uint32_t fan_in = std::max(1u, in_c * k_h * k_w);
    float stddev = std::sqrt(2.0f / static_cast<float>(fan_in));
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, stddev);
    for (auto& v : w.data) v = dist(rng);
    return w;
}

static void apply_relu_inplace(Tensor& t) {
    for (auto& v : t.data) {
        if (v < 0.0f) v = 0.0f;
    }
}

static void apply_zero_skipping_inplace(Tensor& t, double keep_ratio, uint32_t seed) {
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

static Tensor simple_pool_max(const Tensor& input,
                              uint32_t kh,
                              uint32_t kw,
                              uint32_t stride,
                              uint32_t pad) {
    uint32_t out_h = 0;
    uint32_t out_w = 0;
    if (kh >= input.height && kw >= input.width) {
        out_h = 1;
        out_w = 1;
    } else {
        out_h = (input.height + 2 * pad - kh) / stride + 1;
        out_w = (input.width + 2 * pad - kw) / stride + 1;
    }

    Tensor out;
    out.batch = input.batch;
    out.channels = input.channels;
    out.height = out_h;
    out.width = out_w;
    out.data.resize(out.size(), 0.0f);

    for (uint32_t c = 0; c < out.channels; ++c) {
        for (uint32_t oh = 0; oh < out.height; ++oh) {
            for (uint32_t ow = 0; ow < out.width; ++ow) {
                float m = 0.0f;
                uint32_t sh = (kh >= input.height) ? 0 : oh * stride;
                uint32_t sw = (kw >= input.width) ? 0 : ow * stride;
                uint32_t eh = (kh >= input.height) ? input.height : kh;
                uint32_t ew = (kw >= input.width) ? input.width : kw;
                for (uint32_t ki = 0; ki < eh; ++ki) {
                    for (uint32_t kj = 0; kj < ew; ++kj) {
                        int ih = static_cast<int>(sh + ki) - static_cast<int>(pad);
                        int iw = static_cast<int>(sw + kj) - static_cast<int>(pad);
                        if (ih >= 0 && ih < static_cast<int>(input.height) &&
                            iw >= 0 && iw < static_cast<int>(input.width)) {
                            m = std::max(m, input.at(
                                static_cast<uint32_t>(ih),
                                static_cast<uint32_t>(iw), c));
                        }
                    }
                }
                out.at(oh, ow, c) = m;
            }
        }
    }
    return out;
}

static Tensor simple_fc(const Tensor& input, uint32_t out_features, uint32_t seed) {
    uint32_t in_features = input.height * input.width * input.channels;
    float stddev = std::sqrt(2.0f / static_cast<float>(std::max(in_features, 1u)));
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, stddev);

    Tensor out;
    out.batch = 1;
    out.channels = out_features;
    out.height = 1;
    out.width = 1;
    out.data.resize(out_features, 0.0f);
    for (uint32_t j = 0; j < out_features; ++j) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < in_features; ++i) {
            sum += input.data[i] * dist(rng);
        }
        out.data[j] = sum;
    }
    return out;
}

static bool dump_layer_ofm(const Tensor& t,
                           uint32_t output_size_bytes,
                           const std::string& path) {
    std::vector<uint8_t> payload(output_size_bytes, 0);
    size_t avail = t.data.size() * sizeof(float);
    size_t n = std::min(payload.size(), avail);
    if (n > 0) {
        std::memcpy(payload.data(), reinterpret_cast<const uint8_t*>(t.data.data()), n);
    }
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) return false;
    if (!payload.empty()) {
        ofs.write(reinterpret_cast<const char*>(payload.data()),
                  static_cast<std::streamsize>(payload.size()));
    }
    return ofs.good();
}

bool generate_ofm_files(const DnnImage& image,
                        const std::string& output_dir,
                        const OfmFileGenConfig& cfg,
                        std::string* err_msg) {
    if (image.empty()) {
        set_err(err_msg, "empty dnn image");
        return false;
    }
    if (output_dir.empty()) {
        set_err(err_msg, "empty output_dir");
        return false;
    }

    DnnImageLoader loader;
    if (!loader.parse(image.data.data(), image.data.size())) {
        set_err(err_msg, "failed to parse dnn image");
        return false;
    }

    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
        set_err(err_msg, "failed to create output_dir: " + output_dir);
        return false;
    }

    Tensor current = random_input_tensor(cfg.input_height, cfg.input_width,
                                         cfg.input_channels, cfg.seed);

    for (uint32_t i = 0; i < loader.num_layers(); ++i) {
        const auto& ld = loader.layer(i);
        switch (ld.type) {
            case FuncType::Conv2D: {
                Tensor weight = he_weight_tensor(ld.output.channels,
                                                 current.channels,
                                                 ld.weight.height,
                                                 ld.weight.width,
                                                 cfg.seed + i * 1000 + 1);
                Conv2dConfig c{};
                c.stride = ld.stride;
                c.padding = ld.padding;
                c.dilation = 1;
                c.groups = 1;
                Conv2dModel m(c);
                current = m.forward(current, weight);
                break;
            }
            case FuncType::DepthwiseConv2D: {
                Tensor weight = he_weight_tensor(current.channels, 1,
                                                 ld.weight.height,
                                                 ld.weight.width,
                                                 cfg.seed + i * 1000 + 1);
                DepthwiseConv2dConfig c{};
                c.stride = ld.stride;
                c.padding = ld.padding;
                c.dilation = 1;
                DepthwiseConv2dModel m(c);
                current = m.forward(current, weight);
                break;
            }
            case FuncType::Pooling: {
                current = simple_pool_max(current, ld.weight.height, ld.weight.width,
                                          ld.stride, ld.padding);
                break;
            }
            case FuncType::FullyConnected: {
                current = simple_fc(current, ld.output.channels, cfg.seed + i * 1000 + 1);
                break;
            }
            case FuncType::Activation: {
                apply_relu_inplace(current);
                break;
            }
            default: {
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
                    apply_zero_skipping_inplace(current, keep_ratio,
                                                cfg.seed + i * 1337 + 7);
                }
                break;
            }
        }

        std::string path = output_dir + "/layer_" + std::to_string(i) + ".bin";
        if (!dump_layer_ofm(current, ld.output.size, path)) {
            set_err(err_msg, "failed to write OFM file: " + path);
            return false;
        }
    }
    return true;
}

} // namespace flexnpu_sim

