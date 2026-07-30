/**
 * @file layer_function_runner.h
 * @brief R10: Layer-level function_model invocation helper.
 *
 * Given a LayerDescriptor and raw input/weight bytes (dtype per header),
 * decode into Tensor, invoke the matching FunctionModelIf, and encode the
 * result back to bytes. Returns false if the layer type is unsupported or
 * OpenCV is absent.
 *
 * Header-only: linked into both flexnpusim_core callers and the SystemC
 * NpuController without additional CMake plumbing.
 */

#pragma once

#include "compiler/dnn_image/dnn_image_format.h"
#include "model/function/function_model_if.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace flexnpu_sim {

// -------------------------- dtype encode/decode --------------------------

inline TensorDType resolve_dtype(TensorDType per_tensor, TensorDType fallback) {
    // per_tensor == 0 (INT8) is a valid value, not "inherit". Spec treats
    // per-tensor as authoritative; 0 inherits default only if reserved==0 is
    // the encoded sentinel. Keep simple: use per_tensor directly.
    (void)fallback;
    return per_tensor;
}

inline void bytes_to_float(const uint8_t* src, size_t nbytes,
                           TensorDType dt, std::vector<float>& out) {
    const uint32_t esz = tensor_elem_bytes(dt);
    const size_t n = (esz > 0) ? nbytes / esz : 0;
    out.resize(n);
    switch (dt) {
        case TensorDType::INT8: {
            for (size_t i = 0; i < n; ++i) {
                int8_t v = static_cast<int8_t>(src[i]);
                out[i] = static_cast<float>(v);
            }
            break;
        }
        case TensorDType::INT16: {
            for (size_t i = 0; i < n; ++i) {
                int16_t v;
                std::memcpy(&v, src + i * 2, 2);
                out[i] = static_cast<float>(v);
            }
            break;
        }
        case TensorDType::FP32: {
            std::memcpy(out.data(), src, n * 4);
            break;
        }
        case TensorDType::FP16:
        default: {
            // FP16 stub: zero-fill. Proper decode in a later phase.
            std::fill(out.begin(), out.end(), 0.0f);
            break;
        }
    }
}

inline void float_to_bytes(const std::vector<float>& src,
                           TensorDType dt, std::vector<uint8_t>& out) {
    const uint32_t esz = tensor_elem_bytes(dt);
    out.assign(src.size() * esz, 0);
    switch (dt) {
        case TensorDType::INT8: {
            for (size_t i = 0; i < src.size(); ++i) {
                float v = std::round(src[i]);
                if (v > 127.0f) v = 127.0f;
                if (v < -128.0f) v = -128.0f;
                int8_t q = static_cast<int8_t>(v);
                out[i] = static_cast<uint8_t>(q);
            }
            break;
        }
        case TensorDType::INT16: {
            for (size_t i = 0; i < src.size(); ++i) {
                float v = std::round(src[i]);
                if (v > 32767.0f) v = 32767.0f;
                if (v < -32768.0f) v = -32768.0f;
                int16_t q = static_cast<int16_t>(v);
                std::memcpy(out.data() + i * 2, &q, 2);
            }
            break;
        }
        case TensorDType::FP32: {
            std::memcpy(out.data(), src.data(), src.size() * 4);
            break;
        }
        case TensorDType::FP16:
        default:
            break;  // zero-filled
    }
}

// -------------------------- layer runner --------------------------

/// Runs the function model for a single layer. `output_bytes` is resized to
/// layer.output.size and populated in layer.output.dtype encoding.
/// Returns false if unsupported (OpenCV absent or unknown type).
inline bool run_layer_function(const LayerDescriptor& layer,
                               const uint8_t* input_bytes, size_t input_size,
                               const uint8_t* weight_bytes, size_t weight_size,
                               std::vector<uint8_t>& output_bytes) {
#ifndef HAVE_OPENCV
    (void)layer; (void)input_bytes; (void)input_size;
    (void)weight_bytes; (void)weight_size; (void)output_bytes;
    return false;
#else
    const TensorDType in_dt  = layer.input.dtype;
    const TensorDType wt_dt  = layer.weight.dtype;
    const TensorDType out_dt = layer.output.dtype;

    // Decode input → Tensor (NCHW, batch=1).
    Tensor in_t;
    in_t.height = layer.input.height;
    in_t.width  = layer.input.width;
    in_t.channels = layer.input.channels;
    in_t.batch = 1;
    bytes_to_float(input_bytes, input_size, in_dt, in_t.data);
    in_t.data.resize(in_t.size(), 0.0f);  // zero-pad if short

    // Decode weight → Tensor (kernel layout depends on layer type).
    Tensor w_t;
    if (weight_size > 0 && weight_bytes) {
        bytes_to_float(weight_bytes, weight_size, wt_dt, w_t.data);
    }

    std::unique_ptr<FunctionModelIf> model;
    Tensor out_t;
    switch (layer.type) {
        case FuncType::Conv2D: {
            w_t.height   = std::max<uint32_t>(1, layer.weight.height);
            w_t.width    = std::max<uint32_t>(1, layer.weight.width);
            w_t.channels = std::max<uint32_t>(1, layer.weight.channels);
            // Kernel count = output channels.
            w_t.batch    = std::max<uint32_t>(1, layer.output.channels);
            w_t.data.resize(w_t.size(), 0.0f);
            Conv2dConfig cfg;
            cfg.stride   = std::max<uint32_t>(1, layer.stride);
            cfg.padding  = layer.padding;
            cfg.dilation = 1;
            cfg.groups   = 1;
            model = std::make_unique<Conv2dModel>(cfg);
            break;
        }
        case FuncType::DepthwiseConv2D: {
            w_t.height   = std::max<uint32_t>(1, layer.weight.height);
            w_t.width    = std::max<uint32_t>(1, layer.weight.width);
            w_t.channels = 1;
            w_t.batch    = std::max<uint32_t>(1, layer.output.channels);
            w_t.data.resize(w_t.size(), 0.0f);
            DepthwiseConv2dConfig cfg;
            cfg.stride   = std::max<uint32_t>(1, layer.stride);
            cfg.padding  = layer.padding;
            cfg.dilation = 1;
            model = std::make_unique<DepthwiseConv2dModel>(cfg);
            break;
        }
        case FuncType::Pooling: {
            PoolingConfig cfg;
            cfg.type        = PoolingType::Max;
            cfg.kernel_size = std::max<uint32_t>(1, layer.weight.height);
            cfg.stride      = std::max<uint32_t>(1, layer.stride);
            cfg.padding     = layer.padding;
            model = std::make_unique<PoolingModel>(cfg);
            break;
        }
        case FuncType::FullyConnected: {
            const uint32_t in_f  = layer.input.height * layer.input.width *
                                   layer.input.channels;
            const uint32_t out_f = layer.output.height * layer.output.width *
                                   layer.output.channels;
            w_t.height = 1; w_t.width = 1;
            w_t.channels = std::max<uint32_t>(1, in_f);
            w_t.batch    = std::max<uint32_t>(1, out_f);
            w_t.data.resize(w_t.size(), 0.0f);
            FullyConnectedConfig cfg;
            cfg.input_features  = in_f;
            cfg.output_features = out_f;
            cfg.use_bias        = false;
            model = std::make_unique<FullyConnectedModel>(cfg);
            break;
        }
        case FuncType::Activation: {
            ActivationConfig cfg;
            cfg.type = ActivationType::ReLU;
            model = std::make_unique<ActivationModel>(cfg);
            break;
        }
        case FuncType::MatMul: {
            // MatMul model reads weight.at(r,c,h,b); shape must be set so that
            // weight tensor mirrors input layout (B=1, H=num_heads(=channels),
            // L=seq_len, D=d_k). LayerDescriptor weight carries [H,W,C].
            w_t.height   = std::max<uint32_t>(1, layer.weight.height);
            w_t.width    = std::max<uint32_t>(1, layer.weight.width);
            w_t.channels = std::max<uint32_t>(1, layer.weight.channels);
            w_t.batch    = 1;
            w_t.data.resize(w_t.size(), 0.0f);
            MatMulConfig cfg;
            cfg.mode = MatMulMode::QKt;
            model = std::make_unique<MatMulModel>(cfg);
            break;
        }
        case FuncType::ElementWise: {
            // EW model uses flat data; shape fields not strictly required, but
            // set them for downstream consistency.
            w_t.height   = std::max<uint32_t>(1, layer.weight.height);
            w_t.width    = std::max<uint32_t>(1, layer.weight.width);
            w_t.channels = std::max<uint32_t>(1, layer.weight.channels);
            w_t.batch    = 1;
            w_t.data.resize(w_t.size(), 0.0f);
            ElementWiseConfig cfg;
            cfg.op = ElementWiseOp::Add;
            model = std::make_unique<ElementWiseModel>(cfg);
            break;
        }
        default:
            return false;
    }

    if (!model) return false;
    out_t = model->forward(in_t, w_t);

    // Encode output → bytes. Target size = layer.output.size.
    std::vector<uint8_t> encoded;
    float_to_bytes(out_t.data, out_dt, encoded);
    output_bytes.assign(layer.output.size, 0);
    const size_t copy_n = std::min<size_t>(encoded.size(), output_bytes.size());
    std::memcpy(output_bytes.data(), encoded.data(), copy_n);
    return true;
#endif
}

}  // namespace flexnpu_sim
