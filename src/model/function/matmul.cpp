/**
 * @file matmul.cpp
 * @brief Matrix Multiplication functional model implementation
 */

#include "model/function/function_model_if.h"
#include <opencv2/core.hpp>

namespace flexnpu_sim {

// ============================================================================
// MatMulModel Implementation
// ============================================================================

MatMulModel::MatMulModel(const MatMulConfig& config)
    : config_(config) {}

Tensor MatMulModel::forward(const Tensor& input, const Tensor& weight) {
    // Input:  [batch, channels=num_heads, height=seq_len, width=d_k]
    // Weight: [batch, channels=num_heads, height=seq_len, width=d_k]
    // QKt mode: out[b,h] = input[b,h] * weight[b,h]^T  → [seq_len, seq_len]
    // AV  mode: out[b,h] = input[b,h] * weight[b,h]    → [seq_len, d_k]

    const uint32_t B = input.batch;
    const uint32_t H = input.channels;   // num_heads
    const uint32_t L = input.height;     // seq_len
    const uint32_t D = input.width;      // d_k

    const uint32_t out_w = (config_.mode == MatMulMode::QKt) ? L : D;

    Tensor output;
    output.batch    = B;
    output.channels = H;
    output.height   = L;
    output.width    = out_w;
    output.data.resize(output.size(), 0.0f);

    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t h = 0; h < H; ++h) {
            // Extract 2D slices for this (batch, head) pair
            cv::Mat a(static_cast<int>(L), static_cast<int>(D), CV_32F);
            cv::Mat k(static_cast<int>(L), static_cast<int>(D), CV_32F);

            for (uint32_t r = 0; r < L; ++r) {
                for (uint32_t c = 0; c < D; ++c) {
                    a.at<float>(static_cast<int>(r), static_cast<int>(c)) =
                        input.at(r, c, h, b);
                    k.at<float>(static_cast<int>(r), static_cast<int>(c)) =
                        weight.at(r, c, h, b);
                }
            }

            cv::Mat result;
            if (config_.mode == MatMulMode::QKt) {
                result = a * k.t();  // [L, L]
            } else {
                result = a * k;      // [L, D]
            }

            // Copy result back into output tensor
            for (uint32_t r = 0; r < L; ++r) {
                for (uint32_t c = 0; c < out_w; ++c) {
                    output.at(r, c, h, b) =
                        result.at<float>(static_cast<int>(r), static_cast<int>(c));
                }
            }
        }
    }

    return output;
}

std::string MatMulModel::name() const {
    return "MatMul";
}

FuncType MatMulModel::layer_type() const {
    return FuncType::MatMul;
}

} // namespace flexnpu_sim
