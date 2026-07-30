/**
 * @file fully_connected.cpp
 * @brief Fully Connected (Dense) layer functional model implementation
 */

#include "model/function/function_model_if.h"
#include "model/function/function_model_utils.h"
#include <opencv2/dnn.hpp>

namespace flexnpu_sim {

// ============================================================================
// FullyConnectedModel Implementation
// ============================================================================

FullyConnectedModel::FullyConnectedModel(const FullyConnectedConfig& config)
    : config_(config) {}

Tensor FullyConnectedModel::forward(const Tensor& input, const Tensor& weight) {
    // Input: [batch, in_features] (may be flattened from [batch, C, H, W])
    // Weight: [out_features, in_features, 1, 1]
    // Output: [batch, out_features, 1, 1]

    const int out_features = static_cast<int>(weight.batch);
    const int in_features  = static_cast<int>(
        input.channels * input.height * input.width);

    // Weight blob shape: [out_features, in_features]
    cv::Mat weight_blob = tensor_to_mat(weight)
                              .reshape(1, {out_features, in_features});

    cv::dnn::LayerParams lp;
    lp.set("num_output", out_features);
    lp.set("bias_term",  false);
    lp.blobs.push_back(weight_blob);

    cv::dnn::Net net;
    net.addLayerToPrev("fc", "InnerProduct", lp);
    net.setInput(tensor_to_mat(input));
    cv::Mat out = net.forward();

    Tensor result = mat_to_tensor(out);
    // InnerProduct outputs [batch, out_features] 2D — remap to [batch, out_features, 1, 1]
    result.channels = static_cast<uint32_t>(out_features);
    result.height = 1;
    result.width = 1;
    return result;
}

std::string FullyConnectedModel::name() const {
    return "FullyConnected";
}

FuncType FullyConnectedModel::layer_type() const {
    return FuncType::FullyConnected;
}

} // namespace flexnpu_sim
