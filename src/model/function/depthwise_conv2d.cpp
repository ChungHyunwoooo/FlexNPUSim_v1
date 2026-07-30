/**
 * @file depthwise_conv2d.cpp
 * @brief Depthwise 2D Convolution functional model implementation
 */

#include "model/function/function_model_if.h"
#include "model/function/function_model_utils.h"
#include <opencv2/dnn.hpp>

namespace flexnpu_sim {

// ============================================================================
// DepthwiseConv2dModel Implementation
// ============================================================================

DepthwiseConv2dModel::DepthwiseConv2dModel(const DepthwiseConv2dConfig& config)
    : config_(config) {}

Tensor DepthwiseConv2dModel::forward(const Tensor& input, const Tensor& weight) {
    // Input: [batch, channels, in_height, in_width]
    // Weight: [channels, 1, kernel_h, kernel_w] (one kernel per channel)
    // Output: [batch, channels, out_height, out_width]

    const int channels = static_cast<int>(input.channels);
    const int kH       = static_cast<int>(weight.height);
    const int kW       = static_cast<int>(weight.width);

    // groups = channels for depthwise: each channel has its own kernel
    // Weight blob shape: [channels, 1, kH, kW]
    cv::Mat weight_blob = tensor_to_mat(weight)
                              .reshape(1, {channels, 1, kH, kW});

    cv::dnn::LayerParams lp;
    lp.set("kernel_h",   kH);
    lp.set("kernel_w",   kW);
    lp.set("num_output", channels);
    lp.set("pad",        static_cast<int>(config_.padding));
    lp.set("stride",     static_cast<int>(config_.stride));
    lp.set("dilation",   static_cast<int>(config_.dilation));
    lp.set("group",      channels);
    lp.set("bias_term",  false);
    lp.blobs.push_back(weight_blob);

    cv::dnn::Net net;
    net.addLayerToPrev("depthwise_conv", "Convolution", lp);
    net.setInput(tensor_to_mat(input));
    cv::Mat out = net.forward();

    return mat_to_tensor(out);
}

std::string DepthwiseConv2dModel::name() const {
    return "DepthwiseConv2D";
}

FuncType DepthwiseConv2dModel::layer_type() const {
    return FuncType::DepthwiseConv2D;
}

} // namespace flexnpu_sim
