/**
 * @file conv2d.cpp
 * @brief 2D Convolution functional model implementation
 */

#include "model/function/function_model_if.h"
#include "model/function/function_model_utils.h"
#include <opencv2/dnn.hpp>

namespace flexnpu_sim {

// ============================================================================
// Conv2dModel Implementation
// ============================================================================

Conv2dModel::Conv2dModel(const Conv2dConfig& config)
    : config_(config) {}

Tensor Conv2dModel::forward(const Tensor& input, const Tensor& weight) {
    // Input: [batch, in_channels, in_height, in_width]
    // Weight: [out_channels, in_channels/groups, kernel_h, kernel_w]
    // Output: [batch, out_channels, out_height, out_width]

    const int out_c  = static_cast<int>(weight.batch);
    const int in_c   = static_cast<int>(input.channels);
    const int groups = static_cast<int>(config_.groups);
    const int kH     = static_cast<int>(weight.height);
    const int kW     = static_cast<int>(weight.width);

    // Reshape weight blob to [out_c, in_c/groups, kH, kW]
    cv::Mat weight_blob = tensor_to_mat(weight)
                              .reshape(1, {out_c, in_c / groups, kH, kW});

    cv::dnn::LayerParams lp;
    lp.set("kernel_h",  kH);
    lp.set("kernel_w",  kW);
    lp.set("num_output", out_c);
    lp.set("pad",      static_cast<int>(config_.padding));
    lp.set("stride",   static_cast<int>(config_.stride));
    lp.set("dilation", static_cast<int>(config_.dilation));
    lp.set("group",    groups);
    lp.set("bias_term", false);
    lp.blobs.push_back(weight_blob);

    cv::dnn::Net net;
    net.addLayerToPrev("conv", "Convolution", lp);
    net.setInput(tensor_to_mat(input));
    cv::Mat out = net.forward();

    return mat_to_tensor(out);
}

std::string Conv2dModel::name() const {
    return "Conv2D";
}

FuncType Conv2dModel::layer_type() const {
    return FuncType::Conv2D;
}

} // namespace flexnpu_sim
