/**
 * @file pooling.cpp
 * @brief Pooling functional model implementation
 */

#include "model/function/function_model_if.h"
#include "model/function/function_model_utils.h"
#include <opencv2/dnn.hpp>

namespace flexnpu_sim {

// ============================================================================
// PoolingModel Implementation
// ============================================================================

PoolingModel::PoolingModel(const PoolingConfig& config)
    : config_(config) {}

Tensor PoolingModel::forward(const Tensor& input, const Tensor& weight) {
    // Input: [batch, channels, in_height, in_width]
    // Weight: not used (empty)
    // Output: [batch, channels, out_height, out_width]

    (void)weight;  // Unused - pooling doesn't use weights

    cv::dnn::LayerParams lp;
    lp.set("pool",        config_.type == PoolingType::Max ? "max" : "ave");
    lp.set("kernel_size", static_cast<int>(config_.kernel_size));
    lp.set("stride",      static_cast<int>(config_.stride));
    lp.set("pad",         static_cast<int>(config_.padding));

    cv::dnn::Net net;
    net.addLayerToPrev("pool", "Pooling", lp);
    net.setInput(tensor_to_mat(input));
    cv::Mat out = net.forward();

    return mat_to_tensor(out);
}

std::string PoolingModel::name() const {
    if (config_.type == PoolingType::Max) {
        return "MaxPooling";
    } else {
        return "AveragePooling";
    }
}

FuncType PoolingModel::layer_type() const {
    return FuncType::Pooling;
}

} // namespace flexnpu_sim
