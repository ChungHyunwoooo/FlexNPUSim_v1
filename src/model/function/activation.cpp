/**
 * @file activation.cpp
 * @brief Activation function functional model implementation
 */

#include "model/function/function_model_if.h"
#include "model/function/function_model_utils.h"
#include <opencv2/dnn.hpp>

namespace flexnpu_sim {

// ============================================================================
// ActivationModel Implementation
// ============================================================================

ActivationModel::ActivationModel(const ActivationConfig& config)
    : config_(config) {}

Tensor ActivationModel::forward(const Tensor& input, const Tensor& weight) {
    // Input: [batch, channels, height, width]
    // Weight: not used (empty)
    // Output: [batch, channels, height, width] (same shape)

    (void)weight;  // Unused - activation doesn't use weights

    cv::dnn::LayerParams lp;
    std::string layer_type_str;

    switch (config_.type) {
        case ActivationType::ReLU:
            layer_type_str = "ReLU";
            break;
        case ActivationType::ReLU6:
            layer_type_str = "ReLU6";
            break;
        case ActivationType::Sigmoid:
            layer_type_str = "Sigmoid";
            break;
        case ActivationType::Tanh:
            layer_type_str = "TanH";
            break;
        default:
            layer_type_str = "ReLU";
            break;
    }

    cv::dnn::Net net;
    net.addLayerToPrev("activation", layer_type_str, lp);
    net.setInput(tensor_to_mat(input));
    cv::Mat out = net.forward();

    return mat_to_tensor(out);
}

std::string ActivationModel::name() const {
    switch (config_.type) {
        case ActivationType::ReLU:
            return "ReLU";
        case ActivationType::ReLU6:
            return "ReLU6";
        case ActivationType::Sigmoid:
            return "Sigmoid";
        case ActivationType::Tanh:
            return "Tanh";
        default:
            return "Activation";
    }
}

FuncType ActivationModel::layer_type() const {
    return FuncType::Activation;
}

} // namespace flexnpu_sim
