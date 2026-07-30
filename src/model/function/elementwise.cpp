/**
 * @file elementwise.cpp
 * @brief Element-wise functional model implementation
 */

#include "model/function/function_model_if.h"
#include <opencv2/core.hpp>

namespace flexnpu_sim {

// ============================================================================
// ElementWiseModel Implementation
// ============================================================================

ElementWiseModel::ElementWiseModel(const ElementWiseConfig& config)
    : config_(config) {}

Tensor ElementWiseModel::forward(const Tensor& input, const Tensor& weight) {
    // Input and weight must have identical shape.
    // Output has the same shape as input.
    // Add:      output = input + weight
    // Multiply: output = input * weight
    // Subtract: output = input - weight

    Tensor output;
    output.batch    = input.batch;
    output.channels = input.channels;
    output.height   = input.height;
    output.width    = input.width;
    output.data.resize(output.size(), 0.0f);

    // Wrap flat data vectors as 1-row cv::Mat for vectorised OpenCV ops
    const int n = static_cast<int>(input.data.size());

    cv::Mat a(1, n, CV_32F, const_cast<float*>(input.data.data()));
    cv::Mat b(1, n, CV_32F, const_cast<float*>(weight.data.data()));
    cv::Mat dst(1, n, CV_32F, output.data.data());

    switch (config_.op) {
        case ElementWiseOp::Add:
            cv::add(a, b, dst);
            break;

        case ElementWiseOp::Multiply:
            cv::multiply(a, b, dst);
            break;

        case ElementWiseOp::Subtract:
            cv::subtract(a, b, dst);
            break;

        default:
            cv::add(a, b, dst);  // Fallback: treat as Add
            break;
    }

    return output;
}

std::string ElementWiseModel::name() const {
    switch (config_.op) {
        case ElementWiseOp::Add:
            return "ElementWise_Add";
        case ElementWiseOp::Multiply:
            return "ElementWise_Multiply";
        case ElementWiseOp::Subtract:
            return "ElementWise_Subtract";
        default:
            return "ElementWise";
    }
}

FuncType ElementWiseModel::layer_type() const {
    return FuncType::ElementWise;
}

} // namespace flexnpu_sim
