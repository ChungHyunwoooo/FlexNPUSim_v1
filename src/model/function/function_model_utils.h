/**
 * @file function_model_utils.h
 * @brief Tensor ↔ cv::Mat conversion utilities
 */

#pragma once

#include "model/function/function_model_if.h"
#include <opencv2/core.hpp>

namespace flexnpu_sim {

/**
 * @brief Convert Tensor (NCHW) to 4D cv::Mat blob
 *
 * @param t Input tensor (NCHW layout)
 * @return 4D cv::Mat [N, C, H, W]
 */
cv::Mat tensor_to_mat(const Tensor& t);

/**
 * @brief Convert 4D cv::Mat blob to Tensor (NCHW)
 *
 * @param m 4D cv::Mat [N, C, H, W]
 * @return Tensor with NCHW layout
 */
Tensor mat_to_tensor(const cv::Mat& m);

} // namespace flexnpu_sim
