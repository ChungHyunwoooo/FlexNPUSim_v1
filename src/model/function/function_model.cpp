/**
 * @file function_model.cpp
 * @brief Tensor utility implementations
 */

#include "model/function/function_model_if.h"
#include <stdexcept>

namespace flexnpu_sim {

// ============================================================================
// Tensor Access Methods
// ============================================================================

float& Tensor::at(uint32_t h, uint32_t w, uint32_t c, uint32_t b) {
    if (b >= batch || c >= channels || h >= height || w >= width) {
        throw std::out_of_range("Tensor index out of bounds");
    }
    // NCHW layout: batch-major, channel-major, height-major, width-major
    size_t idx = static_cast<size_t>(b) * channels * height * width +
                 static_cast<size_t>(c) * height * width +
                 static_cast<size_t>(h) * width +
                 static_cast<size_t>(w);
    return data[idx];
}

const float& Tensor::at(uint32_t h, uint32_t w, uint32_t c, uint32_t b) const {
    if (b >= batch || c >= channels || h >= height || w >= width) {
        throw std::out_of_range("Tensor index out of bounds");
    }
    size_t idx = static_cast<size_t>(b) * channels * height * width +
                 static_cast<size_t>(c) * height * width +
                 static_cast<size_t>(h) * width +
                 static_cast<size_t>(w);
    return data[idx];
}

} // namespace flexnpu_sim
