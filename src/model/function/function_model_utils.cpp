/**
 * @file function_model_utils.cpp
 * @brief Tensor ↔ cv::Mat conversion implementation
 */

#include "model/function/function_model_utils.h"
#include <cstring>

namespace flexnpu_sim {

cv::Mat tensor_to_mat(const Tensor& t) {
    int dims[] = {
        static_cast<int>(t.batch),
        static_cast<int>(t.channels),
        static_cast<int>(t.height),
        static_cast<int>(t.width)
    };
    cv::Mat mat(4, dims, CV_32F);

    // Copy data — both use NCHW layout
    if (!t.data.empty()) {
        std::memcpy(mat.data, t.data.data(), t.data.size() * sizeof(float));
    }

    return mat;
}

Tensor mat_to_tensor(const cv::Mat& m) {
    Tensor t;

    if (m.dims == 4) {
        t.batch    = static_cast<uint32_t>(m.size[0]);
        t.channels = static_cast<uint32_t>(m.size[1]);
        t.height   = static_cast<uint32_t>(m.size[2]);
        t.width    = static_cast<uint32_t>(m.size[3]);
    } else if (m.dims == 2) {
        t.batch    = 1;
        t.channels = 1;
        t.height   = static_cast<uint32_t>(m.rows);
        t.width    = static_cast<uint32_t>(m.cols);
    } else if (m.dims == 3) {
        t.batch    = 1;
        t.channels = static_cast<uint32_t>(m.size[0]);
        t.height   = static_cast<uint32_t>(m.size[1]);
        t.width    = static_cast<uint32_t>(m.size[2]);
    }

    size_t total = t.size();
    t.data.resize(total);

    if (m.isContinuous()) {
        std::memcpy(t.data.data(), m.data, total * sizeof(float));
    } else {
        // For non-contiguous mats, copy element by element
        const float* src = reinterpret_cast<const float*>(m.data);
        std::memcpy(t.data.data(), src, total * sizeof(float));
    }

    return t;
}

} // namespace flexnpu_sim
