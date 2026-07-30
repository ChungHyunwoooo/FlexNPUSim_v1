/**
 * @file ofm_file_generator.h
 * @brief Functional OFM precompute and per-layer file dump.
 */

#pragma once

#include "compiler/dnn_image/dnn_image_compiler.h"
#include <cstdint>
#include <string>

namespace flexnpu_sim {

struct OfmFileGenConfig {
    uint32_t input_height = 224;
    uint32_t input_width = 224;
    uint32_t input_channels = 3;
    uint32_t seed = 42;
};

/**
 * @brief Precompute layer outputs using function model and dump OFM files.
 *
 * Output naming:
 *   <output_dir>/layer_<flattened_layer_idx>.bin
 *
 * The file payload is raw bytes aligned to LayerDescriptor::output.size.
 */
bool generate_ofm_files(const DnnImage& image,
                        const std::string& output_dir,
                        const OfmFileGenConfig& cfg,
                        std::string* err_msg = nullptr);

} // namespace flexnpu_sim

