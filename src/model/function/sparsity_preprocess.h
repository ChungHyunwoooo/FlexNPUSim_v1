/**
 * @file sparsity_preprocess.h
 * @brief Zero-skipping preprocessing for Nullhop-class NPUs.
 *
 * Rewrites each function descriptor's fetch_operand0 (and prefetch_in)
 * in the packet image using measured input sparsity, so the latency model's
 * count ratios reflect skipped zeros. This is the one value->count bridge the
 * latency model permits (sparsity measurement). No-op unless built with OpenCV.
 */

#pragma once

#include "compiler/dnn_image/dnn_image_compiler.h"  // DnnImage
#include "compiler/dnn_image/dnn_image_generator.h"  // NetworkConfig

namespace flexnpu_sim {

void apply_sparsity(DnnImage& image, const NetworkConfig& cfg);

}  // namespace flexnpu_sim
