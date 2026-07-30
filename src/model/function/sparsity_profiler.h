/**
 * @file sparsity_profiler.h
 * @brief Per-layer activation sparsity profiler
 *
 * Runs forward inference with He-initialized random weights
 * through a compiled DNN image. After each Conv/DWConv layer,
 * applies ReLU and measures the fraction of zero activations.
 *
 * Zero-skipping NPUs (e.g., Nullhop) use these ratios to
 * reduce effective fetch_operand0: adjusted = fetch_operand0 * (1 - input_sparsity).
 *
 * Pure C++ (no SystemC dependency).
 */

#pragma once

#include "model/function/function_model_if.h"
#include "compiler/dnn_image/dnn_image_compiler.h"
#include "compiler/dnn_image/dnn_image_format.h"
#include <vector>
#include <cstdint>

namespace flexnpu_sim {

class SparsityProfiler {
public:
    /**
     * @brief Per-layer sparsity measurement result
     */
    struct LayerSparsity {
        uint32_t layer_id;
        FuncType type;
        double output_sparsity;   ///< Fraction of zeros in output (after ReLU)
    };

    /**
     * @brief Profile per-layer output sparsity for a compiled DNN image
     *
     * Runs forward inference layer by layer:
     *   Conv/DWConv: function model forward + ReLU
     *   Pool: max pooling (preserves non-negative sparsity)
     *   FC: matrix multiply + ReLU
     *
     * @param image      Compiled DNN image
     * @param input_h    Input height
     * @param input_w    Input width
     * @param input_c    Input channels
     * @param seed       Random seed for reproducibility
     * @return Per-layer output sparsity vector
     */
    std::vector<LayerSparsity> profile(const DnnImage& image,
                                        uint32_t input_h, uint32_t input_w,
                                        uint32_t input_c,
                                        unsigned seed = 42);

    /**
     * @brief Compute sparsity ratio (fraction of zeros)
     */
    static double compute_sparsity(const Tensor& t);

    /**
     * @brief Adjust fetch_operand0 for zero-skipping NPUs
     *
     * @param original_fi  Original fetch_operand0 (dense)
     * @param input_sparsity  Fraction of zeros in input activations
     * @return Adjusted fetch_operand0 = original_fi * (1 - input_sparsity)
     */
    static uint32_t adjust_fi(uint32_t original_fi, double input_sparsity);
};

} // namespace flexnpu_sim
