/**
 * @file dnn_image_generator.h
 * @brief DNN Image Generator for Standard Networks
 *
 * Generates DNN Image files for common network architectures
 * (VGG, MobileNet, ResNet, etc.).
 * Pure C++ (no SystemC dependency).
 */

#pragma once

#include "compiler/dnn_image/dnn_image_compiler.h"
#include "common/tile_descriptor.h"
#include "common/types.h"
#include <string>
#include <cstdint>
#include <vector>

namespace flexnpu_sim {

// ============================================================================
// Network Configuration
// ============================================================================

/**
 * @brief Network generation configuration
 */
struct NetworkConfig {
    uint32_t input_height = 224;    ///< Input image height
    uint32_t input_width = 224;     ///< Input image width
    uint32_t input_channels = 3;    ///< Input channels (RGB)
    uint32_t num_classes = 1000;    ///< Number of output classes
    bool include_weights = false;   ///< Generate weight placeholders
    uint32_t base_dram_addr = 0;    ///< Base DRAM address for tensors
};

// ============================================================================
// DNN Image Generator
// ============================================================================

/**
 * @brief DNN Image Generator for standard networks
 *
 * Usage:
 *   DnnImageGenerator gen;
 *   DnnImage image = gen.generate_vgg16();
 *   // or with DSE (specific NPU preset)
 *   gen.configure(pe_config, Dataflow::WS, 256, 16, 16);
 *   DnnImage image = gen.generate_mobilenet_v2(config);
 */
class DnnImageGenerator {
public:
    enum class PeGroupingMode : uint8_t {
        Fixed = 0,          ///< Use one global PeConfig for all layers
        AdaptiveByOps = 1   ///< Derive effective ops_per_pass/parallel_passes/num_output_lanes per layer from n (per-output ops)
    };

    DnnImageGenerator() = default;

    /**
     * @brief Configure hardware parameters for latency model derivation
     *
     * When configured, generators use set_tiled_latency_params with
     * the given HW config. Otherwise, set_default_latency_params is used.
     */
    void configure(const PeConfig& pe, Dataflow df, uint32_t l2_kb,
                   double threshold_ratio = 1.0) {
        pe_config_ = pe;
        dataflow_ = df;
        l2_buffer_kb_ = l2_kb;
        threshold_ratio_ = threshold_ratio;
        configured_ = true;
    }

    /// Reset to default (unconfigured) state
    void reset_config() { configured_ = false; }

    /**
     * @brief Set PE grouping mode (fixed vs. layer-adaptive)
     */
    void set_pe_grouping_mode(PeGroupingMode mode) { pe_grouping_mode_ = mode; }

    /**
     * @brief Register per-layer-type PE override
     *
     * Override has priority over adaptive mode.
     */
    void set_layer_pe_override(FuncType type, const PeConfig& pe);

    /**
     * @brief Register per-layer-index PE override
     *
     * layer_index is 0-based in generation order.
     * Override has highest priority.
     */
    void set_layer_pe_override(uint32_t layer_index, const PeConfig& pe);

    /**
     * @brief Clear all per-layer-type PE overrides
     */
    void clear_layer_pe_overrides() { layer_pe_overrides_.clear(); }

    /**
     * @brief Clear all per-layer-index PE overrides
     */
    void clear_layer_index_pe_overrides() { layer_index_pe_overrides_.clear(); }

    /**
     * @brief Generate single Conv2D layer
     */
    DnnImage generate_single_conv2d(
        uint32_t in_h, uint32_t in_w, uint32_t in_c, uint32_t out_c,
        uint32_t kernel_size = 3, uint32_t stride = 1, uint32_t padding = 1);

    /**
     * @brief Generate VGG-16 network (21 layers)
     */
    DnnImage generate_vgg16(const NetworkConfig& config = NetworkConfig());

    /**
     * @brief Generate VGG-19 network
     */
    DnnImage generate_vgg19(const NetworkConfig& config = NetworkConfig());

    /**
     * @brief Generate MobileNet-V1 network (29 layers)
     */
    DnnImage generate_mobilenet_v1(const NetworkConfig& config = NetworkConfig());

    /**
     * @brief Generate MobileNet-V2 network (54 layers)
     */
    DnnImage generate_mobilenet_v2(const NetworkConfig& config = NetworkConfig());

    /**
     * @brief Generate ResNet-18 network (23 layers)
     */
    DnnImage generate_resnet18(const NetworkConfig& config = NetworkConfig());

    /**
     * @brief Generate ResNet-50 network
     */
    DnnImage generate_resnet50(const NetworkConfig& config = NetworkConfig());

    /**
     * @brief Generate simple test network (3 layers)
     */
    DnnImage generate_test_network(const NetworkConfig& config = NetworkConfig());

    /**
     * @brief Set latency parameters using tile-based Mode A derivation
     */
    void set_tiled_latency_params(LayerSpec& spec,
                                   uint32_t tile_height = 0,
                                   uint32_t tile_width = 0,
                                   uint32_t l2_buffer_kb = 256,
                                   Dataflow dataflow = Dataflow::WS,
                                   PeConfig pe_config = PeConfig()) const;

private:
    // --- HW Configuration (set via configure()) ---
    bool configured_ = false;
    PeConfig pe_config_;
    Dataflow dataflow_ = Dataflow::WS;
    uint32_t l2_buffer_kb_ = 256;
    double threshold_ratio_ = 1.0;  ///< prefetch_in scaling: 1.0=preload, <1.0=streaming
    PeGroupingMode pe_grouping_mode_ = PeGroupingMode::Fixed;

    struct LayerPeOverride {
        FuncType type;
        PeConfig pe;
    };
    std::vector<LayerPeOverride> layer_pe_overrides_;

    struct LayerIndexPeOverride {
        uint32_t layer_index;
        PeConfig pe;
    };
    std::vector<LayerIndexPeOverride> layer_index_pe_overrides_;
    mutable uint32_t layer_sequence_index_ = 0;

    /**
     * @brief Apply latency parameters using current configuration
     *
     * If configure() was called, uses set_tiled_latency_params.
     * Otherwise falls back to set_default_latency_params.
     */
    void apply_latency_params(LayerSpec& spec) const;

    /**
     * @brief Resolve effective PE config for a layer
     */
    PeConfig resolve_pe_config(const LayerSpec& spec, uint32_t layer_index) const;

    /**
     * @brief Estimate per-output operations n from layer spec
     */
    static uint32_t estimate_per_output_ops(const LayerSpec& spec);

    /**
     * @brief Clamp invalid PE config combinations to safe bounds
     */
    static PeConfig sanitize_pe_config(PeConfig pe);

    /**
     * @brief Add MobileNetV2 inverted residual block
     *
     * Updates in_h, in_w, in_c to output dimensions.
     */
    void add_inverted_residual_block(
        DnnImageCompiler& compiler,
        uint32_t& current_addr,
        uint32_t& in_h, uint32_t& in_w, uint32_t& in_c,
        uint32_t out_c, uint32_t expansion_factor, uint32_t stride);

    /**
     * @brief Add depthwise separable block (MobileNetV1 building block)
     *
     * DW Conv 3x3 + PW Conv 1x1 (2 layers).
     * Updates in_h, in_w, in_c to output dimensions.
     */
    void add_depthwise_separable_block(
        DnnImageCompiler& compiler,
        uint32_t& current_addr,
        uint32_t& in_h, uint32_t& in_w, uint32_t& in_c,
        uint32_t out_channels, uint32_t stride);

    /**
     * @brief Add VGG block (N conv layers + 1 pool layer)
     *
     * Updates in_h, in_w, in_c to output dimensions (post-pool).
     */
    void add_vgg_block(
        DnnImageCompiler& compiler,
        uint32_t& current_addr,
        uint32_t& in_h, uint32_t& in_w,
        uint32_t& in_c, uint32_t out_c, uint32_t num_conv_layers);

    /**
     * @brief Add ResNet BasicBlock (2 conv3x3 + optional shortcut)
     *
     * Updates in_h, in_w, in_c to output dimensions.
     */
    void add_residual_block(
        DnnImageCompiler& compiler,
        uint32_t& current_addr,
        uint32_t& in_h, uint32_t& in_w,
        uint32_t& in_c, uint32_t out_c, bool downsample);

    /**
     * @brief Calculate tensor size in bytes
     */
    uint32_t tensor_size(uint32_t h, uint32_t w, uint32_t c) const;

    /**
     * @brief Create default latency parameters for a layer
     */
    void set_default_latency_params(LayerSpec& spec,
                                    const PeConfig& pe_config = PeConfig()) const;
};

} // namespace flexnpu_sim
