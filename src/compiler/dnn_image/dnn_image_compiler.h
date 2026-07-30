/**
 * @file dnn_image_compiler.h
 * @brief DNN Image Compiler
 *
 * Compiles a network definition (list of layer specs) into
 * the canonical packet-based DNN Image format
 * (DnnImageHeader + LayerPacketHeader + FunctionDescriptor[]).
 * Pure C++ (no SystemC dependency).
 */

#pragma once

#include "compiler/dnn_image/dnn_image_format.h"
#include "compiler/dnn_image/dnn_image_packet.h"
#include "compiler/dnn_image/tile_math.h"
#include "common/types.h"
#include <vector>
#include <string>
#include <cstdint>

namespace flexnpu_sim {

// ============================================================================
// MappingParams — Mode B (derived) input
// ============================================================================

/// Paper §3 mapping degrees of freedom. Passed to add_layer_derived alongside
/// LayerGeometry and TileShape. Latency descriptor is derived from these.
struct MappingParams {
    Dataflow dataflow = Dataflow::WS;
    PeType   pe_type  = PeType::Systolic;   ///< used to derive l when l_override=0
    double   delta    = 0.0;                ///< δ ∈ [0,1] writeback fraction
    double   beta_i   = 0.0;                ///< β_I input-path routing
    double   beta_w   = 0.0;                ///< β_W weight-path routing
    double   beta_o   = 1.0;                ///< β_O = 1 - β_I - β_W
    double   alpha    = 1.0;                ///< α pre-fetch ratio ∈ (0,1]
    uint32_t k        = 1;                  ///< PEs per PEG (= ops_per_pass)
    uint32_t g        = 1;                  ///< PEGs per PES (= n_min)
    uint32_t s        = 1;                  ///< PES per array
    uint32_t l_override = 0;                ///< 0 = derive from (pe_type, k); >0 = explicit
};

/// Validate MappingParams. Returns empty string on success, error description
/// on failure. Checks: β sum = 1, δ/α ranges, k/g/s ≥ 1.
std::string validate_mapping(const MappingParams& m);

/// Derive pass latency l from (pe_type, k). l_override takes precedence if nonzero.
uint32_t derive_pass_latency(const MappingParams& m);

// ============================================================================
// Layer Specification (High-Level)
// ============================================================================

/**
 * @brief High-level layer specification
 *
 * This is the input to the compiler. It gets compiled into a
 * FunctionDescriptor (binary format) in the DNN Image packet.
 */
struct LayerSpec {
    std::string name;           ///< Layer name (for debugging)
    FuncType type;             ///< Layer type
    uint32_t stride;            ///< Convolution/pooling stride
    uint32_t padding;           ///< Zero padding
    uint32_t dilation;          ///< Dilation factor (conv only)

    // Input tensor shape
    uint32_t input_height;
    uint32_t input_width;
    uint32_t input_channels;

    // Kernel tensor shape
    uint32_t kernel_height;
    uint32_t kernel_width;
    uint32_t kernel_channels;   ///< Output channels for conv
    uint32_t kernel_count;      ///< Number of kernels

    // Memory addresses (DRAM offsets)
    uint32_t input_address;     ///< Input tensor DRAM address
    uint32_t weight_address;    ///< Weight DRAM address
    uint32_t output_address;    ///< Output tensor DRAM address

    // Latency model parameters
    uint32_t fetch_operand0;
    uint32_t fetch_operand1;
    uint32_t write_output;   ///< pass-completion budget of the packet compute
    uint32_t prefetch_in;
    uint32_t prefetch_wt;
    uint32_t n_min; ///< positive pass-issue width floor
    uint32_t n_max; ///< pass-issue width upper bound
    uint32_t issue_latency;        ///< issue-to-completion-ready pipeline cycles
    uint32_t post_completion_cycles = 0;  ///< controller-accounted layer-end overhead
    uint32_t enable_gran = 0;///< tile-level issue-width statistic override
    uint32_t ops_per_output = 1;    ///< ops_per_output: ops per output
    uint32_t sparsity_mode = 0;     ///< SparsityMode::None

    /**
     * @brief Default constructor
     */
    LayerSpec()
        : type(FuncType::Conv2D), stride(1), padding(0), dilation(1),
          input_height(0), input_width(0), input_channels(0),
          kernel_height(0), kernel_width(0), kernel_channels(0), kernel_count(0),
          input_address(0), weight_address(0), output_address(0),
          fetch_operand0(0), fetch_operand1(0),
          write_output(0), prefetch_in(1), prefetch_wt(1),
          n_min(1), n_max(1),
          issue_latency(0), ops_per_output(1),
          sparsity_mode(0) {}
};

// ============================================================================
// DNN Image Blob
// ============================================================================

/**
 * @brief Compiled DNN Image binary blob
 */
struct DnnImage {
    std::vector<uint8_t> data;  ///< Binary data

    /**
     * @brief Get total size in bytes
     */
    size_t size() const { return data.size(); }

    /**
     * @brief Check if empty
     */
    bool empty() const { return data.empty(); }

    /**
     * @brief Get raw data pointer
     */
    const uint8_t* raw() const { return data.data(); }
};

// ============================================================================
// DNN Image Compiler
// ============================================================================

/**
 * @brief DNN Image Compiler
 *
 * Usage:
 *   DnnImageCompiler compiler;
 *   compiler.add_layer(layer_spec_1);
 *   compiler.add_layer(layer_spec_2);
 *   DnnImage image = compiler.compile();
 *   compiler.save("network.dnn");
 */
class DnnImageCompiler {
public:
    DnnImageCompiler() = default;

    /**
     * @brief Add a layer to the network (Mode A — direct, legacy LayerSpec path)
     * @param spec High-level layer specification
     */
    void add_layer(const LayerSpec& spec);

    /**
     * @brief Add a layer — Mode A (direct) using geometry + latency + tile.
     *
     * Caller supplies the 6/8 latency-model parameters directly. Tile shape is
     * honored on the runtime side when flags.tile_enable is set.
     * @param geom layer geometry (shape, stride, addresses)
     * @param latency LatencyDescriptor (F^I, F^W, F^O, l, n_max, n_min, ...)
     * @param tile TileShape (zero axis = full extent)
     * @param ops_per_output ops-per-output for MAC stats
     * @param sparsity sparsity mode selector
     */
    void add_layer_direct(const LayerGeometry& geom,
                          const LatencyDescriptor& latency,
                          const TileShape& tile,
                          uint32_t ops_per_output,
                          SparsityMode sparsity = SparsityMode::None);

    /**
     * @brief Add a layer — Mode B (derived) from mapping params + tile shape.
     *
     * Computes LatencyDescriptor from LayerGeometry + MappingParams + TileShape
     * following paper §3. flags.derived_mode is set.
     *
     * @note Step 4 provides the API; full derivation (per-tile |I_j|/|W_j|
     *       summation with edge handling and traversal-specific reuse) is
     *       wired up in Step 7 alongside the running-example regression test.
     */
    void add_layer_derived(const LayerGeometry& geom,
                           const MappingParams& mapping,
                           const TileShape& tile);

    // ========================================================================
    // R10 — image blob bundling (weight data, first-layer input fmap, dtype)
    // ========================================================================

    /// Attach a weight byte blob to a layer. build_dnn_image_packets() embeds
    /// it right after the layer's function descriptors. If multiple layers
    /// share a layer_id, the latest attachment wins.
    void attach_weights(uint32_t layer_id, std::vector<uint8_t> bytes);

    /// Attach the first-layer input feature map blob. Stored at
    /// DnnImageHeader.input_fmap_offset after build.
    void set_first_layer_input(std::vector<uint8_t> bytes);

    /// Default numerical dtype recorded in DnnImageHeader.default_dtype.
    /// Per-tensor override lives on TensorDescriptor.dtype.
    void set_default_dtype(TensorDType dt) { default_dtype_ = dt; }
    TensorDType default_dtype() const { return default_dtype_; }

    /**
     * @brief Clear all layers
     */
    void clear();

    /**
     * @brief Get number of layers
     */
    size_t num_layers() const { return layer_specs_.size(); }

    /**
     * @brief Compile network into DNN Image binary
     * @return DNN Image binary blob
     */
    DnnImage compile() const;

    /**
     * @brief Compile and save to file
     * @param output_path Output file path
     * @return true if successful
     */
    bool save(const std::string& output_path) const;

    /**
     * @brief Validate layer specifications
     * @return Vector of error messages (empty if valid)
     */
    std::vector<std::string> validate() const;

private:
    struct PendingTile {
        TileShape tile{};
        uint32_t  flags = 0;
    };
    std::vector<LayerSpec> layer_specs_;
    std::vector<PendingTile> pending_tiles_;  ///< parallel to layer_specs_; zero-init for legacy add_layer()

    // R10 blobs
    std::vector<LayerKernelBlob> kernel_blobs_;
    std::vector<uint8_t> input_fmap_blob_;
    TensorDType default_dtype_ = TensorDType::INT8;

    /**
     * @brief Convert LayerSpec to LayerDescriptor
     */
    LayerDescriptor compile_layer(uint32_t layer_id,
                                   const LayerSpec& spec) const;

    /**
     * @brief Calculate output tensor dimensions
     */
    void calculate_output_shape(const LayerSpec& spec,
                                 uint32_t& out_height,
                                 uint32_t& out_width,
                                 uint32_t& out_channels) const;
};

} // namespace flexnpu_sim
