/**
 * @file dnn_image_compiler.cpp
 * @brief DNN Image Compiler Implementation
 */

#include "compiler/dnn_image/dnn_image_compiler.h"
#include "compiler/dnn_image/dnn_image_packet.h"
#include <fstream>
#include <cstring>
#include <cmath>
#include <sstream>

namespace flexnpu_sim {

// ============================================================================
// MappingParams helpers
// ============================================================================

std::string validate_mapping(const MappingParams& m) {
    std::ostringstream oss;
    const double bsum = m.beta_i + m.beta_w + m.beta_o;
    if (std::abs(bsum - 1.0) > 1e-6) {
        oss << "beta_i + beta_w + beta_o must equal 1 (got " << bsum << ")";
        return oss.str();
    }
    if (m.delta < 0.0 || m.delta > 1.0) return "delta out of [0,1]";
    if (m.alpha <= 0.0 || m.alpha > 1.0) return "alpha out of (0,1]";
    if (m.beta_i < 0.0 || m.beta_w < 0.0 || m.beta_o < 0.0) return "beta_* must be >= 0";
    if (m.k == 0 || m.g == 0 || m.s == 0) return "k, g, s must be >= 1";
    return {};
}

uint32_t derive_pass_latency(const MappingParams& m) {
    if (m.l_override != 0) return m.l_override;
    if (m.pe_type == PeType::Systolic) return m.k;
    // Adder tree: l = 1 + ceil(log2(k)); k==1 → l = 1.
    uint32_t log2_ceil = 0;
    uint32_t v = (m.k > 1) ? (m.k - 1) : 0;
    while (v > 0) { log2_ceil++; v >>= 1; }
    return 1u + log2_ceil;
}

static FuncType to_function_type(FuncType t) {
    switch (t) {
        case FuncType::Conv2D:          return FuncType::Conv2D;
        case FuncType::DepthwiseConv2D: return FuncType::DepthwiseConv2D;
        case FuncType::FullyConnected:  return FuncType::FullyConnected;
        case FuncType::Pooling:         return FuncType::Pooling;
        case FuncType::Activation:      return FuncType::Activation;
        case FuncType::MatMul:          return FuncType::MatMul;
        case FuncType::ElementWise:     return FuncType::ElementWise;
        default:                         return FuncType::ElementWise;
    }
}

void DnnImageCompiler::add_layer(const LayerSpec& spec) {
    layer_specs_.push_back(spec);
    pending_tiles_.push_back({});  // legacy path: no tile, no flags
}

// ============================================================================
// Mode A — add_layer_direct (geometry + explicit latency descriptor + tile)
// ============================================================================

void DnnImageCompiler::add_layer_direct(const LayerGeometry& geom,
                                        const LatencyDescriptor& latency,
                                        const TileShape& tile,
                                        uint32_t ops_per_output,
                                        SparsityMode sparsity) {
    LayerSpec spec;
    spec.name                   = "";
    spec.type                   = geom.type;
    spec.stride                 = geom.stride;
    spec.padding                = geom.padding;
    spec.dilation               = geom.dilation;
    spec.input_height           = geom.input_h;
    spec.input_width            = geom.input_w;
    spec.input_channels         = geom.input_c;
    spec.kernel_height          = geom.kernel_h;
    spec.kernel_width           = geom.kernel_w;
    spec.kernel_channels        = geom.input_c;       // filled for descriptor readability
    spec.kernel_count           = geom.kernel_oc;
    spec.input_address          = geom.input_addr;
    spec.weight_address         = geom.weight_addr;
    spec.output_address         = geom.output_addr;
    spec.fetch_operand0   = latency.fetch_operand0;
    spec.fetch_operand1  = latency.fetch_operand1;
    spec.write_output    = latency.write_output;
    spec.prefetch_in        = latency.prefetch_in;
    spec.prefetch_wt       = latency.prefetch_wt;
    spec.n_min  = latency.n_min;
    spec.n_max  = latency.n_max;
    spec.issue_latency         = latency.issue_latency;
    spec.post_completion_cycles = latency.post_completion_cycles;
    spec.enable_gran     = latency.enable_gran;
    spec.ops_per_output         = ops_per_output;
    spec.sparsity_mode          = static_cast<uint32_t>(sparsity);

    // Stash tile into a pending slot. LayerSpec doesn't carry TileShape yet,
    // so we record it in parallel and splice it in at compile() time.
    layer_specs_.push_back(spec);
    pending_tiles_.push_back({tile, /*flags=*/
                              (tile.t_r || tile.t_c || tile.t_oc || tile.t_ic)
                                  ? FuncFlag::TileEnable : 0u});
}

// ============================================================================
// Mode B — add_layer_derived (stub; full derivation filled by Step 7 tests)
// ============================================================================

void DnnImageCompiler::add_layer_derived(const LayerGeometry& geom,
                                         const MappingParams& mapping,
                                         const TileShape& tile) {
    // Validate early; we treat failures as "produce a zero-traffic placeholder"
    // so that compile() callers can inspect validate() before emitting.
    (void)validate_mapping(mapping);

    OutputShape out = compute_output_shape(geom);
    TileAxes axes{};
    std::string terr = resolve_tile(geom, out, tile, axes);
    (void)terr;  // full error plumbing comes with Step 7

    const uint64_t output_elems = layer_output_element_count(out);
    const uint32_t l   = derive_pass_latency(mapping);
    const uint32_t nmax = mapping.g * mapping.s;
    const uint32_t nmin = mapping.g;

    // Derive n(o): ops per output element. Conv: kH*kW*IC; MatMul: K; etc.
    uint64_t n_per_out = 1;
    switch (geom.type) {
        case FuncType::Conv2D:
            n_per_out = 1ull * geom.kernel_h * geom.kernel_w * geom.input_c;
            break;
        case FuncType::DepthwiseConv2D:
            n_per_out = 1ull * geom.kernel_h * geom.kernel_w;
            break;
        case FuncType::FullyConnected:
            n_per_out = geom.input_c;
            break;
        case FuncType::MatMul:
            n_per_out = geom.input_w;  // K axis
            break;
        case FuncType::Pooling:
            n_per_out = 1ull * geom.kernel_h * geom.kernel_w;
            break;
        case FuncType::Activation:
        case FuncType::ElementWise:
            n_per_out = 1;
            break;
        // Normalization / reduction ops reduce over the feature (channel) axis:
        // each output touches all C inputs (softmax sum, layernorm mean+var,
        // reduce). Convention: reduce axis = channels (shape_dim[2]/input_c),
        // the op-general generalization of FullyConnected's n(o)=input_c.
        case FuncType::Softmax:
        case FuncType::LayerNorm:
        case FuncType::Reduction:
            n_per_out = geom.input_c;
            break;
        case FuncType::Gather:  // index-select copy: one input per output
            n_per_out = 1;
            break;
        default:
            n_per_out = 1;
            break;
    }
    const uint32_t m_passes = static_cast<uint32_t>((n_per_out + mapping.k - 1) / mapping.k);

    // Base traversal traffic: halo-aware summation over all tiles (edge tiles
    // counted at their true, smaller extents). The δ-driven writeback-reload
    // term is added on top below.
    PerTileVolumes pv = per_tile_volumes(geom, axes);
    AggregateVolumes agg = aggregate_operand_volumes(geom, out, axes);
    const uint64_t bar_I = agg.input_elems;
    const uint64_t bar_W = agg.weight_elems;

    const uint64_t F_O = static_cast<uint64_t>(m_passes) * output_elems;
    const uint64_t reload_term = static_cast<uint64_t>(
        mapping.delta * (m_passes > mapping.g ? m_passes - mapping.g : 0) *
        static_cast<double>(output_elems));
    const uint64_t F_I = bar_I + static_cast<uint64_t>(mapping.beta_i * reload_term);
    const uint64_t F_W = bar_W + static_cast<uint64_t>(mapping.beta_w * reload_term);

    LatencyDescriptor lat{};
    lat.fetch_operand0   = static_cast<uint32_t>(F_I);
    lat.fetch_operand1  = static_cast<uint32_t>(F_W);
    lat.write_output    = static_cast<uint32_t>(F_O);
    lat.prefetch_in        = static_cast<uint32_t>(
        mapping.alpha * pv.input_elems_per_tile);
    lat.prefetch_wt       = static_cast<uint32_t>(
        mapping.alpha * pv.weight_elems_per_tile);
    lat.n_min  = nmin;
    lat.n_max  = nmax;
    lat.issue_latency         = l;
    lat.post_completion_cycles = 0;

    add_layer_direct(geom, lat, tile,
                     static_cast<uint32_t>(n_per_out),
                     SparsityMode::None);
    // Mark the last-added record as derived-mode.
    if (!pending_tiles_.empty()) {
        pending_tiles_.back().flags |= FuncFlag::DerivedMode;
    }
}

void DnnImageCompiler::clear() {
    layer_specs_.clear();
    pending_tiles_.clear();
    kernel_blobs_.clear();
    input_fmap_blob_.clear();
    default_dtype_ = TensorDType::INT8;
}

void DnnImageCompiler::attach_weights(uint32_t layer_id, std::vector<uint8_t> bytes) {
    for (auto& kb : kernel_blobs_) {
        if (kb.layer_id == layer_id) { kb.bytes = std::move(bytes); return; }
    }
    kernel_blobs_.push_back({layer_id, std::move(bytes)});
}

void DnnImageCompiler::set_first_layer_input(std::vector<uint8_t> bytes) {
    input_fmap_blob_ = std::move(bytes);
}

DnnImage DnnImageCompiler::compile() const {
    DnnImage image;

    std::vector<FunctionRecord> funcs;
    funcs.reserve(layer_specs_.size());
    for (uint32_t i = 0; i < layer_specs_.size(); ++i) {
        LayerDescriptor ld = compile_layer(i, layer_specs_[i]);
        FunctionRecord rec{};
        rec.layer_id = i;          // one packet per layer by default
        rec.desc.function_id = 0;  // single function descriptor in packet
        rec.desc.type = to_function_type(ld.type);
        rec.desc.operand_src[0] = 0xFFFFFFFFu;
        rec.desc.operand_src[1] = 0xFFFFFFFFu;
        rec.desc.operand_src[2] = 0xFFFFFFFFu;
        rec.desc.operand_src[3] = 0xFFFFFFFFu;
        rec.desc.operand_cnt = 0;
        rec.desc.stride = layer_specs_[i].stride;
        rec.desc.padding = layer_specs_[i].padding;
        rec.desc.dilation = layer_specs_[i].dilation;
        rec.desc.preprocess_type = PreprocessType::None;
        rec.desc.preprocess_param0 = 0;
        rec.desc.preprocess_param1 = 0;
        rec.desc.input = ld.input;
        rec.desc.weight = ld.weight;
        rec.desc.output = ld.output;
        rec.desc.latency = ld.latency;
        rec.desc.ops_per_output = ld.ops_per_output;
        rec.desc.sparsity_mode = ld.sparsity_mode;
        if (i < pending_tiles_.size()) {
            rec.desc.tile  = pending_tiles_[i].tile;
            rec.desc.flags = pending_tiles_[i].flags;
        } else {
            rec.desc.tile  = {};
            rec.desc.flags = 0;
        }
        rec.desc.reserved[0] = rec.desc.reserved[1] = rec.desc.reserved[2] = 0;
        funcs.push_back(rec);
    }

    std::string err;
    const std::vector<uint8_t>* input_blob =
        input_fmap_blob_.empty() ? nullptr : &input_fmap_blob_;
    const std::vector<LayerKernelBlob>* kblobs =
        kernel_blobs_.empty() ? nullptr : &kernel_blobs_;
    if (!build_dnn_image_packets(funcs, image.data, &err, kblobs, input_blob,
                                 default_dtype_)) {
        image.data.clear();
    }
    return image;
}

bool DnnImageCompiler::save(const std::string& output_path) const {
    DnnImage image = compile();

    std::ofstream file(output_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.write(reinterpret_cast<const char*>(image.data.data()), image.size());
    file.close();

    return file.good();
}

std::vector<std::string> DnnImageCompiler::validate() const {
    std::vector<std::string> errors;

    for (size_t i = 0; i < layer_specs_.size(); ++i) {
        const auto& spec = layer_specs_[i];

        // Check input dimensions
        if (spec.input_height == 0 || spec.input_width == 0 || spec.input_channels == 0) {
            errors.push_back("Layer " + std::to_string(i) + ": Invalid input dimensions");
        }

        // Check kernel dimensions
        if (spec.type == FuncType::Conv2D || spec.type == FuncType::DepthwiseConv2D) {
            if (spec.kernel_height == 0 || spec.kernel_width == 0) {
                errors.push_back("Layer " + std::to_string(i) + ": Invalid kernel dimensions");
            }
            if (spec.kernel_count == 0) {
                errors.push_back("Layer " + std::to_string(i) + ": kernel_count must be > 0");
            }
        }

        // Check stride
        if (spec.stride == 0) {
            errors.push_back("Layer " + std::to_string(i) + ": stride must be > 0");
        }

        // Check latency parameters
        if (spec.write_output == 0) {
            errors.push_back("Layer " + std::to_string(i) + ": write_output must be > 0");
        }

        if (spec.n_max == 0) {
            errors.push_back("Layer " + std::to_string(i) + ": n_max must be > 0");
        }

        if (spec.n_min > spec.n_max) {
            errors.push_back("Layer " + std::to_string(i) + ": n_min > n_max");
        }
    }

    return errors;
}

LayerDescriptor DnnImageCompiler::compile_layer(uint32_t layer_id, const LayerSpec& spec) const {
    LayerDescriptor desc;

    // Basic info
    desc.layer_id = layer_id;
    desc.type = spec.type;
    desc.stride = spec.stride;
    desc.padding = spec.padding;

    // Input tensor
    desc.input.height = spec.input_height;
    desc.input.width = spec.input_width;
    desc.input.channels = spec.input_channels;
    desc.input.address = spec.input_address;
    desc.input.size = spec.input_height * spec.input_width * spec.input_channels * sizeof(float);

    // Calculate output dimensions
    uint32_t out_h, out_w, out_c;
    calculate_output_shape(spec, out_h, out_w, out_c);

    // Output tensor
    desc.output.height = out_h;
    desc.output.width = out_w;
    desc.output.channels = out_c;
    desc.output.address = spec.output_address;
    desc.output.size = out_h * out_w * out_c * sizeof(float);

    // Weight tensor
    desc.weight.height = spec.kernel_height;
    desc.weight.width = spec.kernel_width;
    desc.weight.channels = spec.kernel_channels;
    desc.weight.address = spec.weight_address;
    desc.weight.size = spec.kernel_height * spec.kernel_width *
                       spec.kernel_channels * spec.kernel_count * sizeof(float);

    // Latency descriptor
    desc.latency.fetch_operand0 = spec.fetch_operand0;
    desc.latency.fetch_operand1 = spec.fetch_operand1;
    desc.latency.write_output = spec.write_output;
    desc.latency.prefetch_in = spec.prefetch_in;
    desc.latency.prefetch_wt = spec.prefetch_wt;
    desc.latency.n_min = spec.n_min;
    desc.latency.n_max = spec.n_max;
    desc.latency.issue_latency = spec.issue_latency;
    desc.latency.post_completion_cycles = spec.post_completion_cycles;
    desc.latency.enable_gran = spec.enable_gran;
    desc.ops_per_output = spec.ops_per_output;
    desc.sparsity_mode = static_cast<SparsityMode>(spec.sparsity_mode);

    // R9 tile / flags — default non-tile (legacy path)
    desc.packet_layer_id = layer_id;
    desc.packet_meta = 0;
    desc.tile = {};
    desc.flags = 0;
    desc.preprocess_param0 = 0;

    return desc;
}

void DnnImageCompiler::calculate_output_shape(const LayerSpec& spec,
                                                uint32_t& out_height,
                                                uint32_t& out_width,
                                                uint32_t& out_channels) const {
    switch (spec.type) {
        case FuncType::Conv2D:
        case FuncType::DepthwiseConv2D:
        case FuncType::Pooling: {
            // Calculate with dilation
            uint32_t effective_kernel_h = spec.kernel_height + (spec.kernel_height - 1) * (spec.dilation - 1);
            uint32_t effective_kernel_w = spec.kernel_width + (spec.kernel_width - 1) * (spec.dilation - 1);

            // Standard convolution/pooling output formula
            out_height = (spec.input_height + 2 * spec.padding - effective_kernel_h) / spec.stride + 1;
            out_width = (spec.input_width + 2 * spec.padding - effective_kernel_w) / spec.stride + 1;

            if (spec.type == FuncType::DepthwiseConv2D) {
                out_channels = spec.input_channels; // Depthwise keeps same channels
            } else if (spec.type == FuncType::Pooling) {
                out_channels = spec.input_channels; // Pooling keeps same channels
            } else {
                out_channels = spec.kernel_count; // Conv2D output channels = kernel count
            }
            break;
        }

        case FuncType::FullyConnected: {
            // FC flattens to 1x1xN
            out_height = 1;
            out_width = 1;
            out_channels = spec.kernel_count; // Output features
            break;
        }

        case FuncType::Activation:
        // Softmax/LayerNorm normalize in place; Gather (index-select) is modeled
        // as a same-shape copy absent an explicit index-count. All preserve shape.
        case FuncType::Softmax:
        case FuncType::LayerNorm:
        case FuncType::Gather: {
            out_height = spec.input_height;
            out_width = spec.input_width;
            out_channels = spec.input_channels;
            break;
        }

        case FuncType::Reduction: {
            // Reduce over the feature (channel) axis: channels collapse to 1,
            // spatial extent preserved.
            out_height = spec.input_height;
            out_width = spec.input_width;
            out_channels = 1;
            break;
        }

        default: {
            // Unknown layer type - preserve input shape
            out_height = spec.input_height;
            out_width = spec.input_width;
            out_channels = spec.input_channels;
            break;
        }
    }
}

} // namespace flexnpu_sim
