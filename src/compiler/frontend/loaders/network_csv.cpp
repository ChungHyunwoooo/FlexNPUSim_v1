#include "compiler/frontend/loaders/network_csv.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <vector>

#include "compiler/dnn_image/dnn_image_packet.h"

namespace flexnpu_sim::frontend {

struct UnifiedLayerEntry {
    uint32_t layer_id = 0;
    LayerSpec spec{};
    uint32_t tile_h = 0;
    uint32_t tile_w = 0;
    PeConfig pe{};
    uint32_t row_order = 0;

    // R9 TileShape (optional; zero = non-tile / legacy path)
    TileShape r9_tile{};
    uint32_t  r9_flags = 0;
    bool      has_r9_tile = false;

    uint32_t function_id = 0;
    bool has_function_id = false;
    FuncType function_type = FuncType::Conv2D;
    uint32_t i1_connect = 0xFFFFFFFFu;
    uint32_t i2_connect = 0xFFFFFFFFu;
    bool has_i1_connect = false;
    bool has_i2_connect = false;
    PreprocessType preprocess_type = PreprocessType::None;
    uint32_t preprocess_param0 = 0;
    uint32_t preprocess_param1 = 0;

    bool has_input_addr = false;
    bool has_kernel_addr = false;
    bool has_output_addr = false;

    // Mode 2 oracle override (paper §III 6 parameters): if has_oracle is set,
    // these values replace the compiler-derived LatencyDescriptor fields after
    // set_tiled_latency_params runs. CSV columns: f_i, f_w, f_o, l_pass,
    // n_max, n_min (required); tau_i, tau_w, post_cycles (optional).
    bool has_oracle = false;
    uint32_t oracle_f_i = 0;
    uint32_t oracle_f_w = 0;
    uint32_t oracle_f_o = 0;
    uint32_t oracle_l   = 0;
    uint32_t oracle_n_max = 0;
    uint32_t oracle_n_min = 0;
    bool has_oracle_tau_i = false; uint32_t oracle_tau_i = 0;
    bool has_oracle_tau_w = false; uint32_t oracle_tau_w = 0;
    bool has_oracle_post  = false; uint32_t oracle_post  = 0;
};

static std::string trim_copy(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) b++;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) e--;
    return s.substr(b, e - b);
}

static std::string to_lower_copy(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) cols.push_back(trim_copy(tok));
    return cols;
}

static bool try_parse_u32(const std::string& s, uint32_t& out) {
    const std::string t = trim_copy(s);
    if (t.empty()) return false;
    try {
        size_t pos = 0;
        const unsigned long v = std::stoul(t, &pos, 10);
        if (pos != t.size()) return false;
        if (v > std::numeric_limits<uint32_t>::max()) return false;
        out = static_cast<uint32_t>(v);
        return true;
    } catch (...) {
        return false;
    }
}

static std::string normalize_header(const std::string& s) {
    std::string t = trim_copy(s);
    for (char& c : t) {
        if (c == ' ' || c == '-' || c == '/' || c == '.') c = '_';
    }
    return t;
}

static int find_column(const std::map<std::string, size_t>& idx,
                       std::initializer_list<const char*> keys) {
    for (const char* k : keys) {
        auto it = idx.find(k);
        if (it != idx.end()) return static_cast<int>(it->second);

        if (std::char_traits<char>::length(k) <= 1) continue;

        const std::string lk = to_lower_copy(k);
        for (const auto& kv : idx) {
            if (to_lower_copy(kv.first) == lk) {
                return static_cast<int>(kv.second);
            }
        }
    }
    return -1;
}

static bool parse_layer_type_token(const std::string& token, FuncType& out) {
    const std::string t = to_lower_copy(trim_copy(token));
    if (t == "conv2d" || t == "conv") { out = FuncType::Conv2D; return true; }
    if (t == "depthwiseconv2d" || t == "depthwise_conv2d" || t == "dwconv" || t == "dw") {
        out = FuncType::DepthwiseConv2D; return true;
    }
    if (t == "fullyconnected" || t == "fully_connected" || t == "fc") {
        out = FuncType::FullyConnected; return true;
    }
    if (t == "pooling" || t == "pool" || t == "maxpool" || t == "avgpool") {
        out = FuncType::Pooling; return true;
    }
    if (t == "activation" || t == "act" || t == "relu" || t == "gelu") {
        out = FuncType::Activation; return true;
    }
    if (t == "matmul" || t == "mm") { out = FuncType::MatMul; return true; }
    if (t == "elementwise" || t == "eltwise" || t == "ew" || t == "preprocess" || t == "pre") {
        out = FuncType::ElementWise; return true;
    }
    if (t == "softmax") { out = FuncType::Softmax; return true; }
    if (t == "layernorm" || t == "layer_norm" || t == "ln") {
        out = FuncType::LayerNorm; return true;
    }
    if (t == "reduction" || t == "reduce") { out = FuncType::Reduction; return true; }
    if (t == "gather") { out = FuncType::Gather; return true; }
    return false;
}

static bool parse_function_type_token(const std::string& token, FuncType& out) {
    const std::string t = to_lower_copy(trim_copy(token));
    if (t == "conv2d" || t == "conv") { out = FuncType::Conv2D; return true; }
    if (t == "depthwiseconv2d" || t == "depthwise_conv2d" || t == "dwconv" || t == "dw") {
        out = FuncType::DepthwiseConv2D; return true;
    }
    if (t == "fullyconnected" || t == "fully_connected" || t == "fc" || t == "linear") {
        out = FuncType::FullyConnected; return true;
    }
    if (t == "pooling" || t == "pool" || t == "maxpool" || t == "avgpool") {
        out = FuncType::Pooling; return true;
    }
    if (t == "activation" || t == "act" || t == "relu" || t == "gelu") {
        out = FuncType::Activation; return true;
    }
    if (t == "matmul" || t == "mm") { out = FuncType::MatMul; return true; }
    if (t == "elementwise" || t == "eltwise" || t == "ew" || t == "add" || t == "mul") {
        out = FuncType::ElementWise; return true;
    }
    if (t == "residual" || t == "skipadd") { out = FuncType::Residual; return true; }
    if (t == "concat" || t == "concatenate") { out = FuncType::Concat; return true; }
    if (t == "deconv" || t == "deconvolution") { out = FuncType::Deconvolution; return true; }
    if (t == "preprocess" || t == "pre" || t == "zero_skip" || t == "zeroskip") {
        out = FuncType::Preprocess; return true;
    }
    if (t == "softmax") { out = FuncType::Softmax; return true; }
    if (t == "layernorm" || t == "layer_norm" || t == "ln") {
        out = FuncType::LayerNorm; return true;
    }
    if (t == "reduction" || t == "reduce") { out = FuncType::Reduction; return true; }
    if (t == "gather") { out = FuncType::Gather; return true; }
    return false;
}

static bool parse_preprocess_type_token(const std::string& token, PreprocessType& out) {
    const std::string t = to_lower_copy(trim_copy(token));
    if (t.empty() || t == "none" || t == "0") {
        out = PreprocessType::None;
        return true;
    }
    if (t == "zero_skipping" || t == "zero-skipping" || t == "zero_skip" || t == "zeroskip" ||
        t == "nullhop") {
        out = PreprocessType::ZeroSkipping;
        return true;
    }
    return false;
}

static bool parse_pe_type_token(const std::string& token, PeType& out) {
    const std::string t = to_lower_copy(trim_copy(token));
    if (t == "systolic" || t == "sa") {
        out = PeType::Systolic;
        return true;
    }
    if (t == "addertree" || t == "adder_tree" || t == "tree") {
        out = PeType::AdderTree;
        return true;
    }
    return false;
}

static void compute_output_shape(const LayerSpec& spec,
                                 uint32_t& out_h,
                                 uint32_t& out_w,
                                 uint32_t& out_c) {
    switch (spec.type) {
        case FuncType::Conv2D:
        case FuncType::DepthwiseConv2D:
        case FuncType::Pooling: {
            const uint32_t effective_kernel_h =
                spec.kernel_height + (spec.kernel_height - 1) * (spec.dilation - 1);
            const uint32_t effective_kernel_w =
                spec.kernel_width + (spec.kernel_width - 1) * (spec.dilation - 1);
            out_h = (spec.input_height + 2 * spec.padding - effective_kernel_h) / spec.stride + 1;
            out_w = (spec.input_width + 2 * spec.padding - effective_kernel_w) / spec.stride + 1;
            if (spec.type == FuncType::DepthwiseConv2D || spec.type == FuncType::Pooling) {
                out_c = spec.input_channels;
            } else {
                out_c = spec.kernel_count;
            }
            return;
        }
        case FuncType::FullyConnected:
            out_h = 1;
            out_w = 1;
            out_c = spec.kernel_count;
            return;
        case FuncType::Activation:
        case FuncType::ElementWise:
            out_h = spec.input_height;
            out_w = spec.input_width;
            out_c = spec.input_channels;
            return;
        case FuncType::MatMul:
            out_h = spec.input_height;
            out_w = spec.input_width;
            out_c = (spec.kernel_count > 0) ? spec.kernel_count : spec.input_channels;
            return;
        case FuncType::Reduction:
            // Reduce over the feature (channel) axis: channels collapse to 1.
            out_h = spec.input_height;
            out_w = spec.input_width;
            out_c = 1;
            return;
        default:
            // Softmax/LayerNorm/Gather and other same-shape ops preserve input.
            out_h = spec.input_height;
            out_w = spec.input_width;
            out_c = spec.input_channels;
            return;
    }
}

static uint64_t kernel_bytes(const LayerSpec& spec) {
    switch (spec.type) {
        case FuncType::Conv2D:
        case FuncType::DepthwiseConv2D:
            return static_cast<uint64_t>(spec.kernel_height) * spec.kernel_width *
                   spec.kernel_channels * spec.kernel_count * sizeof(float);
        case FuncType::FullyConnected:
        case FuncType::MatMul:
            return static_cast<uint64_t>(spec.kernel_channels) * spec.kernel_count * sizeof(float);
        default:
            return 0;
    }
}

static std::vector<UnifiedLayerEntry> load_layer_spec_csv(const std::string& path,
                                                          const PeConfig& base_pe,
                                                          uint64_t* out_footprint = nullptr) {
    std::vector<UnifiedLayerEntry> out;
    if (path.empty()) return out;

    std::ifstream ifs(path);
    if (!ifs.is_open()) return out;

    std::string header_line;
    std::map<std::string, size_t> idx;
    uint32_t line_no = 0;
    bool header_ready = false;

    while (std::getline(ifs, header_line)) {
        line_no++;
        std::string t = trim_copy(header_line);
        if (t.empty() || t[0] == '#') continue;
        auto cols = split_csv_line(t);
        for (size_t i = 0; i < cols.size(); ++i) {
            idx[normalize_header(cols[i])] = i;
        }
        header_ready = true;
        break;
    }

    if (!header_ready) return out;

    const int c_layer_id = find_column(idx, {"layer_id", "id"});
    const int c_name = find_column(idx, {"layer_name", "name"});
    const int c_type = find_column(idx, {"layer_type", "type", "op", "operation"});
    const int c_in_h = find_column(idx, {"input_h", "in_h"});
    const int c_in_w = find_column(idx, {"input_w", "in_w"});
    const int c_in_c = find_column(idx, {"input_c", "in_c"});
    const int c_kh = find_column(idx, {"kernel_h", "k_h", "kh"});
    const int c_kw = find_column(idx, {"kernel_w", "k_w", "kw"});
    const int c_kc = find_column(idx, {"kernel_c", "k_c", "kc"});
    const int c_kcnt = find_column(idx, {"kernel_count", "out_c", "k_count", "kernel_n"});
    const int c_stride = find_column(idx, {"stride", "s"});
    const int c_padding = find_column(idx, {"padding", "pad", "p"});
    const int c_dilation = find_column(idx, {"dilation", "d"});
    const int c_tile_h = find_column(idx, {"tile_h", "t_h", "th"});
    const int c_tile_w = find_column(idx, {"tile_w", "t_w", "tw"});
    const int c_k = find_column(idx, {"ops_per_pass", "k"});
    const int c_g = find_column(idx, {"parallel_passes", "g"});
    const int c_s = find_column(idx, {"num_output_lanes", "s", "pes_count", "num_pes"});
    const int c_psum_kb = find_column(idx, {"psum_kb", "psum_capacity_kb"});
    const int c_pe_type = find_column(idx, {"pe_type", "petype"});
    const int c_function_id = find_column(idx, {"function_id", "fid"});
    const int c_function_type = find_column(idx, {"function_type", "ftype", "func_type"});
    const int c_i1_connect = find_column(idx, {"i1_connect", "input1_connect"});
    const int c_i2_connect = find_column(idx, {"i2_connect", "input2_connect"});
    const int c_pre_type = find_column(idx, {"preprocess_type", "preprocess", "pre_type"});
    const int c_pre_p0 = find_column(idx, {"preprocess_param0", "pre_param0", "skip_permille"});
    const int c_pre_p1 = find_column(idx, {"preprocess_param1", "pre_param1"});
    // Per-layer 𝓔 override — optional. Values: lockstep | independent |
    // grouped:N (N = emit-width floor). Empty/auto/invalid = inherit global.
    const int c_eg = find_column(idx, {"enable_granularity", "eg"});
    const int c_in_addr = find_column(idx, {"input_addr", "input_address"});
    const int c_k_addr = find_column(idx, {"kernel_addr", "kernel_address"});
    const int c_out_addr = find_column(idx, {"output_addr", "output_address"});

    // Mode 2 oracle (paper 6 params) — all optional. If f_i/f_w/f_o/l/n_max/n_min
    // are all present, the row uses oracle values instead of compiler-derived.
    const int c_oracle_f_i    = find_column(idx, {"f_i", "F_I", "fetch_operand0", "total_input_operands"});
    const int c_oracle_f_w    = find_column(idx, {"f_w", "F_W", "fetch_operand1", "total_weight_operands"});
    const int c_oracle_f_o    = find_column(idx, {"f_o", "F_O", "write_output", "total_output_writes"});
    const int c_oracle_l      = find_column(idx, {"l_pass", "issue_latency", "pipeline_depth"});
    const int c_oracle_n_max  = find_column(idx, {"n_max", "n_max", "max_outputs_per_cycle"});
    const int c_oracle_n_min  = find_column(idx, {"n_min", "n_min", "min_outputs_per_cycle"});
    const int c_oracle_tau_i  = find_column(idx, {"tau_i", "threshold_input"});
    const int c_oracle_tau_w  = find_column(idx, {"tau_w", "threshold_weight"});
    const int c_oracle_post   = find_column(idx, {"post_cycles", "post_completion_cycles"});

    // R9 TileShape — all optional; presence of any enables tile_enable flag.
    const int c_t_r          = find_column(idx, {"t_r"});
    const int c_t_c          = find_column(idx, {"t_c"});
    const int c_t_oc         = find_column(idx, {"t_oc"});
    const int c_t_ic         = find_column(idx, {"t_ic"});
    const int c_t_kern       = find_column(idx, {"t_kern"});
    const int c_intra_order  = find_column(idx, {"intra_order"});

    // The layer type may be given as `layer_type` or (readable CSVs) as
    // `function_type`. The PE-mapping columns k/g/s are optional: when absent
    // the preset values stand, and oracle rows (f_i..n_min) ignore them.
    const int c_type_eff = (c_type >= 0) ? c_type : c_function_type;
    if (c_layer_id < 0 || c_type_eff < 0 || c_in_h < 0 || c_in_w < 0 ||
        c_in_c < 0 || c_tile_h < 0 || c_tile_w < 0) {
        return out;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        line_no++;
        std::string t = trim_copy(line);
        if (t.empty() || t[0] == '#') continue;

        auto cols = split_csv_line(t);
        if (cols.size() < idx.size()) cols.resize(idx.size());

        auto get_col = [&](int c) -> std::string {
            if (c < 0) return "";
            const size_t ci = static_cast<size_t>(c);
            if (ci >= cols.size()) return "";
            return cols[ci];
        };

        uint32_t layer_id = 0;
        if (!try_parse_u32(get_col(c_layer_id), layer_id)) continue;

        FuncType lt = FuncType::Conv2D;
        if (!parse_layer_type_token(get_col(c_type_eff), lt)) continue;

        UnifiedLayerEntry e;
        e.layer_id = layer_id;
        e.row_order = line_no;
        e.spec.name = get_col(c_name).empty() ? ("layer_" + std::to_string(layer_id)) : get_col(c_name);
        e.spec.type = lt;
        e.spec.stride = 1;
        e.spec.padding = 0;
        e.spec.dilation = 1;
        e.pe = base_pe;
        e.function_type = FuncType::Conv2D;
        e.preprocess_type = PreprocessType::None;

        uint32_t v = 0;
        if (!try_parse_u32(get_col(c_in_h), e.spec.input_height) ||
            !try_parse_u32(get_col(c_in_w), e.spec.input_width) ||
            !try_parse_u32(get_col(c_in_c), e.spec.input_channels)) {
            continue;
        }
        if (try_parse_u32(get_col(c_stride), v)) e.spec.stride = std::max(1u, v);
        if (try_parse_u32(get_col(c_padding), v)) e.spec.padding = v;
        if (try_parse_u32(get_col(c_dilation), v)) e.spec.dilation = std::max(1u, v);

        e.spec.kernel_height = 1;
        e.spec.kernel_width = 1;
        e.spec.kernel_channels = e.spec.input_channels;
        e.spec.kernel_count = e.spec.input_channels;
        if (try_parse_u32(get_col(c_kh), v)) e.spec.kernel_height = std::max(1u, v);
        if (try_parse_u32(get_col(c_kw), v)) e.spec.kernel_width = std::max(1u, v);
        if (try_parse_u32(get_col(c_kc), v)) e.spec.kernel_channels = std::max(1u, v);
        if (try_parse_u32(get_col(c_kcnt), v)) e.spec.kernel_count = std::max(1u, v);

        if (e.spec.type == FuncType::DepthwiseConv2D) {
            // A depthwise filter spans ONE input channel (K_H·K_W·1), and there
            // are C_out = C_in such filters. kernel_channels is the per-filter
            // input-channel depth = 1, NOT C_in — otherwise kernel_bytes and the
            // baked fetch_operand1 become K_H·K_W·C_in·C_out (C_in× too large,
            // violating F^wt ≤ A^wt). kernel_count = C_out = input_channels.
            e.spec.kernel_channels = 1;
            e.spec.kernel_count = e.spec.input_channels;
        } else if (e.spec.type == FuncType::Pooling ||
                   e.spec.type == FuncType::Activation ||
                   e.spec.type == FuncType::ElementWise) {
            e.spec.kernel_channels = 0;
            e.spec.kernel_count = e.spec.input_channels;
        }

        if (try_parse_u32(get_col(c_function_id), v)) {
            e.function_id = v;
            e.has_function_id = true;
        }
        FuncType ft = FuncType::Conv2D;
        if (parse_function_type_token(get_col(c_function_type), ft)) {
            e.function_type = ft;
        } else {
            switch (e.spec.type) {
                case FuncType::Conv2D:          e.function_type = FuncType::Conv2D; break;
                case FuncType::DepthwiseConv2D: e.function_type = FuncType::DepthwiseConv2D; break;
                case FuncType::FullyConnected:  e.function_type = FuncType::FullyConnected; break;
                case FuncType::Pooling:         e.function_type = FuncType::Pooling; break;
                case FuncType::Activation:      e.function_type = FuncType::Activation; break;
                case FuncType::MatMul:          e.function_type = FuncType::MatMul; break;
                case FuncType::ElementWise:     e.function_type = FuncType::ElementWise; break;
                default:                         e.function_type = FuncType::ElementWise; break;
            }
        }

        if (try_parse_u32(get_col(c_i1_connect), v)) {
            e.i1_connect = v;
            e.has_i1_connect = true;
        }
        if (try_parse_u32(get_col(c_i2_connect), v)) {
            e.i2_connect = v;
            e.has_i2_connect = true;
        }

        PreprocessType pre_t = PreprocessType::None;
        if (parse_preprocess_type_token(get_col(c_pre_type), pre_t)) e.preprocess_type = pre_t;
        if (try_parse_u32(get_col(c_pre_p0), v)) e.preprocess_param0 = v;
        if (try_parse_u32(get_col(c_pre_p1), v)) e.preprocess_param1 = v;

        if (!try_parse_u32(get_col(c_tile_h), e.tile_h) ||
            !try_parse_u32(get_col(c_tile_w), e.tile_w)) {
            continue;
        }
        e.tile_h = std::max(1u, e.tile_h);
        e.tile_w = std::max(1u, e.tile_w);

        // k/g/s override the preset PE mapping when present; absent columns
        // leave the preset defaults (already in e.pe) in place.
        try_parse_u32(get_col(c_k), e.pe.ops_per_pass);
        try_parse_u32(get_col(c_g), e.pe.parallel_passes);
        try_parse_u32(get_col(c_s), e.pe.num_output_lanes);
        e.pe.ops_per_pass = std::max(1u, e.pe.ops_per_pass);
        e.pe.parallel_passes = std::max(1u, e.pe.parallel_passes);
        e.pe.num_output_lanes = std::max(1u, e.pe.num_output_lanes);
        {
            uint32_t pv = 0;
            if (c_psum_kb >= 0 && try_parse_u32(get_col(c_psum_kb), pv)) {
                e.pe.partial_sum_buffer_kb = pv;
            }
        }

        PeType row_pe_type = e.pe.pe_type;
        if (parse_pe_type_token(get_col(c_pe_type), row_pe_type)) e.pe.pe_type = row_pe_type;

        if (try_parse_u32(get_col(c_in_addr), v)) {
            e.spec.input_address = v;
            e.has_input_addr = true;
        }
        if (try_parse_u32(get_col(c_k_addr), v)) {
            e.spec.weight_address = v;
            e.has_kernel_addr = true;
        }
        if (try_parse_u32(get_col(c_out_addr), v)) {
            e.spec.output_address = v;
            e.has_output_addr = true;
        }

        // R9 TileShape — optional. Any present column enables tile_enable.
        bool r9_present = false;
        if (try_parse_u32(get_col(c_t_r),  v)) { e.r9_tile.t_r  = v; r9_present = true; }
        if (try_parse_u32(get_col(c_t_c),  v)) { e.r9_tile.t_c  = v; r9_present = true; }
        if (try_parse_u32(get_col(c_t_oc), v)) { e.r9_tile.t_oc = v; r9_present = true; }
        if (try_parse_u32(get_col(c_t_ic), v)) { e.r9_tile.t_ic = v; r9_present = true; }
        if (try_parse_u32(get_col(c_t_kern), v))      { e.r9_tile.t_kern      = v; r9_present = true; }
        if (try_parse_u32(get_col(c_intra_order), v)) { e.r9_tile.intra_order = v; r9_present = true; }
        if (r9_present) {
            e.has_r9_tile = true;
            e.r9_flags = FuncFlag::TileEnable;
        }

        // Per-layer 𝓔 (enable granularity): encoded (mode<<24)|N into the
        // LatencyDescriptor; 0 = inherit hw.compute.enable_gran.
        {
            const std::string eg_tok = to_lower_copy(trim_copy(get_col(c_eg)));
            if (eg_tok == "lockstep") {
                e.spec.enable_gran = 1u << 24;
            } else if (eg_tok == "independent") {
                e.spec.enable_gran = 2u << 24;
            } else if (eg_tok.rfind("grouped:", 0) == 0) {
                uint32_t n = 0;
                if (try_parse_u32(eg_tok.substr(8), n) && n > 0 && n < (1u << 24))
                    e.spec.enable_gran = (3u << 24) | n;
            }
        }

        // Mode 2 oracle: require all 6 core params to be present together.
        bool ok_fi   = try_parse_u32(get_col(c_oracle_f_i),   e.oracle_f_i);
        bool ok_fw   = try_parse_u32(get_col(c_oracle_f_w),   e.oracle_f_w);
        bool ok_fo   = try_parse_u32(get_col(c_oracle_f_o),   e.oracle_f_o);
        bool ok_l    = try_parse_u32(get_col(c_oracle_l),     e.oracle_l);
        bool ok_nmax = try_parse_u32(get_col(c_oracle_n_max), e.oracle_n_max);
        bool ok_nmin = try_parse_u32(get_col(c_oracle_n_min), e.oracle_n_min);
        if (ok_fi && ok_fw && ok_fo && ok_l && ok_nmax && ok_nmin) {
            e.has_oracle = true;
            e.has_oracle_tau_i = try_parse_u32(get_col(c_oracle_tau_i), e.oracle_tau_i);
            e.has_oracle_tau_w = try_parse_u32(get_col(c_oracle_tau_w), e.oracle_tau_w);
            e.has_oracle_post  = try_parse_u32(get_col(c_oracle_post),  e.oracle_post);
        }

        out.push_back(e);
    }

    std::stable_sort(out.begin(), out.end(),
                     [](const UnifiedLayerEntry& a, const UnifiedLayerEntry& b) {
                         if (a.layer_id != b.layer_id) return a.layer_id < b.layer_id;
                         if (a.has_function_id != b.has_function_id) return a.has_function_id > b.has_function_id;
                         if (a.has_function_id && b.has_function_id && a.function_id != b.function_id) {
                             return a.function_id < b.function_id;
                         }
                         return a.row_order < b.row_order;
                     });

    uint32_t current_layer = std::numeric_limits<uint32_t>::max();
    uint32_t auto_fid = 0;
    uint32_t prev_fid = 0xFFFFFFFFu;
    for (auto& e : out) {
        if (e.layer_id != current_layer) {
            current_layer = e.layer_id;
            auto_fid = 0;
            prev_fid = 0xFFFFFFFFu;
        }
        if (!e.has_function_id) {
            e.function_id = auto_fid++;
        } else if (e.function_id >= auto_fid) {
            auto_fid = e.function_id + 1;
        }
        if (!e.has_i1_connect) e.i1_connect = (prev_fid == 0xFFFFFFFFu) ? 0xFFFFFFFFu : prev_fid;
        if (!e.has_i2_connect) e.i2_connect = 0xFFFFFFFFu;
        prev_fid = e.function_id;
    }

    uint64_t cursor = 0;
    // Producer→consumer address chaining: a consumer reads the tensor its
    // producer wrote. Without this every row got a fresh input buffer and
    // the DRAM image was address-disconnected across layers (inter-layer
    // retention/reuse could never see the producer's output).
    std::map<uint32_t, uint32_t> layer_out_addr;  // fid → out addr (this packet)
    uint32_t chain_layer = 0xFFFFFFFFu;
    uint32_t prev_packet_out = 0xFFFFFFFFu;   // last output of previous packet
    uint32_t cur_packet_last_out = 0xFFFFFFFFu;
    for (auto& e : out) {
        if (e.layer_id != chain_layer) {
            chain_layer = e.layer_id;
            prev_packet_out = cur_packet_last_out;
            layer_out_addr.clear();
        }
        const uint64_t in_bytes = static_cast<uint64_t>(e.spec.input_height) *
                                  e.spec.input_width * e.spec.input_channels * sizeof(float);
        if (!e.has_input_addr) {
            auto it = layer_out_addr.find(e.i1_connect);
            if (it != layer_out_addr.end()) {
                e.spec.input_address = it->second;       // in-packet chain
            } else if (layer_out_addr.empty() &&
                       prev_packet_out != 0xFFFFFFFFu) {
                e.spec.input_address = prev_packet_out;   // packet-to-packet
            } else {
                e.spec.input_address = static_cast<uint32_t>(cursor);
            }
        }
        cursor = std::max(cursor, static_cast<uint64_t>(e.spec.input_address) + in_bytes);

        const uint64_t k_bytes = kernel_bytes(e.spec);
        if (!e.has_kernel_addr) e.spec.weight_address = static_cast<uint32_t>(cursor);
        cursor = std::max(cursor, static_cast<uint64_t>(e.spec.weight_address) + k_bytes);

        uint32_t out_h = 0, out_w = 0, out_c = 0;
        compute_output_shape(e.spec, out_h, out_w, out_c);
        const uint64_t out_bytes = static_cast<uint64_t>(out_h) * out_w * out_c * sizeof(float);
        if (!e.has_output_addr) e.spec.output_address = static_cast<uint32_t>(cursor);
        cursor = std::max(cursor, static_cast<uint64_t>(e.spec.output_address) + out_bytes);
        layer_out_addr[e.function_id] = e.spec.output_address;
        cur_packet_last_out = e.spec.output_address;
    }

    // `cursor` is the high-water mark of the (image-relative) placement — the
    // linker's region size. The caller asserts it against the DRAM region.
    if (out_footprint) *out_footprint = cursor;

    return out;
}

bool generate_network_from_layer_spec_csv(const std::string& path,
                                          const config::FlexNpuSimConfig& cfg,
                                          UnifiedCsvModel& out,
                                          std::string& err) {
    const PeConfig pe        = config::pe_config_from(cfg.hw.compute);
    const Dataflow dataflow  = config::dataflow_from(cfg.hw.dataflow);
    const uint32_t l2_kb     = cfg.hw.buffers.global.capacity_kb;
    const double   threshold = config::effective_threshold_ratio(cfg.hw.compute);

    uint64_t footprint = 0;
    auto entries = load_layer_spec_csv(path, pe, &footprint);
    if (entries.empty()) {
        err = "layer spec csv parse failed or empty: " + path;
        return false;
    }
    // Linker bounds check: the placed image must fit the DRAM region declared
    // by the memory map. Overflow here is a mis-sized DRAM config, not a bug in
    // a layer — fail loudly the way a linker reports a region overflow.
    if (footprint > cfg.address_map.dram_size) {
        err = "network footprint " + std::to_string(footprint) +
              " B exceeds DRAM region size " +
              std::to_string(cfg.address_map.dram_size) + " B (address_map.dram_size)";
        return false;
    }

    out = UnifiedCsvModel{};
    out.net_cfg.input_height = entries.front().spec.input_height;
    out.net_cfg.input_width = entries.front().spec.input_width;
    out.net_cfg.input_channels = entries.front().spec.input_channels;
    out.net_cfg.num_classes = entries.back().spec.kernel_count;

    DnnImageGenerator gen;
    gen.configure(pe, dataflow, l2_kb, threshold);
    std::vector<FunctionRecord> packet_records;
    packet_records.reserve(entries.size());

    uint32_t layer_packet_count = 0;
    uint32_t prev_layer_id = std::numeric_limits<uint32_t>::max();
    for (const auto& e : entries) {
        LayerSpec spec = e.spec;
        gen.set_tiled_latency_params(spec, e.tile_h, e.tile_w,
                                     l2_kb, dataflow, e.pe);

        // Mode 2 oracle override: replace compiler-derived 6-param descriptor
        // with values supplied directly in the CSV row.
        if (e.has_oracle) {
            spec.fetch_operand0  = e.oracle_f_i;
            spec.fetch_operand1 = e.oracle_f_w;
            spec.write_output   = e.oracle_f_o;
            spec.issue_latency        = e.oracle_l;
            spec.n_max = e.oracle_n_max;
            spec.n_min = e.oracle_n_min;
            if (e.has_oracle_tau_i) spec.prefetch_in        = e.oracle_tau_i;
            if (e.has_oracle_tau_w) spec.prefetch_wt       = e.oracle_tau_w;
            if (e.has_oracle_post)  spec.post_completion_cycles = e.oracle_post;
        }

        if (e.layer_id != prev_layer_id) {
            layer_packet_count++;
            prev_layer_id = e.layer_id;
        }

        uint32_t out_h = 0, out_w = 0, out_c = 0;
        compute_output_shape(spec, out_h, out_w, out_c);

        FunctionRecord rec{};
        rec.layer_id = e.layer_id;
        rec.desc.function_id = e.function_id;
        rec.desc.type = e.function_type;
        rec.desc.operand_src[0] = e.i1_connect;
        rec.desc.operand_src[1] = e.i2_connect;
        rec.desc.operand_src[2] = 0xFFFFFFFFu;
        rec.desc.operand_src[3] = 0xFFFFFFFFu;
        rec.desc.operand_cnt = (e.i2_connect != 0xFFFFFFFFu) ? 2 : (e.i1_connect != 0xFFFFFFFFu ? 1 : 0);
        rec.desc.stride = spec.stride;
        rec.desc.padding = spec.padding;
        rec.desc.dilation = spec.dilation;
        rec.desc.preprocess_type = e.preprocess_type;
        rec.desc.preprocess_param0 = e.preprocess_param0;
        rec.desc.preprocess_param1 = e.preprocess_param1;
        rec.desc.input.height = spec.input_height;
        rec.desc.input.width = spec.input_width;
        rec.desc.input.channels = spec.input_channels;
        rec.desc.input.address = spec.input_address;
        rec.desc.input.size = spec.input_height * spec.input_width * spec.input_channels * sizeof(float);
        rec.desc.weight.height = spec.kernel_height;
        rec.desc.weight.width = spec.kernel_width;
        rec.desc.weight.channels = spec.kernel_channels;
        rec.desc.weight.address = spec.weight_address;
        rec.desc.weight.size = static_cast<uint32_t>(kernel_bytes(spec));
        rec.desc.output.height = out_h;
        rec.desc.output.width = out_w;
        rec.desc.output.channels = out_c;
        rec.desc.output.address = spec.output_address;
        rec.desc.output.size = out_h * out_w * out_c * sizeof(float);
        rec.desc.latency.fetch_operand0 = spec.fetch_operand0;
        rec.desc.latency.fetch_operand1 = spec.fetch_operand1;
        rec.desc.latency.write_output = spec.write_output;
        rec.desc.latency.prefetch_in = spec.prefetch_in;
        rec.desc.latency.prefetch_wt = spec.prefetch_wt;
        rec.desc.latency.n_min = spec.n_min;
        rec.desc.latency.n_max = spec.n_max;
        rec.desc.latency.issue_latency = spec.issue_latency;
        rec.desc.latency.post_completion_cycles = spec.post_completion_cycles;
        rec.desc.latency.enable_gran = spec.enable_gran;
        rec.desc.ops_per_output = spec.ops_per_output;
        rec.desc.sparsity_mode = static_cast<SparsityMode>(spec.sparsity_mode);
        if (e.has_r9_tile) {
            rec.desc.tile  = e.r9_tile;
            rec.desc.flags = e.r9_flags;
        } else if (e.tile_h < spec.input_height || e.tile_w < spec.input_width) {
            // Legacy tile_h/tile_w < input dim: tiling is implicitly active
            // and the latency model already accounts for tile-induced refetch
            // (set_tiled_latency_params). Skip whole-layer GB fit check by
            // setting TileEnable. Per-tile fit is the compiler's responsibility.
            rec.desc.tile        = {};
            rec.desc.tile.t_r    = e.tile_h;
            rec.desc.tile.t_c    = e.tile_w;
            rec.desc.tile.t_kern = spec.kernel_height * spec.kernel_width;
            rec.desc.flags       = FuncFlag::TileEnable;
        } else {
            rec.desc.tile  = {};
            rec.desc.flags = 0;
        }
        rec.desc.reserved[0] = rec.desc.reserved[1] = rec.desc.reserved[2] = 0;
        packet_records.push_back(rec);
    }

    std::string packet_err;
    if (!build_dnn_image_packets(packet_records, out.image.data, &packet_err)) {
        err = "failed to build packet image from csv: " + packet_err;
        out.image.data.clear();
        return false;
    }

    out.num_layers = static_cast<uint32_t>(entries.size());
    out.num_layer_packets = layer_packet_count;
    return true;
}

}  // namespace flexnpu_sim::frontend

