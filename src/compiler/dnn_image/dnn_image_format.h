/**
 * @file dnn_image_format.h
 * @brief DNN Image binary format definition
 *
 * A DNN Image is a binary serialization of the network's layer information.
 * The simulator reads this image from DRAM and extracts per-layer parameters.
 *
 * Structure (canonical):
 *   [DnnImageHeader]
 *   [input fmap blob]                                            (optional)
 *   [LayerPacketHeader + FunctionDescriptor x num_functions] x num_layer_packets
 *   [weight section: kernel blobs]                               (optional)
 * Kernel blobs are not embedded in the packets; they are gathered in the
 * weight section at the end of the image, and
 * LayerPacketHeader.kernel_data_offset points to them as an image-global
 * offset.
 */

#pragma once

#include <cstdint>

namespace flexnpu_sim {

// ============================================================================
// Magic Number & Version
// ============================================================================

constexpr uint32_t DNN_IMAGE_MAGIC   = 0x4E505544;  // "NPUD"
// v0x00060000: TensorDescriptor.dtype + DnnImageHeader.default_dtype &
// input_fmap_offset (spec-aligned quantization support).
constexpr uint32_t DNN_IMAGE_VERSION = 0x00070000;  // R11: +LatencyDescriptor.enable_gran

// ============================================================================
// Tensor Data Type (R10, spec-aligned)
// ============================================================================

/// Per-tensor numerical representation. Spec FlexNPUSim: INT8/INT16/FP16.
/// FP32 added for functional-model reference verification (non-hardware path).
enum class TensorDType : uint32_t {
    INT8 = 0,
    INT16 = 1,
    FP16 = 2,
    FP32 = 3
};

inline uint32_t tensor_elem_bytes(TensorDType dt) {
    switch (dt) {
        case TensorDType::INT8:  return 1;
        case TensorDType::INT16: return 2;
        case TensorDType::FP16:  return 2;
        case TensorDType::FP32:  return 4;
    }
    return 1;
}

// ============================================================================
// DNN Image Header (32 bytes, R10 v0.6)
// ============================================================================

struct DnnImageHeader {
    uint32_t magic;             // DNN_IMAGE_MAGIC
    uint32_t version;           // DNN_IMAGE_VERSION
    uint32_t num_layers;        // layer count
    uint32_t total_size;        // whole-image size (bytes)
    TensorDType default_dtype;  // image-level default numerical dtype
    uint32_t input_fmap_offset; // offset (bytes) of first-layer input fmap blob
    uint32_t input_fmap_size;   // size (bytes) of first-layer input fmap blob
    uint32_t weight_section_offset; // image-relative start of the global weight
                                    // section (0 if no weights embedded)
};

static_assert(sizeof(DnnImageHeader) == 32, "DnnImageHeader must be 32 bytes");

// ============================================================================
// Tensor Descriptor (24 bytes, R10 v0.6)
// ============================================================================

// shape_dim[] is the op-general (ISA) view; height/width/channels are the
// conv interpretation of the same 3 slots (matmul reads them as M/N/K, etc.).
struct TensorDescriptor {
    union {
        uint32_t shape_dim[3];
        struct { uint32_t height; uint32_t width; uint32_t channels; };
    };
    uint32_t address;      // image-relative offset or DRAM abs (runtime resolves)
    uint32_t size;         // byte count
    TensorDType dtype;     // per-tensor override; 0 inherits default_dtype
};

static_assert(sizeof(TensorDescriptor) == 24, "TensorDescriptor must be 24 bytes");

// ============================================================================
// Latency Descriptor (40 bytes)
// ============================================================================

struct LatencyDescriptor {
    uint32_t fetch_operand0;    // whole-layer input operand count
    uint32_t fetch_operand1;   // whole-layer weight operand count
    uint32_t write_output;     // pass-completion event budget of the packet compute
    uint32_t prefetch_in;         // GB-resident input gate before compute starts
    uint32_t prefetch_wt;        // GB-resident weight gate before compute starts
    uint32_t n_min;   // positive pass-issue width floor per cycle
    uint32_t n_max;   // pass-issue width ceiling per cycle
    uint32_t issue_latency;          // pipeline cycles from issue to completion-ready
    uint32_t post_completion_cycles;  // controller-accounted layer-end overhead
    // Per-layer granularity override for tile-level effective issue-width stats.
    // 0 = inherit the global hw.compute.enable_gran knob. Encoding:
    // (mode<<24)|N with
    // mode 1=lockstep, 2=independent, 3=grouped; N = pass-issue group floor, used
    // by grouped only (consumer clamps N to [1, n_max]).
    uint32_t enable_gran;
};

static_assert(sizeof(LatencyDescriptor) == 40, "LatencyDescriptor must be 40 bytes (R11: +enable_gran)");

// ============================================================================
// Layer Type
// ============================================================================

// Unified opcode (merges the former LayerType/FunctionType pair, plus the
// Transformer opcodes). Values 0-10 preserve the legacy FunctionType encoding.
enum class FuncType : uint32_t {
    Conv2D          = 0,
    DepthwiseConv2D = 1,
    FullyConnected  = 2,
    Pooling         = 3,
    Activation      = 4,
    MatMul          = 5,    ///< matrix multiply (QK^T, AV)
    ElementWise     = 6,    ///< element-wise op (add, mul, sub); residual add
    Preprocess      = 7,    ///< e.g., zero-skipping preprocessor
    Residual        = 8,
    Concat          = 9,
    Deconvolution   = 10,
    Softmax         = 11,   ///< Transformer
    LayerNorm       = 12,
    Reduction       = 13,
    Gather          = 14
};

enum class PreprocessType : uint32_t {
    None         = 0,
    ZeroSkipping = 1
};

enum class SparsityMode : uint32_t {
    None              = 0,  // ignore sparsity; use the original traffic
    RuntimeProfiled   = 1,  // read the actual data; count effective operands at runtime
    Precomputed       = 2   // traffic pre-scaled by the sparsity ratio at compile time
};

// ============================================================================
// Tile Shape (24 bytes) — R9 tile abstraction
// ============================================================================

/// Axis traversal order for tile iteration. Reserved value 0xFF = Custom (future
/// extension). R9 scope covers IC-first (NVDLA/Nullhop WS), OC-first (IS-like),
/// Spatial-first (Eyeriss RS, MIDAP feature-map stationary).
enum class TileTraversal : uint32_t {
    IcFirstOcLast = 0,
    OcFirstIcLast = 1,
    SpatialFirst  = 2,
    KernelFirst   = 3,
    Custom        = 0xFF
};

/// 5-axis superset tile shape. Axis meaning is FuncType-specific:
///   Conv2D/Deconv       : (T_R, T_C, T_OC, T_IC, T_KERN=kH*kW)
///   DepthwiseConv2D     : T_OC == T_IC enforced by validator
///   FullyConnected      : (T_R=1, T_C=1, T_OC=T_M, T_IC=T_K)
///   MatMul              : (T_R=T_M, T_C=T_N, T_OC=1, T_IC=T_K)
///   Pooling/Activation  : (T_R, T_C, T_OC=T_CH, T_IC=0)
///   ElementWise/Residual: (T_R, T_C, T_OC=T_CH, T_IC=0)
///   Concat/Preprocess   : axes ignored; use flags.tile_enable=0
/// Axis value 0 means "unused for this function" (validator enforces). All zeros
/// + flags.tile_enable=0 is the legacy non-tile path.
// tile_dim[] is the op-general (ISA) view; the named fields are the conv/matmul
// interpretation of the same slots. Both alias the same 6 words (byte-identical).
struct TileShape {
    union {
        uint32_t tile_dim[6];
        struct {
            uint32_t t_r;          ///< tile_dim[0] output row tile
            uint32_t t_c;          ///< tile_dim[1] output col tile
            uint32_t t_oc;         ///< tile_dim[2] output-channel tile (MatMul: T_N)
            uint32_t t_ic;         ///< tile_dim[3] reduction tile (MatMul: T_K)
            uint32_t t_kern;       ///< tile_dim[4] kernel tile = kH*kW
            uint32_t intra_order;  ///< tile_dim[5] TileTraversal enum value
        };
    };
};

static_assert(sizeof(TileShape) == 24, "TileShape must be 24 bytes");

// Descriptor flag bits (LayerDescriptor/FunctionDescriptor `flags` field).
namespace FuncFlag {
constexpr uint32_t TileEnable        = 1u << 0;  ///< runtime tile loop on
constexpr uint32_t DerivedMode       = 1u << 1;  ///< built by add_layer_derived (Mode B)
constexpr uint32_t FunctionModelOn   = 1u << 2;  ///< R10 reserved: emit tensor output
constexpr uint32_t WritebackRequired = 1u << 3;  ///< R10 reserved: partial-sum writeback
constexpr uint32_t ReservedMask      = 0xFFFFFFF0u;  ///< bits 4-31 must be zero
}  // namespace FuncFlag

// ============================================================================
// Layer Descriptor (160 bytes, 4-byte aligned)
// ============================================================================

struct LayerDescriptor {
    uint32_t         layer_id;
    FuncType        type;
    uint32_t         stride;
    uint32_t         padding;
    TensorDescriptor input;            // 20 bytes
    TensorDescriptor weight;           // 20 bytes
    TensorDescriptor output;           // 20 bytes
    LatencyDescriptor latency;         // 36 bytes
    uint32_t         ops_per_output;   // ops per output (for MAC stats)
    SparsityMode     sparsity_mode;    // sparsity handling mode
    uint32_t         packet_layer_id;  // source packet layer id (loader-filled)
    uint32_t         packet_meta;      // (fn_type<<24)|(prep_type<<16)|function_id
    TileShape        tile;             // 24 bytes (R9)
    uint32_t         flags;            // FuncFlag bits (R9)
    // Zero-skip permille of the layer's INPUT stream (0 = none). Loader-filled
    // from FunctionDescriptor.preprocess_param0 when preprocess_type is
    // ZeroSkipping; the tile executor scales input fetch bytes by keep/1000.
    // (Was the always-zero `reserved` pad — same 4-byte slot.)
    uint32_t         preprocess_param0;
};

static_assert(sizeof(LayerDescriptor) == 176, "LayerDescriptor must be 176 bytes (R11: +LatencyDescriptor.enable_gran)");

// ============================================================================
// Layer Packet Structures
// ============================================================================

struct LayerPacketHeader {
    uint32_t layer_id;
    uint32_t num_functions;
    uint32_t packet_size;           // header + function descriptors (no weights)
    uint32_t kernel_data_offset;    // image-global offset into weight section (0 if absent)
    uint32_t kernel_data_size;      // bytes of kernel blob (0 if absent)
    uint32_t reserved;              // must be zero
};

static_assert(sizeof(LayerPacketHeader) == 24, "LayerPacketHeader must be 24 bytes (R10)");

struct FunctionDescriptor {
    uint32_t function_id;
    FuncType type;
    uint32_t operand_src[4];           // per-operand source: producer func_id,
                                       // special source id, or 0xFFFFFFFF (none).
                                       // [0]/[1] = first/second operand.
    uint32_t stride;
    uint32_t padding;
    uint32_t dilation;
    PreprocessType preprocess_type;    // optional preprocessor module selector
    uint32_t preprocess_param0;        // e.g., zero-skip ratio (permille)
    uint32_t preprocess_param1;        // reserved for preprocessor extension
    TensorDescriptor input;
    TensorDescriptor weight;
    TensorDescriptor output;
    LatencyDescriptor latency;         // 36 bytes
    uint32_t ops_per_output;           // ops per output (for MAC stats)
    SparsityMode sparsity_mode;        // sparsity handling mode
    TileShape tile;                    // 24 bytes (R9)
    uint32_t flags;                    // FuncFlag bits (R9)
    uint8_t  operand_cnt;              // valid operand_src entries
    uint8_t  reserved[3];              // pad, must be zero
};

static_assert(sizeof(FunctionDescriptor) == 200, "FunctionDescriptor must be 200 bytes (operand_src[4] regroup)");

} // namespace flexnpu_sim
