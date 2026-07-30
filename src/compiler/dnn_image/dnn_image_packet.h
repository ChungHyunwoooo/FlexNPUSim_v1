/**
 * @file dnn_image_packet.h
 * @brief Packet-based DNN Image utilities
 */

#pragma once

#include "compiler/dnn_image/dnn_image_format.h"
#include <cstdint>
#include <string>
#include <vector>

namespace flexnpu_sim {

struct FunctionRecord {
    uint32_t layer_id = 0;
    FunctionDescriptor desc{};
};

/// Per-layer kernel data blob. R10: compiler bundles kernel bytes into the
/// layer packet body. Attached to the first FunctionRecord of the layer;
/// subsequent records in the same layer leave this empty.
struct LayerKernelBlob {
    uint32_t layer_id = 0;
    std::vector<uint8_t> bytes;
};

struct FunctionExecutionMeta {
    uint32_t packet_layer_id = 0;
    uint32_t packet_index = 0;
    uint32_t function_id = 0;
    uint32_t i1_connect = 0xFFFFFFFFu;
    uint32_t i2_connect = 0xFFFFFFFFu;
    FuncType function_type = FuncType::ElementWise;
    PreprocessType preprocess_type = PreprocessType::None;
    bool i1_from_packet = false;
    bool i2_from_packet = false;
    bool has_internal_consumer = false;
};

FuncType map_function_type_to_layer_type(FuncType type);

/// Build a DNN image binary. R10 variant accepts optional per-layer kernel
/// blobs and an optional first-layer input fmap blob to embed in the image.
bool build_dnn_image_packets(const std::vector<FunctionRecord>& functions,
                             std::vector<uint8_t>& out_image,
                             std::string* err_msg = nullptr,
                             const std::vector<LayerKernelBlob>* kernel_blobs = nullptr,
                             const std::vector<uint8_t>* input_fmap_blob = nullptr,
                             TensorDType default_dtype = TensorDType::INT8);

bool parse_dnn_image_packets_to_layers(const uint8_t* data,
                                       size_t length,
                                       std::vector<LayerDescriptor>& out_layers,
                                       uint32_t* out_num_layer_packets = nullptr,
                                       std::string* err_msg = nullptr,
                                       std::vector<FunctionExecutionMeta>* out_exec_meta = nullptr);

} // namespace flexnpu_sim
