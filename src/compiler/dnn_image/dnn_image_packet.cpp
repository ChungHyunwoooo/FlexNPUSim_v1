/**
 * @file dnn_image_packet.cpp
 * @brief Packet-based DNN Image utilities
 */

#include "compiler/dnn_image/dnn_image_packet.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <unordered_map>

namespace flexnpu_sim {

FuncType map_function_type_to_layer_type(FuncType type) {
    switch (type) {
        case FuncType::Conv2D:          return FuncType::Conv2D;
        case FuncType::DepthwiseConv2D: return FuncType::DepthwiseConv2D;
        case FuncType::FullyConnected:  return FuncType::FullyConnected;
        case FuncType::Pooling:         return FuncType::Pooling;
        case FuncType::Activation:      return FuncType::Activation;
        case FuncType::MatMul:          return FuncType::MatMul;
        case FuncType::ElementWise:     return FuncType::ElementWise;
        case FuncType::Preprocess:      return FuncType::ElementWise;
        case FuncType::Residual:        return FuncType::ElementWise;
        case FuncType::Concat:          return FuncType::ElementWise;
        case FuncType::Deconvolution:   return FuncType::Conv2D;
        default:                              return FuncType::ElementWise;
    }
}

static void set_err(std::string* err, const std::string& msg) {
    if (err) *err = msg;
}

bool build_dnn_image_packets(const std::vector<FunctionRecord>& functions,
                             std::vector<uint8_t>& out_image,
                             std::string* err_msg,
                             const std::vector<LayerKernelBlob>* kernel_blobs,
                             const std::vector<uint8_t>* input_fmap_blob,
                             TensorDType default_dtype) {
    out_image.clear();
    if (functions.empty()) {
        set_err(err_msg, "empty function list");
        return false;
    }

    std::vector<FunctionRecord> sorted = functions;
    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const FunctionRecord& a, const FunctionRecord& b) {
                         if (a.layer_id != b.layer_id) return a.layer_id < b.layer_id;
                         return a.desc.function_id < b.desc.function_id;
                     });

    std::vector<std::pair<uint32_t, std::vector<FunctionDescriptor>>> packets;
    for (const auto& fr : sorted) {
        if (packets.empty() || packets.back().first != fr.layer_id) {
            packets.push_back({fr.layer_id, {}});
        }
        packets.back().second.push_back(fr.desc);
    }

    // Look up per-layer kernel blob (optional).
    auto find_kernel = [&](uint32_t layer_id) -> const std::vector<uint8_t>* {
        if (!kernel_blobs) return nullptr;
        for (const auto& kb : *kernel_blobs) {
            if (kb.layer_id == layer_id && !kb.bytes.empty()) return &kb.bytes;
        }
        return nullptr;
    };

    size_t total_size = sizeof(DnnImageHeader);
    if (input_fmap_blob) total_size += input_fmap_blob->size();
    for (const auto& p : packets) {
        size_t pkt = sizeof(LayerPacketHeader) +
                     p.second.size() * sizeof(FunctionDescriptor);
        if (auto kb = find_kernel(p.first)) pkt += kb->size();
        total_size += pkt;
    }
    if (total_size > 0xFFFFFFFFu) {
        set_err(err_msg, "image too large");
        return false;
    }

    out_image.resize(total_size, 0);
    size_t cursor = 0;

    // Defer header write until input_fmap_offset is known.
    const size_t header_pos = cursor;
    cursor += sizeof(DnnImageHeader);

    uint32_t input_fmap_offset = 0;
    uint32_t input_fmap_size = 0;
    if (input_fmap_blob && !input_fmap_blob->empty()) {
        input_fmap_offset = static_cast<uint32_t>(cursor);
        input_fmap_size = static_cast<uint32_t>(input_fmap_blob->size());
        std::memcpy(out_image.data() + cursor, input_fmap_blob->data(), input_fmap_size);
        cursor += input_fmap_size;
    }

    // Global weight section: all kernel blobs live after the packet region
    // (header + descriptors), addressed by an image-relative offset. Packets
    // carry no embedded weight bytes; `kernel_data_offset` is now global.
    size_t weight_cursor = cursor;
    for (const auto& p : packets)
        weight_cursor += sizeof(LayerPacketHeader)
                       + p.second.size() * sizeof(FunctionDescriptor);
    const uint32_t weight_section_start = static_cast<uint32_t>(weight_cursor);
    bool any_weight = false;

    for (const auto& p : packets) {
        const uint32_t num_f = static_cast<uint32_t>(p.second.size());
        const auto* kblob = find_kernel(p.first);
        const uint32_t kblob_size = kblob ? static_cast<uint32_t>(kblob->size()) : 0u;
        const uint32_t desc_region = static_cast<uint32_t>(
            p.second.size() * sizeof(FunctionDescriptor));

        LayerPacketHeader ph{};
        ph.layer_id = p.first;
        ph.num_functions = num_f;
        ph.packet_size = static_cast<uint32_t>(sizeof(LayerPacketHeader) + desc_region);
        ph.kernel_data_offset = kblob ? static_cast<uint32_t>(weight_cursor) : 0u;
        ph.kernel_data_size = kblob_size;
        ph.reserved = 0;
        std::memcpy(out_image.data() + cursor, &ph, sizeof(ph));
        cursor += sizeof(ph);

        for (const auto& fd : p.second) {
            std::memcpy(out_image.data() + cursor, &fd, sizeof(fd));
            cursor += sizeof(fd);
        }
        if (kblob) {
            std::memcpy(out_image.data() + weight_cursor, kblob->data(), kblob_size);
            weight_cursor += kblob_size;
            any_weight = true;
        }
    }

    DnnImageHeader hdr{};
    hdr.magic = DNN_IMAGE_MAGIC;
    hdr.version = DNN_IMAGE_VERSION;
    hdr.num_layers = static_cast<uint32_t>(packets.size());
    hdr.total_size = static_cast<uint32_t>(total_size);
    hdr.default_dtype = default_dtype;
    hdr.input_fmap_offset = input_fmap_offset;
    hdr.input_fmap_size = input_fmap_size;
    hdr.weight_section_offset = any_weight ? weight_section_start : 0;
    std::memcpy(out_image.data() + header_pos, &hdr, sizeof(hdr));

    return true;
}

bool parse_dnn_image_packets_to_layers(const uint8_t* data,
                                       size_t length,
                                       std::vector<LayerDescriptor>& out_layers,
                                       uint32_t* out_num_layer_packets,
                                       std::string* err_msg,
                                       std::vector<FunctionExecutionMeta>* out_exec_meta) {
    out_layers.clear();
    if (out_num_layer_packets) *out_num_layer_packets = 0;
    if (out_exec_meta) out_exec_meta->clear();
    if (!data || length < sizeof(DnnImageHeader)) {
        set_err(err_msg, "buffer too small");
        return false;
    }

    DnnImageHeader hdr{};
    std::memcpy(&hdr, data, sizeof(hdr));
    if (hdr.magic != DNN_IMAGE_MAGIC) {
        set_err(err_msg, "invalid magic");
        return false;
    }
    if (hdr.version != DNN_IMAGE_VERSION) {
        set_err(err_msg, "unsupported image version");
        return false;
    }
    if (hdr.total_size > length) {
        set_err(err_msg, "truncated image");
        return false;
    }

    size_t cursor = sizeof(DnnImageHeader);
    // R10: skip first-layer input fmap blob if present.
    if (hdr.input_fmap_size > 0) {
        if (static_cast<size_t>(hdr.input_fmap_offset) != cursor ||
            cursor + hdr.input_fmap_size > hdr.total_size) {
            set_err(err_msg, "invalid input_fmap region");
            return false;
        }
        cursor += hdr.input_fmap_size;
    }
    uint32_t packet_count = 0;
    uint32_t synthetic_idx = 0;

    while (cursor + sizeof(LayerPacketHeader) <= hdr.total_size &&
           packet_count < hdr.num_layers) {
        LayerPacketHeader ph{};
        std::memcpy(&ph, data + cursor, sizeof(ph));
        if (ph.packet_size < sizeof(LayerPacketHeader)) {
            set_err(err_msg, "invalid packet_size");
            return false;
        }
        if (cursor + ph.packet_size > hdr.total_size) {
            set_err(err_msg, "packet out of range");
            return false;
        }

        // R10: packet = header + descriptors + optional kernel blob.
        if (ph.kernel_data_size > ph.packet_size ||
            (ph.kernel_data_size > 0 && ph.kernel_data_offset < sizeof(LayerPacketHeader))) {
            set_err(err_msg, "invalid kernel_data offset/size");
            return false;
        }
        const uint32_t desc_region = ph.packet_size
                                   - static_cast<uint32_t>(sizeof(LayerPacketHeader))
                                   - ph.kernel_data_size;
        if ((desc_region % sizeof(FunctionDescriptor)) != 0) {
            set_err(err_msg, "invalid function descriptor region size");
            return false;
        }
        uint32_t num_fd_by_size = desc_region / static_cast<uint32_t>(sizeof(FunctionDescriptor));
        if (num_fd_by_size < ph.num_functions) {
            set_err(err_msg, "packet function count mismatch");
            return false;
        }

        std::vector<FunctionDescriptor> fds;
        fds.reserve(ph.num_functions);
        size_t fd_cursor = cursor + sizeof(LayerPacketHeader);
        for (uint32_t i = 0; i < ph.num_functions; ++i) {
            FunctionDescriptor fd{};
            std::memcpy(&fd, data + fd_cursor, sizeof(fd));
            fd_cursor += sizeof(fd);
            fds.push_back(fd);
        }

        std::unordered_map<uint32_t, uint32_t> function_id_to_local_idx;
        function_id_to_local_idx.reserve(fds.size());
        for (uint32_t i = 0; i < fds.size(); ++i) {
            function_id_to_local_idx.emplace(fds[i].function_id, i);
        }
        std::unordered_map<uint32_t, uint32_t> internal_consumer_count;
        internal_consumer_count.reserve(fds.size());
        for (const auto& fd : fds) {
            if (function_id_to_local_idx.find(fd.operand_src[0]) != function_id_to_local_idx.end()) {
                internal_consumer_count[fd.operand_src[0]]++;
            }
            if (function_id_to_local_idx.find(fd.operand_src[1]) != function_id_to_local_idx.end()) {
                internal_consumer_count[fd.operand_src[1]]++;
            }
        }

        for (const auto& fd : fds) {
            LayerDescriptor ld{};
            ld.layer_id = synthetic_idx++;
            ld.type = map_function_type_to_layer_type(fd.type);
            ld.stride = fd.stride;
            ld.padding = fd.padding;
            ld.input = fd.input;
            ld.weight = fd.weight;
            ld.output = fd.output;
            ld.latency = fd.latency;
            ld.ops_per_output = fd.ops_per_output;
            ld.sparsity_mode = fd.sparsity_mode;
            // preserve packet/function metadata for debug/profiling
            ld.packet_layer_id = ph.layer_id;
            ld.packet_meta =
                ((static_cast<uint32_t>(fd.type) & 0xFFu) << 24) |
                ((static_cast<uint32_t>(fd.preprocess_type) & 0xFFu) << 16) |
                (fd.function_id & 0xFFFFu);
            ld.tile = fd.tile;
            ld.flags = fd.flags;
            // Carry the zero-skip ratio to the runtime layer so the tile
            // executor can scale input traffic (sparse streams move less).
            ld.preprocess_param0 =
                (fd.preprocess_type == PreprocessType::ZeroSkipping)
                    ? std::min<uint32_t>(fd.preprocess_param0, 1000u) : 0;

            // Auto preprocess instantiation: zero-skipping as a dedicated function
            // (param0 = skip ratio in permille)
            if (fd.type == FuncType::Preprocess &&
                fd.preprocess_type == PreprocessType::ZeroSkipping) {
                uint32_t skip_permille = std::min<uint32_t>(fd.preprocess_param0, 1000u);
                uint32_t keep_permille = 1000u - skip_permille;
                uint64_t fi = ld.latency.fetch_operand0;
                uint64_t fo = ld.latency.write_output;
                uint64_t th = ld.latency.prefetch_in;
                ld.latency.write_output =
                    static_cast<uint32_t>(std::max<uint64_t>(1u, (fo * keep_permille) / 1000u));
                ld.latency.prefetch_in =
                    static_cast<uint32_t>(std::max<uint64_t>(1u, (th * keep_permille) / 1000u));
                ld.latency.fetch_operand1 = 0;
                ld.latency.prefetch_wt = 0;
            }

            out_layers.push_back(ld);
            if (out_exec_meta) {
                FunctionExecutionMeta meta{};
                meta.packet_layer_id = ph.layer_id;
                meta.packet_index = packet_count;
                meta.function_id = fd.function_id;
                meta.i1_connect = fd.operand_src[0];
                meta.i2_connect = fd.operand_src[1];
                meta.function_type = fd.type;
                meta.preprocess_type = fd.preprocess_type;
                meta.i1_from_packet =
                    (function_id_to_local_idx.find(fd.operand_src[0]) != function_id_to_local_idx.end());
                meta.i2_from_packet =
                    (function_id_to_local_idx.find(fd.operand_src[1]) != function_id_to_local_idx.end());
                meta.has_internal_consumer = (internal_consumer_count[fd.function_id] > 0);
                out_exec_meta->push_back(meta);
            }
        }

        cursor += ph.packet_size;
        packet_count++;
    }

    if (packet_count != hdr.num_layers) {
        set_err(err_msg, "layer packet count mismatch");
        return false;
    }
    if (out_num_layer_packets) *out_num_layer_packets = packet_count;
    return true;
}

} // namespace flexnpu_sim
