/**
 * @file dnn_image_loader.h
 * @brief DNN Image parser (pure C++, header-only)
 *
 * Takes a memory buffer (data read from DRAM), parses the canonical
 * packet-based DNN Image, and flattens each FunctionDescriptor into an
 * execution-unit LayerDescriptor.
 */

#pragma once

#include "compiler/dnn_image/dnn_image_format.h"
#include "compiler/dnn_image/dnn_image_packet.h"
#include "model/latency/latency_model.h"
#include <vector>
#include <cstring>

namespace flexnpu_sim {

class DnnImageLoader {
public:
    /**
     * @brief Parse a DNN Image from a memory buffer
     * @param data pointer to the binary data
     * @param length data length (bytes)
     * @return true on successful parse
     */
    bool parse(const uint8_t* data, size_t length) {
        parsed_ = false;
        layers_.clear();
        exec_meta_.clear();
        num_layer_packets_ = 0;

        if (!data || length < sizeof(DnnImageHeader)) {
            return false;
        }

        std::memcpy(&header_, data, sizeof(DnnImageHeader));

        if (header_.magic != DNN_IMAGE_MAGIC) {
            return false;
        }

        if (header_.version != DNN_IMAGE_VERSION) {
            return false;
        }

        std::string err;
        if (!parse_dnn_image_packets_to_layers(
                data, length, layers_, &num_layer_packets_, &err, &exec_meta_)) {
            last_error_ = err;
            return false;
        }

        // R10: extract per-layer kernel blobs and first-layer input fmap.
        // Walk packets once more to record (offset, size) into source buffer.
        extract_blobs(data, length);

        // Runtime executes one latency model per function descriptor.
        header_.num_layers = static_cast<uint32_t>(layers_.size());
        parsed_ = true;
        return true;
    }

    /// First-layer input feature map blob (may be empty).
    const std::vector<uint8_t>& input_fmap_bytes() const { return input_fmap_bytes_; }

    /// Per-layer-packet kernel blob. `packet_idx` refers to the packet order in
    /// the image, not the flattened function index. Returns empty vector if the
    /// packet has no kernel data.
    const std::vector<uint8_t>& kernel_bytes(uint32_t packet_idx) const {
        static const std::vector<uint8_t> empty;
        if (packet_idx >= kernel_bytes_per_packet_.size()) return empty;
        return kernel_bytes_per_packet_[packet_idx];
    }

    TensorDType default_dtype() const { return header_.default_dtype; }

    /**
     * @brief Whether parsing has completed
     */
    bool is_parsed() const { return parsed_; }
    const std::string& error() const { return last_error_; }

    /**
     * @brief Header accessor
     */
    const DnnImageHeader& header() const { return header_; }

    /**
     * @brief Layer count
     */
    uint32_t num_layers() const { return header_.num_layers; }

    /**
     * @brief Layer packet count
     */
    uint32_t num_layer_packets() const { return num_layer_packets_; }

    /**
     * @brief Layer descriptor accessor
     */
    const LayerDescriptor& layer(uint32_t idx) const {
        return layers_.at(idx);
    }

    /**
     * @brief Function execution metadata for flattened runtime index.
     */
    const FunctionExecutionMeta& function_meta(uint32_t idx) const {
        return exec_meta_.at(idx);
    }

    /**
     * @brief Convert LayerDescriptor → LmParams
     *
     * Produces the form LmModel accepts.
     * Ratios are restored from fixed-point (x1000) to double.
     */
    LmParams to_latency_params(uint32_t idx) const {
        const auto& ld = layers_.at(idx).latency;

        LmParams params;
        params.fetch_operand0  = ld.fetch_operand0;
        params.fetch_operand1 = ld.fetch_operand1;
        params.write_output   = ld.write_output;
        params.prefetch_in       = ld.prefetch_in;
        params.prefetch_wt      = ld.prefetch_wt;
        params.n_min = ld.n_min;
        params.n_max = ld.n_max;
        params.issue_latency        = ld.issue_latency;

        return params;
    }

private:
    DnnImageHeader header_{};
    std::string last_error_;
    std::vector<LayerDescriptor> layers_;
    std::vector<FunctionExecutionMeta> exec_meta_;
    bool parsed_ = false;
    uint32_t num_layer_packets_ = 0;

    // R10 blobs
    std::vector<uint8_t> input_fmap_bytes_;
    std::vector<std::vector<uint8_t>> kernel_bytes_per_packet_;

    void extract_blobs(const uint8_t* data, size_t length) {
        input_fmap_bytes_.clear();
        kernel_bytes_per_packet_.clear();
        if (!data || length < sizeof(DnnImageHeader)) return;

        if (header_.input_fmap_size > 0 &&
            header_.input_fmap_offset + header_.input_fmap_size <= length) {
            input_fmap_bytes_.assign(
                data + header_.input_fmap_offset,
                data + header_.input_fmap_offset + header_.input_fmap_size);
        }

        // Walk packets — mirror parser logic but only capture blob spans.
        size_t cursor = sizeof(DnnImageHeader);
        if (header_.input_fmap_size > 0) cursor += header_.input_fmap_size;
        for (uint32_t p = 0; p < header_.num_layers; ++p) {
            if (cursor + sizeof(LayerPacketHeader) > header_.total_size) break;
            LayerPacketHeader ph{};
            std::memcpy(&ph, data + cursor, sizeof(ph));
            std::vector<uint8_t> kblob;
            // kernel_data_offset is image-global (into the weight section that
            // follows the packet region), not packet-relative.
            if (ph.kernel_data_size > 0 &&
                ph.kernel_data_offset >= header_.weight_section_offset &&
                ph.kernel_data_offset + ph.kernel_data_size <= header_.total_size) {
                const uint8_t* kptr = data + ph.kernel_data_offset;
                kblob.assign(kptr, kptr + ph.kernel_data_size);
            }
            kernel_bytes_per_packet_.push_back(std::move(kblob));
            cursor += ph.packet_size;
        }
    }
};

} // namespace flexnpu_sim
