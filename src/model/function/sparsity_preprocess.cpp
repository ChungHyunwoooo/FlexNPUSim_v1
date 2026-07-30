#include "model/function/sparsity_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "compiler/dnn_image/dnn_image_format.h"
#include "compiler/dnn_image/dnn_image_packet.h"
#include "compiler/dnn_image/dnn_image_loader.h"
#ifdef HAVE_OPENCV
#include "model/function/sparsity_profiler.h"
#endif

namespace flexnpu_sim {

void apply_sparsity([[maybe_unused]] DnnImage& image,
                    [[maybe_unused]] const NetworkConfig& cfg) {
#ifdef HAVE_OPENCV
    SparsityProfiler profiler;
    auto sparsity = profiler.profile(image,
        cfg.input_height, cfg.input_width, cfg.input_channels);

    uint8_t* raw = image.data.data();
    DnnImageLoader loader;
    if (!loader.parse(raw, image.data.size())) return;

    uint32_t layer_count = loader.num_layers();

    // Patch packet image in place.
    DnnImageHeader hdr{};
    std::memcpy(&hdr, raw, sizeof(hdr));
    if (hdr.version != DNN_IMAGE_VERSION) return;

    size_t cursor = sizeof(DnnImageHeader);
    uint32_t flat_idx = 0;
    while (cursor + sizeof(LayerPacketHeader) <= hdr.total_size &&
           flat_idx < layer_count) {
        LayerPacketHeader ph{};
        std::memcpy(&ph, raw + cursor, sizeof(ph));
        if (ph.packet_size < sizeof(LayerPacketHeader)) break;
        if (cursor + ph.packet_size > hdr.total_size) break;
        size_t fd_cursor = cursor + sizeof(LayerPacketHeader);
        for (uint32_t fi = 0; fi < ph.num_functions; ++fi) {
            if (fd_cursor + sizeof(FunctionDescriptor) > cursor + ph.packet_size) break;
            FunctionDescriptor fd{};
            std::memcpy(&fd, raw + fd_cursor, sizeof(fd));

            double input_sp = 0.0;
            if (flat_idx > 0 && flat_idx - 1 < sparsity.size()) {
                input_sp = sparsity[flat_idx - 1].output_sparsity;
            }
            uint32_t orig = fd.latency.fetch_operand0;
            fd.latency.fetch_operand0 = SparsityProfiler::adjust_fi(orig, input_sp);
            if (orig > 0 && fd.latency.fetch_operand0 < orig) {
                double r = static_cast<double>(fd.latency.fetch_operand0) / orig;
                fd.latency.prefetch_in =
                    std::max(1u, static_cast<uint32_t>(std::round(fd.latency.prefetch_in * r)));
            }
            // ratio_input_fixed removed in format v4; ratio is derived at
            // runtime from write_output / fetch_operand0.
            std::memcpy(raw + fd_cursor, &fd, sizeof(fd));
            fd_cursor += sizeof(FunctionDescriptor);
            flat_idx++;
        }
        cursor += ph.packet_size;
    }
#endif
}

}  // namespace flexnpu_sim
