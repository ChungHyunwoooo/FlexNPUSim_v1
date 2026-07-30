/**
 * @file tile_descriptor.cpp
 * @brief Tile Scheduling Algorithm Implementation
 */

#include "common/tile_descriptor.h"
#include <algorithm>
#include <cmath>

namespace flexnpu_sim {

/**
 * @brief Compute tile schedule for a layer
 *
 * Partitions the output feature map into tiles that fit within L2 buffer capacity.
 * Each tile may require multiple passes to handle input channel slicing.
 *
 * @param layer_name Layer identifier
 * @param layer_id Layer index
 * @param output_height Full output height
 * @param output_width Full output width
 * @param output_channels Full output channels
 * @param input_channels Full input channels
 * @param kernel_height Convolution kernel height
 * @param kernel_width Convolution kernel width
 * @param stride Convolution stride
 * @param padding Convolution padding
 * @param l2_buffer_capacity_kb L2 Global Buffer capacity (KB)
 * @param tile_height Target tile height (0 = auto)
 * @param tile_width Target tile width (0 = auto)
 * @return Computed tile schedule
 */
TileSchedule compute_tile_schedule(
    const std::string& layer_name,
    uint32_t layer_id,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t output_channels,
    uint32_t input_channels,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride,
    uint32_t padding,
    uint32_t l2_buffer_capacity_kb,
    uint32_t tile_height,
    uint32_t tile_width,
    uint32_t input_budget_kb,
    uint32_t kernel_budget_kb,
    uint32_t output_budget_kb,
    bool is_depthwise) {

    TileSchedule schedule;
    schedule.layer_name = layer_name;
    schedule.layer_id = layer_id;
    schedule.layer_output_height = output_height;
    schedule.layer_output_width = output_width;
    schedule.layer_output_channels = output_channels;
    schedule.layer_input_channels = input_channels;
    schedule.scheduling_policy = "row_major";

    // Auto-determine tile size if not specified
    if (tile_height == 0 || tile_width == 0) {
        // Default: 8x8 tiles (can be optimized based on L2 capacity)
        tile_height = std::min(8u, output_height);
        tile_width = std::min(8u, output_width);
    }

    schedule.tile_height = tile_height;
    schedule.tile_width = tile_width;

    // Calculate number of tiles in each dimension
    uint32_t num_tiles_h = (output_height + tile_height - 1) / tile_height;
    uint32_t num_tiles_w = (output_width + tile_width - 1) / tile_width;
    schedule.num_tiles = num_tiles_h * num_tiles_w;

    // L2 buffer capacity in bytes
    uint64_t l2_capacity_bytes = static_cast<uint64_t>(l2_buffer_capacity_kb) * 1024;

    // Per-region budgets: explicit partition sizes when provided (fixed GB
    // partition), otherwise the legacy conservative 40/40/20 split.
    uint64_t l2_input_budget = input_budget_kb
        ? static_cast<uint64_t>(input_budget_kb) * 1024
        : l2_capacity_bytes * 40 / 100;
    uint64_t l2_kernel_budget = kernel_budget_kb
        ? static_cast<uint64_t>(kernel_budget_kb) * 1024
        : l2_capacity_bytes * 40 / 100;
    uint64_t l2_output_budget = output_budget_kb
        ? static_cast<uint64_t>(output_budget_kb) * 1024
        : l2_capacity_bytes * 20 / 100;

    // Generate tiles in row-major order
    uint32_t tile_id = 0;
    for (uint32_t th = 0; th < num_tiles_h; ++th) {
        for (uint32_t tw = 0; tw < num_tiles_w; ++tw) {
            TileDescriptor tile;
            tile.tile_id = tile_id++;

            // Output tile dimensions
            tile.output_height_start = th * tile_height;
            tile.output_width_start = tw * tile_width;
            tile.output_height = std::min(tile_height, output_height - tile.output_height_start);
            tile.output_width = std::min(tile_width, output_width - tile.output_width_start);
            tile.output_channels = output_channels;

            // Input tile dimensions (with receptive field)
            tile.input_height_start = (tile.output_height_start * stride) - padding;
            tile.input_width_start = (tile.output_width_start * stride) - padding;
            tile.input_height = tile.output_height * stride + kernel_height - 1;
            tile.input_width = tile.output_width * stride + kernel_width - 1;
            tile.input_channels = input_channels;

            // Kernel dimensions
            tile.kernel_height = kernel_height;
            tile.kernel_width = kernel_width;

            // Calculate passes based on L2 capacity
            // Output buffer requirement (fixed per tile)
            uint64_t output_bytes = static_cast<uint64_t>(tile.output_height) *
                                    tile.output_width * output_channels * sizeof(float);

            // Check if output fits in budget
            if (output_bytes > l2_output_budget) {
                // Output too large - would need tile splitting (not implemented)
                tile.num_passes = 1;
            } else {
                // Calculate max input channels per pass
                uint64_t input_bytes_per_channel = static_cast<uint64_t>(tile.input_height) *
                                                    tile.input_width * sizeof(float);
                // Depthwise: one Kh·Kw filter per (input==output) channel, so a
                // channel's kernel footprint is Kh·Kw — not Kh·Kw·Cout.
                uint64_t kernel_bytes_per_input_channel = static_cast<uint64_t>(kernel_height) *
                                                          kernel_width *
                                                          (is_depthwise ? 1u : output_channels) *
                                                          sizeof(float);

                uint64_t bytes_per_channel = input_bytes_per_channel + kernel_bytes_per_input_channel;

                // Max channels that fit in L2
                uint64_t available_budget = std::min(l2_input_budget, l2_kernel_budget);
                uint32_t max_channels_per_pass = static_cast<uint32_t>(available_budget / bytes_per_channel);

                if (max_channels_per_pass == 0) {
                    max_channels_per_pass = 1; // At least 1 channel per pass
                }

                // Calculate number of passes
                tile.num_passes = (input_channels + max_channels_per_pass - 1) / max_channels_per_pass;
            }

            // Generate pass descriptors
            tile.passes.resize(tile.num_passes);
            uint32_t remaining_channels = input_channels;
            uint32_t channel_offset = 0;

            for (uint32_t p = 0; p < tile.num_passes; ++p) {
                PassDescriptor& pass = tile.passes[p];
                pass.pass_id = p;
                pass.is_first_pass = (p == 0);
                pass.is_last_pass = (p == tile.num_passes - 1);

                // Distribute channels evenly across passes
                uint32_t channels_this_pass = std::min(remaining_channels,
                                                        (input_channels + tile.num_passes - 1) / tile.num_passes);

                pass.input_channel_start = channel_offset;
                pass.input_channel_end = channel_offset + channels_this_pass;
                pass.input_channel_count = channels_this_pass;

                // Memory traffic for this pass
                pass.input_fetch_bytes = static_cast<uint64_t>(tile.input_height) *
                                         tile.input_width * channels_this_pass * sizeof(float);

                pass.kernel_fetch_bytes = static_cast<uint64_t>(kernel_height) * kernel_width *
                                          channels_this_pass *
                                          (is_depthwise ? 1u : output_channels) * sizeof(float);

                // Output writeback only on last pass
                pass.output_writeback_bytes = pass.is_last_pass ? output_bytes : 0;

                channel_offset += channels_this_pass;
                remaining_channels -= channels_this_pass;
            }

            // Total memory traffic
            tile.total_input_fetch_bytes = 0;
            tile.total_kernel_fetch_bytes = 0;
            tile.total_output_writeback_bytes = 0;

            for (const auto& pass : tile.passes) {
                tile.total_input_fetch_bytes += pass.input_fetch_bytes;
                tile.total_kernel_fetch_bytes += pass.kernel_fetch_bytes;
                tile.total_output_writeback_bytes += pass.output_writeback_bytes;
            }

            tile.total_memory_traffic_bytes = tile.total_input_fetch_bytes +
                                               tile.total_kernel_fetch_bytes +
                                               tile.total_output_writeback_bytes;

            schedule.tiles.push_back(tile);
        }
    }

    // Generate execution order (row-major for now)
    schedule.execution_order.resize(schedule.num_tiles);
    for (uint32_t i = 0; i < schedule.num_tiles; ++i) {
        schedule.execution_order[i] = i;
    }

    // Compute aggregate statistics
    schedule.compute_statistics();

    return schedule;
}

// ============================================================================
// lower_tile_schedule — bake dataflow into a runtime transaction list
// ============================================================================

std::vector<TileTxn> lower_tile_schedule(const TileSchedule& sched,
                                         Dataflow dataflow,
                                         uint32_t input_total_operands,
                                         uint32_t weight_total_operands,
                                         uint32_t element_bytes,
                                         uint32_t compute_passes_per_output) {
    OperandResidency r;
    r.weight_resident = (dataflow == Dataflow::WS);
    r.input_resident  = (dataflow == Dataflow::IS);
    return lower_tile_schedule(sched, r, input_total_operands,
                               weight_total_operands, element_bytes,
                               compute_passes_per_output);
}

std::vector<TileTxn> lower_tile_schedule(const TileSchedule& sched,
                                         OperandResidency residency,
                                         uint32_t input_total_operands,
                                         uint32_t weight_total_operands,
                                         uint32_t element_bytes,
                                         uint32_t compute_passes_per_output) {
    const uint32_t elem = (element_bytes > 0) ? element_bytes : 1;
    const uint32_t cpp  = (compute_passes_per_output > 0) ? compute_passes_per_output : 1;
    // compute_tile_schedule accumulates its *_bytes footprints in sizeof(float)
    // units regardless of the real element size, so operand (element) counts are
    // recovered by dividing by that unit — NOT by `elem`. Dividing by `elem`
    // inflated every operand count by sizeof(float)/elem (4x at int8, 2x at
    // int16), which over-charged compute (F^O/n_max) and DRAM traffic.
    constexpr uint32_t kSchedElemBytes = sizeof(float);
    std::vector<TileTxn> txns;
    txns.reserve(sched.execution_order.size());

    for (size_t k = 0; k < sched.execution_order.size(); ++k) {
        const uint32_t tid = sched.execution_order[k];
        if (tid >= sched.tiles.size()) continue;
        const TileDescriptor& td = sched.tiles[tid];
        const bool first = (k == 0);

        // Raw per-tile footprints (dataflow-agnostic) from the schedule, in
        // element (operand) units.
        const uint32_t tile_in_ops  =
            static_cast<uint32_t>(td.total_input_fetch_bytes / kSchedElemBytes);
        const uint32_t tile_wt_ops  =
            static_cast<uint32_t>(td.total_kernel_fetch_bytes / kSchedElemBytes);
        const uint32_t tile_out_ops =
            static_cast<uint32_t>(td.total_output_writeback_bytes / kSchedElemBytes);

        TileTxn t{};
        t.num_passes            = td.num_passes;
        // Compute target = pass completions (F^O contribution) the latency model
        // must generate for this tile = |O_tile| · compute_passes_per_output
        // (n/k), reconciling the memory-pass schedule with the compute-pass
        // descriptor. The physical DRAM writeback is |O_tile| bytes (final only).
        t.output_write_operands = tile_out_ops * cpp;
        t.output_write_bytes    = tile_out_ops * elem;   // physical DRAM write

        // Apply the stationary-operand rule per operand: a resident operand
        // is streamed once (first tile) and held; a non-resident operand is
        // streamed per tile. WS/IS/OS are the three named points of this
        // 2-bit space (both-resident covers fully on-chip small layers).
        t.input_fetch_operands  = residency.input_resident
            ? (first ? input_total_operands : 0u) : tile_in_ops;
        t.weight_fetch_operands = residency.weight_resident
            ? (first ? weight_total_operands : 0u) : tile_wt_ops;
        t.input_fetch_bytes  = t.input_fetch_operands  * elem;
        t.weight_fetch_bytes = t.weight_fetch_operands * elem;
        txns.push_back(t);
    }
    return txns;
}

} // namespace flexnpu_sim
