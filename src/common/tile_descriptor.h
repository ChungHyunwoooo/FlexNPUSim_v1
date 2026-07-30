/**
 * @file tile_descriptor.h
 * @brief Tile Execution Model Types
 *
 * Based on v2.0 spec Section 6: Tile Execution Model.
 *
 * Tiling decomposes large output feature maps into smaller tiles that fit
 * within the L2 Global Buffer. Each tile may be computed in multiple passes
 * to handle input channel dimension that exceeds buffer capacity.
 *
 * Key concepts:
 * - Tile: A spatial partition of the output (H_tile x W_tile x C_out)
 * - Pass: A temporal partition within a tile (slices input channels)
 * - Schedule: Ordered sequence of tiles + passes for one layer
 * - Traffic: Memory fetch requirement per tile
 */

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include "common/types.h"

namespace flexnpu_sim {

// ============================================================================
// Pass Descriptor
// ============================================================================

/**
 * @brief Pass descriptor (input channel slice within a tile)
 *
 * A pass processes a slice of input channels to produce partial outputs.
 * Multiple passes accumulate to produce the final tile output.
 *
 * Example: If a tile has C_in=512 but L2 can only hold 128 channels,
 *          the tile requires 4 passes (128 channels each).
 */
struct PassDescriptor {
    uint32_t pass_id;                  ///< Pass index within tile (0-based)
    uint32_t input_channel_start;      ///< Start channel index (inclusive)
    uint32_t input_channel_end;        ///< End channel index (exclusive)
    uint32_t input_channel_count;      ///< Number of input channels in this pass

    bool is_first_pass;                ///< First pass (zero partial sums)
    bool is_last_pass;                 ///< Last pass (write final output)

    // --- Memory Traffic ---
    uint64_t input_fetch_bytes;        ///< Input data to fetch (bytes)
    uint64_t kernel_fetch_bytes;       ///< Kernel weights to fetch (bytes)
    uint64_t output_writeback_bytes;   ///< Output data to write (bytes)

    /**
     * @brief Default constructor
     */
    PassDescriptor()
        : pass_id(0), input_channel_start(0), input_channel_end(0),
          input_channel_count(0), is_first_pass(true), is_last_pass(true),
          input_fetch_bytes(0), kernel_fetch_bytes(0), output_writeback_bytes(0) {}
};

// ============================================================================
// Tile Descriptor
// ============================================================================

/**
 * @brief Tile descriptor (spatial partition of output feature map)
 *
 * A tile is a rectangular region of the output feature map that can fit
 * within the L2 Global Buffer along with required input/kernel data.
 *
 * Tile execution model:
 *   1. Load input/kernel data for pass 0
 *   2. Compute partial outputs (accumulate in L2 output buffer)
 *   3. Repeat for pass 1, 2, ..., N-1
 *   4. Write final tile output to DRAM
 */
struct TileDescriptor {
    uint32_t tile_id;                  ///< Tile index within layer (0-based)

    // --- Output Tile Dimensions ---
    uint32_t output_height_start;      ///< Start row in output feature map
    uint32_t output_width_start;       ///< Start column in output feature map
    uint32_t output_height;            ///< Tile height (pixels)
    uint32_t output_width;             ///< Tile width (pixels)
    uint32_t output_channels;          ///< Number of output channels (typically full C_out)

    // --- Input Tile Dimensions (including padding/stride) ---
    uint32_t input_height_start;       ///< Start row in input feature map
    uint32_t input_width_start;        ///< Start column in input feature map
    uint32_t input_height;             ///< Required input height (includes receptive field)
    uint32_t input_width;              ///< Required input width (includes receptive field)
    uint32_t input_channels;           ///< Total input channels (all passes combined)

    // --- Kernel Dimensions ---
    uint32_t kernel_height;            ///< Convolution kernel height
    uint32_t kernel_width;             ///< Convolution kernel width

    // --- Pass Decomposition ---
    uint32_t num_passes;               ///< Number of passes to complete this tile
    std::vector<PassDescriptor> passes; ///< Pass descriptors (size = num_passes)

    // --- Memory Traffic (Total across all passes) ---
    uint64_t total_input_fetch_bytes;    ///< Total input fetch (fetch_operand0)
    uint64_t total_kernel_fetch_bytes;   ///< Total kernel fetch (Eq: F^K)
    uint64_t total_output_writeback_bytes; ///< Total output writeback
    uint64_t total_memory_traffic_bytes; ///< Sum of all traffic

    /**
     * @brief Default constructor
     */
    TileDescriptor()
        : tile_id(0), output_height_start(0), output_width_start(0),
          output_height(0), output_width(0), output_channels(0),
          input_height_start(0), input_width_start(0),
          input_height(0), input_width(0), input_channels(0),
          kernel_height(0), kernel_width(0), num_passes(1),
          total_input_fetch_bytes(0), total_kernel_fetch_bytes(0),
          total_output_writeback_bytes(0), total_memory_traffic_bytes(0) {}
};

// ============================================================================
// Fetch Traffic Model
// ============================================================================

/**
 * @brief Fetch traffic calculation
 *
 * Computes memory fetch requirements for a tile with multiple passes.
 *
 * Per-tile traffic:
 *   fetch_operand0  = initial_input + spill_ratio * input_reuse * (passes_per_output - parallel_passes) * output_size
 *   fetch_operand1 = initial_weight + spill_ratio * weight_reuse * (passes_per_output - parallel_passes) * output_size
 *
 * where:
 *   initial_input/weight  = base fetch for tile j
 *   output_size           = output element count for tile j
 *   passes_per_output     = number of passes (ceil(ops_per_output / ops_per_pass))
 *   parallel_passes       = PEGs per PES
 *   spill_ratio * reuse   = reuse factor (0 if reused across passes, 1 if refetched)
 *   num_tiles             = total number of tiles
 *
 * Total layer traffic:
 *   total = sum over all tiles (input_fetch + weight_fetch + output_size)
 */
struct FetchTraffic {
    uint64_t initial_input_bytes;      ///< initial input fetch for tile
    uint64_t initial_weight_bytes;     ///< initial weight fetch for tile
    uint64_t output_bytes;             ///< output size for tile
    uint32_t num_passes;               ///< passes_per_output: number of passes (ceil(ops_per_output / ops_per_pass))
    uint32_t pegs_per_pes;             ///< parallel_passes: PEGs per PES

    double input_reuse_factor;         ///< spill_ratio * beta_input: 0 (reuse) or 1 (refetch)
    double weight_reuse_factor;        ///< spill_ratio * beta_weight: 0 (reuse) or 1 (refetch)

    Dataflow dataflow;                 ///< Stationary operand dataflow type

    /**
     * @brief Compute total input fetch bytes
     *
     * total_input = initial_input + input_reuse_factor * (passes - parallel_passes) * output_size
     */
    uint64_t compute_input_fetch() const {
        uint32_t reload_count = (num_passes > pegs_per_pes) ? (num_passes - pegs_per_pes) : 0;
        uint64_t reuse_penalty = static_cast<uint64_t>(
            input_reuse_factor * output_bytes * reload_count
        );
        return initial_input_bytes + reuse_penalty;
    }

    /**
     * @brief Compute total weight fetch bytes
     *
     * total_weight = initial_weight + weight_reuse_factor * (passes - parallel_passes) * output_size
     */
    uint64_t compute_weight_fetch() const {
        uint32_t reload_count = (num_passes > pegs_per_pes) ? (num_passes - pegs_per_pes) : 0;
        uint64_t reuse_penalty = static_cast<uint64_t>(
            weight_reuse_factor * output_bytes * reload_count
        );
        return initial_weight_bytes + reuse_penalty;
    }

    /**
     * @brief Compute total memory traffic for this tile
     *
     * total_traffic = total_input + total_weight + output_size
     */
    uint64_t compute_total_traffic() const {
        return compute_input_fetch() + compute_weight_fetch() + output_bytes;
    }

    /**
     * @brief Default constructor (no reuse penalty)
     */
    FetchTraffic()
        : initial_input_bytes(0), initial_weight_bytes(0), output_bytes(0),
          num_passes(1), pegs_per_pes(1),
          input_reuse_factor(0.0), weight_reuse_factor(0.0),
          dataflow(Dataflow::WS) {}
};

// ============================================================================
// Tile Schedule
// ============================================================================

/**
 * @brief Tile schedule (ordered sequence of tiles for one layer)
 *
 * The scheduler partitions a layer's output into tiles and determines
 * execution order. Tiles may execute in various orders (row-major,
 * Z-order, etc.) depending on data reuse optimization.
 */
struct TileSchedule {
    std::string layer_name;            ///< Layer identifier
    uint32_t layer_id;                 ///< Layer index in network

    // --- Layer Dimensions ---
    uint32_t layer_output_height;      ///< Full layer output height
    uint32_t layer_output_width;       ///< Full layer output width
    uint32_t layer_output_channels;    ///< Full layer output channels
    uint32_t layer_input_channels;     ///< Full layer input channels

    // --- Tiling Configuration ---
    uint32_t tile_height;              ///< Standard tile height
    uint32_t tile_width;               ///< Standard tile width
    uint32_t num_tiles;                ///< Total number of tiles
    std::vector<TileDescriptor> tiles; ///< Tile descriptors (size = num_tiles)

    // --- Execution Order ---
    std::vector<uint32_t> execution_order; ///< Tile execution order (tile IDs)
    std::string scheduling_policy;     ///< Policy name (row_major, z_order, etc.)

    // --- Aggregate Statistics ---
    uint64_t total_layer_input_bytes;     ///< Total input fetch across all tiles
    uint64_t total_layer_kernel_bytes;    ///< Total kernel fetch across all tiles
    uint64_t total_layer_output_bytes;    ///< Total output writeback
    uint64_t total_layer_memory_traffic;  ///< Sum of all memory traffic

    /**
     * @brief Default constructor
     */
    TileSchedule()
        : layer_id(0), layer_output_height(0), layer_output_width(0),
          layer_output_channels(0), layer_input_channels(0),
          tile_height(0), tile_width(0), num_tiles(0),
          scheduling_policy("row_major"),
          total_layer_input_bytes(0), total_layer_kernel_bytes(0),
          total_layer_output_bytes(0), total_layer_memory_traffic(0) {}

    /**
     * @brief Compute aggregate statistics from tile descriptors
     *
     * Sums up memory traffic across all tiles.
     */
    void compute_statistics() {
        total_layer_input_bytes = 0;
        total_layer_kernel_bytes = 0;
        total_layer_output_bytes = 0;

        for (const auto& tile : tiles) {
            total_layer_input_bytes += tile.total_input_fetch_bytes;
            total_layer_kernel_bytes += tile.total_kernel_fetch_bytes;
            total_layer_output_bytes += tile.total_output_writeback_bytes;
        }

        total_layer_memory_traffic = total_layer_input_bytes +
                                      total_layer_kernel_bytes +
                                      total_layer_output_bytes;
    }
};

/**
 * @brief Compute tile schedule for a layer
 *
 * Mode A: automatically derives tile/pass decomposition and memory traffic
 * from layer dimensions, hardware config, and tile shape.
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
    uint32_t tile_height = 0,
    uint32_t tile_width = 0,
    // Optional per-region L2 budgets (KB). 0 = legacy conservative split
    // (40% input / 40% kernel / 20% output of the whole capacity). Wired from
    // buffers.global.*_partition_kb so a fixed GB partition shapes the tile
    // pass schedule instead of the hard-coded split.
    uint32_t input_budget_kb = 0,
    uint32_t kernel_budget_kb = 0,
    uint32_t output_budget_kb = 0,
    // Depthwise conv (from the CSV layer type): each output channel has one
    // Kh·Kw filter over a single input channel, so weight volume is Kh·Kw·Cout,
    // not the full-conv Kh·Kw·Cin·Cout. Drops the Cin over-count on the kernel
    // fetch (up to Cin×, e.g. 1024× on the last MobileNet layers).
    bool is_depthwise = false);

// ============================================================================
// Tile Transaction — the compiler's baked, runtime-executable schedule
// ============================================================================
//
// The runtime data controller executes a *list* of these in order. The
// compiler has already applied the dataflow's stationary-operand rule during
// lowering (e.g. WS ⇒ weight streamed only on the first tile, resident after),
// so the runtime is dataflow-agnostic: it just issues the per-tile DMA counts
// and drives the compute model with the per-tile output/pass counts. Each tile
// is sized to fit the on-chip buffer, so there is no whole-tensor streaming and
// no circular-buffer wrap.
//
// Operand counts are the *logical* counts consumed by the latency model
// (the same unit as F^I/F^K/F^O). *_bytes are the physical DMA sizes used for
// AXI/DRAM timing.
struct TileTxn {
    uint32_t input_fetch_operands  = 0;  ///< streamed input operands this tile (0 = resident)
    uint32_t weight_fetch_operands = 0;  ///< streamed weight operands this tile (0 = resident)
    uint32_t output_write_operands = 0;  ///< output operands this tile produces
    uint32_t num_passes            = 1;  ///< reduction passes for this tile
    uint32_t input_fetch_bytes     = 0;  ///< physical input DMA bytes (AXI/DRAM timing)
    uint32_t weight_fetch_bytes    = 0;  ///< physical weight DMA bytes
    uint32_t output_write_bytes    = 0;  ///< physical output DMA bytes
    uint32_t input_addr_offset     = 0;  ///< byte offset into the input tensor
    uint32_t output_addr_offset    = 0;  ///< byte offset into the output tensor
    // psum spill: when reduction passes cannot keep this tile's partial output
    // resident, partials round-trip DRAM between passes. These are extra DMA
    // bytes on top of the final output write (0 when partials stay on-chip).
    uint32_t psum_spill_read_bytes  = 0;  ///< partial-output re-reads (passes 2..P)
    uint32_t psum_spill_write_bytes = 0;  ///< partial-output writes (passes 1..P-1)
};

/**
 * @brief Lower a compile-time TileSchedule into the runtime transaction list,
 *        applying the dataflow's stationary-operand rule.
 *
 * WS: weight resident — streamed on the first tile only; input streamed per tile.
 * IS: input resident — streamed on the first tile only; weight streamed per tile.
 * OS: both streamed per tile.
 *
 * By construction, summing the per-tile counts reproduces the layer descriptor:
 *   Σ input_fetch_operands  == F^I,
 *   Σ weight_fetch_operands == F^K,
 *   Σ output_write_operands == F^O,
 * so the executed transactions are consistent with the derived descriptor.
 *
 * @param input_total_operands  full input-tensor operand count (H·W·Cin) — used
 *                              as the resident load on the first tile for IS.
 * @param weight_total_operands full weight-tensor operand count (kh·kw·Cin·Cout)
 *                              — the resident load on the first tile for WS.
 * @param element_bytes         bytes per operand (for the physical *_bytes fields).
 */
std::vector<TileTxn> lower_tile_schedule(const TileSchedule& sched,
                                         Dataflow dataflow,
                                         uint32_t input_total_operands,
                                         uint32_t weight_total_operands,
                                         uint32_t element_bytes,
                                         uint32_t compute_passes_per_output = 1);

/// Per-operand residency: the stationary rule the Dataflow enum encodes,
/// exposed as two independent bits (hierarchy v1, D1). resident = streamed
/// once on the first tile and held on-chip; !resident = streamed per tile.
/// WS = {weight}, IS = {input}, OS = {} — the enum overload maps to this.
struct OperandResidency {
    bool input_resident  = false;
    bool weight_resident = false;
};

std::vector<TileTxn> lower_tile_schedule(const TileSchedule& sched,
                                         OperandResidency residency,
                                         uint32_t input_total_operands,
                                         uint32_t weight_total_operands,
                                         uint32_t element_bytes,
                                         uint32_t compute_passes_per_output = 1);

} // namespace flexnpu_sim
