/**
 * @file npu_controller_internal.h
 * @brief Free helpers shared across the NpuController translation units.
 *
 * NpuController's implementation is split across npu_controller*.cpp (one file
 * per responsibility). Two file-local helpers are used by more than one of
 * those units; they live here as `inline` so every unit sees one definition.
 * Unit-local helpers stay `static` in their own .cpp.
 */

#pragma once

#include "compiler/dnn_image/dnn_image_format.h"  // FuncType

#include <cstdint>
#include <cstdlib>

namespace flexnpu_sim {

// DMA transaction/byte tallies. Incremented by the DMA unit (trigger_rdma/wdma)
// and read by the read path for its chunk-size profiling stat, so they are
// shared across those two translation units (inline → one instance).
inline uint64_t g_rdma_count = 0;
inline uint64_t g_wdma_count = 0;
inline uint64_t g_rdma_bytes = 0;
inline uint64_t g_wdma_bytes = 0;

// pe_buffers lookup key — a latency model is 1:1 with a function descriptor;
// per-unit buffer overrides are keyed by the CSV function-type strings.
inline const char* function_buffer_key(FuncType t) {
    switch (t) {
        case FuncType::Conv2D:          return "conv2d";
        case FuncType::DepthwiseConv2D: return "dwconv";
        case FuncType::FullyConnected:  return "fully_connected";
        case FuncType::Pooling:         return "pooling";
        case FuncType::Activation:      return "activation";
        case FuncType::MatMul:          return "matmul";
        case FuncType::ElementWise:     return "elementwise";
        default:                         return "";
    }
}

// Per-layer console line mode. Default = short human progress line; the
// machine-readable @LAYER CSV line is emitted when FLEXNPUSIM_LAYER_CSV=1
// (set by demo/run.py and parser scripts) or FLEXNPUSIM_VERBOSE=1.
inline bool layer_line_csv() {
    static const bool on = (std::getenv("FLEXNPUSIM_LAYER_CSV") != nullptr) ||
                           (std::getenv("FLEXNPUSIM_VERBOSE") != nullptr);
    return on;
}

}  // namespace flexnpu_sim
