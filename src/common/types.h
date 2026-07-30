/**
 * @file types.h
 * @brief FlexNPUSim shared types, memory map, NPU registers, FSM states
 *
 * Follows v2.0 spec Sections 9, 10, Table 6, Table 13.
 */

#pragma once

#include <cstdint>

namespace flexnpu_sim {

// ============================================================================
// Base types
// ============================================================================

using addr_t   = uint64_t;
using data32_t = uint32_t;
using data64_t = uint64_t;

// ============================================================================
// Memory Map (v2.0 Table 6)
// ============================================================================

namespace memory_map {

constexpr addr_t ADDR_SRAM      = 0x20000000;  // 1MB
constexpr addr_t ADDR_NPU       = 0x40000000;  // 64KB
constexpr addr_t ADDR_DRAM_CTRL = 0x50000000;  // 64KB
constexpr addr_t ADDR_PROC      = 0x60000000;  // 64KB
constexpr addr_t ADDR_DRAM      = 0x80000000;  // 2GB

constexpr uint32_t SIZE_SRAM      = 0x00100000;  // 1MB
constexpr uint32_t SIZE_NPU       = 0x00010000;  // 64KB
constexpr uint32_t SIZE_DRAM_CTRL = 0x00010000;  // 64KB
constexpr uint32_t SIZE_PROC      = 0x00010000;  // 64KB
constexpr uint64_t SIZE_DRAM      = 0x80000000;  // 2GB

} // namespace memory_map

// ============================================================================
// NPU register offsets (v2.0 Table 13, 13 in total)
// ============================================================================

namespace npu_reg {

constexpr uint32_t NPUCR     = 0x00;  // R/W  Control
constexpr uint32_t NPUSR     = 0x04;  // RO   Status
constexpr uint32_t DNNSA     = 0x08;  // R/W  DNN image base (low 32b)
constexpr uint32_t DNNSA_MSB = 0x0C;  // R/W  DNN image base (high 32b)
constexpr uint32_t NPUIER    = 0x10;  // R/W  Interrupt Enable (stub)
constexpr uint32_t NPUISR    = 0x14;  // R/W1C Interrupt Status (stub)
constexpr uint32_t TILE_CFG  = 0x18;  // R/W  Tile config (stub)
constexpr uint32_t PASS_CFG  = 0x1C;  // R/W  Pass config (stub)
constexpr uint32_t BUF_CFG   = 0x20;  // R/W  Buffer config
constexpr uint32_t PERF_CNT0 = 0x24;  // RO   Total cycles (low 32b)
constexpr uint32_t PERF_CNT1 = 0x28;  // RO   Total MACs (low 32b)
constexpr uint32_t PERF_CNT2 = 0x2C;  // RO   Memory read bytes (low 32b)
constexpr uint32_t PERF_CNT3 = 0x30;  // RO   Memory write bytes (low 32b)
// High 32b of each 64-bit perf counter. Large networks (e.g. VGG-16 at
// 15.4 G MACs, or multi-GB DRAM traffic) exceed 2^32, so the counters are
// exposed as {HI:LO} pairs and recombined by the host.
constexpr uint32_t PERF_CNT0_HI = 0x34;  // RO   Total cycles (high 32b)
constexpr uint32_t PERF_CNT1_HI = 0x38;  // RO   Total MACs (high 32b)
constexpr uint32_t PERF_CNT2_HI = 0x3C;  // RO   Memory read bytes (high 32b)
constexpr uint32_t PERF_CNT3_HI = 0x40;  // RO   Memory write bytes (high 32b)

constexpr uint32_t REG_COUNT = 17;
constexpr uint32_t REG_SPACE = 0x44;  // last register + 4

// NPUCR bit fields
constexpr uint32_t NPUCR_START   = (1u << 0);
constexpr uint32_t NPUCR_RESET   = (1u << 1);
constexpr uint32_t NPUCR_TILE_EN = (1u << 2);
constexpr uint32_t NPUCR_WB_EN   = (1u << 3);
constexpr uint32_t NPUCR_MODE_MASK = (0x3u << 4);

// NPUSR bit fields
constexpr uint32_t NPUSR_HALT     = (1u << 0);
constexpr uint32_t NPUSR_IDLE     = (1u << 1);
constexpr uint32_t NPUSR_ERROR    = (1u << 2);
constexpr uint32_t NPUSR_WB_ACTIVE = (1u << 3);

} // namespace npu_reg

// ============================================================================
// NPU FSM states
// ============================================================================

enum class NpuState : uint8_t {
    Idle,
    LoadLayerDesc,
    DmaReadInput,
    DmaReadKernel,
    DmaAndCompute,     ///< DMA read overlapped with compute (Fix #7)
    Compute,
    DmaWriteOutput,
    LayerComplete,
    AllDone
};

// ============================================================================
// PE Type
// ============================================================================

enum class PeType : uint8_t {
    Systolic,    ///< Systolic array — issue_latency = ops_per_pass
    AdderTree    ///< Adder tree — issue_latency = 1 + ceil(log2(ops_per_pass))
};

// ============================================================================
// Dataflow
// ============================================================================

enum class Dataflow : uint8_t {
    WS,   ///< Weight Stationary  — weights pinned, inputs resent
    IS,   ///< Input Stationary   — inputs pinned, weights resent
    OS    ///< Output Stationary  — outputs pinned, inputs+weights resent
};

// ============================================================================
// PE Configuration
// ============================================================================

struct PeConfig {
    PeType   pe_type              = PeType::Systolic;
    uint32_t ops_per_pass         = 1;   // unit operations one pass handles
    uint32_t parallel_passes      = 1;   // passes that can run in parallel for one output
    uint32_t num_output_lanes     = 16;  // independent lanes working on distinct outputs
    uint32_t partial_sum_buffer_kb = 0;  // local partial-sum capacity (KB). 0 = none
    uint32_t element_size_bytes   = 1;   // bytes per operand (INT8=1, FP16=2, FP32=4)

    uint32_t n_max() const { return num_output_lanes * parallel_passes; }
    // n_min: minimum positive pass-issue width per cycle.
    // A PES with g PEGs issues g passes per cycle when active.
    // A single active PES issues `parallel_passes` passes.
    // For NVDLA (g=1), n_min=1. For MAERI (g>1), n_min=g.
    uint32_t n_min() const { return parallel_passes; }
    uint32_t total_execution_units() const { return num_output_lanes * parallel_passes; }
};

// ============================================================================
// AXI config types (SystemC_AXI library Config aliases)
// ============================================================================
// CPU ↔ NPU registers: 32-bit addr, 32-bit data
// DMA ↔ Memory:        32-bit addr, 64-bit data
// The AXIConfig template itself lives in axi_config.h.
// The config aliases FlexNPUSim uses are defined in the SystemC module headers.

} // namespace flexnpu_sim
