/**
 * @file npu_controller.cpp
 * @brief NPU Controller implementation
 *
 * v2.0 Architecture:
 *   - fast_dma fully removed: every DMA goes through trigger_rdma/wdma
 *     (real AXI transfers)
 *   - DRAM timing: MemoryAxiSlave applies it on the AXI response (removed
 *     from the NPU)
 *   - double buffer: config-driven cross-layer pipelining
 *
 *   M1  — DMA/Compute/Write run concurrently (SystemC scheduler interleave)
 *   M2  — Compute advances real simulation time
 *   M3  — Operands arrive incrementally (chunked DMA → l2_*_ready grows)
 *   M4  — write_thread performs DMA writes independently
 *   M5  — cross-layer pipelining (write/read overlap when L2 has room)
 *   M7  — AXI bus width parameterized
 *   M12 — fetch_rate = n_max (independent of DMA BW)
 *   M13 — L2 buffer management (GbState tracking)
 *   M16 — perf_cycles = total DMA + compute + stall + write time
 */

#include "systemc/npu/npu_controller.h"
#include "system/packet_execution_policy.h"
#include "common/debug_log.h"
#include "compiler/dnn_image/tile_search.h"
#include "model/function/layer_function_runner.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "model/latency/latency_model_chain.h"
#include "systemc/memory/model/peak_bw_layer_model.h"

namespace flexnpu_sim {

// ============================================================================
// Constructor / Destructor
// ============================================================================

NpuController::NpuController(sc_module_name name, NpuGlobalBuffer& gb,
                            DmaChannel& rdma_ch, DmaChannel& wdma_ch, Ufb& ufb,
                            uint32_t npu_base)
    : sc_module(name),
      rdma_ch_(rdma_ch),
      wdma_ch_(wdma_ch),
      global_buffer_(gb),
      state(NpuState::Idle),
      ufb_(ufb)
{
    npu_base_ = npu_base;
    std::memset(regs, 0, sizeof(regs));

    slave_if = new AXI_slave_if<NpuRegConfig>(
        "slave_if",
        npu_base_,
        npu_reg::REG_SPACE
    );
    slave_if->clk(clk);

    // The DMA channels (engines + command signals) are siblings owned by the Npu
    // container; it clocks the engines, hands them the shared GB backing store,
    // and attaches the data-path transports. The controller only drives the
    // command signals (trigger_rdma / trigger_wdma / wdma_issue_thread).

    SC_THREAD(slave_read_thread);
    SC_THREAD(slave_write_thread);
    SC_THREAD(read_thread);
    SC_THREAD(compute_thread);
    SC_THREAD(write_thread);
    SC_THREAD(wdma_issue_thread);
}

NpuController::~NpuController() {
    delete slave_if;
}

void NpuController::set_gb_mode(NpuGlobalBuffer::Mode mode) {
    global_buffer_.set_mode(mode);
}

void NpuController::set_gb_cache_line_size(uint32_t bytes) {
    global_buffer_.set_cache_line_size(bytes);
}

void NpuController::set_timeline_output(const std::string& path) {
    timeline_output_path_ = path;
    timeline_enabled_ = !timeline_output_path_.empty();
}
uint32_t NpuController::input_stream_capacity_bytes() const {
    if (!double_buffer_enabled_) return gb_size_bytes();
    const uint32_t reserved = std::min(gb_.output.available(), gb_size_bytes());
    const uint32_t available = gb_size_bytes() - reserved;
    return available > 0 ? available : gb_size_bytes();
}

uint32_t NpuController::output_region_base_offset() const {
    if (!double_buffer_enabled_) return 0;
    const uint32_t reserved = std::min(gb_.output.available(), gb_size_bytes());
    // reserved == 0 means there is no separate GB output partition (e.g.
    // output_location=accumulator): the output does not occupy a reserved GB
    // region, so stage WDMA from offset 0. Returning gb_size here would place
    // the WDMA at the end of the buffer and overflow (off == gb_size).
    if (reserved == 0) return 0;
    return gb_size_bytes() - reserved;
}

uint32_t NpuController::output_region_capacity_bytes() const {
    const uint32_t base = output_region_base_offset();
    const uint32_t capacity = gb_size_bytes() - base;
    return capacity > 0 ? capacity : gb_size_bytes();
}

// ============================================================================
// bind
// ============================================================================

void NpuController::bind(AXI_SIGNALS<NpuRegConfig>& signals) {
    bind_port_signal(&slave_if->ports, signals);
}


}  // namespace flexnpu_sim
