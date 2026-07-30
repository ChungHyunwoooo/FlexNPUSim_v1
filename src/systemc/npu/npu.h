/**
 * @file npu.h
 * @brief Npu SC_MODULE — the accelerator container.
 *
 * The NPU is a container that INSTANCEs its sub-blocks and wires them:
 *   - NpuGlobalBuffer gb   — on-chip SRAM (shared)                  [done]
 *   - DmaChannel rdma/wdma — rDMAC / wDMAC + command signals        [done]
 *   - NpuController        — sequencing (registers, FSM, DMA/UFB orchestration)
 *   - Ufb                  — unit function block (compute datapath) [Phase C: pending]
 *
 * The controller DRIVES the sub-blocks (holds them by reference); it no longer
 * owns them. The external interface flexnpusim_system talks to lives here, so
 * internal rewiring does not touch the caller.
 */

#pragma once

#include <systemc.h>

#include "systemc/npu/npu_controller.h"
#include "systemc/npu/dma_channel.h"
#include "systemc/bus/memory_mapped_transport.h"

namespace flexnpu_sim {

SC_MODULE(Npu) {
    sc_in_clk clk;   // distributed to the sub-blocks

    // Sub-blocks, declared in construction order (each binds to earlier ones).
    NpuGlobalBuffer gb_;                 // on-chip SRAM (shared)
    DmaChannel      rdma_ch_, wdma_ch_;  // read / write DMA channels
    Ufb             ufb_;                // unit function block (PE compute datapath)
    NpuController   controller_;         // driver (references gb_ + channels + ufb_)

    Npu(sc_module_name name,
        uint32_t gb_bytes = NpuController::GB_SIZE_DEFAULT,
        uint32_t npu_base = memory_map::ADDR_NPU)
        : sc_module(name),
          gb_(gb_bytes),
          rdma_ch_("rdma_ch"),
          wdma_ch_("wdma_ch"),
          controller_("controller", gb_, rdma_ch_, wdma_ch_, ufb_, npu_base) {
        controller_.clk(clk);
        // Clock the DMA engines and hand them the shared GB backing store.
        rdma_ch_.engine.clk(clk);
        wdma_ch_.engine.clk(clk);
        rdma_ch_.engine.gb_data = gb_.data();
        rdma_ch_.engine.gb_size = gb_.size();
        wdma_ch_.engine.gb_data = gb_.data();
        wdma_ch_.engine.gb_size = gb_.size();
    }

    /// Direct access to the controller (for result reads: layer_records, perf_*).
    NpuController&       controller()       { return controller_; }
    const NpuController& controller() const { return controller_; }

    // ---- Setup interface ----
    void bind(AXI_SIGNALS<NpuRegConfig>& signals) { controller_.bind(signals); }
    void attach_dma_transports(transport::MemoryMappedTransport* rd,
                               transport::MemoryMappedTransport* wr) {
        rdma_ch_.engine.attach_transport(rd);
        wdma_ch_.engine.attach_transport(wr);
    }
    // DMA IP flow-control: FIFO depth (bytes) + explicit AXI issuing capability.
    void set_dma_flow_control(uint32_t fifo_bytes,
                              uint32_t max_issuing_reads,
                              uint32_t max_issuing_writes) {
        rdma_ch_.engine.set_flow_control(fifo_bytes, max_issuing_reads,
                                         max_issuing_writes);
        wdma_ch_.engine.set_flow_control(fifo_bytes, max_issuing_reads,
                                         max_issuing_writes);
    }
    void set_double_buffer(bool enable)         { controller_.set_double_buffer(enable); }
    void set_axi_bus_width(uint32_t bits)       { controller_.set_axi_bus_width(bits); }
    void set_clock_period_ns(double ns)         { controller_.set_clock_period_ns(ns); }
    void set_system_config(const config::FlexNpuSimConfig& cfg) {
        controller_.set_system_config(cfg);
    }
    void set_functional_ofm_mode(bool enable)   { controller_.set_functional_ofm_mode(enable); }
    void set_functional_ofm_dir(const std::string& dir) {
        controller_.set_functional_ofm_dir(dir);
    }
    void set_gb_mode(NpuGlobalBuffer::Mode mode) { controller_.set_gb_mode(mode); }
    void set_gb_cache_line_size(uint32_t bytes)  { controller_.set_gb_cache_line_size(bytes); }
    void set_timeline_output(const std::string& path) {
        controller_.set_timeline_output(path);
    }
};

}  // namespace flexnpu_sim
