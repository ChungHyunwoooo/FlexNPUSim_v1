/**
 * @file npu_controller_regs.cpp
 * @brief AXI slave register interface (reg R/W, slave threads, NPUSR)
 *
 * Split from npu_controller.cpp (behaviour-neutral); see
 * npu_controller.h for the class and npu_controller_internal.h for
 * the shared free helpers.
 */

#include "systemc/npu/npu_controller.h"
#include "systemc/npu/npu_controller_internal.h"
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
// Register Access
// ============================================================================

uint32_t NpuController::reg_read(uint32_t offset) const {
    uint32_t idx = offset / 4;
    if (idx >= npu_reg::REG_COUNT) return 0;

    switch (offset) {
        case npu_reg::NPUSR:
            return regs[idx];
        case npu_reg::PERF_CNT0:
            return static_cast<uint32_t>(perf_cycles);
        case npu_reg::PERF_CNT1:
            return static_cast<uint32_t>(perf_macs);
        case npu_reg::PERF_CNT2:
            return static_cast<uint32_t>(perf_mem_reads);
        case npu_reg::PERF_CNT3:
            return static_cast<uint32_t>(perf_mem_writes);
        case npu_reg::PERF_CNT0_HI:
            return static_cast<uint32_t>(perf_cycles >> 32);
        case npu_reg::PERF_CNT1_HI:
            return static_cast<uint32_t>(perf_macs >> 32);
        case npu_reg::PERF_CNT2_HI:
            return static_cast<uint32_t>(perf_mem_reads >> 32);
        case npu_reg::PERF_CNT3_HI:
            return static_cast<uint32_t>(perf_mem_writes >> 32);
        default:
            return regs[idx];
    }
}

void NpuController::reg_write(uint32_t offset, uint32_t value) {
    uint32_t idx = offset / 4;
    if (idx >= npu_reg::REG_COUNT) return;

    // Writes to RO registers are ignored
    if (offset == npu_reg::NPUSR ||
        offset == npu_reg::PERF_CNT0 ||
        offset == npu_reg::PERF_CNT1 ||
        offset == npu_reg::PERF_CNT2 ||
        offset == npu_reg::PERF_CNT3 ||
        offset == npu_reg::PERF_CNT0_HI ||
        offset == npu_reg::PERF_CNT1_HI ||
        offset == npu_reg::PERF_CNT2_HI ||
        offset == npu_reg::PERF_CNT3_HI) {
        return;
    }

    // W1C: NPUISR
    if (offset == npu_reg::NPUISR) {
        regs[idx] &= ~value;
        return;
    }

    regs[idx] = value;

    // Detect NPUCR.START
    if (offset == npu_reg::NPUCR && (value & npu_reg::NPUCR_START)) {
        start_event.notify();
    }

    // NPUCR.RESET
    if (offset == npu_reg::NPUCR && (value & npu_reg::NPUCR_RESET)) {
        set_state(NpuState::Idle);
        perf_cycles = 0;
        perf_macs = 0;
        perf_mem_reads = 0;
        perf_mem_writes = 0;
        perf_read_txns = 0;
        perf_write_txns = 0;
        gb_.reset_for_layer();
        layer_input_loaded_ = false;
        layer_weight_loaded_ = false;
        layer_compute_done_ = false;
        global_buffer_.reset_presence();
        regs[npu_reg::NPUCR / 4] &= ~npu_reg::NPUCR_RESET;
        update_status_reg();
    }
}

void NpuController::update_status_reg() {
    uint32_t sr = 0;
    if (state == NpuState::Idle || state == NpuState::AllDone) {
        sr |= npu_reg::NPUSR_IDLE;
    }
    if (state == NpuState::AllDone) {
        sr |= npu_reg::NPUSR_HALT;
    }
    sr |= ((current_layer_idx_ & 0xFF) << 8);
    sr |= ((total_layers_ & 0xFF) << 16);
    regs[npu_reg::NPUSR / 4] = sr;
}

// ============================================================================
// Slave Read Thread — serves CPU register reads
// ============================================================================

void NpuController::slave_read_thread() {
    while (true) {
        wait(slave_if->read_reg_event);

        if (slave_if->readQueue.empty()) continue;

        auto* trans = slave_if->readQueue.front();
        slave_if->readQueue.pop();

        uint32_t addr = trans->addr.to_uint();
        uint32_t offset = addr - npu_base_;
        uint32_t len = trans->len.to_uint();

        slave_if->Rid = trans->ID;
        slave_if->Rlen = trans->len;

        uint32_t current_offset = offset;
        for (uint32_t beat = 0; beat <= len; ++beat) {
            uint32_t data = reg_read(current_offset);
            slave_if->read_buffer = data;
            slave_if->set_read_data_event.notify();
            wait(slave_if->get_read_data_event);
            current_offset += 4;
        }

        delete trans;
    }
}

// ============================================================================
// Slave Write Thread — serves CPU register writes
// ============================================================================

void NpuController::slave_write_thread() {
    while (true) {
        wait(slave_if->write_reg_event);

        if (slave_if->writeQueue.empty()) continue;

        auto* trans = slave_if->writeQueue.front();
        slave_if->writeQueue.pop();

        uint32_t addr = trans->addr.to_uint();
        uint32_t offset = addr - npu_base_;
        uint32_t len = trans->len.to_uint();

        slave_if->Wid = trans->ID;
        slave_if->Wlen = trans->len;

        uint32_t current_offset = offset;
        for (uint32_t beat = 0; beat <= len; ++beat) {
            wait(slave_if->set_write_data_event);
            uint32_t data = slave_if->write_buffer.to_uint();
            reg_write(current_offset, data);

            if (beat < len) {
                slave_if->write_resp_event.notify();
            }
            current_offset += 4;
        }

        delete trans;
    }
}

// ============================================================================

}  // namespace flexnpu_sim
