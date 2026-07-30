/**
 * @file dma_channel.h
 * @brief One NPU DMA channel — a DmaEngine plus its command signals.
 *
 * The engine and its command signals are bound together here so the NPU
 * container can own read/write DMA channels as siblings of the controller.
 * The controller DRIVES a channel through its signals (trigger_rdma/wdma); the
 * engine moves bytes between DRAM (its attached transport) and the global buffer.
 */

#pragma once

#include <systemc.h>
#include "systemc/dma/dma_engine.h"

namespace flexnpu_sim {

SC_MODULE(DmaChannel) {
    sc_signal<uint32_t> src_addr, dst_addr, transfer_size;
    sc_signal<bool>     start, busy, done, read_mode;
    DmaEngine engine;

    explicit DmaChannel(sc_module_name nm) : sc_module(nm), engine("engine") {
        engine.src_addr(src_addr);
        engine.dst_addr(dst_addr);
        engine.transfer_size(transfer_size);
        engine.start(start);
        engine.busy(busy);
        engine.done(done);
        engine.read_mode(read_mode);
    }
};

}  // namespace flexnpu_sim
