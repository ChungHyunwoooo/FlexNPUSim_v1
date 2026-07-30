/**
 * @file processor.h
 * @brief Processor SC_MODULE — the host CPU that programs and babysits the NPU.
 *
 * Consolidates the three former processor abstractions (CpuDriver +
 * ProcessorModel + ProcessorAxiMaster) into one SystemC model, built on the
 * DSL-lab "fractal_processor" idiom:
 *
 *   - The firmware C/C++ runs NATIVELY on the SystemC host (not on FPGA).
 *   - INTERNAL vs EXTERNAL accesses are decoupled (as in fractal_processor):
 *       · EXTERNAL — peripheral/MMIO (NPU registers) are MANUAL AXI
 *         transactions (reg_read/reg_write → master_if->read_single/write_single).
 *         These generate a real bus transaction; their latency IS the AXI
 *         handshake + slave response time. They do NOT go through the cache.
 *       · INTERNAL — the CPU's own data-memory loads/stores are counted by
 *         software_delay(store,load,...) and priced by mem_access(), a purely
 *         statistical cache hit/miss penalty with NO bus transaction.
 *   - Compute latency comes from software_delay(store,load,jmp,mul,div,gen):
 *     instruction-class counts hand-derived from the disassembly of the
 *     equivalent ARM firmware (branch/mul/div/generic cycle costs).
 *
 * Job: hand the DNN image base to the NPU, start it, poll to completion, then
 * read back the performance counters — same control path CpuDriver drove, now
 * with an explicit CPU-side latency model.
 */

#pragma once

#include <systemc.h>
#include <cstdlib>
#include <iostream>

#include "systemc/npu/npu_controller.h"        // NpuRegConfig, npu_reg, memory_map, AXI_*
#include "systemc/bus/axi/axi_signal_ports.h"  // bind_port_signal

namespace flexnpu_sim {

SC_MODULE(Processor) {
    sc_in_clk clk;
    AXI_master_if<NpuRegConfig>* master_if;
    sc_event test_done;
    bool completed = false;

    // NPU performance counters, read back over AXI at completion. Report
    // interface preserved from CpuDriver so flexnpusim_system is unchanged.
    uint64_t cnt0 = 0, cnt1 = 0, cnt2 = 0, cnt3 = 0;

    // This model's own contribution: CPU-side cycles spent programming and
    // polling the NPU (compute + memory-access latency accumulated below).
    uint64_t cpu_cycles = 0;

    uint32_t npu_base  = memory_map::ADDR_NPU;   ///< set by caller (config-driven)
    uint32_t dram_base = memory_map::ADDR_DRAM;

    // Latency model parameters (defaults mirror config::ProcessorCfg). Set from
    // the hw JSON via configure(); see software_delay()/mem_access() below.
    uint32_t cost_jump    = 5;    ///< branch/jump cycles
    uint32_t cost_mul     = 4;    ///< integer multiply cycles
    uint32_t cost_div     = 10;   ///< integer divide cycles
    uint32_t cost_generic = 1;    ///< ALU/mov/etc. cycles
    // Cache hit/miss model for INTERNAL data-memory accesses only (mem_access);
    // EXTERNAL MMIO never goes through this. cache_miss_ratio is the target miss
    // fraction the statistical model converges to.
    double   cache_miss_ratio = 1.0;   ///< target fraction of accesses that miss
    uint32_t hit_latency      = 1;     ///< cycles on an L1 hit
    uint32_t miss_latency     = 100;   ///< cycles on a miss (DRAM round-trip)
    uint64_t hit_count = 0, miss_count = 0;  ///< running tallies (feedback model)

    SC_HAS_PROCESS(Processor);

    Processor(sc_module_name name) : sc_module(name) {
        master_if = new AXI_master_if<NpuRegConfig>("cpu_master");
        SC_THREAD(run);
    }
    ~Processor() { delete master_if; }

    void bind(AXI_SIGNALS<NpuRegConfig>& sig) {
        master_if->clk(clk);
        bind_port_signal(&master_if->ports, sig);
    }

    /// Load latency-model parameters from the hw JSON's `processor` section.
    void configure(const config::ProcessorCfg& pc) {
        cost_jump        = pc.cost_jump;
        cost_mul         = pc.cost_mul;
        cost_div         = pc.cost_div;
        cost_generic     = pc.cost_generic;
        hit_latency      = pc.cache_hit_latency;
        miss_latency     = pc.cache_miss_latency;
        cache_miss_ratio = pc.cache_miss_ratio;
    }

    // ------------------------------------------------------------------
    // Firmware: the C/C++ an ARM host would run to drive the NPU. Executes
    // natively here; every MMIO access is a manual AXI transaction.
    // ------------------------------------------------------------------
    void run() {
        wait(10, SC_NS);
        // Region A (entry/program): materialize base+constants, init loop vars.
        // AArch64 -O1: 8 generic (mov×7, movk×1) + 1 branch (enter loop). The 3
        // stores are the NPU-register writes below → EXTERNAL (reg_write), so
        // store=load=0 here; no internal data memory in this region.
        software_delay(/*store=*/0, /*load=*/0, /*jmp=*/1, /*mul=*/0, /*div=*/0, /*gen=*/8);
        uint32_t base = npu_base;

        // Set DNN image base address + START.
        reg_write(base + npu_reg::DNNSA,     dram_base);
        reg_write(base + npu_reg::DNNSA_MSB, 0);
        reg_write(base + npu_reg::NPUCR,     npu_reg::NPUCR_START);

        // Poll NPUSR for completion (reactive) with a relaxed per-layer
        // watchdog. sc_main runs until this thread calls sc_stop().
        uint32_t watchdog = 0, last_layer = 0xFF;
        while (watchdog < 20000) {            // 20000 × 10ms = 200s per layer
            wait(10, SC_MS);
            uint32_t sr = reg_read(base + npu_reg::NPUSR);   // EXTERNAL (real txn)
            // Region B (one poll body): AArch64 -O1 gives 7 generic (add/ubfx/
            // cmp×2/csinc/mov/lsr) + 3 branch (idle break, watchdog break,
            // back-edge). No mul/div — /4 and the 20000 compare strength-reduce.
            // The NPUSR read is EXTERNAL (reg_read above) → store=load=0.
            software_delay(/*store=*/0, /*load=*/0, /*jmp=*/3, /*mul=*/0, /*div=*/0, /*gen=*/7);
            if (sr & npu_reg::NPUSR_IDLE) { completed = true; break; }

            uint32_t cur = (sr >> 8) & 0xFF;
            if (cur != last_layer) {
                if (std::getenv("FLEXNPUSIM_VERBOSE"))
                    std::cerr << "  Layer " << cur << "/" << ((sr >> 16) & 0xFF) << "\n";
                last_layer = cur; watchdog = 0;
            } else {
                watchdog++;
            }
        }

        if (!completed)
            std::cerr << "ERROR: NPU timeout at layer " << (int)last_layer << "\n";

        // Read the 64-bit performance counters ({HI:LO} pairs).
        auto rd64 = [&](uint32_t lo_off, uint32_t hi_off) -> uint64_t {
            uint64_t lo = reg_read(base + lo_off);
            uint64_t hi = reg_read(base + hi_off);
            return (hi << 32) | lo;
        };
        cnt0 = rd64(npu_reg::PERF_CNT0, npu_reg::PERF_CNT0_HI);
        cnt1 = rd64(npu_reg::PERF_CNT1, npu_reg::PERF_CNT1_HI);
        cnt2 = rd64(npu_reg::PERF_CNT2, npu_reg::PERF_CNT2_HI);
        cnt3 = rd64(npu_reg::PERF_CNT3, npu_reg::PERF_CNT3_HI);

        if (std::getenv("FLEXNPUSIM_VERBOSE"))
            std::cerr << "  CPU-side cycles (program+poll): " << cpu_cycles << "\n";

        test_done.notify();
        sc_stop();
    }

private:
    // ------------------------------------------------------------------
    // Timing annotation (fractal_processor software_delay). jmp/mul/div/gen are
    // instruction-class counts hand-derived from the firmware disassembly;
    // store/load are the region's INTERNAL data-memory accesses, each priced by
    // mem_access() (statistical cache, no bus). NPU-register accesses are
    // EXTERNAL — they are reg_read/reg_write, not counted here.
    // ------------------------------------------------------------------
    void software_delay(int store, int load, int jmp, int mul, int div, int gen) {
        uint32_t cyc = jmp * cost_jump + mul * cost_mul
                     + div * cost_div + gen * cost_generic;
        cpu_cycles += cyc;
        for (uint32_t i = 0; i < cyc; ++i) wait(clk.posedge_event());
        for (int i = 0; i < store + load; ++i) mem_access();  // internal memory
    }

    // One INTERNAL data-memory access: statistical cache hit/miss, NO bus.
    // Feedback model (fractal_processor): if the running hit ratio is above the
    // target, this access misses (pulling the ratio back), else it hits — so the
    // realised miss fraction converges to cache_miss_ratio.
    void mem_access() {
        uint64_t total = hit_count + miss_count;
        double   hit_ratio = total ? static_cast<double>(hit_count) / total : 0.0;
        uint32_t cyc;
        if (hit_ratio > 1.0 - cache_miss_ratio) { ++miss_count; cyc = miss_latency; }
        else                                    { ++hit_count;  cyc = hit_latency; }
        cpu_cycles += cyc;
        for (uint32_t i = 0; i < cyc; ++i) wait(clk.posedge_event());
    }

    // EXTERNAL MMIO access points: manual AXI transactions only. The real bus
    // handshake + NPU-slave response IS the latency; no cache penalty applied.
    uint32_t reg_read(uint32_t addr) {
        return master_if->read_single(addr).to_uint();
    }
    void reg_write(uint32_t addr, uint32_t data) {
        master_if->write_single(addr, data);
    }
};

}  // namespace flexnpu_sim
