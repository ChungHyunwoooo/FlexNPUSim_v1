#ifndef AXI_H
#define AXI_H

/**
 * AXI4 Template Library
 *
 * Complete AXI4 protocol implementation with:
 * - Parameterized bit widths (ID, ADDR, DATA)
 * - Configurable master/slave count
 * - Runtime memory map configuration
 * - Full AXI4 signal set (LOCK, CACHE, PROT, QoS, REGION, BRESP, RRESP, WSTRB)
 * - AXI3/AXI4 support via templates
 *
 * Quick Start:
 *   #include "axi.h"
 *
 *   // Use default AXI4 configuration
 *   AXI_SIGNALS<> signals;
 *   AXI_master_if<> master("master");
 *   AXI_slave_if<> slave("slave", 0x0, 0x1000);
 *
 *   // Custom configuration
 *   using MyAXI = AXIConfig<8, 64, 512>;  // 8-bit ID, 64-bit addr, 512-bit data
 *   AXI_SIGNALS<MyAXI> sig;
 *   AXI_master_if<MyAXI> m("m");
 *
 *   // Memory map
 *   DynamicMemoryMap<sc_uint<32>> mmap({
 *       {0x0000, 0x1000, 0, "RAM"},
 *       {0x1000, 0x100,  1, "GPIO"}
 *   });
 *   AXI_BUS<MyAXI> bus("bus", 2, 2, mmap);
 */

#include "axi_types.h"
#include "systemc/bus/axi/axi_config.h"
#include "axi_config_loader.h"
#include "axi_transaction.h"
#include "axi_memory_map.h"
#include "systemc/bus/axi/axi_signal_ports.h"
#include "axi_logger.h"
#include "axi_master_if.h"
#include "axi_slave_if.h"
#include "axi_memory_slave.h"
#include "profiler/axi_profiler.h"
#include "axi_exclusive_monitor.h"
#include "bus/axi_bus.h"

#endif // AXI_H
