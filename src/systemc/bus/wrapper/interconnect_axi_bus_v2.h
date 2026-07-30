/**
 * @file interconnect_axi_bus_v2.h
 * @brief R5 interconnect wrapper — thin SC_MODULE over transport::axi::Bus.
 *
 * Legacy InterconnectAxiBus wraps the external AXI_BUS. V2 wraps the in-tree
 * R4 Bus<Spec,NumM,NumS>, inheriting its 5-policy arbiter, DECERR synthesis,
 * 4 KiB boundary + range check, and per-master skid-buffer R/B handshake.
 *
 * NumM / NumS are compile-time (R4 Bus requirement). For the data bus in
 * flexnpusim_system, NumS=1 (single DRAM slave) and NumM ranges 1..8 across
 * DMAC configurations; callers dispatch on NumM via a switch. A future
 * system-composition config v2 (docs/reference/system/topology.md) will own assembly
 * directly without this wrapper.
 *
 * API mirrors the legacy wrapper's bind_master / bind_slave so existing
 * wiring code ports with minimal edits.
 */

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <systemc.h>
#include <vector>

#include "systemc/bus/axi/axi_signal_ports.h"
#include "systemc/bus/axi/bus/axi_arbiter.h"
#include "systemc/bus/axi/bus/axi_bus.h"

namespace flexnpu_sim {

template <class Spec, unsigned NumM, unsigned NumS = 1>
class InterconnectAxiBusV2 : public sc_core::sc_module {
public:
    using BusT = transport::axi::Bus<Spec, NumM, NumS>;

    sc_core::sc_in<bool> clk{"clk"};

    InterconnectAxiBusV2(
        sc_core::sc_module_name name,
        transport::axi::ArbitPolicy policy,
        const std::vector<transport::axi::MasterArbitCfg>& master_cfgs,
        const std::array<typename BusT::SlaveRange, NumS>& slave_ranges)
        : sc_core::sc_module(name),
          bus_("bus", policy, master_cfgs, slave_ranges)
    {
        bus_.clk(clk);
    }

    // Port-to-signal binding — callers materialise their own AXI_SIGNALS
    // bundle per master / slave and pass it here, matching the legacy shape.
    void bind_master(unsigned master_id, AXI_SIGNALS<Spec>& signals) {
        if (master_id >= NumM) {
            SC_REPORT_ERROR("InterconnectAxiBusV2",
                ("Master ID " + std::to_string(master_id) + " out of range").c_str());
            return;
        }
        bind_port_signal(&bus_.m_ports[master_id], signals);
    }

    void bind_slave(unsigned slave_id, AXI_SIGNALS<Spec>& signals) {
        if (slave_id >= NumS) {
            SC_REPORT_ERROR("InterconnectAxiBusV2",
                ("Slave ID " + std::to_string(slave_id) + " out of range").c_str());
            return;
        }
        bind_port_signal(&bus_.s_ports[slave_id], signals);
    }

    BusT&       bus()       { return bus_; }
    const BusT& bus() const { return bus_; }

private:
    BusT bus_;
};

}  // namespace flexnpu_sim
