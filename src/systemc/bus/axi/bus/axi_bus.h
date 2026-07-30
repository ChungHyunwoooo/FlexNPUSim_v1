#ifndef FLEXNPU_TRANSPORT_AXI_BUS_AXI_BUS_H
#define FLEXNPU_TRANSPORT_AXI_BUS_AXI_BUS_H

// N×M AXI interconnect (R4').
//
// Parallels Xilinx AXI Interconnect / ARM NIC-400:
//   * One address decoder with 4 KiB boundary rule and per-slave range
//     check. Unmapped / violating bursts are converted into DECERR
//     responses synthesized inside the bus — offending requests never
//     reach a slave.
//   * Per-slave AR/AW arbiters (reusing the 5-policy axi::Arbiter) pick
//     one granted master per slave per cycle. Bursts targeting
//     different slaves proceed in parallel — genuine non-blocking
//     N×M crossbar.
//   * W channel follows AW order per slave via a per-slave service queue
//     (w_order_, populated at AW forwarding time) — AXI4 W interleaving
//     is impossible by construction.
//   * R and B responses are demultiplexed via per-slave FIFOs of master
//     indices populated at AR/AW forwarding time.
//
// Handshake timing (this revision):
//   * AR/AW/B VALID pulses are 1 cycle. Assumes upstream master and
//     downstream slave keep READY tied high (our R2 Master / R3 Slave
//     contract) on those channels.
//   * R and W data channels are combinational passthrough: VALID flows
//     downstream, READY flows upstream (no combinational VALID<-READY
//     path, AXI A3.3.1), so data beats stream back-to-back at 1/cycle —
//     the interconnect adds no per-beat cycle, matching real fabrics
//     (register slices keep full throughput via skid buffers). The old
//     clocked W loop forwarded 1 beat per 2 cycles (WREADY forced low
//     every other cycle), silently halving write bandwidth.
//
// Cross-slave parallelism:
//   * AR / AW / W / R / B threads loop over all slaves per cycle and
//     forward independently. A per-master "already forwarded this
//     cycle" guard prevents two slaves from colliding on the same
//     master's response wires; the second slave waits one cycle.

#include <systemc.h>
#include <sysc/kernel/sc_spawn.h>
#include <sysc/kernel/sc_dynamic_processes.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

#include "systemc/bus/axi/axi_signal_ports.h"
#include "systemc/bus/axi/bus/axi_arbiter.h"
#include "systemc/bus/axi/bus/axi_contention_profile.h"
#include "systemc/bus/axi/spec/axi_spec.h"

namespace flexnpu_sim::transport::axi {

template <class Spec, unsigned NumM, unsigned NumS>
class Bus : public sc_core::sc_module {
public:
    static_assert(NumM >= 1, "Bus requires at least one master port");
    static_assert(NumS >= 1, "Bus requires at least one slave port");

    using id_t    = typename Spec::id_t;
    using addr_t  = typename Spec::addr_t;
    using data_t  = typename Spec::data_t;
    using strb_t  = typename Spec::strb_t;
    using len_t   = typename Spec::len_t;
    using resp_t  = typename Spec::resp_t;

    struct SlaveRange {
        uint64_t base = 0;
        uint64_t size = 0;
    };

    sc_core::sc_in<bool> clk{"clk"};
    SLAVE_PORTS<Spec>    m_ports[NumM];
    MASTER_PORTS<Spec>   s_ports[NumS];

    Bus(sc_core::sc_module_name name,
        ArbitPolicy policy,
        const std::vector<MasterArbitCfg>& master_cfgs,
        const std::array<SlaveRange, NumS>& slave_ranges)
        : sc_core::sc_module(name),
          slave_ranges_(slave_ranges),
          profile_(NumM, NumS)
    {
        if (master_cfgs.size() != NumM) {
            throw std::invalid_argument("Bus: master_cfgs.size() must equal NumM");
        }
        ar_arbs_.reserve(NumS);
        aw_arbs_.reserve(NumS);
        for (unsigned s = 0; s < NumS; ++s) {
            ar_arbs_.emplace_back(policy, master_cfgs);
            aw_arbs_.emplace_back(policy, master_cfgs);
        }
        SC_HAS_PROCESS(Bus);
        SC_THREAD(ar_thread);
        SC_THREAD(aw_thread);
        SC_THREAD(concurrency_sampler);
        // W channel: combinational passthrough + clocked advance, the exact
        // r_forward/r_advance idiom (see R channel comment below) applied to
        // the write-data direction. w_forward drives each serving slave's W
        // inputs from the master at the front of its w_order_ queue and that
        // master's WREADY from the slave's WREADY; w_advance retires bursts
        // on the WLAST handshake. Beats stream at 1/cycle.
        SC_METHOD(w_forward);
        sensitive << w_route_changed_;
        for (unsigned m = 0; m < NumM; ++m) {
            sensitive << m_ports[m].WVALID << m_ports[m].WDATA
                      << m_ports[m].WSTRB  << m_ports[m].WLAST;
            if constexpr (Spec::HAS_WID) sensitive << m_ports[m].WID;
        }
        for (unsigned s = 0; s < NumS; ++s)
            sensitive << s_ports[s].WREADY;

        SC_METHOD(w_advance);
        sensitive << clk.pos();
        dont_initialize();
        // Per-master R and B drain threads give spec-compliant
        // `do { wait(posedge); } while (!READY)` handshakes without
        // serializing cross-master forwards. Each master's drain
        // thread owns its own m_ports[m].{R,B}VALID wires, so
        // writes do not collide across masters. Slave-side RREADY /
        // BREADY is pulsed from the owning master's thread only while
        // read_order_[s] / write_order_[s] front index equals its
        // master id — mutual exclusion is therefore queue-based, not
        // lock-based.
        // R channel: combinational passthrough (a real RTL interconnect). The
        // old per-master r_drain thread + per-slave RREADY pulser forwarded R at
        // 1 beat / 3 cycles (scan + master-accept + slave-ACK round-trip),
        // throttling read data below the bus rate. r_forward drives each master's
        // R outputs from its serving slave and each served slave's RREADY from
        // its consuming master — VALID depends only on upstream VALID, READY only
        // on downstream READY, so there is no combinational VALID<-READY path
        // (AXI A3.3.1) and beats stream at 1/cycle. r_advance pops the routing
        // queue on the RLAST handshake.
        SC_METHOD(r_forward);
        sensitive << route_changed_;
        for (unsigned s = 0; s < NumS; ++s)
            sensitive << s_ports[s].RVALID << s_ports[s].RDATA
                      << s_ports[s].RLAST  << s_ports[s].RID << s_ports[s].RRESP;
        for (unsigned m = 0; m < NumM; ++m)
            sensitive << m_ports[m].RREADY;

        SC_METHOD(r_advance);
        sensitive << clk.pos();
        dont_initialize();

        for (unsigned m = 0; m < NumM; ++m) {
            sc_core::sc_spawn(
                sc_core::sc_bind(&Bus::b_drain_master, this, m),
                ("b_drain_" + std::to_string(m)).c_str());
        }
        for (unsigned s = 0; s < NumS; ++s) {
            sc_core::sc_spawn(
                sc_core::sc_bind(&Bus::slave_bready_pulser, this, s),
                ("b_ack_" + std::to_string(s)).c_str());
        }
    }

    const ContentionProfile& contention() const { return profile_; }

private:
    enum class DecodeResult : uint8_t {
        Ok            = 0,
        Unmapped      = 1,
        Boundary4K    = 2,
        RangeOverflow = 3,
    };
    struct DecodeOut {
        DecodeResult res       = DecodeResult::Unmapped;
        int          slave_idx = -1;
    };

    DecodeOut decode(uint64_t addr, unsigned len_plus_1, unsigned beat_size_bytes) const {
        DecodeOut out;
        const uint64_t total      = static_cast<uint64_t>(len_plus_1) * beat_size_bytes;
        const uint64_t end        = addr + total;
        const uint64_t page_start = addr & ~uint64_t(0xFFFu);
        if (total > 0 && end > page_start + 0x1000u) {
            out.res = DecodeResult::Boundary4K;
            return out;
        }
        for (unsigned s = 0; s < NumS; ++s) {
            const uint64_t b = slave_ranges_[s].base;
            const uint64_t e = b + slave_ranges_[s].size;
            if (addr >= b && addr < e) {
                if (end <= e) {
                    out.res       = DecodeResult::Ok;
                    out.slave_idx = static_cast<int>(s);
                } else {
                    out.res = DecodeResult::RangeOverflow;
                }
                return out;
            }
        }
        out.res = DecodeResult::Unmapped;
        return out;
    }

    // ---- Per-slave routing queues --------------------------------------
    std::array<std::deque<unsigned>, NumS> read_order_;
    std::array<std::deque<unsigned>, NumS> write_order_;      // for B demux
    std::array<std::deque<unsigned>, NumS> w_order_;          // W service order
                                                               // per slave (= AW
                                                               // order); enforces
                                                               // the AXI4 no-W-
                                                               // interleave rule
    std::array<std::deque<unsigned>, NumM> master_w_targets_; // pending AW slaves
                                                               // per master
    std::array<std::deque<id_t>, NumM>     master_err_awid_;  // AWID saved when
                                                               // target == NumS,
                                                               // consumed on WLAST
                                                               // to build BRESP

    // ACK coordination for the WRITE (B) channel: the per-master b_drain thread
    // requests a 1-cycle BREADY pulse toward slave s; the per-slave pulser owns
    // the port write so SystemC sees a single driver per sc_signal.
    std::array<sc_core::sc_event, NumS> s_bready_req_;
    std::array<sc_core::sc_event, NumS> s_bready_done_;

    // R channel routing-state change (a read_order_/err_reads_ pop) — retriggers
    // the combinational r_forward so it re-evaluates the serving relationships.
    sc_core::sc_event route_changed_;

    // W channel routing-state change (w_order_/master_w_targets_ push or pop) —
    // retriggers the combinational w_forward. Needed because queue mutations
    // are not signal changes (the DECERR sink in particular has no slave-side
    // WREADY edge to wake the method).
    sc_core::sc_event w_route_changed_;

    // ---- DECERR synthesis ----------------------------------------------
    struct ErrorRead  {
        unsigned master_idx;
        id_t     id;
        unsigned remaining_beats;
    };
    struct ErrorWrite { unsigned master_idx; id_t id; };
    std::deque<ErrorRead>  err_reads_;
    std::deque<ErrorWrite> err_writes_pending_b_;  // awaiting B synthesis

    std::array<SlaveRange, NumS> slave_ranges_;
    std::vector<Arbiter>         ar_arbs_;
    std::vector<Arbiter>         aw_arbs_;
    ContentionProfile            profile_;

    // -------------------------------------------------------------------
    // AR thread
    // -------------------------------------------------------------------
    void ar_thread() {
        for (unsigned i = 0; i < NumM; ++i) m_ports[i].ARREADY.write(false);
        for (unsigned s = 0; s < NumS; ++s) s_ports[s].ARVALID.write(false);

        while (true) {
            sc_core::wait(clk.posedge_event());

            // Decode every requesting master.
            std::array<int,          NumM> tgt{};   tgt.fill(-1);
            std::array<DecodeResult, NumM> dres{};
            dres.fill(DecodeResult::Unmapped);
            for (unsigned i = 0; i < NumM; ++i) {
                if (!m_ports[i].ARVALID.read()) continue;
                const uint64_t a   = m_ports[i].ARADDR.read().to_uint64();
                const unsigned lp1 = m_ports[i].ARLEN.read().to_uint() + 1u;
                const unsigned sb  = 1u << m_ports[i].ARSIZE.read().to_uint();
                const auto d = decode(a, lp1, sb);
                dres[i] = d.res;
                tgt[i]  = d.slave_idx;
            }

            // Per-slave arbitration.
            std::array<int, NumS> grant{}; grant.fill(-1);
            for (unsigned s = 0; s < NumS; ++s) {
                std::vector<bool> req(NumM, false);
                for (unsigned i = 0; i < NumM; ++i) {
                    if (tgt[i] == static_cast<int>(s)) req[i] = true;
                }
                grant[s] = ar_arbs_[s].grant(req);
            }

            // Wait-cycle accounting.
            for (unsigned i = 0; i < NumM; ++i) {
                if (!m_ports[i].ARVALID.read()) continue;
                if (tgt[i] < 0) continue;
                if (grant[tgt[i]] != static_cast<int>(i)) {
                    profile_.record_ar_wait(i, static_cast<unsigned>(tgt[i]));
                }
            }

            // DECERR path: accept AR with one-cycle ARREADY pulse, queue
            // synthesized R beats.
            std::array<bool, NumM> ack_master{}; ack_master.fill(false);
            for (unsigned i = 0; i < NumM; ++i) {
                if (!m_ports[i].ARVALID.read()) continue;
                if (dres[i] == DecodeResult::Ok) continue;
                switch (dres[i]) {
                    case DecodeResult::Unmapped:      profile_.record_decode(true, 0); break;
                    case DecodeResult::Boundary4K:    profile_.record_decode(true, 1); break;
                    case DecodeResult::RangeOverflow: profile_.record_decode(true, 2); break;
                    default: break;
                }
                const unsigned lp1 = m_ports[i].ARLEN.read().to_uint() + 1u;
                err_reads_.push_back(ErrorRead{i, m_ports[i].ARID.read(), lp1});
                ack_master[i] = true;
            }

            // Forward granted AR to each slave.
            for (unsigned s = 0; s < NumS; ++s) {
                if (grant[s] < 0) continue;
                const unsigned g = static_cast<unsigned>(grant[s]);
                s_ports[s].ARID    .write(m_ports[g].ARID.read());
                s_ports[s].ARADDR  .write(m_ports[g].ARADDR.read());
                s_ports[s].ARLEN   .write(m_ports[g].ARLEN.read());
                s_ports[s].ARSIZE  .write(m_ports[g].ARSIZE.read());
                s_ports[s].ARBURST .write(m_ports[g].ARBURST.read());
                s_ports[s].ARLOCK  .write(m_ports[g].ARLOCK.read());
                s_ports[s].ARCACHE .write(m_ports[g].ARCACHE.read());
                s_ports[s].ARPROT  .write(m_ports[g].ARPROT.read());
                if constexpr (Spec::HAS_QOS)    s_ports[s].ARQOS   .write(m_ports[g].ARQOS.read());
                if constexpr (Spec::HAS_REGION) s_ports[s].ARREGION.write(m_ports[g].ARREGION.read());
                s_ports[s].ARVALID.write(true);
                ack_master[g] = true;
                read_order_[s].push_back(g);
                profile_.record_ar_grant(g, s);
                profile_.record_slave_channel(s, 0);
            }

            // Pulse master-side ARREADYs for masters whose AR we accepted.
            for (unsigned i = 0; i < NumM; ++i) {
                if (ack_master[i]) m_ports[i].ARREADY.write(true);
            }

            sc_core::wait(clk.posedge_event());

            for (unsigned i = 0; i < NumM; ++i) m_ports[i].ARREADY.write(false);
            for (unsigned s = 0; s < NumS; ++s) s_ports[s].ARVALID.write(false);
        }
    }

    // -------------------------------------------------------------------
    // AW thread
    // -------------------------------------------------------------------
    void aw_thread() {
        for (unsigned i = 0; i < NumM; ++i) m_ports[i].AWREADY.write(false);
        for (unsigned s = 0; s < NumS; ++s) s_ports[s].AWVALID.write(false);

        while (true) {
            sc_core::wait(clk.posedge_event());

            std::array<int,          NumM> tgt{};  tgt.fill(-1);
            std::array<DecodeResult, NumM> dres{}; dres.fill(DecodeResult::Unmapped);
            for (unsigned i = 0; i < NumM; ++i) {
                if (!m_ports[i].AWVALID.read()) continue;
                const uint64_t a   = m_ports[i].AWADDR.read().to_uint64();
                const unsigned lp1 = m_ports[i].AWLEN.read().to_uint() + 1u;
                const unsigned sb  = 1u << m_ports[i].AWSIZE.read().to_uint();
                const auto d = decode(a, lp1, sb);
                dres[i] = d.res;
                tgt[i]  = d.slave_idx;
            }

            std::array<int, NumS> grant{}; grant.fill(-1);
            for (unsigned s = 0; s < NumS; ++s) {
                std::vector<bool> req(NumM, false);
                for (unsigned i = 0; i < NumM; ++i) {
                    if (tgt[i] == static_cast<int>(s)) req[i] = true;
                }
                grant[s] = aw_arbs_[s].grant(req);
            }
            for (unsigned i = 0; i < NumM; ++i) {
                if (!m_ports[i].AWVALID.read()) continue;
                if (tgt[i] < 0) continue;
                if (grant[tgt[i]] != static_cast<int>(i)) {
                    profile_.record_aw_wait(i, static_cast<unsigned>(tgt[i]));
                }
            }

            std::array<bool, NumM> ack_master{}; ack_master.fill(false);
            for (unsigned i = 0; i < NumM; ++i) {
                if (!m_ports[i].AWVALID.read()) continue;
                if (dres[i] == DecodeResult::Ok) continue;
                switch (dres[i]) {
                    case DecodeResult::Unmapped:      profile_.record_decode(false, 0); break;
                    case DecodeResult::Boundary4K:    profile_.record_decode(false, 1); break;
                    case DecodeResult::RangeOverflow: profile_.record_decode(false, 2); break;
                    default: break;
                }
                // Mark this master's upcoming W burst as DECERR-bound; the
                // actual BRESP=DECERR is emitted by b_thread once
                // w_thread finishes draining the W beats (see
                // master_w_targets_ == NumS branch below). AWID is
                // saved alongside so the synthesized B carries the
                // right tag.
                master_w_targets_[i].push_back(NumS);   // sentinel
                master_err_awid_[i].push_back(m_ports[i].AWID.read());
                ack_master[i] = true;
                w_route_changed_.notify(sc_core::SC_ZERO_TIME);
            }

            for (unsigned s = 0; s < NumS; ++s) {
                if (grant[s] < 0) continue;
                const unsigned g = static_cast<unsigned>(grant[s]);
                s_ports[s].AWID    .write(m_ports[g].AWID.read());
                s_ports[s].AWADDR  .write(m_ports[g].AWADDR.read());
                s_ports[s].AWLEN   .write(m_ports[g].AWLEN.read());
                s_ports[s].AWSIZE  .write(m_ports[g].AWSIZE.read());
                s_ports[s].AWBURST .write(m_ports[g].AWBURST.read());
                s_ports[s].AWLOCK  .write(m_ports[g].AWLOCK.read());
                s_ports[s].AWCACHE .write(m_ports[g].AWCACHE.read());
                s_ports[s].AWPROT  .write(m_ports[g].AWPROT.read());
                if constexpr (Spec::HAS_QOS)    s_ports[s].AWQOS   .write(m_ports[g].AWQOS.read());
                if constexpr (Spec::HAS_REGION) s_ports[s].AWREGION.write(m_ports[g].AWREGION.read());
                s_ports[s].AWVALID.write(true);
                ack_master[g] = true;
                master_w_targets_[g].push_back(s);
                write_order_[s].push_back(g);
                w_order_[s].push_back(g);
                w_route_changed_.notify(sc_core::SC_ZERO_TIME);
                profile_.record_aw_grant(g, s);
                profile_.record_slave_channel(s, 1);
            }

            for (unsigned i = 0; i < NumM; ++i) {
                if (ack_master[i]) m_ports[i].AWREADY.write(true);
            }

            sc_core::wait(clk.posedge_event());

            for (unsigned i = 0; i < NumM; ++i) m_ports[i].AWREADY.write(false);
            for (unsigned s = 0; s < NumS; ++s) s_ports[s].AWVALID.write(false);
        }
    }

    // -------------------------------------------------------------------
    // W channel — combinational passthrough interconnect (mirror of the R
    // channel below, write-data direction).
    //
    // w_forward (combinational): each slave is served by exactly the master
    // at the front of its w_order_ queue (= AW acceptance order), provided
    // that master's own front target agrees (it may first have to drain a
    // DECERR burst). W data/VALID pass downstream, WREADY passes upstream,
    // so there is no combinational VALID<-READY path (AXI A3.3.1) and write
    // beats stream at 1 beat per cycle. Queue-based pairing makes AXI4
    // cross-master W interleave impossible and keeps arbitration
    // deterministic. Slave backpressure is honored: WREADY low at the slave
    // reaches the master combinationally, which then holds its beat.
    //
    // w_advance (clocked): retire a burst on the WLAST beat handshake —
    // pop w_order_/master_w_targets_, or enqueue the synthesized DECERR B
    // response for sink bursts.
    // -------------------------------------------------------------------
    void w_forward() {
        std::array<bool, NumM> mready{}; mready.fill(false);

        for (unsigned s = 0; s < NumS; ++s) {
            int serving = -1;
            if (!w_order_[s].empty()) {
                const unsigned m = w_order_[s].front();
                if (!master_w_targets_[m].empty() &&
                    master_w_targets_[m].front() == s) {
                    serving = static_cast<int>(m);
                }
            }
            if (serving >= 0) {
                const unsigned m = static_cast<unsigned>(serving);
                if constexpr (Spec::HAS_WID) {
                    s_ports[s].WID.write(m_ports[m].WID.read());
                }
                s_ports[s].WDATA .write(m_ports[m].WDATA.read());
                s_ports[s].WSTRB .write(m_ports[m].WSTRB.read());
                s_ports[s].WLAST .write(m_ports[m].WLAST.read());
                s_ports[s].WVALID.write(m_ports[m].WVALID.read());
                mready[m] = s_ports[s].WREADY.read();
            } else {
                s_ports[s].WVALID.write(false);
                s_ports[s].WLAST .write(false);
            }
        }

        // DECERR sink: the master's front burst decodes to no slave — swallow
        // its beats at full rate (READY held high; b_drain synthesizes BRESP).
        for (unsigned m = 0; m < NumM; ++m) {
            if (!master_w_targets_[m].empty() &&
                master_w_targets_[m].front() == NumS) {
                mready[m] = true;
            }
            m_ports[m].WREADY.write(mready[m]);
        }
    }

    void w_advance() {
        // Normal bursts: retire on the WLAST beat handshake at the slave.
        for (unsigned s = 0; s < NumS; ++s) {
            if (!s_ports[s].WVALID.read() || !s_ports[s].WREADY.read()) continue;
            profile_.record_slave_channel(s, 2);
            if (!s_ports[s].WLAST.read()) continue;
            if (!w_order_[s].empty()) {
                const unsigned m = w_order_[s].front();
                w_order_[s].pop_front();
                if (!master_w_targets_[m].empty()) master_w_targets_[m].pop_front();
            }
            w_route_changed_.notify(sc_core::SC_ZERO_TIME);
        }
        // DECERR sink bursts: retire on the master-side WLAST handshake.
        for (unsigned m = 0; m < NumM; ++m) {
            if (master_w_targets_[m].empty() ||
                master_w_targets_[m].front() != NumS) continue;
            if (!m_ports[m].WVALID.read() || !m_ports[m].WREADY.read() ||
                !m_ports[m].WLAST.read()) continue;
            master_w_targets_[m].pop_front();
            id_t bid = id_t(0);
            if (!master_err_awid_[m].empty()) {
                bid = master_err_awid_[m].front();
                master_err_awid_[m].pop_front();
            }
            err_writes_pending_b_.push_back(ErrorWrite{m, bid});
            w_route_changed_.notify(sc_core::SC_ZERO_TIME);
        }
    }

    // -------------------------------------------------------------------
    // R channel — combinational passthrough interconnect.
    //
    // r_forward (combinational): for each master m, drive its R outputs from
    // the slave whose read_order_ front is m (or a DECERR synthesized beat,
    // which takes priority); each slave being consumed this cycle sees RREADY
    // from its consuming master. VALID passes downstream (m.RVALID <- s.RVALID),
    // READY passes upstream (s.RREADY <- m.RREADY), so there is no combinational
    // VALID<-READY dependency (AXI A3.3.1) and read data streams at 1 beat per
    // cycle — the interconnect adds no per-beat cycle. Master backpressure is
    // honored: with the master's RREADY low the slave's RREADY is low, so the
    // slave holds its beat (no loss) and the master holds its VALID.
    //
    // r_advance (clocked): pop the routing queue / retire a DECERR read on the
    // RLAST beat's handshake (VALID && READY both high at the edge).
    // -------------------------------------------------------------------
    void r_forward() {
        std::array<int, NumS> slave_consumer{}; slave_consumer.fill(-1);

        for (unsigned m = 0; m < NumM; ++m) {
            // DECERR synthesized beat has priority (mirrors the legacy order).
            bool decerr = false;
            for (const auto& e : err_reads_) {
                if (e.master_idx != m) continue;
                m_ports[m].RID   .write(e.id);
                m_ports[m].RDATA .write(0);
                m_ports[m].RRESP .write(resp_t(0x3));   // AXI DECERR
                m_ports[m].RLAST .write(e.remaining_beats == 1);
                m_ports[m].RVALID.write(true);
                decerr = true;
                break;
            }
            if (decerr) continue;

            // Normal: the first slave whose read_order_ front is m.
            int ss = -1;
            for (unsigned s = 0; s < NumS; ++s) {
                if (!read_order_[s].empty() && read_order_[s].front() == m) {
                    ss = static_cast<int>(s);
                    break;
                }
            }
            if (ss >= 0) {
                const unsigned s = static_cast<unsigned>(ss);
                m_ports[m].RID   .write(s_ports[s].RID.read());
                m_ports[m].RDATA .write(s_ports[s].RDATA.read());
                m_ports[m].RRESP .write(s_ports[s].RRESP.read());
                m_ports[m].RLAST .write(s_ports[s].RLAST.read());
                m_ports[m].RVALID.write(s_ports[s].RVALID.read());
                if (s_ports[s].RVALID.read()) slave_consumer[s] = static_cast<int>(m);
            } else {
                m_ports[m].RVALID.write(false);
            }
        }

        // Upstream READY passthrough: a slave sees RREADY only while a master is
        // actively consuming its current beat.
        for (unsigned s = 0; s < NumS; ++s) {
            if (slave_consumer[s] >= 0)
                s_ports[s].RREADY.write(
                    m_ports[static_cast<unsigned>(slave_consumer[s])].RREADY.read());
            else
                s_ports[s].RREADY.write(false);
        }
    }

    void r_advance() {
        // Normal reads: pop the routing queue on the RLAST beat handshake.
        for (unsigned s = 0; s < NumS; ++s) {
            if (read_order_[s].empty()) continue;
            if (s_ports[s].RVALID.read() && s_ports[s].RREADY.read()) {
                profile_.record_slave_channel(s, 3);
                if (s_ports[s].RLAST.read()) {
                    read_order_[s].pop_front();
                    route_changed_.notify(sc_core::SC_ZERO_TIME);
                }
            }
        }
        // DECERR reads: retire the front err_read of each master on its accepted
        // beat (err has forward priority, RVALID always driven, so a beat
        // transfers whenever the master's RREADY is high).
        std::array<bool, NumM> m_done{}; m_done.fill(false);
        for (auto it = err_reads_.begin(); it != err_reads_.end(); ) {
            const unsigned m = it->master_idx;
            if (m_done[m]) { ++it; continue; }   // only the front err per master
            m_done[m] = true;
            if (m_ports[m].RREADY.read()) {
                --it->remaining_beats;
                route_changed_.notify(sc_core::SC_ZERO_TIME);  // rb changed -> re-forward
                if (it->remaining_beats == 0) { it = err_reads_.erase(it); continue; }
            }
            ++it;
        }
    }

    // -------------------------------------------------------------------
    // Concurrency sampler — each cycle, count how many slaves have any
    // channel carrying activity this edge. A slave is considered active
    // if any of AR/AW/W/R/B VALID wires are high on either side of it
    // (so DECERR-synthesized beats are excluded — they don't touch a
    // slave). The histogram is the direct measure of cross-slave
    // parallelism.
    void concurrency_sampler() {
        while (true) {
            sc_core::wait(clk.posedge_event());
            unsigned active = 0;
            for (unsigned s = 0; s < NumS; ++s) {
                const bool busy =
                    s_ports[s].ARVALID.read() || s_ports[s].AWVALID.read() ||
                    s_ports[s].WVALID .read() || s_ports[s].RVALID .read() ||
                    s_ports[s].BVALID .read();
                if (busy) ++active;
            }
            profile_.record_concurrency(active);
        }
    }

    // -------------------------------------------------------------------
    // B drain (per-master) — mirror of r_drain_master.
    // -------------------------------------------------------------------
    void b_drain_master(unsigned m) {
        m_ports[m].BVALID.write(false);
        while (true) {
            sc_core::wait(clk.posedge_event());

            bool handled = false;
            for (auto it = err_writes_pending_b_.begin();
                 it != err_writes_pending_b_.end(); ) {
                if (it->master_idx != m) { ++it; continue; }
                m_ports[m].BID   .write(it->id);
                m_ports[m].BRESP .write(resp_t(0x3));   // AXI DECERR
                m_ports[m].BVALID.write(true);

                do {
                    sc_core::wait(clk.posedge_event());
                } while (!m_ports[m].BREADY.read());

                m_ports[m].BVALID.write(false);
                err_writes_pending_b_.erase(it);
                handled = true;
                break;
            }
            if (handled) continue;

            for (unsigned s = 0; s < NumS; ++s) {
                if (!s_ports[s].BVALID.read()) continue;
                if (write_order_[s].empty()) continue;
                if (write_order_[s].front() != m) continue;

                m_ports[m].BID   .write(s_ports[s].BID.read());
                m_ports[m].BRESP .write(s_ports[s].BRESP.read());
                m_ports[m].BVALID.write(true);

                do {
                    sc_core::wait(clk.posedge_event());
                } while (!m_ports[m].BREADY.read());

                m_ports[m].BVALID.write(false);

                s_bready_req_[s].notify(sc_core::SC_ZERO_TIME);
                sc_core::wait(s_bready_done_[s]);

                profile_.record_slave_channel(s, 4);
                write_order_[s].pop_front();
                break;
            }
        }
    }

    void slave_bready_pulser(unsigned s) {
        s_ports[s].BREADY.write(false);
        while (true) {
            sc_core::wait(s_bready_req_[s]);
            s_ports[s].BREADY.write(true);
            sc_core::wait(clk.posedge_event());
            s_ports[s].BREADY.write(false);
            s_bready_done_[s].notify(sc_core::SC_ZERO_TIME);
        }
    }
};

}  // namespace flexnpu_sim::transport::axi

#endif  // FLEXNPU_TRANSPORT_AXI_BUS_AXI_BUS_H
