# Bus

The bus subsystem is the accelerator's path to DRAM: a signal-level AXI stack
(master → N×M interconnect → slave) plus the **DmaEngine** that turns a logical
transfer into protocol-legal bursts and drives them across it. Everything the
NPU reads or writes to memory — operands, weights, output feature maps, the DNN
image itself — crosses this subsystem.

- Source: `src/systemc/bus/` (the AXI stack) and `src/systemc/dma/` (the DMA
  engine + its behavioral model and profiles)
- Drives: the [memory](../memory/) tier (the slave's timing port hands each
  accepted transaction to a DRAM timing model)
- Consumed by: the [NPU](../npu/) controller (one DmaEngine per rDMAC/wDMAC
  channel) and `flexnpusim_system`, which assembles the masters, the interconnect,
  and the memory slave

## Why it exists

A cycle-level DSE run needs realistic memory transport: burst splitting at the
AXI 4 KiB page rule, finite outstanding-transaction budgets, arbitration when
several DMA channels contend for one DRAM slave, and the DMA↔memory overlap that
hides latency. Modelling that as a signal-level AXI fabric — real VALID/READY
handshakes on five channels, not a latency formula — reproduces the contention
and backpressure the hardware has, and lets the same code run at RTL parity
(`ModelingMode::Full`) or as an analytic latency model.

## The AXI stack

```
   NpuController                                  DRAM timing tier
        │  start/size/addr                              ▲
        ▼                                               │ submit / tick /
  ┌───────────┐  BurstReq   ┌──────────────┐            │ pop_completed
  │ DmaEngine │────────────▶│ AxiTransport │            │
  │ (per      │  issue_read │  (adapter to │      ┌───────────┐
  │  channel) │  issue_write│   R2 Master) │      │  Slave    │ MemPort
  └───────────┘◀────────────└──────┬───────┘      │  (R3)     │
   window cap                       │ 5 channels   └─────▲─────┘
   = min(FIFO,                      ▼ AR R AW W B        │ s_ports[0]
     AXI tickets)            ┌──────────────┐            │
                            │  Master (R2) │  m_ports    │
                            └──────┬───────┘             │
                                   ▼                     │
                            ┌──────────────────────────────────┐
                            │  Bus<Spec,NumM,NumS>  (R4 N×M)    │
                            │  decode · per-slave arbiter ·     │
                            │  R/W passthrough · DECERR synth   │
                            └──────────────────────────────────┘
```

- **DmaEngine** (`dma/dma_engine.h`) — command-driven SC_MODULE. On a
  start pulse it asks its `DmaModel` for a `BurstPlan` (4 KiB + `max_burst_bytes`
  split) and issues each burst through a runtime-polymorphic
  `MemoryMappedTransport`, keeping a sliding in-flight window.
- **AxiTransport** (`axi/axi_transport.h`) — adapts the abstract `BurstReq` /
  `Ticket` interface onto the ticket-based R2 master, packing bytes into beats.
- **Master** (`axi/master/axi_master.h`) — five independent SC_THREADs (AR / R /
  AW / W / B) and a `TicketPool` bounding outstanding reads and writes.
- **Bus** (`axi/bus/axi_bus.h`) — the N×M crossbar: address decode with the
  4 KiB rule and per-slave range check, per-slave AR/AW arbiters, combinational
  R/W data passthrough, and in-fabric DECERR synthesis for unmapped or
  boundary-crossing bursts. Wrapped for assembly by
  `wrapper/interconnect_axi_bus_v2.h`.
- **Slave** (`axi/slave/axi_slave.h`) — five SC_THREADs plus a per-cycle memory
  tick; a `MemoryBackend` holds bytes, a `MemPort` supplies real DRAM timing
  (no latency formula).

`Spec<Protocol, ID_W, ADDR_W, DATA_W, USER_W>` (`axi/spec/axi_spec.h`) is the
single source of truth for signal widths and AXI3/AXI4 feature flags; every
module above is templated on it.

## Documents

| File | Contents |
|------|----------|
| [`modeling.md`](modeling.md) | the timing model: the VALID/READY handshakes, R/W combinational passthrough at 1 beat/cycle (the recent W rewrite), the AR/AW/B one-cycle pulses, per-slave arbitration, W-follows-AW ordering, DECERR synthesis, and the DMA in-flight window cap that fixed the narrow-bus livelock |
| [`parameters.md`](parameters.md) | the config the subsystem consumes — `axi.*`, `hw.dma.*`, and `topology.*` — as tables with JSON key paths |
| [`protocols.md`](protocols.md) | protocol overview: the five AXI channels and the subset the model exercises |

## One-paragraph summary

The DmaEngine splits each transfer into spec-legal bursts and streams them
through an AXI master whose ticket pool caps outstanding transactions; the N×M
interconnect decodes each burst's address, arbitrates one master per slave per
cycle, forwards read and write data as a combinational passthrough (1 beat/cycle,
no fabric-added bubble), keeps write data in AW-acceptance order per slave, and
synthesizes DECERR for anything unmapped or crossing a 4 KiB page — while the
slave hands every accepted transaction to a DRAM timing model and streams the
response back in arrival order. The DMA's in-flight window is capped by the
master's outstanding-ticket budget so a narrow bus (which doubles the FIFO burst
count) cannot try to issue past the pool and deadlock.
