# Bus — Timing Model

This document explains *how the AXI stack turns burst traffic into cycles*. The
config that parameterizes it is in [`parameters.md`](parameters.md).

The bus is modelled at the **signal level**: five AXI channels (AR / R / AW / W /
B) with real VALID/READY handshakes on `sc_signal` wires, clocked by the shared
`clk`. There is no latency *formula* — cycle count emerges from how the
handshakes serialize under finite outstanding budgets, arbitration, and memory
backpressure. (`ModelingMode::LatencyOnly` exists as an alternative fidelity but
the shipped DSE path runs `Full`.)

## Handshake semantics

A beat transfers on the clock edge where both VALID and READY are high. The
stack keeps two channel styles, chosen to match how real fabrics behave:

- **AR / AW / B — one-cycle-pulse handshakes.** The interconnect drives the
  channel VALID for one cycle and pulses the corresponding READY, assuming the
  upstream master and downstream slave hold their READY tied high on these
  channels (the R2 Master / R3 Slave contract). Address and write-response
  channels carry one transfer, so a pulse is sufficient and adds no throughput
  penalty.
- **R / W — combinational passthrough.** VALID flows downstream and READY flows
  upstream with no clocked stage in between, so a data beat crosses the fabric in
  the same cycle it is offered. Crucially, VALID depends only on the upstream
  VALID and READY only on the downstream READY — there is **no combinational
  VALID←READY path** (AXI A3.3.1) — so beats stream back-to-back at **1
  beat/cycle** and the interconnect adds no per-beat cycle, exactly as a real
  register-sliced fabric does (skid buffers preserve full throughput).

### The W-channel rewrite (1 beat/cycle)

The write-data channel was recently rewritten. The old clocked `w_thread`
forwarded **1 beat per 2 cycles** — it forced WREADY low every other cycle —
which silently halved effective write bandwidth to width/2 per cycle (measured
4 B/cyc at 64 b, 8 B/cyc at 128 b). No real AXI fabric ships that. The rewrite
(commit `32e77ac`) reimplements W as the same combinational-passthrough idiom the
R channel already used: `w_forward` pairs each slave with the master at the front
of its per-slave `w_order_` queue and passes WREADY upstream; `w_advance` retires
a burst on the WLAST handshake. See the header comment in
`src/systemc/bus/axi/bus/axi_bus.h` and the commit message. The change was
applied uniformly to all accelerators with results kept regardless of direction.

## Address decode and the 4 KiB rule

Before any arbitration, `Bus::decode` (in `axi_bus.h`) classifies each requesting
master's AR/AW address against the compile-time `slave_ranges_`:

- **Boundary4K** — the burst (`len+1` beats × beat bytes) crosses a 4 KiB page.
- **Unmapped** — the start address hits no slave range.
- **RangeOverflow** — the address is in a slave's range but the burst runs past
  its end.
- **Ok** — otherwise, with the matched slave index.

The DMA's own `DmaModel::calculate_bursts` already splits every transfer at the
4 KiB boundary and at `max_burst_bytes` (and clamps the beat size to the address
alignment), so well-formed DMA traffic decodes `Ok`; the decode path is the
fabric's guard against a malformed burst reaching a slave.

## Per-slave arbitration

Each slave has its own AR arbiter and AW arbiter (`axi/bus/axi_arbiter.h`), all
sharing one policy but independent state. Every cycle, `ar_thread` / `aw_thread`
decode all requesting masters, build a per-slave requester bitset, and call
`grant()`; a burst targeting a different slave proceeds in parallel — a genuine
non-blocking N×M crossbar, not a shared bus. Five policies are supported:

| Policy | Grant rule |
|--------|-----------|
| `RoundRobin` | equal-share rotation through requesters |
| `WeightedRR` | per-master credits; drain on consecutive grants, replenish on rotation |
| `FixedPriority` | lowest `priority` value wins (can starve lower tiers) |
| `PriorityRR` | highest-priority tier first, round-robin within it |
| `QoSAware` | highest live QOS first (per-transaction), round-robin within a QOS level |

A master whose request is not granted this cycle has its wait recorded in the
`ContentionProfile` (`axi/bus/axi_contention_profile.h`), which also tracks
per-slave channel-busy cycles, decode violations, and a cross-slave concurrency
histogram — the direct measure of crossbar parallelism.

## W follows AW order (no interleave)

At AW-grant time the granted master is pushed onto that slave's `w_order_` queue.
`w_forward` serves a slave only from the master at the front of its `w_order_`,
so write data arrives in exactly AW-acceptance order and **AXI4 cross-master W
interleave is impossible by construction** — no interleave-tag bookkeeping is
needed. R and B responses are demultiplexed the same way, through per-slave FIFOs
of master indices (`read_order_`, `write_order_`) populated at AR/AW forwarding
time. A per-master "already forwarded this cycle" guard keeps two slaves from
colliding on one master's response wires (the second waits a cycle).

## DECERR synthesis

Unmapped, boundary-crossing, and range-overflow bursts never reach a slave; the
bus synthesizes their error responses internally. A bad AR is accepted with a
one-cycle ARREADY pulse and its beats are queued in `err_reads_`; `r_forward`
drives synthesized R beats with `RRESP = DECERR` (0x3), RLAST on the last beat.
A bad AW marks the master's W burst with a `NumS` sentinel; `w_forward` swallows
its W beats at full rate, and on WLAST `w_advance` enqueues a synthesized
`BRESP = DECERR` that a per-master B-drain thread emits. The master's ID is
preserved so the synthesized response carries the right tag.

## The DMA in-flight window (livelock fix)

The DmaEngine issues bursts in a sliding window and drains the head before
issuing past the window (`execute_read_bursts` / `execute_write_bursts`, and the
persistent `pipe_*` variants for the tiled path). The window size is the DMA IP's
own resource — `fifo_bytes / burst_bytes`, optionally capped by an explicit AXI
issuing capability — **hard-capped by the AXI master's outstanding-transaction
budget**:

```
window = min( fifo_bytes / max_burst_bytes  [· max_issuing],
              caps().max_outstanding_{reads,writes} )
```

The cap is load-bearing. The master's `TicketPool::acquire()` *blocks* when full,
and it is called inside `issue`. If the window were wider than the pool, the pipe
would try to issue an (N+1)-th burst before draining one: `acquire()` blocks, the
pipe never drains to release a ticket, and the DMA deadlocks. This bit whenever
`fifo_bytes / burst_bytes > max_outstanding` — e.g. a narrow bus halves
`burst_bytes`, doubling the FIFO window past the 4-deep write pool. Physically a
DMA cannot have more transactions in flight than its AXI interface tracks, so the
cap (`read_window()` / `write_window()` in `dma_engine.h`) is the true bound.

Head-drain is order-safe because all DMA issues use one AXI ID (`qos=0`), and the
same-ID AXI rule guarantees responses return in issue order — so the oldest
in-flight ticket is always the first to complete, keeping the GB copy order
aligned with burst-issue order.

## What it models — and what it does not

Models: the five-channel signal-level handshake, 1 beat/cycle R/W streaming,
N×M non-blocking crossbar arbitration under five policies, finite outstanding
budgets and their backpressure, 4 KiB / range / unmapped decode with in-fabric
DECERR, W-follows-AW ordering, and (through the slave's `MemPort`) real
memory-side timing rather than a latency constant.

Does not model: write-data interleaving (structurally excluded), out-of-order
same-ID completion (same-ID stays in order by contract), or multi-slave data
buses on the shipped DSE path — `flexnpusim_system` instances `NumS = 1` (a single
DRAM slave) with `NumM` = read+write channels, so cross-slave parallelism is
exercised by the profiler and tests but not the default sweep.
