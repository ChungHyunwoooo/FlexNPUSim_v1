# Memory — Modeling

The memory subsystem separates **real hardware** from an **approximate model**
along one interface. Everything the AXI slave does — ready/valid handshakes,
per-transaction state, one-beat-per-cycle read-data streaming — is real. The one
place the simulator permits an approximation is the memory-controller *timing*,
and it is confined to the `DramTiming` model. That model's internals may
approximate, but its interface is the real memory-controller one: backpressure,
submit, tick, completion (`src/systemc/memory/dram/dram_timing.h`).

## The `MemPort` contract

The slave drives the model through four functions bound at construction
(`memory_axi_slave_v2.h::install_mem_port` → `Slave::set_mem_port`,
`axi_slave.h:83`). Left unset, the slave is a zero-latency memory: it accepts
always and completes at once (`axi_slave.h:152-158`).

| Function | Meaning | Contract |
|----------|---------|----------|
| `will_accept(addr, is_write)` | Can another transaction enter this cycle? | `false` deasserts ARREADY/AWREADY — real ready/valid backpressure. |
| `submit(token, addr, size, is_write)` | Hand over an accepted transaction. | `token` is opaque; the caller matches it against `pop_completed`. Precondition: `will_accept()` was true. |
| `tick()` | Advance one system cycle. | Called once per clock by the slave's `mem_tick` SC_METHOD. |
| `pop_completed(token)` | Drain one finished transaction. | Sets `token` and returns `true`, or `false` when none are ready. |

The slave ticks the model every cycle and moves every popped token into a
`completed_` set; a response channel then streams only once its token is in that
set (`axi_slave.h:161-167`).

## Transaction lifecycle

A read and a write enter the model at different points, because the AXI
transaction itself resolves at different points.

**Read — submitted at AR.** The `ar_thread` asserts ARREADY only while
`will_accept(addr, false)` is true *and* the read queue has space. On the
accepted AR it pre-fetches every beat from the byte store immediately, tags the
op with a fresh token, and `submit()`s it with `size = beats × stride`
(`axi_slave.h:170-193`). The data is already read; the token only gates *when*
the `r_thread` is allowed to stream it. The `r_thread` waits for the token to
complete, then drives the beats one per cycle over the R channel.

**Write — submitted at WLAST.** The `aw_thread` latches the AW and enqueues a
`WriteOp`; the `w_thread` pairs it with the incoming W beats and commits each
beat to the byte store as it lands. Only after WLAST does it `submit()` the whole
write for timing — first blocking until `will_accept(addr, true)`
(`axi_slave.h:214-241`). The `b_thread` waits for that token, then raises BVALID.
So a write's data reaches memory during the W burst, but its *timing* (and its B
response) is charged as one transaction at WLAST.

## Ordering guarantees

Responses leave in **arrival order**, per direction, enforced by the FIFO queues
(`rd_q_`, `aw_q_`/`b_q_`), not by the model. The `r_thread` and `b_thread` read
their queues in order and block on the head's token, so a later burst that the
model happens to finish first still waits its turn (`axi_slave.h:16-22`). The DMA
relies on this same-ID FIFO ordering for its head-drain. Note this means
`pop_completed` may return tokens out of order — the model is free to — but the
slave only *acts* on a token once it is at the head of its response queue.

## The bank-occupancy engine (`ideal` / `bandwidth` / `bank`)

The three internal tiers share `AnalyticalDram`
(`src/systemc/memory/dram/analytical/analytical_dram.h`), which supplies real
memory-controller behavior around an approximate per-access latency:

- **Concurrency = `banks` parallel servers.** `will_accept` is
  `in_flight < banks`. A transaction enters only while a server is free; beyond
  that it stalls at the AXI handshake. When the AXI slave's outstanding window
  exceeds `banks`, effective concurrency is `min(outstanding, banks)` — it
  *emerges* from occupancy, it is never computed.
- **Each accepted transaction finishes `access_latency()` cycles later.**
  `submit()` records `done = now + access_latency(...)`; `pop_completed()`
  returns the first pending whose `done ≤ now`.
- **The shared read-data bus is not modelled here.** The AXI R channel serializes
  read data for real (one beat per cycle); there is no bus-contention term in the
  latency.

Subclasses supply only `access_latency()`, the tier-specific approximate part:

| Tier | `access_latency(address, size, is_write)` |
|------|-------------------------------------------|
| `ideal` (`ideal/dram_ideal.cpp`) | `is_write ? write_cyc : read_cyc` — constant, address- and size-independent. |
| `bandwidth` (`bandwidth/dram_bandwidth.cpp`) | read: `read_cyc + size/bw`; write: `write_cyc + (include_writes ? size/bw : 0)`. `bw = 0` ⇒ base latency only. |
| `bank` (`bank/dram_bank.cpp`) | write: `write_latency_cyc`; read: hit `row_hit_cyc` / miss `row_miss_cyc`. |

The `bank` tier decodes each address to `(bank, row)` with a line = one DRAM row,
rows interleaved across banks so consecutive lines land in different banks
(`bank/bank_address.cpp`). A read to a bank's currently open row hits; any access
(read or write) leaves its row open, so a following same-row access hits.

## The `cycle` tier — DRAMSim3

`cycle` (`src/systemc/memory/dram/cycle/dram_cycle.cpp`) wraps DRAMSim3 driven
*loaded*, the way DRAMSim3's own frontend drives it: submit as transactions
arrive, tick every cycle, take completions from the callback — no blocking
drain, so bank parallelism, queueing, and refresh are actually modelled.

- `will_accept` maps to an in-flight cap (`accept_cap_`, DRAMSim3's queue size).
- `submit` breaks a burst into bus-sized DRAMSim3 transactions
  (`bus_bytes_ = bus_width/8 × burst_length`); the burst's token completes only
  once all its transactions have.
- `tick` advances the DRAM clock by `sys_clock_ns / tCK` DRAM cycles per system
  cycle, carrying the fractional remainder (`dram_accum_`).

Built only with `HAVE_DRAMSIM3` (always set by the build). Without it the tier is
a 0-latency stub that completes every transaction immediately.

## The layer-level model — `peak_bw`

Selected by `dram.tiled_dma_timing` (**not** `dram.tier`), and living beside the
per-transaction tiers in `src/systemc/memory/model/`, `peak_bw` charges a whole
tiled layer's memory time analytically instead of per burst
(`layer_timing_strategy.h`, `peak_bw_layer_model.h`):

- `"sim"` (default, `SimDmaTiming`) — the DMA/AXI/DRAM transaction sim already
  advanced time; the layer's memory term is that elapsed time.
- `"peak_bw"` (`PeakBwTiming`) — suppress per-burst DMA timing and model a
  fully-pipelined stream: `read_latency_cyc + bytes / bandwidth_bytes_per_cycle`,
  honoring `include_writes` (per-burst latency hidden). Used by the controller's
  tiled path (`npu_controller_read.cpp:150`, `npu_controller_tile.cpp:473`).

This lives in the memory subsystem, not the controller, so that changing memory
behavior stays a config/model change rather than a compute-side edit.
