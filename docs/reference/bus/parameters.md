# Bus — Parameters

The subsystem draws config from three JSON blocks: `axi.*` (the AXI master/slave
runtime knobs and signal width), `hw.dma.*` (the DMA IP), and `topology.*` (the
crossbar's per-master arbitration). All are parsed in
`src/common/config_loader.cpp` into `AxiConfig` / `DmaCfg` / `TopologyConfig`
(`src/common/flexnpu_config.h`) and wired in `src/system/flexnpusim_system.cpp`.

## `axi.*` — AXI transport

Parsed by `parse_axi`. Feeds `axi::CommonConfig` (master/slave outstanding
budgets) and, at runtime, the bus beat width.

| JSON key | Default | Meaning | Consumed by |
|----------|---------|---------|-------------|
| `axi.protocol` | `"AXI4"` | `"AXI3"` (LEN 4b, WID, max 16 beats) or `"AXI4"` (LEN 8b, QOS/REGION, max 256 beats) | `Spec` / `ProtocolTraits` |
| `axi.data_width_bits` | `64` | bus beat width in bits; runtime `bus_bits`, sets beat bytes = `bus_bits/8` | `caps().max_burst_bytes`, DMA burst split |
| `axi.addr_width_bits` | `32` | address width | `Spec` |
| `axi.id_width_bits` | `4` | AXI ID width | `Spec` |
| `axi.max_burst_length` | `256` | max beats per burst (AXI3=16, AXI4=256) | `caps().max_beats_per_burst` |
| `axi.max_outstanding_reads` | `8` | read ticket-pool depth (master) → read in-flight cap | `TicketPool`, `caps()`, `read_window()` |
| `axi.max_outstanding_writes` | `4` | write ticket-pool depth (master) → write in-flight cap | `TicketPool`, `caps()`, `write_window()` |
| `axi.modeling_mode` | `"Full"` | `"Full"` (signal-level) or `"LatencyOnly"` | transport fidelity |
| `axi.default_arbit_policy` | `"RoundRobin"` | the data-bus arbiter policy (5 names below) | `Bus` per-slave arbiters |

`data_width_bits` is a runtime knob: the compiled `DataSpecT` signal is sized to
a fixed maximum and the live beat width comes from `bus_bits`. The DSE sweep
validates it against a supported set before wiring it in.

**Arbitration policy names** (accepted by `parse_bus_arbit`, snake_case aliases
in parentheses): `RoundRobin` (`round_robin`, default), `WeightedRR`
(`weighted_rr`), `FixedPriority` (`fixed_priority`), `PriorityRR`
(`priority_rr`), `QoSAware` (`qos_aware`).

> Note: the *active* data-bus policy is taken from `axi.default_arbit_policy`.
> `topology.interconnect.arbit_policy` (below) is parsed into the schema but is
> not read by the data-bus assembly in `flexnpusim_system`.

## `hw.dma.*` — DMA engine

Parsed by `parse_dma`. A `profile` name expands into per-field defaults
(`resolve_profile` in `dma/model/dma_profile.h`); explicit keys then override
each field.

| JSON key | Default (generic) | Meaning | Consumed by |
|----------|-------------------|---------|-------------|
| `hw.dma.profile` | `"generic"` | preset: `"generic"`, `"nvdla_cdma"`, `"arm_dma330"` | `resolve_profile` |
| `hw.dma.buffer_size_kb` | `16` | internal read/write FIFO depth (KiB); sets the in-flight window | `set_flow_control` → `fifo_bytes_` |
| `hw.dma.max_burst_length` | `8` | max beats per burst in the burst planner | `DmaModel` / `DmaConfig` |
| `hw.dma.max_issuing_reads` | `0` | explicit AXI read issuing cap (`0` = derive from FIFO) | `read_window()` |
| `hw.dma.max_issuing_writes` | `0` | explicit AXI write issuing cap (`0` = derive) | `write_window()` |
| `hw.dma.read_channels` | `1` | read DMA channels (extra masters on the bus) | master assembly |
| `hw.dma.write_channels` | `1` | write DMA channels | master assembly |

The window math (`outstanding_window`): `fifo_bytes / burst_bytes`, capped by
`max_issuing` when set, floored at 1 — then hard-capped by
`axi.max_outstanding_{reads,writes}` inside `read_window()`/`write_window()` (see
[`modeling.md`](modeling.md), the livelock fix). `buffer_size_kb` is therefore a
live timing knob: a deeper FIFO widens the window and hides more latency, up to
the AXI outstanding budget.

Profile presets (from `resolve_profile`):

| Profile | `max_burst_beats` | `fifo_bytes` | `max_issuing_{r,w}` | `{read,write}_channels` |
|---------|-------------------|--------------|---------------------|-------------------------|
| `generic` | 8 | 16 KiB | 0 / 0 (derive) | 1 / 1 |
| `nvdla_cdma` | 4 | 4 KiB | 0 / 0 | 2 / 1 |
| `arm_dma330` | 16 | 4 KiB | 8 / 8 | 8 / 8 |

## `topology.*` — crossbar wiring

Parsed by `parse_topology`. On the shipped DSE data bus, per-master
arbitration attributes are read from `topology.masters` (in declaration order)
and injected into each `MasterArbitCfg`; the other topology fields describe the
N×M wiring schema.

| JSON key | Default | Meaning | Consumed by |
|----------|---------|---------|-------------|
| `topology.masters[].qos` | `0` | static QOS (per-transaction override possible) | `QoSAware` arbiter |
| `topology.masters[].weight` | `1` | `WeightedRR` credits per rotation | `WeightedRR` arbiter |
| `topology.masters[].priority` | `0` | `FixedPriority`/`PriorityRR` tier (lower wins) | priority arbiters |
| `topology.masters[].default_qos` | `0` | `QoSAware` fallback when no live QOS is set | `QoSAware` arbiter |
| `topology.masters[].port_bits` | `64` | master port width | topology schema |
| `topology.masters[].max_outstanding_{reads,writes}` | `8` / `0` | per-master outstanding budget | topology schema |
| `topology.slaves[].addr_base` / `addr_size` | `0` / `0` | slave decode range (base, size) | `Bus::SlaveRange` |
| `topology.slaves[].base_{read,write}_latency_cyc` | `0` | slave-side fixed latency | topology schema |
| `topology.slaves[].port_bits` | `128` | slave port width | topology schema |
| `topology.links[].{master,slave}` | — | master→slave connection | topology schema |
| `topology.links[].width_converter` | — | e.g. `"64to128"` | topology schema |
| `topology.links[].bridge_extra_cyc` | `0` | per-link bridge latency | topology schema |
| `topology.interconnect.type` | `"axi_crossbar"` | interconnect kind | topology schema |
| `topology.interconnect.arbit_policy` | `"RoundRobin"` | schema policy (see note above — data bus uses `axi.default_arbit_policy`) | topology schema |
| `topology.interconnect.extra_latency_per_hop_cyc` | `0` | per-hop latency | topology schema |
| `topology.interconnect.preemption` | `false` | preemptive arbitration flag | topology schema |

TODO: several `topology.*` fields (`port_bits`, per-master
`max_outstanding_*`, `slaves[].base_*_latency_cyc`, `links[].*`,
`interconnect.extra_latency_per_hop_cyc`, `preemption`) are parsed into the
config but not yet consumed by the `flexnpusim_system` data-bus assembly; the live
data-bus geometry is `NumS = 1` with `NumM` = `read_channels + write_channels`.
Their intended wiring belongs to the future system-composition config
(`docs/reference/system/topology.md`) and is not verifiable from the current bus
code.
