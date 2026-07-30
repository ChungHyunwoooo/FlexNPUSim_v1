# Reuse-scheduling model

> **Status: known gap — not yet implemented.** Notes from the MIDAP reference
> comparison; the residual it explains is documented in
> [`../../../validation/results.md`](../../../validation/results.md).

## Conclusion

The model in `FlexNPUSim_model_core_v3.md` defines the reuse quantities $R^x_{\mathrm{GB}}$, $R^x_{\mathrm{LM}}$ and the transfer quantities $E^y$ as identities (Section 4.3, eq 7–10), but it does not define the schedule that produces their values. This document adds that missing layer: a capacity-driven residency schedule that derives $F^x_{\mathrm{GB}}$ and $E^y_{\mathrm{GB}}$ — the two DRAM-traffic terms — from the on-chip capacity and the traversal order. MIDAP's compiler (`MidapSim/midap_software`) is the reference; its cost model is transcribed and mapped onto FlexNPUSim's quantities below.

## The gap between the accounting and the schedule

Section 4.3 states $A^x = F^x_{\mathrm{GB}} + R^x_{\mathrm{GB}} + R^x_{\mathrm{LM}}$: every operand use is a DRAM supply, a GB reuse, or an LM reuse. The split is an identity — it holds for any schedule — so it fixes what the quantities MEAN but not what they EQUAL. Two schedules with the same partition give different $F^x_{\mathrm{GB}}$: keeping a feature map on-chip assigns the next layer's input to $R^{\mathrm{in}}_{\mathrm{GB}}$; evicting it assigns the same use to $F^{\mathrm{in}}_{\mathrm{GB}}$ (a DRAM read) plus the producer's $E^{\mathrm{out}}_{\mathrm{GB}}$ (a DRAM write). The model does not say which.

The implementation filled this gap ad hoc, and inconsistently across the two execution paths, which is the measured MIDAP gap:

- Tiled path forwards a GB-resident input (activation reuse) but refetches weights per output tile (input-stationary, weight non-resident).
- Non-tiled path reads weights once but never forwards its input (always a DRAM read).

Neither path keeps BOTH resident, so MobileNetV1 under `layer_fusion=auto` reads 9.3 MB where MIDAP reads ~5.6 MB. The residual is weight refetch (tiled) plus activation round-trip (non-tiled).

## MIDAP's schedule as a cost model

`layer_block._determine_path_order(available_fmem)` chooses the block traversal order and the input/output residency ("stationary") to minimize DRAM access. `_get_dram_access_size` is the cost it minimizes, with two terms per layer `v`:

- **Output spill.** `2 · max(require_total_fsize − num_available_banks · fsize, 0)`. The output stays in FMEM when it fits the free banks (cost 0); otherwise the excess round-trips DRAM (write + read = 2×).
- **Weight reload.** `weight_load_num = 1 if is_weight_in_wmem else div_ceil(input_banks, num_available_banks + 1)`, times `get_weight_size()`. A WMEM-resident weight loads once; a non-resident weight reloads once per input-bank group.

`fmem_info` allocates and evicts feature banks explicitly (`discard_data_by_layer`), so residency is a decision, not a side effect. The schedule is therefore: partition on-chip memory into feature banks (FMEM) and weight banks (WMEM), then order layers and choose residency so that outputs stay resident for their consumer and weights stay resident across tiles, spilling only what does not fit.

## Mapping onto FlexNPUSim quantities

The two MIDAP cost terms are exactly FlexNPUSim's two DRAM-traffic quantities:

- $E^{\mathrm{out}}_{\mathrm{GB}}$ (GB→DRAM output write) = the output-spill term. Zero when the output fits the feature region and its consumer runs next; otherwise the tile output round-trips.
- $F^{\mathrm{wt}}_{\mathrm{GB}}$ (DRAM→GB weight supply) = weight-size × reload count. One reload when the weight fits the weight region ($R^{\mathrm{wt}}_{\mathrm{GB}}$ absorbs the rest); `div_ceil(reduction, capacity)` reloads otherwise.

A retained output makes its consumer's input a $R^{\mathrm{in}}_{\mathrm{GB}}$ (GB reuse), removing both the producer's $E^{\mathrm{out}}_{\mathrm{GB}}$ and the consumer's $F^{\mathrm{in}}_{\mathrm{GB}}$ — the symmetric read/write pair the retention fixes already handle once the residency decision is made.

## Design

Replace the two ad-hoc mechanisms — `layer_fusion` (all-or-nothing output residency) and dataflow-tied `residency.weight_resident` — with one capacity-driven residency decision, computed once per layer against the fixed GB partition:

1. `weight_resident = weight_volume ≤ weight_partition_bytes` → $F^{\mathrm{wt}}_{\mathrm{GB}}$ = weight_volume when true, else weight_volume × `div_ceil(C_in, feed_capacity)`.
2. `output_stationary = output_volume ≤ feature_partition_bytes AND next layer consumes it` → $E^{\mathrm{out}}_{\mathrm{GB}} = 0$ and the consumer's input becomes $R^{\mathrm{in}}_{\mathrm{GB}}$; else the output spills and the consumer reads DRAM.

Two properties make this correct where the current code is not:

- **One schedule, both paths.** The residency decision is computed before execution and produces the $F/R/E$ the model defines; tiled and non-tiled executors both read those values instead of each re-deriving traffic. This removes the tiled/non-tiled inconsistency and the duplicate depthwise weight computation that diverged (compiler vs `tile_descriptor`).
- **Weight and feature resident together.** Separate feature and weight partitions let both stay resident when capacity allows, which is what MIDAP's FMEM+WMEM does and what no single current path does.

## Implementation plan

1. Scheduler pass (compiler side, per Section 3.2): given the GB partition, emit per-layer `weight_resident` and `output_stationary` flags and the resulting $F^{\mathrm{wt}}_{\mathrm{GB}}$, $E^{\mathrm{out}}_{\mathrm{GB}}$ into the descriptor. Verify → the emitted $F^x_{\mathrm{GB}} \le A^x$ (the invariant the depthwise bug violated).
2. Runtime executes the baked residency instead of the `layer_fusion` knob and the dataflow weight-residency. Verify → MobileNetV1 read approaches ~5.6 MB (weights once + first input), MIDAP config.
3. Add a scheduling subsection to `FlexNPUSim_model_core_v3.md` between 4.3 and 4.4 deriving $R/E$ from the residency decision, so the model states the schedule it currently omits.

## Cross-architecture check

The schedule must express the range, not just MIDAP. NVDLA is the opposite corner: tile-streaming with no cross-layer feature retention, so `output_stationary` is false by design and `weight_resident` holds only within a layer — which is FlexNPUSim's current tiled behavior and matches NVDLA RTL at 2.62% mean cycle error. MIDAP sets both residency flags true. One scheduler with two capacity/order settings must reproduce both corners.
