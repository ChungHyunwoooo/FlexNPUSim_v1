# validation/scalesim/ — SCALE-Sim cross-check

Compares FlexNPUSim against **SCALE-Sim v2** (an open systolic-array simulator:
array + IFMAP/Filter/OFMAP SRAM + DRAM, **no bus/NoC**) on one controlled conv
layer, for all three dataflows (WS/OS/IS). SCALE-Sim's memory-only structure is
already FlexNPUSim's default (the single-NPU path adds zero interconnect latency),
so no bus-removal setup is needed — the hw JSON just matches the array + SRAM sizes.

## Run

```sh
./validation/scalesim/compare.sh        # after a build
```

Prints, per dataflow × metric: sim, SCALE-Sim reference, % error, verdict.

## What it found (2026-07-11)

- **WS: fully validated (<1%).** F^I/F^W/F^O match SCALE-Sim SRAM access counts,
  and -report DRAM read matches SCALE-Sim DRAM to 0.3%.
- **DRAM read matches for WS & OS (0.3%)**; F^W matches for all three (<0.7%).
- **OS psum write — generalized (now PASS).** Added a psum-residency-aware
  `psum_sram_wr` metric (LayerPerfRecord, @LAYER field 19, report section 5):
  `psum_sram_wr = psum_resident ? out_elems : F^O`, where `psum_resident =
  (dataflow==OS) or output_location in {accumulator, pe_local}`. OS now 25,088 vs
  28,416 (-11.7%, residual = systolic drain, same family as compute); WS/IS
  903,168 = 903,168. F^O itself is unchanged (it drives the compute model).
- **2 remaining gaps (→ next F generalization work):**
  1. `IS fetch_input` — F^I=16,384 vs 112,896. Input-stationary reuses the input
     from SRAM across output channels; FlexNPUSim collapses F^I to the DRAM-fetch
     count under IS, losing the on-chip reuse count.
  2. `IS dram_read` — 139,512 vs 90,112 (+55%). FlexNPUSim over-reads DRAM under IS
     (streamed-weight refetch) when everything actually fits.
- **compute is a KNOWN-GAP**: FlexNPUSim compute cycles are dataflow-invariant
  (56,464 = MACs/peak) while SCALE-Sim varies (69,695/63,023/81,431) via
  dataflow-specific mapping utilization + systolic fill/drain — not modeled here.

Level correspondence learned: FlexNPUSim `@LAYER` F^I/F^W/F^O = **on-chip array-feed**
counts (compare to SCALE-Sim SRAM); `-report` rd/wr_bytes = **DRAM-boundary** bytes
(compare to SCALE-Sim DRAM). Matching the wrong level makes WS look 54x off when it
is actually <1%.

## Regenerate the SCALE-Sim reference numbers

`scalesim_reference.csv` holds recorded numbers (SCALE-Sim is not vendored). To
regenerate:

```sh
git clone --depth 1 https://github.com/scalesim-project/scale-sim-v2   # @ 9f98c437
cd scale-sim-v2
for df in ws os is; do
  PYTHONPATH="$PWD" python3 scalesim/scale.py \
    -c ../validation/scalesim/scalesim_configs/$df.cfg \
    -t ../validation/scalesim/scalesim_configs/topology.csv \
    -p /tmp/ss_out -i conv
done
# read <run_name>/DETAILED_ACCESS_REPORT.csv (SRAM/DRAM access) and
#      <run_name>/COMPUTE_REPORT.csv (Total Cycles = compute).
```

`scalesim_configs/{ws,os,is}.cfg` = 16x16 array, 512KB per SRAM, one MemoryBank.
`topology.csv` = the one_conv layer (SCALE-Sim topology format). The matched
FlexNPUSim inputs are `model/hw/npu/scalesim-{ws,os,is}.json` +
`model/sw/cnn/scalesim/one_conv.csv`.
