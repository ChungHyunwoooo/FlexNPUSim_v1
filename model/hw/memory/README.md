# DRAMSim3 ini preset directory

This directory holds DRAMSim3 configuration ini files used by FlexNPUSim's Tier B
(DSE / new-result) runs. Each preset is a standard DRAMSim3 config file that
defines DDR protocol, timing, and organizational parameters.

## Prerequisite

DRAMSim3 library must be built with `HAVE_DRAMSIM3` enabled. See
`src/memory/model/dram_dramsim3.h` for integration.

## Presets currently populated

| File | DDR | Data rate | Target usage |
|---|---|---|---|
| `DDR4_3200_x8.ini` | DDR4 | 3200 MT/s | Eyeriss v2, MAERI Tier B; upstream JEDEC |
| `LPDDR4_2400_x16.ini` | LPDDR4 | 2400 MT/s | NVDLA / MIDAP / Nullhop / Gemmini Tier B; mobile-class deployment |
| `DDR3_1600_x16.ini` | DDR3 | 1600 MT/s | reference slot; Zynq-style Tier A candidate for Nullhop |

## Sourcing

All presets copied verbatim from upstream DRAMSim3 `configs/` directory at:
`/home/hwchung/workspace/02_full_system/external/DRAMsim3/configs/`.

Do not hand-author timing parameters. Re-sync when upstream releases new JEDEC
timing updates. If a needed preset (e.g. LPDDR4-3200, HBM2) is added upstream,
copy it here and update this table.

## Missing-preset notes

- LPDDR4-3200 requested by some HW JSONs originally, not present upstream. All
  LPDDR4-using Tier B JSONs now reference LPDDR4-2400 instead.

## Referencing from HW JSON

Each HW JSON selects the cycle tier and one preset via the `dram` block:

```json
{
  "dram": { "tier": "cycle", "ini": "model/hw/memory/LPDDR4_2400_x16.ini" }
}
```

The path is resolved relative to the project root. Other tiers (`ideal`,
`bandwidth`, `bank`) need no `.ini` — see `docs/reference/memory/parameters.md`.
