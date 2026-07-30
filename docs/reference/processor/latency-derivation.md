# Processor latency — `software_delay()` count derivation

The `Processor` model (`src/systemc/processor/processor.h`) prices CPU-side
latency with two DECOUPLED mechanisms (matching the DSL-lab `fractal_processor`):

- **EXTERNAL** — peripheral/MMIO accesses (the NPU registers) are manual AXI
  transactions (`reg_read`/`reg_write`). They generate a real bus transaction;
  their latency IS the AXI handshake + NPU-slave response. They do **not** go
  through the cache.
- **INTERNAL** — the CPU's own data-memory loads/stores are counted by the
  `store`/`load` args of `software_delay(store, load, jmp, mul, div, gen)` and
  priced by `mem_access()`, a purely statistical cache hit/miss penalty with
  **no** bus transaction.
- Compute is `software_delay`'s `jmp/mul/div/gen` counts × per-class cost
  (config `processor.cost_*`).

The counts are **hand-derived from the disassembly** of an ARM-host mirror of
the firmware (the C runs natively; only the instruction *mix* is used for
timing). Because the firmware only pokes NPU registers (all EXTERNAL) and has no
internal data memory, its `store`/`load` args are 0. This note records the
derivation so the values are reproducible and reviewable.

## Firmware mirror

Mirrors `Processor::run()`: program DNNSA/START, then poll NPUSR to IDLE with a
per-layer watchdog.

```c
typedef unsigned u32;
static volatile u32* const NPU = (u32*)0x40000000u;
enum { NPUCR = 0x00, NPUSR = 0x04, DNNSA = 0x08, DNNSA_MSB = 0x0C };
enum { NPUCR_START = 1u, NPUSR_IDLE = 2u };   // IDLE = bit1

u32 firmware(u32 dram_base) {
    NPU[DNNSA / 4]     = dram_base;      // str → EXTERNAL reg_write  ─┐ region A
    NPU[DNNSA_MSB / 4] = 0;              // str → EXTERNAL reg_write   │
    NPU[NPUCR / 4]     = NPUCR_START;    // str → EXTERNAL reg_write  ─┘
    u32 polls = 0, last_layer = 0xFF, watchdog = 0;
    while (watchdog < 20000) {
        u32 sr = NPU[NPUSR / 4];         // ldr → EXTERNAL reg_read
        ++polls;                          // ─┐ region B (one poll body)
        if (sr & NPUSR_IDLE) break;       //  │
        u32 cur = (sr >> 8) & 0xFF;       //  │
        if (cur != last_layer) { last_layer = cur; watchdog = 0; }
        else                     watchdog++;  // ─┘
    }
    return polls;
}
```

Disassemble:

```
clang --target=aarch64-linux-gnu -ffreestanding -O1 -S firmware.c
```

## AArch64 -O1 (non-`.`-directive lines)

```asm
    mov   w10, #4                 ; region A ─┐
    mov   w8,  w0                 ;            │
    mov   w9,  wzr                ;            │
    mov   w0,  wzr                ;            │  8 generic
    movk  w10, #16384, lsl #16    ;            │
    mov   w12, #1073741824        ;            │
    mov   w13, #1                 ;            │
    mov   w11, #255               ;           ─┘
    str   w8,  [x10, #4]          ; mem_access (DNNSA)
    str   wzr, [x10, #8]          ; mem_access (DNNSA_MSB)
    str   w13, [x12]              ; mem_access (NPUCR START)
    b     .LBB0_2                 ; 1 branch (enter loop)
    add   w0,  w0, #1             ; region B ─┐ polls++
    tbnz  w8,  #1, .LBB0_5        ;            │ branch: idle break
    lsr   w8,  w9, #5             ;            │ watchdog<20000 (625<<5)
    cmp   w8,  #624               ;            │
    b.hi  .LBB0_5                 ;            │ branch: watchdog break
    ldr   w8,  [x10]              ; mem_access (NPUSR read)
    tbnz  w8,  #1, .LBB0_1        ;            │ (idle recheck, rotation)
    ubfx  w12, w8, #8, #8         ;            │ cur = (sr>>8)&0xFF
    cmp   w12, w11                ;            │
    mov   w11, w12               ;            │
    csinc w9,  wzr, w9, ne        ;            │ watchdog update
    b     .LBB0_1                 ;           ─┘ branch: back-edge
.LBB0_5:
    ret
```

## Classification → counts

The `str`/`ldr` to `NPU[...]` are EXTERNAL peripheral accesses → modelled as
`reg_write`/`reg_read` (real AXI transactions), NOT `mem_access`. There is no
internal data memory, so `store`/`load` are 0. Only the surrounding compute is
counted here. `software_delay(store, load, jmp, mul, div, gen)`:

| Region | store | load | generic | branch | mul | div | `software_delay(...)` |
|--------|:---:|:---:|:------:|:------:|:---:|:---:|-----------------------|
| A — entry/program | 0 | 0 | 8 (`mov`×7, `movk`×1) | 1 (`b`) | 0 | 0 | `software_delay(0,0,1,0,0,8)` |
| B — one poll body | 0 | 0 | 7 (`add`,`lsr`,`cmp`×2,`ubfx`,`mov`,`csinc`) | 3 (`tbnz`,`b.hi`,`b`) | 0 | 0 | `software_delay(0,0,3,0,0,7)` |

Notes:
- **`mul=div=0` is empirical**: the `/4` index scaling folds into addressing,
  and `watchdog < 20000` strength-reduces to `(watchdog>>5) <= 624` — no divide.
- Region B counts the logical single iteration; `-O1` loop rotation emits a
  duplicate idle `tbnz`, which is not double-charged.

## Resulting CPU-side cycles

`cpu_cycles` counts only INTERNAL work (`software_delay` compute + `mem_access`).
EXTERNAL MMIO latency is the real AXI transaction's own sim-time, not tallied
here. With config defaults (jmp 5, gen 1) and one poll before IDLE:

```
region A : 1*5 + 8*1        = 13
1 poll   : 3*5 + 7          = 22
                     total  = 35    (matches sim)
```

The 3 register writes, the NPUSR polls, and the 8 perf-counter reads are all
EXTERNAL — real AXI transactions whose latency is the bus handshake + NPU-slave
response, not a `cpu_cycles` cache charge.
