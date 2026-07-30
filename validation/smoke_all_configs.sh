#!/usr/bin/env bash
# All-config completion smoke test — runs every model/hw/npu/*.json on a single
# small conv (one_conv) and asserts each completes (no livelock/crash) with the
# dataflow-invariant MAC count. Catches datapath regressions that pass the
# MIDAP/NVDLA anchors but silently break other accelerators — the exact gap that
# let the 2026-07-18 narrow-bus write-window livelock reach 9 configs unnoticed.
#
#   ./validation/smoke_all_configs.sh        # exit 0 iff every config completes
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/sim/flexnpusim"
NET="$ROOT/model/sw/cnn/scalesim/one_conv.csv"
EXPECT_MACS=14450688          # one_conv MAC count (network property, config-invariant)
PER_CONFIG_TIMEOUT=60         # a single small conv completes in <1s; 60s = livelock guard

[ -x "$BIN" ] || { echo "build first: cmake --build $ROOT/build"; exit 2; }
[ -f "$NET" ] || { echo "missing workload: $NET"; exit 2; }

pass=0; fail=0
printf "%-16s %-12s %-12s %s\n" CONFIG CYCLES MACs VERDICT
for f in "$ROOT"/model/hw/npu/*.json; do
    n="$(basename "$f" .json)"
    rep="$ROOT/out/smoke_${n}.txt"; mkdir -p "$ROOT/out"
    out="$(timeout "$PER_CONFIG_TIMEOUT" "$BIN" -hw_conf "$f" -network "$NET" \
             -o "$ROOT/out/smoke_tmp.csv" -report "$rep" 2>&1)"; rc=$?
    cyc="$(printf '%s' "$out" | grep -oE 'Cycles:[[:space:]]+[0-9]+' | grep -oE '[0-9]+')"
    macs="$(grep -oE 'MACs[[:space:]]+:[[:space:]]+[0-9]+' "$rep" 2>/dev/null | grep -oE '[0-9]+$')"
    done="$(printf '%s' "$out" | grep -qE 'Completed:[[:space:]]+YES' && echo YES || echo NO)"

    verdict=PASS
    if [ "$rc" -eq 124 ]; then verdict="FAIL (livelock/timeout)"; cyc="TIMEOUT"
    elif [ "$done" != YES ]; then verdict="FAIL (did not complete)"
    elif [ "${macs:-0}" != "$EXPECT_MACS" ]; then verdict="FAIL (MAC $macs != $EXPECT_MACS)"
    fi
    [ "$verdict" = PASS ] && pass=$((pass+1)) || fail=$((fail+1))
    printf "%-16s %-12s %-12s %s\n" "$n" "${cyc:-none}" "${macs:-?}" "$verdict"
done

echo "----"
echo "PASS=$pass FAIL=$fail"
[ "$fail" -eq 0 ]
