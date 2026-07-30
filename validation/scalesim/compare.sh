#!/usr/bin/env bash
# SCALE-Sim cross-check — runs FlexNPUSim on the `one_conv` layer for WS/OS/IS
# and compares each quantity against the recorded SCALE-Sim numbers
# (scalesim_reference.csv). Prints a per-metric table with % error and verdict.
#
# "Memory-only, no bus" holds by default (single-NPU path adds zero interconnect
# latency), so this is a clean array + SRAM + DRAM comparison.
#
#   ./validation/scalesim/compare.sh
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/build/sim/flexnpusim"
REF="$ROOT/validation/scalesim/scalesim_reference.csv"
NET="$ROOT/model/sw/cnn/scalesim/one_conv.csv"
[ -x "$BIN" ] || { echo "build first: cmake --build $ROOT/build"; exit 2; }
mkdir -p "$ROOT/out"

# Run each dataflow once, cache @LAYER line + report path.
declare -A LAYER RPT
for df in ws os is; do
    r="$ROOT/out/ss_cmp_$df.txt"
    LAYER[$df]="$(FLEXNPUSIM_LAYER_CSV=1 "$BIN" -hw_conf "$ROOT/model/hw/npu/scalesim-$df.json" \
        -network "$NET" -o "$ROOT/out/ss_cmp_$df.csv" -report "$r" 2>&1 | grep '@LAYER')"
    RPT[$df]="$r"
done

extract() { # dataflow recipe
    local df="$1" rec="$2"
    case "$rec" in
        L0.f*) echo "${LAYER[$df]}" | cut -d, -f"${rec#L0.f}" ;;
        RD) grep -i "read traffic"  "${RPT[$df]}" | grep -oE '[0-9]+ B' | head -1 | tr -d ' B' ;;
        WR) grep -i "write traffic" "${RPT[$df]}" | grep -oE '[0-9]+ B' | head -1 | tr -d ' B' ;;
        *) echo "" ;;
    esac
}

printf "%-3s %-14s %-26s %11s %11s %8s  %s\n" \
    DF METRIC LEVEL SIM SCALESIM ERR% VERDICT
pass=0; fail=0; gap=0
while IFS=, read -r df metric ext ref tol level; do
    case "$df" in \#*|"") continue ;; esac
    sim="$(extract "$df" "$ext")"
    [ -z "$sim" ] && { printf "%-3s %-14s %-26s %11s %11s %8s  FAIL(no sim)\n" \
        "$df" "$metric" "$level" "-" "$ref" "-"; fail=$((fail+1)); continue; }
    err="$(awk -v s="$sim" -v r="$ref" 'BEGIN{printf "%.2f",(r!=0)?100*(s-r)/r:0}')"
    if [ "$metric" = compute ]; then
        v="KNOWN-GAP"; gap=$((gap+1))          # MACs/peak model is dataflow-invariant by design
    else
        v="$(awk -v e="$err" -v t="$tol" 'BEGIN{ea=(e<0)?-e:e;print(ea<=t)?"PASS":"FAIL"}')"
        [ "$v" = PASS ] && pass=$((pass+1)) || fail=$((fail+1))
    fi
    printf "%-3s %-14s %-26s %11s %11s %7s%%  %s\n" \
        "$df" "$metric" "$level" "$sim" "$ref" "$err" "$v"
done < "$REF"
echo ""
echo "PASS=$pass  FAIL=$fail  KNOWN-GAP=$gap"
