#!/usr/bin/env bash
# Table-driven validation harness — compares simulator output against published
# paper numbers within a per-row tolerance. Generalizes check_nvdla_anchors.sh
# (which pins two sim numbers byte-exactly) to a %-tolerance comparison across
# accelerators, dataflows, and metrics. Anchors live in validation/references.csv.
#
# "Memory-only" comparison (SCALE-Sim style, no bus): the single-NPU path already
# adds zero interconnect latency by default (axi base/hop latency = 0, one master
# pair, no arbitration contention), so no bus-removal setup is required — match
# the reference's array + SRAM sizes in the hw JSON and run.
#
#   ./validation/run_validation.sh            # run every non-TODO anchor
#   exit 0 iff no row FAILs (TODO rows are SKIP, not FAIL).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/sim/flexnpusim"
REF="$ROOT/validation/references.csv"
[ -x "$BIN" ] || { echo "no binary at $BIN — build first (cmake --build build)"; exit 2; }
[ -f "$REF" ] || { echo "no reference table at $REF"; exit 2; }
mkdir -p "$ROOT/out"

extract() { # logfile recipe -> value on stdout
    local log="$1" rec="$2"
    case "$rec" in
        L*.f*) local n="${rec#L}"; n="${n%%.f*}"; local f="${rec#*.f}"
               awk -F, -v n="$n" -v f="$f" \
                   '$1=="@LAYER" && $2==n {print $f; exit}' "$log" ;;
        SUM.f*) local f="${rec#SUM.f}"
                awk -F, -v f="$f" \
                    '$1=="@LAYER"{s+=$f} END{print s}' "$log" ;;
        *) echo "" ;;
    esac
}

pass=0; fail=0; skip=0
printf "%-13s %-20s %-14s %13s %13s %8s %5s  %s\n" \
    ACCELERATOR NETWORK METRIC SIM PUBLISHED ERR% TOL VERDICT

# The trailing `source` field may contain commas; `read` puts the remainder in
# the last variable, so leave `src` last and it captures the citation verbatim.
while IFS=, read -r acc cfg net df metric ext pub tol conf src; do
    [ -z "${acc:-}" ] && continue
    case "$acc" in \#*) continue ;; esac

    net_short="$(basename "$net" .csv 2>/dev/null)"
    if [ "$pub" = "TODO" ]; then
        printf "%-13s %-20s %-14s %13s %13s %8s %5s  SKIP (awaiting citation)\n" \
            "$acc" "$net_short" "$metric" "-" "-" "-" "$tol"
        skip=$((skip+1)); continue
    fi

    log="$ROOT/out/val_${acc}_${net_short}_${metric}.log"
    FLEXNPUSIM_LAYER_CSV=1 "$BIN" -hw_conf "$ROOT/$cfg" -network "$ROOT/$net" \
        -o "$ROOT/out/val_tmp.csv" > "$log" 2>&1
    sim="$(extract "$log" "$ext")"

    if [ -z "$sim" ]; then
        printf "%-13s %-20s %-14s %13s %13s %8s %5s  FAIL (no sim value; see %s)\n" \
            "$acc" "$net_short" "$metric" "-" "$pub" "-" "$tol" "$log"
        fail=$((fail+1)); continue
    fi

    err="$(awk -v s="$sim" -v p="$pub" \
        'BEGIN{printf "%.2f", (p!=0)?100*(s-p)/p:0}')"
    verdict="$(awk -v e="$err" -v t="$tol" \
        'BEGIN{ea=(e<0)?-e:e; print (ea<=t)?"PASS":"FAIL"}')"
    printf "%-13s %-20s %-14s %13s %13s %7s%% %4s%%  %s [%s]\n" \
        "$acc" "$net_short" "$metric" "$sim" "$pub" "$err" "$tol" "$verdict" "$conf"
    if [ "$verdict" = PASS ]; then pass=$((pass+1)); else fail=$((fail+1)); fi
done < "$REF"

echo ""
echo "PASS=$pass  FAIL=$fail  SKIP=$skip"
[ "$fail" -eq 0 ]
