#!/usr/bin/env python3
"""Full experiment suite on the faithful config-controlled single datapath.
Sweeps real hardware knobs (DRAM tier, bus width, bandwidth, DMA burst/fifo)
and cross-accelerator configs. peak_bw formula is NOT used except as the anchor.
"""
import json, subprocess, re, copy, os, sys

ROOT = "/home/hwchung/workspace/FlexNPUSim"
SCR  = os.path.dirname(os.path.abspath(__file__))
BIN  = f"{ROOT}/build/sim/flexnpusim"
NET_MIDAP = f"{ROOT}/model/sw/cnn/midap/mobilenetv1_midap.csv"
BASE = json.load(open(f"{ROOT}/model/hw/npu/midap_single.json"))

def setpath(d, path, val):
    cur = d
    *head, last = path.split(".")
    for k in head:
        cur = cur.setdefault(k, {})
    cur[last] = val

def run(cfg, network=NET_MIDAP, tag="cfg"):
    p = f"{SCR}/_run.json"
    json.dump(cfg, open(p, "w"))
    try:
        r = subprocess.run([BIN, "-hw_conf", p, "-network", network,
                            "-o", "/tmp/exp.csv"],
                           capture_output=True, text=True, timeout=340)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT", "-", "-")
    out = r.stdout + r.stderr
    cyc = re.search(r'Cycles:\s+(\d+)', out)
    done = "YES" if re.search(r'Completed:\s+YES', out) else "NO"
    macs = re.search(r'MACs?:\s+([\d,]+)', out)
    return (int(cyc.group(1)) if cyc else None, done,
            macs.group(1) if macs else "-")

def variant(**overrides):
    c = copy.deepcopy(BASE)
    for k, v in overrides.items():
        setpath(c, k.replace("__", "."), v)
    return c

PEAK_BW = 920819
results = {}

# E1. DRAM tier ladder (fidelity)
print("E1 DRAM tier ladder", file=sys.stderr)
results["E1"] = []
for tier in ["ideal", "bandwidth", "bank", "cycle"]:
    cyc, done, _ = run(variant(**{"dram__tier": tier}))
    results["E1"].append((tier, cyc, done))
    print(f"  tier={tier:10} cyc={cyc} {done}", file=sys.stderr)

# E2. AXI bus width sweep (bandwidth tier)
json.dump(results, open(f"{SCR}/exp_results.json","w"), indent=2)
print("E2 bus width", file=sys.stderr)
results["E2"] = []
for w in [128, 256, 512, 1024]:
    cyc, done, _ = run(variant(**{"axi__data_width_bits": w}))
    results["E2"].append((w, cyc, done))
    print(f"  bus={w:5}b cyc={cyc} {done}", file=sys.stderr)

# E3. DRAM bandwidth sweep (B/cycle)
json.dump(results, open(f"{SCR}/exp_results.json","w"), indent=2)
print("E3 DRAM bandwidth", file=sys.stderr)
results["E3"] = []
for bw in [13, 26, 52, 104]:
    cyc, done, _ = run(variant(**{"dram__bandwidth_bytes_per_cycle": bw}))
    results["E3"].append((bw, cyc, done))
    print(f"  bw={bw:4} B/cyc cyc={cyc} {done}", file=sys.stderr)

# E4. DMA burst length + fifo depth
json.dump(results, open(f"{SCR}/exp_results.json","w"), indent=2)
print("E4 DMA burst/fifo", file=sys.stderr)
results["E4"] = []
for burst in [8, 16, 32, 64]:
    cyc, done, _ = run(variant(**{"hw__dma__max_burst_length": burst}))
    results["E4"].append(("burst", burst, cyc, done))
    print(f"  burst={burst:4} cyc={cyc} {done}", file=sys.stderr)
for fifo in [16, 32, 64, 128]:
    cyc, done, _ = run(variant(**{"hw__dma__buffer_size_kb": fifo}))
    results["E4"].append(("fifo_kb", fifo, cyc, done))
    print(f"  fifo={fifo:4}kb cyc={cyc} {done}", file=sys.stderr)

# E5. Cross-accelerator (same datapath, config-only) on same network
json.dump(results, open(f"{SCR}/exp_results.json","w"), indent=2)
print("E5 cross-accelerator", file=sys.stderr)
results["E5"] = []
for name in ["midap_peakbw", "nvdla-large", "eyeriss-v1", "gemmini", "tpu-v1", "simba"]:
    cfgp = f"{ROOT}/model/hw/npu/{name}.json"
    if not os.path.exists(cfgp):
        continue
    r = subprocess.run([BIN, "-hw_conf", cfgp, "-network", NET_MIDAP,
                        "-o", "/tmp/exp.csv"], capture_output=True, text=True, timeout=340)
    out = r.stdout + r.stderr
    cyc = re.search(r'Cycles:\s+(\d+)', out)
    done = "YES" if re.search(r'Completed:\s+YES', out) else "NO"
    results["E5"].append((name, int(cyc.group(1)) if cyc else None, done))
    print(f"  {name:14} cyc={cyc.group(1) if cyc else None} {done}", file=sys.stderr)

json.dump(results, open(f"{SCR}/exp_results.json", "w"), indent=2)
print("DONE -> exp_results.json", file=sys.stderr)
