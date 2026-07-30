#!/usr/bin/env python3
"""Regenerate ALL results.md §3 DSE numbers after the 2026-07-20 bus W-channel
fix. Reproduces the original experiment matrix (same oracle-strip, same DRAM
tier=bandwidth 26 B/cyc + 100cyc latency):
  3a: MIDAP x MobileNet x {64/128,64/256,128/512,256/1024}
  3b: MIDAP+NVDLA x MobileNet x {256/1024,128/512,64/128} + 64b GB sweep {128,256,1024}
  3c: 6 accels x MobileNet x {64/128,128/512,256/1024}
  3d: MIDAP+NVDLA x ResNet18 x 3 configs; TPU/MIDAP/Gemmini x {MobileNet,ResNet18,VGG16} @256/1024
Parallel 6-way; per-run unique files (old script reused /tmp/g.csv — race)."""
import json, subprocess, re, os, csv, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN  = f"{ROOT}/build/sim/flexnpusim"
SCR  = os.path.join(ROOT, "out", "dse_regen"); os.makedirs(SCR, exist_ok=True)

def strip_oracle(src, dst):
    rows = list(csv.reader(open(src))); hdr = rows[0]
    drop = {c for c in ('pe_type','k','g','s','psum_kb','f_i','f_w','f_o',
                        'n_max','n_min') if c in hdr}
    keep = [i for i,c in enumerate(hdr) if c not in drop]
    with open(dst,'w',newline='') as f:
        w=csv.writer(f)
        for r in rows: w.writerow([r[i] for i in keep])
    return dst

NETS = {
  'mobilenet': strip_oracle(f"{ROOT}/model/sw/cnn/midap/mobilenetv1_midap.csv", f"{SCR}/_r_mn.csv"),
  'resnet18':  strip_oracle(f"{ROOT}/model/sw/cnn/midap/resnet18.csv",          f"{SCR}/_r_rn.csv"),
  'vgg16':     strip_oracle(f"{ROOT}/model/sw/cnn/midap/vgg16.csv",             f"{SCR}/_r_vg.csv"),
}

def variant(accel, axi, gb):
    c = json.load(open(f"{ROOT}/model/hw/npu/{accel}.json"))
    c['axi']['data_width_bits'] = axi
    c['hw']['global_buffer']['capacity_kb'] = gb
    d = c.setdefault('dram',{}); d['tier']='bandwidth'
    d['bandwidth_bytes_per_cycle']=26; d['read_latency_cyc']=100
    p=f"{SCR}/_r_{accel}_{axi}_{gb}.json"; json.dump(c,open(p,"w"))
    return p

def run(job):
    accel, net, axi, gb, tmo = job
    tag=f"{accel}_{net}_{axi}_{gb}"
    rep=f"{SCR}/_r_{tag}.rep"
    try:
        subprocess.run([BIN,"-hw_conf",variant(accel,axi,gb),"-network",NETS[net],
                        "-o",f"{SCR}/_r_{tag}.out","-report",rep],
                       capture_output=True,text=True,timeout=tmo)
    except subprocess.TimeoutExpired:
        return tag, dict(note='TIMEOUT')
    txt=open(rep).read() if os.path.exists(rep) else ""
    def g(pat):
        m=re.search(pat,txt); return m.group(1) if m else None
    cyc=g(r'total cycles\s*:\s*(\d+)')
    return tag, dict(cyc=int(cyc) if cyc else None,
                     peu=g(r'PE utilization\s*:\s*([\d.]+)'),
                     busu=g(r'DRAM bus util\s*:\s*([\d.]+)'),
                     cmp_b=len(re.findall(r'compute-bound',txt)),
                     mem_b=len(re.findall(r'memory-bound',txt)))

ACC6 = ['midap_single','nvdla-large','nullhop','tpu-v1','gemmini','eyeriss-v2']
jobs = []
# mobilenet full grid (covers 3a/3b/3c): 5 configs x 6 accels
for (axi,gb) in [(256,1024),(128,512),(64,1024),(64,256),(64,128)]:
    for a in ACC6:
        jobs.append((a,'mobilenet',axi,gb,300))
# 3d resnet18: midap+nvdla crossover 3 configs; tpu/gemmini @256/1024
for (axi,gb) in [(256,1024),(128,512),(64,128)]:
    for a in ['midap_single','nvdla-large']:
        jobs.append((a,'resnet18',axi,gb,300))
for a in ['tpu-v1','gemmini']:
    jobs.append((a,'resnet18',256,1024,300))
# 3d + Backup G(레이어별 util) vgg16 @256/1024: 6 가속기 전부
for a in ACC6:
    jobs.append((a,'vgg16',256,1024,600))

print(f"total jobs: {len(jobs)}", flush=True)
results={}
with ThreadPoolExecutor(max_workers=6) as ex:
    for tag,res in ex.map(run, jobs):
        results[tag]=res
        print(f"  {tag:38} {res}", flush=True)
        json.dump(results,open(f"{SCR}/regen_dse_all.json","w"),indent=2)
print("REGEN_DONE", flush=True)
