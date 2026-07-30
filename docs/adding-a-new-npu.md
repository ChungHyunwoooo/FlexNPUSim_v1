# 신규 NPU 온보딩 가이드

새 NPU를 **소스 수정 없이 config JSON + network CSV만으로** FlexNPUSim에 올리는
절차. 준비물 → 추출 → JSON → CSV → 실행/검증 → 캘리브레이션 순서로 진행한다.
(정식화 근거와 필드 배선은 `docs/reference/latency-model/`의
`f-derivation.md`·`hw-derivation-examples.md`·`parameters.md` 참조.)

---

## 0. 준비물 체크리스트 (9개)

| 급 | # | 항목 | 출처 |
|---|---|---|---|
| **필수** | 1 | datapath 구조 3값: accumulator 수 / update당 fan-in / enable 방식 | 마이크로아키텍처 블록도 |
| | 2 | element size (INT8/16/FP16) | 스펙 |
| | 3 | on-chip 버퍼: GB 용량 (+partition, psum 위치, accumulator 용량) | 스펙 SRAM 표 |
| | 4 | dataflow(WS/OS/IS) + preload/streaming 여부 | 실행 방식 서술 |
| | 5 | 워크로드: 레이어 dims (PyTorch면 `src/compiler/pytorch/pytorch_frontend.py`) | 모델 정의 |
| 권장 | 6 | pipeline depth l (모르면 §2에서 유도) | 스펙/실측 |
| | 7 | GB↔PE 포트 폭, PE 버퍼 용량, double buffering | 스펙 |
| | 8 | sparsity/압축률, 컴파일러 tile 스케줄 | 스펙/컴파일러 |
| 선택 | 9 | **vendor 성능 참조** (perf 모델 산식 > per-layer 실측 > 총 latency) | §5의 신뢰 앵커 |

## 1. Step 1 — datapath에서 (A, k, 𝓔, l) 읽기

여기만 "생각"이 필요하다. 나머지는 표 옮겨적기.

- **A (accumulator 수)**: partial output에 대한 **독립 read–modify–write 경로**의
  수. 판별 규칙: *나중 pass가 다시 읽어 누산하는 저장값*이 accumulator다. 한 pass
  안에서 다음 stage로 전달만 되는 레지스터(spatial reduction)는 아니다.
  → JSON `num_output_lanes` (=n_max의 원천).
- **k (pass 폭)**: accumulator 1회 update로 수렴하는 contribution 수(fan-in,
  "k-way update"). → JSON `ops_per_pass`.
- **𝓔 (enable 방식)**: accumulator들이 **함께** 켜지는가(lockstep — NVDLA CACC),
  **독립**으로 켜지는가. → JSON `enable_granularity: "lockstep" | "independent"`.
  레이어마다 다르면(재구성형) CSV `enable_granularity` 컬럼으로 per-layer
  오버라이드: `lockstep | independent | grouped:N`(N = emit 폭 하한).
  ⚠️ **최다 실수 지점**: k/g/s 유도는 n_min=g를 주므로 lockstep 하드웨어(진짜
  n_min=n_max)를 표현하지 못한다 — lockstep이면 반드시 명시하라 (NVDLA n_min
  1→16 버그의 원인).
- **l (pipeline depth)**: issue→retire 상수. 실측이 없으면 chain 구조 l=k,
  tree 구조 l=1+⌈log₂k⌉로 유도(unit_type이 이걸 정한다). GB→PE 전송 시간은
  arrival trace에 이미 있으므로 **l에 넣으면 이중 계상**.

**추출 예 4종**: WS systolic(column-end bank 4개, 4-stage chain)
→ (4,4,2^A,4) / NVDLA nv_full INT16 → |A|=16(Atomic-K), k=64(Atomic-C), lockstep
/ Nullhop N_out=M → |A|=128, k=1, lockstep / MAERI → layer마다 변동 —
per-layer (A,k,l)은 CSV `k/g/s` 컬럼, per-layer 𝓔는 CSV `enable_granularity`
컬럼으로 표현한다(계층 v1 S3).

## 2. Step 2 — hw JSON 작성

`model/hw/npu/acme-x1.json`처럼 새 파일 하나. `preset_name`이 내장 preset이 아니면
자동으로 **config-driven 모드**로 돈다(모든 값이 JSON에서). 전체 스키마 표는
`docs/reference/common/parameters.md`; 오타 키는 `[config] warning: unknown key`로
즉시 잡힌다(무음 무시 없음).

새 DRAM 등급이 registry에 없으면:
```json
"dram": { "tier": "bank", "banks": 8, "line_bytes": 64,
          "row_hit_cyc": 20, "row_miss_cyc": 45, "write_latency_cyc": 30 }
```
(cycle-정확 경로가 필요하면 `tier: dramsim3` + vendor `.ini`.)

## 3. Step 3 — network CSV

- **Mode 2 (auto, 기본)**: dims만 쓰면 descriptor를 §1의 값에서 유도.
- **Mode 1 (oracle)**: vendor 컴파일러가 `f_i,f_w,f_o,l_pass,n_max,n_min`을 직접
  줄 때 — 유도 없이 그대로 실행 (새 op/새 lowering의 탈출구이기도 함).
- 예시 쌍: `demo/networks/vgg16_mode_1.csv` / `vgg16_mode_2.csv`.

⚠️ **우선순위 함정 2개**:
1. CSV의 per-layer `k/g/s`는 hw JSON을 **오버라이드**한다. 다른 NPU용으로 만든
   CSV를 재사용하면 그 NPU의 mapping이 따라온다 — 새 NPU 첫 실행에선 CSV에서
   k/g/s 컬럼을 빼거나 값을 맞춰라.
2. 런타임 tile 입력은 `t_r/t_c` 컬럼이고(`tile_h/w`는 생성기용), **`model_spill:
   true`면 CSV tile을 무시하고 GB 용량에서 재유도**한다(의도된 spill 의미론).

## 4. Step 4 — 실행과 sanity 읽기

```bash
build/sim/flexnpusim -hw_conf model/hw/npu/acme-x1.json -network net.csv -o out.csv
```
- `@LAYER` 라인: F^I/F^K/F^O, total, compute, MACs, max_output_ratio(f16),
  avg_output_ratio(f17).
- **3지표 분해**로 병목 위치 판독: `PE utilization = duty × issue폭 × pass폭`
  — util = MACs/(cycles·k·n_max), duty = 시간 점유, avg_output_ratio = emit 폭
  (lockstep이면 항상 n_max가 정상), pass폭 = MACs/(F^O·k) (첫 레이어 C_in<k에서
  깎이는 게 정상 — VGG conv1_1 42%).
- roofline 감각: latency = max(memory, compute). config가 traffic을 바꿔도
  compute-bound 구간에선 latency가 안 움직인다 — 버그가 아니라 정의.
- per-tile 디버그: `FLEXNPUSIM_EMIT_TRACE=1` (tile별 in/wt/out·ideal·feed·emit·
  floor — floor가 그 레이어의 실효 𝓔).
- `-report`의 §5(Output Stream)가 partial output(F^O, accumulator 관측)과 boundary
  output write(final store + psum spill)를 분해한다. ⚠️ psum spill은 **기본
  40/40/20 예산에서는 구조적으로 발화 불가**(출력이 psum store를 넘을 크기면 출력
  예산도 넘어 단일-pass로 강제됨) — spill을 모델하려면 `*_partition_kb`로 출력
  예산을 크게, 입력/weight 예산을 작게 명시해 다중 pass를 유도해야 한다.

### 4b. 데이터 경로 분류표 (리포트의 byte가 어느 경로인지)

| ID | 경로 | 발동 조건 | 시간 모델 |
|---|---|---|---|
| I1 | input DRAM→GB, tile당 | WS/OS | ✓ |
| I2 | input 1회 상주 | IS | ✓ |
| I3 | 이전 레이어 출력 재fetch | 기본. `layer_fusion:"auto"`면 직전 출력이 GB에 잔존 ∧ 주소 연쇄일 때 겹치는 만큼 **GB-hit로 전환**(fetch 생략) | ✓ |
| W1 | weight 1회 상주 | WS ∧ working set이 GB에 듦 — 초과 시 per-tile로 **강등**(거짓 상주 없음, 용량-인지) | ✓ |
| W2 | weight tile마다 전량 재fetch | OS/IS, 또는 WS 강등 | ✓ |
| P1 | psum on-chip 상주 | out ≤ psum_store 또는 단일 pass | — |
| P3 | **psum DRAM 왕복** (write+reload) | model_spill ∧ 다중 pass ∧ out > psum_store (partition 분리 필요) | ✓ (양방향) |
| O1 | final output GB→DRAM | 무조건 | ✓ |
| O2 | 출력 on-chip forwarding (직전 1건) | `layer_fusion:"auto"` — I3의 GB-hit와 동일 사건의 출력측 표현 | ✓ (다음 레이어 fetch 생략) |

미모델: P4(GB-중간층 spill) — 다층 retention·eviction 정책과 함께 v2 범위.
**읽기·쓰기 모두 물리**(같은 AXI/DRAM 경로, wdma) — DRAM write 타이밍과 MC
R↔W turnaround가 실제로 발화한다. 단 roofline상 compute-bound 레이어에선 write
시간이 compute floor에 흡수되어 total에 안 보일 수 있다(정상 — memory-bound
구간에서만 표면화).

## 5. Step 5 — 검증·캘리브레이션 (참조가 있을 때)

**원칙: fit 금지, 구조 유도 우선.** NVDLA 선례: perf-model 산식을 하드웨어
상수만으로 <1% 재현(§9 입력이 정확하면 자동으로 나온다). RTL testbench 총
cycle 같은 아티팩트에 자유 파라미터를 맞추는 것은 interpolation이지 검증이
아니다(held-out에서 18% 실패한 전례).

1. compute 참조(perf 모델류) → `--field compute`로 대조:
   ```bash
   build/sim/flexnpusim ... 2> sim.log
   compare_reference.py(재생성 예정 — git 히스토리 97afdba) --log sim.log --ref ref.csv --field compute --tol-pct 2
   ```
2. 오차가 남으면 **overhead 2노브만** 보정: `issue_interval_q8`(지속 발행률,
   256=1.0), `packet_post_cycles`(레이어 고정 오버헤드) — 물리 근거를 적고 넣을 것.
3. end-to-end 실측 참조는 memory 환경을 동일하게 맞춘 뒤 `--field total`.
4. 참조가 없으면: 결과를 **상대 비교용**으로 명시하고 진행(절대 앵커는 NVDLA뿐).

## 6. 현재 한계 (이 가이드로 안 되는 것)

- **구조가 novelty인 NPU**(per-level 재사용·다층 on-chip·row-stationary·
  per-layer 재구성): 계층 v1 대상 — 현재 모델 범위 밖(로드맵: `docs/reference/npu/reuse-scheduling-gap.md`).
- AXI 폭은 런타임 노브(`axi.data_width_bits`, 단일 바이너리, 컴파일 최대
  `FLEXNPUSIM_AXI_MAX_DATA_W` 이하; full network ≥512b는 기존 버스트 버그로
  res3급만 검증), tree/multicast 분배 latency는 미연결, layout(CHWN/stride)
  미모델.
