# F 파라미터 유도 — 코드 검증 + 논문 대조 (f유도)

이 문서는 트래픽 파라미터 F^I, F^W, F^O를 **시뮬레이터 코드가 실제로 어떻게
계산하는지** 라인 단위로 확정하고, 그 공식을 논문(LaTeX §3.4 / v4 §4.3 / f-model)의
유도와 대조한다. 모든 숫자는 실측(one_conv, 필요 시 k·g 변형)으로 검증했다.
컴퓨트-org 파라미터(l, n_min, n_max)는 `hw-derivation-examples.md` 참조.

워크로드 one_conv: 입력 16×16×64, 3×3, 128 필터 → 출력 14×14×128, |O|=25,088, n̄=576.

## 0. 코드가 실제로 계산하는 공식 (검증 완료)

| 파라미터 | 코드 공식 (`dnn_image_generator.cpp`) | 줄 |
|---|---|---|
| pass p | p = ⌈n̄ / k⌉ = ⌈ops_per_output / ops_per_pass⌉ | 1160 |
| **F^O** | **F^O = p · \|O\|** | 1163 |
| 상주 오퍼랜드 | WS: F^W = \|W\| ; IS: F^I = Ho·Wo·Cin·Kh·Kw (im2col) | 1193, 1204 |
| 비상주 오퍼랜드 base | tile-summed refetch (`schedule.total_layer_*_bytes`) | 1195, 1208 |
| **δ (spill 비율)** | psum 없으면 1; concurrent>psum_cap이면 1−psum_cap/concurrent; 아니면 0 | 1222–1235 |
| **reload(writeback)** | δ · **(p − g)** · \|O\|, 비상주 오퍼랜드에 가산 | 1240–1247 |

즉 코드의 완전한 형태:

```
F^O          = p·|O|                         (p = ⌈n̄/k⌉)
F^(resident) = |resident once|               (WS→|W|, IS→im2col)
F^(streamed) = base_tile_fetch + δ·(p−g)·|O| (β 라우팅: WS→input, IS→weight)
```

## 1. F^O = p·|O| — pass 완료 수 (pass-step 아님)

**코드**(1160,1163): `passes_per_output = ⌈ops_per_output/ops_per_pass⌉`,
`write_output = passes_per_output · out_h·out_w·out_c`. 즉 F^O = p·|O|.

**검증**:
- one_conv (k=16, g=1): p=⌈576/16⌉=36 → 36·25,088 = **903,168** ✓ (실측).
- k=64: p=9 → 9·25,088 = **225,792** ✓.
- **g=2로 변형 실행**: F^O = **903,168** (= p·|O|), **451,584(=h·|O|) 아님** ✓.

**논문 대조 — 불일치 ①**:
- v4 §4.2 eq 5: f_o = |P_pkt| = pass 수 → **p·|O|. 코드와 일치.** ✓
- LaTeX §3.4: F^O = |O| + F^P, F^P = (⌈p/g⌉−1)|O| = (h−1)|O| → **F^O = h·|O|.**
- g=1이면 h=p라 같지만 **g>1이면 다르다**: Nullhop(g=2) 코드 903,168 vs LaTeX 451,584.
- → **F^O를 "pass 완료 수(p·|O|)"로 통일**해야 한다(latency model이 누적하는 게 pass
  완료이므로 코드/v4가 맞다). LaTeX의 |O|+F^P 표기는 pass-step(stored result) 수라
  F^O가 아니라 G^ps+G^out(=h·|O|)에 해당한다.

## 2. 상주 오퍼랜드 = 1회 로드 (dataflow가 결정)

**코드**(1191–1215): WS면 weight 상주 → `F^W = Kh·Kw·Kc·Kcount = |W|`; IS면 input 상주 →
`F^I = Ho·Wo·Cin·Kh·Kw`(im2col footprint, halo 전개).

**검증**:
- WS(NVDLA): F^W = **73,728** = 3·3·64·128 = |W| ✓ (weight 딱 1회).
- IS(MIDAP): F^I = **112,896** = 14·14·64·9 = im2col base ✓ (feature 딱 1회).

**논문 대조**: LaTeX/v4 모두 상주 오퍼랜드를 1회 base로 둔다 — 일치. (단 IS의 input은
im2col 전개 크기를 써야 함이 코드 주석 1198–1203의 핵심; 물리 H·W·C가 아니라 P·Q·C·R·S.
이 통일이 과거 IS F^I 오류의 수정이었다 — [[validation-program-direction]].)

## 3. 비상주 오퍼랜드 = tile refetch + writeback-reload

**코드**(1195/1208 + 1237–1247): 비상주 오퍼랜드 = base(tile-summed refetch) +
**δ·(p−g)·|O|**. reload는 WS면 F^I(input)에, IS면 F^W(weight)에 가산(β 라우팅, 1242–1247).

**검증 (WS, NVDLA one_conv, 오라클 psum=0 → δ=1, g=1)**:
- base = raw input 16×16×64 = 16,384 (1 tile, GB에 맞음).
- reload = (p−1)·|O|: k=16 → 35·25,088 = 878,080; k=64 → 8·25,088 = 200,704.
- F^I = 16,384 + reload → k=16: **894,464** ✓, k=64: **217,088** ✓.

**검증 (IS, MIDAP one_conv, δ=1, g=1)**:
- F^W = base |W| 73,728 + reload 878,080 = **951,808** ✓ (실측).

**논문 대조 — 불일치 ②**:
- 코드 reload = **(p − g)·|O|** (f-model의 `reload = max(0, pass−g)`와 일치).
- LaTeX §3.4: writeback 항에 F^P = (⌈p/g⌉−1)|O| = (h−1)|O| 사용.
- g=1이면 p−1 = h−1이라 같지만 **g>1이면 (p−g) ≠ (h−1)**: 예 p=36,g=2 → 코드 34, LaTeX 17.
- → reload를 **(p−g)**로 통일(코드/f-model)하거나, LaTeX의 (h−1) 유도로 코드를 바꿔야 한다.
  현재 코드는 f-model 유도를 따른다.

## 4. δ (spill 비율) = 로컬 psum 용량 vs concurrent outputs

**코드**(1222–1235): p>g일 때 psum_cap = partial_sum_buffer_kb·1024/4,
concurrent = min(|O|, n_max). psum_cap=0이면 δ=1; concurrent>psum_cap이면
δ=1−psum_cap/concurrent; 아니면 δ=0.

**검증**: nvdla-large hw config는 psum=16KB(=4096 elems), concurrent=min(25088,16)=16 <
4096 → **hw만 쓰면 δ=0**. 그런데 `one_conv.csv`가 `psum_kb=0`을 오라클로 실어
partial_sum_buffer_kb=0 → **δ=1**(전량 writeback). 이래서 §3의 reload가 붙었다.

**함의**: "NVDLA CACC → δ=0"은 이상적 하드웨어 사실이고 hw config로도 δ=0이지만,
**검증 워크로드 CSV의 오라클 psum=0이 δ=1로 덮어쓴다.** 논문의 트래픽 표가 어느 δ
(hw 유도 δ=0 vs 오라클 δ=1)로 계산됐는지 명시해야 한다.

## 5. 두 레벨: array-feed(SRAM) vs 물리 DRAM traffic

논문이 스스로 경고하는 "두 레벨"이 여기서 갈린다. 코드의 F^I(WS) =
base_tile_fetch(GB 경계 트래픽, DRAM급) + reload(psum writeback). v4 §4.3은 이를
F_GB^x(DRAM→GB)와 F_LM^x(GB→LM array-feed)로 **분리**하고, descriptor의 f_i = |Q_i|는
GB→LM delivery 수다. LaTeX §3.4는 F^I 하나로 **혼동**한다. 제출 시 F^I가 어느 레벨인지
(array-feed delivery인지 boundary traffic인지) 못박아야 한다 — latency model의 입력은
array-feed delivery(f_i=|Q_i|)이고, DRAM traffic은 별도(-report의 rd/wr_bytes).

## 6. Worked example — NVDLA WS on one_conv (end-to-end, 전 숫자 검증)

| 단계 | 값 | 출처 |
|---|---|---|
| 워크로드 | \|O\|=25,088, n̄=576 | shape |
| pass p = ⌈576/16⌉ | 36 | 코드 1160 |
| **F^O = p·\|O\|** | **903,168** | 코드 1163, 실측 ✓ |
| 상주 weight F^W = \|W\| | **73,728** | 코드 1193, 실측 ✓ |
| δ (오라클 psum=0) | 1.0 | 코드 1229 |
| reload = (p−1)·\|O\| | 878,080 | 코드 1240 |
| base input (1 tile) | 16,384 | 코드 1195 |
| **F^I = base + δ·reload** | **894,464** | 실측 ✓ |
| **descriptor** | (894,464, 73,728, 903,168, l, n_min, n_max) | |

## 6.5 SystemC 없이 손으로 유도하는 완전 recipe

시뮬레이터를 돌리지 않고 shape + hw 파라미터만으로 F^I·F^W·F^O를 손계산한다. 유일하게
비자명한 부분(비상주 base tile-fetch)은 tile schedule(`tile_descriptor.cpp`)을 손으로
전개해 얻는다.

### 입력 (one_conv, NVDLA WS)
H_in=16, W_in=16, C_in=64, Kh=Kw=3, C_out=128, stride=1, pad=0, tile=14×14,
GB=512 KB, dataflow=WS, k=16, g=1, psum(오라클)=0. 원소 단위는 float(4 B).

### Step 1 — 출력 shape와 |O|
Ho = ⌊(H_in + 2·pad − Kh)/stride⌋ + 1 = (16−3)/1 + 1 = **14**, Wo = 14.
|O| = Ho·Wo·C_out = 14·14·128 = **25,088**.

### Step 2 — 출력당 unit op n̄
컨볼루션 n̄ = Kh·Kw·C_in = 3·3·64 = **576**.

### Step 3 — pass 수
p = ⌈n̄/k⌉ = ⌈576/16⌉ = **36**.

### Step 4 — F^O
**F^O = p·|O| = 36·25,088 = 903,168.**

### Step 5 — 타일 전개 (tile_descriptor.cpp 손계산)
- 타일 수: num_tiles = ⌈Ho/tile_h⌉·⌈Wo/tile_w⌉ = ⌈14/14⌉·⌈14/14⌉ = **1** (출력 전체가 1 타일).
- 타일의 입력 수용영역: input_h = tile_out_h·stride + Kh − 1 = 14·1 + 3 − 1 = **16**, input_w = 16.
- 채널 분할(pass) 여부: output_bytes = tile_out_h·tile_out_w·C_out·4 = 14·14·128·4 = 1,003,520 B.
  output budget = GB·20% = 512·1024·0.2 = 104,857 B. output_bytes > budget →
  **채널 분할 없이 1 pass**(코드 fallback, line 122–124), 즉 64 채널 한 번에.
  (일반식: output이 budget에 맞으면 max_ch/pass = min(input_budget,kernel_budget)/
  (input_bytes_per_ch + kernel_bytes_per_ch), passes = ⌈C_in/max_ch⌉.)

### Step 6 — 비상주 base tile-fetch (Σ 타일 Σ pass)
1 타일·1 pass이므로:
- input fetch = input_h·input_w·C_in = 16·16·64 = **16,384** (operand).
- kernel fetch = Kh·Kw·C_in·C_out = 3·3·64·128 = **73,728** (= |W|).

일반식: base_input = Σ_tiles Σ_passes (input_h_tile · input_w_tile · ch_this_pass),
base_kernel = Σ_tiles Σ_passes (Kh·Kw·ch_this_pass·C_out). 타일이 여럿이면 수용영역 halo와
채널 재읽기가 refetch로 누적된다. one_conv은 1 타일이라 base = 원입력·|W|.

### Step 7 — 상주 vs 스트림 (dataflow가 라우팅)
WS → weight 상주(1회): **F^W_base = |W| = 73,728**. input 스트림: F^I_base = 16,384.
(IS면 반대: F^I_base = im2col = Ho·Wo·C_in·Kh·Kw = 14·14·64·9 = 112,896을 1회,
F^W_base = |W| = 73,728을 스트림.)

### Step 8 — spill δ
p(36) > g(1)이고 psum(오라클)=0 → **δ = 1** (로컬 psum 없음 → 전량 writeback).
(hw psum=16 KB면: concurrent = min(|O|, n_max)=min(25088,16)=16 ≤ 4096 = 16 KB/4 →
**δ=0**. 즉 δ는 psum 출처에 좌우된다.)

### Step 9 — reload (writeback-and-reload)
reload = δ·(p − g)·|O| = 1·(36 − 1)·25,088 = **878,080**. WS는 스트림 오퍼랜드(input)에
가산, IS는 weight에 가산.

### Step 10 — 최종 조립
- **F^I = F^I_base + reload = 16,384 + 878,080 = 894,464.**
- **F^W = 73,728** (상주, reload 없음).
- **F^O = 903,168.**
- descriptor = (894,464, 73,728, 903,168; l, n_min, n_max) — 전부 시뮬레이터 실측과 일치.

**IS(MIDAP) 손계산 대조**: F^I = 112,896(im2col, 상주 1회), F^W = 73,728 + 878,080 =
951,808, F^O = 903,168 — 실측과 일치.

### 손유도 공식 요약 (컨볼루션)
```
Ho = (H_in+2·pad−Kh)/stride + 1     |O| = Ho·Wo·C_out     n̄ = Kh·Kw·C_in
p  = ⌈n̄/k⌉                          F^O = p·|O|
base_input  = Σtiles Σpass (input_h_tile·input_w_tile·ch)   # 1타일이면 H_in·W_in·C_in
base_weight = |W| = Kh·Kw·C_in·C_out
δ = (psum=0)?1 : (concurrent>psum_cap)?1−psum_cap/concurrent : 0
reload = δ·(p−g)·|O|
WS:  F^W=base_weight,           F^I=base_input + reload
IS:  F^I=Ho·Wo·C_in·Kh·Kw,      F^W=base_weight + reload
```

## 7. 제출 전 해결해야 할 논문↔코드 불일치 (요약)

| # | 항목 | 코드 (실측) | LaTeX §3.4 | v4/f-model | g=1 | g>1 |
|---|---|---|---|---|---|---|
| ① | **F^O** | **p·\|O\|** | \|O\|+F^P = h·\|O\| | v4 eq5=p·\|O\| ✓ | 같음 | **다름** |
| ② | **reload** | **(p−g)·\|O\|** | (h−1)·\|O\| | f-model=(p−g) ✓ | 같음 | **다름** |
| ③ | **F^I 레벨** | tile+reload (혼합) | 단일 F^I | v4 F_GB/F_LM 분리 | — | — |
| ④ | **δ provenance** | 오라클 psum=0→1 | 서술 없음 | hw psum→0 | — | — |

**권고**: 코드는 일관되게 **f-model 유도**(F^O=p·|O|, reload=(p−g)|O|)를 따른다. LaTeX
§3.4의 (|O|+F^P, F^P=(h−1)|O|) 서술을 **코드/v4/f-model에 맞춰 F^O=p·|O|,
reload=(p−g)|O|로 통일**하고, F^I를 array-feed(f_i=|Q_i|) 레벨로 명시하며, 트래픽 표의
δ가 hw 유도(psum 반영)인지 오라클(psum=0)인지 캡션에 적으면 f유도가 코드와 정합한다.
이 넷을 맞추면 유도 서술이 시뮬레이터 출력과 line-by-line 일치한다.
