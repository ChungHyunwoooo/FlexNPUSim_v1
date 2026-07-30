# 실제 하드웨어 동작 → 파라미터 모델링

## 왜 "파라미터"인가

NPU 시뮬레이터는 한 레이어가 몇 사이클 걸리는지 구한다. 데이터가 DRAM에서 온칩 버퍼로
오는 **메모리 쪽**은 사이클마다 실제로 흉내 낸다. 그러나 PE 배열이 곱셈-덧셈을 반복하는
**연산 쪽**을 사이클마다 흉내 내면 느리다. 대신 연산의 진행을 **여섯 개의 숫자**로 요약한다.
이 문서는 그 여섯 숫자가 가속기의 **어떤 물리적 동작에서 나오는지**를 실제 가속기로 하나씩
보인다. 여섯이 뭔지 미리 알 필요는 없다 — 아래에서 PE 하나부터 쌓으면 저절로 나온다.

예시 레이어는 `one_conv` 하나로 고정한다: 입력 16×16(세로·가로)×64(채널), 3×3 커널,
필터 128개, stride 1, 패딩 없음. 출력은 (16−3)/1+1 = 14, 즉 14×14×128.
- **출력 원소 수 |O|** = 14·14·128 = **25,088**.
- **출력 하나에 필요한 곱셈-누산(MAC) 수 n̄** = Kh·Kw·Cin = 3·3·64 = **576**
  (출력 픽셀 하나 = 3×3 커널 × 64 채널 = 576개 곱의 합).

## 하드웨어를 바닥부터 쌓기 — 여기서 6개 숫자가 나온다

**PE(processing element)** 하나는 한 사이클에 MAC 하나를 한다. 출력 하나가 576 MAC이니 PE
하나로는 576 사이클. 그래서 하드웨어는 PE를 세 층으로 묶는다.

**① k개 PE = PEG (PE group).** 한 번에 k개 MAC을 함께 받는다. 이 "한 번에 받는 k개 MAC
덩어리"를 **pass**라 부른다. 출력 하나(576 MAC)를 k개씩 나누면
→ **pass 수 p = ⌈n̄/k⌉** [첫째 숫자에 필요]. (NVDLA k=64면 9번, Gemmini k=16이면 36번.)

**PEG를 어떻게 배치하냐가 결과 지연을 정한다.** k개 MAC이 한 pass로 들어가 결과가 나오기까지
걸리는 사이클을 **l (pipeline latency)**라 한다.
- **시스톨릭**: k개 PE가 한 줄로 서서 데이터를 옆으로 밀며 누산 → 마지막 PE까지 k사이클 → l=k.
- **애더 트리**: k개 곱을 동시에 하고 트리로 합 → l = 1+⌈log₂k⌉.
→ **l** [둘째 숫자].

**② g개 PEG = PES (PE set).** g개 PEG가 한 출력에 대해 **함께(lock-step) 움직인다** — 한
사이클에 g개 pass가 통째로 나가거나 아무것도 안 나간다(부분집합 불가). 그래서 한 사이클에
낼 수 있는 최소 진행 = g.
→ **n_min = g** [셋째 숫자].

**③ s개 PES = 배열.** s개 PES가 각자 다른 출력을 동시에 만든다 → 한 사이클 최대 g·s개 출력.
→ **n_max = g·s** [넷째 숫자].

남은 둘은 "데이터가 얼마나 오가나"다.

**출력 이벤트 예산 F^O.** pass 하나가 끝날 때마다 진행을 1 센다. 전체 출력의 전체 pass 수 =
p·|O|. (한 출력의 부분합을 p번 갱신하니까.)
→ **F^O = p·|O|** [다섯째 숫자].

**입력·가중치 트래픽 F^I, F^W.** 온칩 버퍼에서 PE로 전달되는 입력/가중치 operand의 수.
어느 오퍼랜드를 온칩에 고정하느냐(**dataflow**)가 이 둘을 가른다 — §4에서 실측으로 보인다.
→ **F^I, F^W** [여섯째].

**정리.** descriptor = **(F^I, F^W, F^O, l, n_min, n_max)**. 이 여섯이 하드웨어 사실
`(k, g, s, PEG배치, dataflow, 버퍼)`와 레이어 shape에서 탐색 없이 나온다. 코드 필드명은
각각 `fetch_operand0, fetch_operand1, write_output, issue_latency, n_min, n_max`이고
하드웨어 입력은 `ops_per_pass(k), num_output_lanes(s), parallel_passes(g), pe_type`이다.
아래 §1~§9가 하나씩 실제 가속기로 근거화하고, 손계산 recipe(f유도 문서 §6.5)로 닫는다.

---

## 1. k와 PEG 배치 → pass 수 p, 파이프라인 지연 l (위 ①의 상세)

pass p = ⌈n̄/k⌉이고 l은 PEG 배치에서 나온다(코드 `dnn_image_compiler.cpp:35`
`Systolic→k`, `AdderTree→1+⌈log₂k⌉`). one_conv(n̄=576):

| 가속기 | 실제 구조 | pe_type | k | pass ⌈576/k⌉ | **l** |
|---|---|---|---|---|---|
| NVDLA nv_full | 64-입력 애더트리(Atomic-C=64) | adder_tree | 64 | 9 | **7** |
| Gemmini 16×16 | 16-길이 시스톨릭 행 | systolic | 16 | 36 | **16** |
| TPU-v1 | 256×256 시스톨릭 | systolic | 256 | 3 | **256** |
| MIDAP | 64-입력 애더트리 | adder_tree | 64 | 9 | **7** |
| MAERI | 8-입력 애더트리 | adder_tree | 8 | 72 | **4** |

**교훈**: NVDLA와 MIDAP은 **같은 64-입력 애더트리(k=64, l=7)**라 배치·l이 동일하고 **오직
dataflow만** 다르다(NVDLA=WS, MIDAP=IS, §4). l이 배치로 갈리는 대비는 **systolic vs 애더트리**에서
본다 — systolic(TPU·Gemmini)은 l=k, 애더트리(NVDLA·MIDAP·MAERI)는 l=1+⌈log₂k⌉. 애더트리가 l을
log로 줄인다.

## 2. 병렬 pass g → 최소 진행 n_min, pass-step

**실제 동작.** PES는 g개 PEG를 **lock-step**으로 구동한다. 한 output에 대해 한 cycle에
g개 pass가 통째로 나가거나 아무것도 안 나간다 — g의 부분집합만은 불가능하다.

**그래서** n_min = g, pass-step = ⌈p/g⌉(g개 pass가 한 stored result로 묶임).
- **우리 7개 config는 모두 단일 배열(g=1) → n_min=1.** 한 배열을 g개의 독립 reduction 그룹으로
  쪼갠 config에서만 g>1 → n_min=g가 되어, "한 cycle에 최소 g개 pass를 통째 발행"이라는 lock-step
  제약이 생긴다.

## 3. lane 수 s → 최대 진행 n_max

**실제 동작.** s개 PES가 각자 다른 output을 독립 생성 → 한 cycle에 최대 g·s개 출력.

**그래서** n_max = g·s: NVDLA/Gemmini/Eyeriss(1×16)=16, MAERI(1×32)=32, TPU(1×256)=256,
Nullhop(1×8)=8.

## 4. Dataflow(상주 오퍼랜드) → 트래픽 F^I, F^W  — 실측 증거

**실제 동작.** 데이터플로우 = "어느 오퍼랜드를 PE 근처에 고정하느냐":
- **Weight-stationary**(NVDLA, Gemmini, TPU): weight를 고정 → **weight 한 번 로드, input
  계속 흘림.**
- **Input-stationary**(MIDAP, Eyeriss): feature를 온칩 고정 → **input 한 번 로드, weight
  계속 흘림.**

**그래서** 같은 one_conv, 같은 F^O인데 트래픽이 **뒤집힌다**. 아래는 시뮬레이터 실제
출력이다.

> **여기서 '오라클'을 처음 만난다.** 검증용 워크로드 CSV(one_conv 등)는 레퍼런스
> 하드웨어의 descriptor 값 — `pe_type, k, g, s`, 그리고 psum 용량 — 을 각 레이어 행에
> **직접 적어둔다.** 왜? NVDLA RTL 같은 레퍼런스와 사이클을 비교하려면 그 하드웨어의
> descriptor를 정확히 재현해야 하는데, hw config에서 유도하다 어긋나기보다 검증된 값을
> 못박는 게 안전하기 때문이다. 이렇게 CSV가 직접 실은 값을 **오라클(oracle)**이라 한다.
> 그래서 one_conv에서는 k·g·s가 CSV값(16,1,16)으로 고정되고 hw JSON의 k(NVDLA는 64)는
> 무시된다 — 즉 아래 표의 F^O·n_max는 전 config 동일하다. **하지만 dataflow는 hw JSON이
> 정하므로**, 오라클이 k를 고정해도 F^I/F^W는 config별로 갈린다(그게 아래 표의 핵심).
> 일반 DSE 워크로드는 이 오라클 컬럼이 없어 hw config에서 유도한다(§8a).

| dataflow | 가속기 | **F^I** | **F^W** | 해석 |
|---|---|---|---|---|
| WS | NVDLA·Gemmini·TPU·MAERI·Nullhop·Simba | **894,464** | **73,728** | weight 한 번(=\|W\|=3·3·64·128), input 재흘림 |
| IS | MIDAP·Eyeriss | **112,896** | **951,808** | input 한 번(=im2col base 14·14·64·9), weight 재적재 |

- WS의 F^W = 73,728 = **정확히 \|W\|** = Kh·Kw·Cin·Cout (weight 딱 한 번 로드).
- IS의 F^I = 112,896 = **im2col base**. (im2col = 출력마다 자기 3×3×64 수용영역을 읽는 것.
  수용영역이 겹쳐서, 상주 입력을 "한 번" 공급해도 PE에 전달되는 총 operand 수는 원입력
  16·16·64=16,384가 아니라 전개된 Ho·Wo·Cin·Kh·Kw = 14·14·64·9 = **112,896**이다. F^I은
  "PE에 전달된 operand 수"라서 이 전개값을 센다 — DRAM에서 읽은 바이트 수가 아니다.)

같은 워크로드·같은 descriptor(k,g,s)에서 오직 hw JSON의 dataflow만 바꿔도 F^I·F^W가
뒤집힌다 = "dataflow가 트래픽을 결정한다"의 직접 증거. 식으로는 F^I = Ī + δβ_I·F^P,
F^W = W̄ + δβ_W·F^P에서 상주 여부가 Ī·W̄(1회 base vs 재fetch 합)를 가른다.

**단, 위 숫자는 descriptor k=16(오라클) 기준이다. dataflow는 split(어느 쪽이 상주=1회냐)을
정하고, k는 흘려지는 쪽의 magnitude를 정한다** — 흘려지는 오퍼랜드의 delivery는 pass 수
⌈n̄/k⌉에 비례하므로, 같은 WS라도 k=64면 F^I=217,088로 준다(§9). 상주 오퍼랜드(WS의 F^W,
IS의 F^I)만 k와 무관하게 1회 base에 머문다.

## 5. Accumulator 거처 → 부분출력 spill 비율 δ

**실제 동작.** 한 pass로 감축이 안 끝나면 pass-step 사이에 부분출력 F^P가 생긴다. 이걸
어디 두느냐:
- **NVDLA CACC**(전용 convolution accumulator): 전체 감축 동안 부분합 온칩 유지 → **DRAM
  왕복 없음, δ=0**. Output-stationary(MIDAP accumulator)도 δ=0.
- accumulator가 넘치면 → **spill 후 재적재, δ>0**.

**그래서** F^P는 항상 pass-step에서 생기지만, **δ=0이면 F^I·F^W에 reload가 더해지지
않는다**. §4 표가 δ=0(CACC/accumulator 상주) 경우라 WS의 F^W가 재적재 없이 \|W\|에 머문다.

## 6. Sparsity(zero-skip) → 레이어 평균 n̄만 감소

**실제 동작.** Nullhop은 0인 activation의 MAC을 건너뛴다 → 출력별 n(o)가 달라진다.

**그래서** sparsity는 **오직 레이어 평균 n̄ = (1/|O|)Σn(o) 한 스칼라로만** 들어간다.
패턴을 cycle마다 동적 시뮬레이션하지 않는다 — n̄가 줄면 p=⌈n̄/k⌉가 줄고 F^O가 줄 뿐,
나머지 유도는 동일. (input-dependent skipping이 모델에 들어가는 유일한 지점.)

## 7. 안 바뀌는 것 — bus width / DRAM tier

**실제 동작.** 128b vs 256b AXI, DDR4 vs LPDDR4는 **memory-side** timing — DRAM→GB 도착
시각을 바꾼다.

**그래서** 6개 compute 파라미터는 bus/DRAM과 **무관**하다. 그 효과는 GB read interface의
per-cycle delivery timing(a_i(t), a_w(t))으로만 들어간다. 이게 memory-side(cycle-accurate)
/ compute-side(파라미터) **분리**의 핵심이다. (실측: bus 폭·tier를 바꿔도 MAC·F^O 불변,
cycle만 변함.)

---

## 8a. 하드웨어 구조 → 유도 파라미터 (순수 유도, k에 따라 변함)

각 가속기의 **hw config (k,g,s,pe_type)에 §1~3 공식을 적용한 값**. one_conv shape로
계산(|O|=25,088, n̄=576). 이건 "이 하드웨어면 이렇게 유도된다"의 순수 계산이다.

| 가속기 | pe_type | k | g | s | pass ⌈576/k⌉ | pass-step ⌈p/g⌉ | **l** | **n_min** | **n_max** | **F^O=p·\|O\|** (코드) |
|---|---|---|---|---|---|---|---|---|---|---|
| NVDLA nv_full | adder_tree | 64 | 1 | 16 | 9 | 9 | 7 | 1 | 16 | **225,792** |
| Gemmini | systolic | 16 | 1 | 16 | 36 | 36 | 16 | 1 | 16 | **903,168** |
| TPU-v1 | systolic | 256 | 1 | 256 | 3 | 3 | 256 | 1 | 256 | **75,264** |
| MAERI | adder_tree | 8 | 1 | 32 | 72 | 72 | 4 | 1 | 32 | **1,806,336** |
| MIDAP | adder_tree | 64 | 1 | 16 | 9 | 9 | 7 | 1 | 16 | **225,792** |
| Eyeriss-v2 | systolic | 16 | 1 | 16 | 36 | 36 | 16 | 1 | 16 | **903,168** |
| Nullhop | systolic | 16 | 1 | 8 | 36 | 36 | 16 | 1 | **8** | **903,168** |

큰 k(TPU=256)는 pass를 줄여 F^O를 줄인다. k=16인 Gemmini·Eyeriss·Nullhop은 F^O=903,168로
오라클 실측과 일치한다.

> **⚠ 논문↔코드 불일치 (f유도에서 해결 필요).** 코드는 **F^O = p·\|O\|** (pass 완료 수,
> `dnn_image_generator.cpp:1163`, v4 eq 5 f_o=\|P_pkt\|와 일치). 하지만 LaTeX §3.4는
> **F^O = \|O\| + F^P = h·\|O\|** (pass-STEP 수)로 정의한다. g=1이면 h=p라 같지만,
> **g>1이면**(예: 한 배열을 g=2로 쪼갠 config: p=36 → h=⌈36/2⌉=18) **코드 903,168(p·\|O\|) vs
> LaTeX 451,584(h·\|O\|)로 갈린다.** 우리 7개 config는 모두 g=1이라 실제로는 일치하지만, 두 정의
> (pass 완료 vs pass-step)를 통일해야 한다. 자세한 건 f유도 문서 참조.

## 8b. Dataflow → 트래픽 (one_conv 실측, descriptor 오라클 k,g,s=16,1,16 고정)

| dataflow | 가속기 | **F^I** | **F^W** | **F^O** | l/n_min/n_max |
|---|---|---|---|---|---|
| WS | NVDLA·Gemmini·TPU·MAERI·Nullhop·Simba | 894,464 | 73,728 | 903,168 | 16 / 1 / 16 |
| IS | MIDAP·Eyeriss | 112,896 | 951,808 | 903,168 | 16 / 1 / 16 |

**8a와 8b의 차이가 핵심이다**: 8a는 hw 구조에서 순수 유도한 값(k별로 다름), 8b는
one_conv 검증 CSV로 실행한 실측값(오라클이 k=16으로 고정 → l·n_max 전부 16). **논문은
어느 층위의 수치인지 명시해야 한다** — 검증 표의 descriptor는 오라클 유래이고, DSE
스윕의 descriptor는 hw 유도 유래다.

## 9. 한 레이어 완전 유도 — NVDLA on one_conv (순수 hw 유도)

"실제 동작 → 파라미터"를 hw config(오라클 무시)로 한 번에 이어 본다.

1. **워크로드**: one_conv → 출력 14×14×128, |O| = **25,088**, n̄ = 3·3·64 = **576**.
2. **하드웨어 (NVDLA nv_full)**: Atomic-C=64 → k=64; PES 1개 → g=1; output lane(Atomic-K) 16 → s=16;
   CMAC **애더트리** 배치.
   → pass p = ⌈576/64⌉ = **9**, pass-step h = ⌈9/1⌉ = **9**.
   → **n_min = g = 1**, **n_max = g·s = 16**, **l = 1+⌈log₂64⌉ = 7**(애더트리).
3. **부분출력**: F^P = (h−1)·|O| = 8·25,088 = **200,704** → F^O = |O|+F^P = **225,792**.
4. **트래픽 (WS)**: weight 상주 → F^W = |W| = **73,728**(1회). input은 비상주 →
   F^I = base(=raw input 16×16×64 = 16,384, 1회) + δ·(p−g)·|O|. `one_conv.csv`가
   `psum_kb=0`을 오라클로 실어 **δ=1**(로컬 psum 없음 → 전량 writeback), g=1이므로
   reload=(p−1)·|O| = (9−1)·25,088 = 200,704 → **F^I = 16,384 + 200,704 = 217,088**(실측).
   → **F^I는 k에 의존**: reload가 pass 수 p=⌈n̄/k⌉에 비례(k=16이면 878,080, k=64면
   200,704). dataflow는 split(어느 쪽이 상주=1회냐)을, k는 비상주 쪽 reload magnitude를
   정한다. (주의: δ=1은 이상적 CACC의 δ=0이 아니라 CSV 오라클 psum=0에서 온 것. hw
   config의 psum=16KB만 쓰면 이 레이어는 δ=0이 된다 — §5·f유도 참조.)
5. **descriptor**: (f_i, f_w, f_o, l, n_min, n_max) = **(217,088, 73,728, 225,792, 7, 1, 16)**
   — 전부 시뮬레이터 실측(k=64로 one_conv 실행). 이 여섯 값 + GB read interface의
   per-cycle delivery로 §5 latency reconstruction이 이 레이어 cycle을 복원한다.

**대조 (오라클 k=16)**: 원본 `one_conv.csv`로 돌리면 descriptor가
**(894,464, 73,728, 903,168, 16, 1, 16)**이 된다(§8b). k만 16↔64로 바뀌어도 F^I(894,464↔
217,088)와 F^O(903,168↔225,792)가 함께 움직인다 — **F^I(reload)·F^O는 pass 구조(k)에
결합돼 있고, 상주 오퍼랜드 F^W(73,728)만 k와 무관하다.**

**요지**: 2~4의 모든 값이 (k,g,s,배치,dataflow,accumulator)라는 하드웨어 사실과 레이어
shape에서 **탐색 없이** 나온다. 이것이 "고정 데이터플로우 → 단일 매핑 → 닫힌 형태
파라미터"의 구체적 실현이며, 논문 §3(파라미터 유도)의 산문은 이 인과 사슬을 그대로 쓰면
된다.
