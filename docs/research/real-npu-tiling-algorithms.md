# 실기 NPU 9종의 Conv/Matmul 타일링 · loop-nest (grounding, 2026-07-11)

Phase D tile-calc 알고리즘을 실기에 맞추기 위한 조사. 각 설계를 1차 출처(논문·공식
스펙·시뮬레이터 소스)로 grounding. 모델 추상화: *(dataflow ∈ {WS,IS,OS}, reduction
width `k`, output-parallel width `m`, on-chip buffer 용량)*.

Loop-nest 표기(7-D conv): `N`=batch, `K`=출력채널, `C`=입력채널(감축), `P×Q`=출력
공간, `R×S`=필터 공간. resident = on-chip 유지·재사용, streams = 매 스텝 재-read.

## 요약 마스터 표

| 설계 | Dataflow(모델 class) | resident | streams | 타일 loop + 순서 | 감축 C | partial sum | k / m / buffer |
|---|---|---|---|---|---|---|---|
| **NVDLA** | WS(stripe)+OS(accum) | 가중치(stripe)+psum | inputs | K-grp→plane(stripes)→C(channel op)→R·S→64-atomic | ⌈C/64⌉ blocks | **resident** CACC, full C 후 unload | 64 / 16(32) / CBUF 512KB |
| **Eyeriss v1** | RS(energy-balanced) | 필터 row(spad) | ifmap row; psum 이동 | R×E 매핑; C/K replicate; C,K,N fold via GLB | spatial(PE열)+temporal fold | 열 내 + **GLB**, GLB 초과 시만 DRAM spill | 1D-row / e·p / GLB 108KB |
| **Eyeriss v2** | RS+(flexible) | 필터 row; NoC 라우팅 | iact/psum mesh | 임의 dim(incl. G) spatial tile; per-layer multicast/unicast | RS fold, compact은 fold↓ | 열 내 + cluster GLB psum bank | SIMD-2 / flexible / GLB ~192KB |
| **TPU v1** | **WS** | 256×256 weight tile | act row; psum↓ | N_out/256 → C/256(weight tile load) → M stream | ⌈K/256⌉ weight tiles | **resident** 4MiB acc | 256 / 256 / UB 24MiB |
| **Gemmini** | **WS or OS** | WS:weight tile / OS:output tile | 나머지 둘 | I(out-row)→J(out-col)→K(reduce), DIM 단위 | K-tiles | **resident** accumulator(`ex_accumulate`) | 16 / 16 / spad 256KB + acc 64KB |
| **MAERI** | flexible(RS+WS+OS-node) | 가중치(MS) | act multicast | VN 크기 + VN replicate onto 고정 MS | fan-in&gt;MS면 fold-over-rows | **spill→Prefetch Buffer**, 재feed | VN-size / #VN / PB 80KB |
| **MIDAP** | **IS(feature-map-stationary)** | 입력 FM(FMEM) | 가중치(WMEM) | K-grp→x→y→kx→C(64-chunk); FM **width x** tile | full C 제자리, mem split 없음 | **resident** adder-tree accum | 64 / 16 / FMEM 1MB + WMEM 256KB |
| **NullHop** | **IS + zero-skip**(kernel WS) | 커널; 성장 psum | 압축 act(SM+NZVL) | vertical stripe; 모든 Ni 먼저→공간 | single pass, 모든 Ni stream | **resident** MAC accum; inter-MAC in PRE | dense-C / 128 / IDP 512KB |
| **Simba** | **hierarchical WS** | 가중치(vector-MAC) | activations | 4-level tile of C,K(default); per-PE r,s→k1→c1→p1→q1 | C를 c0/c1/c2/c3 분할 | PE acc temporal resident; **cross-PE reduction** spatial | 8/vMAC / spread-K / distributed |

## 핵심 — WS/IS/OS가 타일링에서 갈리는 지점

모든 게 **"어떤 operand를 재-fetch 안 하느냐"**로 결정된다. 그게 outermost loop,
타일 크기를 제약하는 buffer, partial sum의 거처를 정한다:

- **WS** (TPU, Simba, NVDLA-stripe, Gemmini-WS): **weight tile이 resident/outermost.**
  `[Tc×Tk×R×S]` weight tile을 **weight buffer/array 용량**에 맞춰 고르고, 배열이 그걸
  재사용하며 **입력 activation을 출력위치(P,Q,N) 걸쳐 stream.** C&gt;k면 감축 weight-tile
  연속 load, **psum accumulator에 resident 누적.** *멀티패스 트리거 = weight tile &gt; weight buffer.*
- **OS** (Gemmini-OS, + NVDLA/TPU/MIDAP/NullHop의 누적 단계): **output tile이 resident**
  in accumulator. `[Tp×Tq×Tk]`를 **accumulator SRAM**에 맞춰 고르고, **weight+input을
  stream하며 full C 감축을 제자리 누적** 후 evict. *멀티패스 트리거 = output tile &gt; accumulator.*
- **IS** (MIDAP, NullHop): **input feature map이 resident**, **모든 filter/출력채널을
  stream.** 타일은 **입력 공간(MIDAP: width x; NullHop: vertical stripe)**을 **feature SRAM**에
  맞춤; C는 한 sweep에 소비되는 innermost 감축. *멀티패스 트리거 = FM tile &gt; feature SRAM.*

WS/IS/OS 외에 모델에 필요한 **직교 2축**:
1. **psum 거처**: *resident accumulator*(NVDLA CACC, TPU/Gemmini acc, MIDAP, NullHop,
   Simba PE-acc) vs *spill-and-refeed to shared buffer*(MAERI fold→PB, Eyeriss GLB→DRAM
   overflow). 이 한 비트가 C-split 비용을 "공짜 재누적"↔"buffer 왕복"으로 가른다.
2. **감축 locality**: *spatial*(배열/트리 걸쳐: Eyeriss PE열, Simba cross-PE, MAERI ART,
   TPU 열) vs *temporal*(cycle 걸쳐 accumulator에 누적: NVDLA, MIDAP, Gemmini-K, NullHop).
   실기는 섞는다(Simba c0=spatial + c1=temporal이 가장 명확).

## 파라메트릭 타일링 알고리즘 (dataflow + buffer 용량 키)

레이어 `(N,K,C,P,Q,R,S)`, 하드웨어 `(dataflow D, k_hw, m_hw, buffers)`:

```
1. 배열 두 축 매핑:  Tc_arr = k_hw (C→감축축),  Tk_arr = m_hw (K→출력병렬축)
2. dataflow D로 RESIDENT 타일 선택 (= outermost/held loop):
   WS: resident = weight tile [Tc×Tk×R×S]; cap = weight_buffer
       stream = act over (N,P,Q);  order = for k_tile: for c_tile(load w): stream P,Q,N
   OS: resident = output tile [Tp×Tq×Tk]; cap = accumulator
       stream = weights,inputs;    order = for (p,q,k)_tile(hold acc): for c: for r,s: MAC
   IS: resident = input tile [Tc×Tih×Tiw]; cap = feature_SRAM
       stream = weights over all K; order = for input_tile(hold): for k: for c,r,s: MAC
3. free 타일 dim(WS:Tp,Tq / OS:Tk / IS:Tih,Tiw)을 resident footprint ≤ cap인 최대로
   (double-buffer면 cap/2).
4. 감축 split: n_Cpass = ceil(C / (Tc_arr * spatial_C_replication)); 각 pass 누적:
     psum_home == RESIDENT_ACC: 제자리 누적 (NVDLA/TPU/Gemmini/MIDAP/NullHop/Simba-temporal)
     psum_home == SPILL_BUFFER: partial을 shared buffer에 쓰고 재feed+add (MAERI, Eyeriss overflow)
5. K/출력 split: n_Kpass = ceil(K / Tk_arr).
6. (옵션 spatial fan-out, Simba): C,K를 PE/chiplet 걸쳐 분할 + spatial-C용 cross-PE reduction.
```

**모델 관점 요약**: loop *순서*는 `D`가 완전 결정(WS→weight outer, OS→output held·C
innermost, IS→input outer·K stream). 타일 *크기*는 **resident operand를 담는 그 한
buffer** 용량(+double-buffer ½)이 결정. `k_hw`가 C를 얼마나 잘게 쪼갤지 정하고,
**`psum_home`(resident acc vs spill-buffer)**이 C-split 비용을 가르는 여분 1비트.
NVDLA·TPU·Gemmini·MIDAP·NullHop·Simba는 partial을 **resident**로; **MAERI(folding)·
Eyeriss(GLB/DRAM overflow)**만 감축 과대 시 **buffer 왕복** — cycle 모델이 다르게
가격 매겨야 할 케이스.

## Confidence
- **High(1차 출처)**: NVDLA(nvdla.org HW spec — atomic/stripe/block/channel/group 계층
  verbatim), TPU v1(ISCA'17), Gemmini(paper+repo Configs.scala/README ISA), MAERI(ASPLOS'18),
  NullHop(TNNLS/arXiv PDF 직독), Simba(MICRO/CACM Listing 1 직독), MIDAP dataflow·loop
  order(저자 시뮬레이터 소스).
- **Medium(플래그)**: MIDAP 메모리/MAC 수치=시뮬레이터 default(virtual prototype, 실리콘
  미공개); Simba per-PE buffer 바이트 수치=2차 요약; Eyeriss v2 GLB-cluster 크기 소스별 편차.

_출처: 백그라운드 리서치 에이전트(웹+소스 조사, 2026-07-11). 상세 per-accelerator
표는 조사 원본 참조._
