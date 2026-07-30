# 타일 탐색 목적함수 — 연구 주제 시드 (future paper topics)

FlexNPUSim의 컴파일러는 레이어마다 **타일**(출력채널 × 공간 H/W × 감축 pass
blocking)을 on-chip buffer(GB)에 맞게 골라야 한다. "어떤 타일이 최선인가"를
정하는 **목적함수** 자체가 연구 문제다. 아래 셋은 각각 독립 논문이 될 후보다.

> 프로덕션 타일러는 **실기 NPU의 dataflow 고정 blocking을 재현**하는 방향으로
> 간다(Phase D, `[[2026-07-10-lm-contract-schema-redesign]]`). 아래 세 목적함수는
> 그와 대비되는 **추상 optimizer** 계열로, 그 자체가 연구 대상이다.

공통 제약: 선택 타일 `t`에 대해 working set `WS(t) ≤ GB`. 표기 `F^I/F^W/F^O` =
tiling에서 유도되는 fetch input/weight, write output 총량(halo refetch·psum spill
포함).

---

## 1. 최소 타일 수 (Minimum Tile-Count)

- **목적**: `min ∏_axis ⌈dim/t_axis⌉  s.t. WS(t) ≤ GB`
- **아이디어**: 타일 수 최소 ≈ loop 오버헤드 최소·재사용 최대. affine WS 모델에서
  타일당 최대 크기가 closed-form이라 탐색이 가볍고, **설계상 GB에 대해 단조**
  (budget↑ → feasible set↑ → min↓)라 DSE 곡선이 깨끗하다.
- **연구 각도**: (i) GB-monotone tiling의 증명 가능한 성질, (ii) affine WS 하
  closed-form 타일, (iii) *tile-count와 실제 traffic/latency의 괴리 상한* —
  언제 tile-count가 성능의 나쁜 proxy가 되는가.
- **장단**: 단순·빠름·단조 보장 ↔ **타일 모양(halo/refetch) 무시**라 traffic
  준최적일 수 있음.
- **열린 질문**: num_tiles↔traffic gap bound; multi-level buffer 확장;
  tie-break이 최종 성능에 주는 영향.

## 2. 최소 DRAM 트래픽 (Minimum DRAM Traffic)

- **목적**: `min (F^I + F^W + F^O)(t)  s.t. WS(t) ≤ GB`, F는 halo refetch +
  partial-sum spill 포함.
- **아이디어**: memory-bound 성능의 직접 동인은 이동 바이트. tall-thin 타일은
  halo↑ → traffic↑로 **모양이 자연 페널티**를 받는다(tile-count가 못 잡는 것).
- **연구 각도**: 해석적 traffic 모델을 tiling 목적함수로; **Timeloop/ZigZag류
  cost-model 탐색과 대비되는 closed-form GB-aware 타일러**; traffic vs on-chip
  area Pareto; sparsity/compression을 count 보정으로 접었을 때의 traffic 최적.
- **장단**: 성능 정렬 ↔ compute-bound·대역 은닉 구간에선 traffic≠latency;
  value-independent count 가정.
- **열린 질문**: 목적함수의 볼록성/전역최적 존재; 다중 operand 재사용의 상호작용;
  dataflow별 traffic 형태의 닫힌 형태 유도.

## 3. 최소 모델 지연 (Minimum Modeled Latency, closed-loop)

- **목적**: `min L(t)`, `L` = 해당 tiling으로 **LM + 메모리/버스 모델을 돌린
  cycle**.
- **아이디어**: 근사 proxy가 아니라 **실제 모델 cycle**을 최적화 —
  compute/memory overlap, 버스 arbitration contention, DRAM row-buffer 효과까지
  포함(이 repo가 이미 모델링하는 것들: burst·5-master 경합 등).
- **연구 각도**: cycle 모델 위에서의 tiling 탐색 = **tiler와 simulator를 루프**;
  후보마다 full sim을 피하기 위한 **surrogate/learned tile predictor**;
  latency-optimal이 traffic-optimal과 갈리는 조건의 특성화.
- **장단**: 가장 정확(overlap·contention 반영) ↔ 후보당 sim이라 비쌈·비해석적·
  탐색공간 폭발.
- **열린 질문**: full sim 없이 latency를 근사하는 surrogate; 탐색공간 가지치기;
  differentiable 완화 가능성.

---

## 대비축 — 실기 blocking과의 efficiency gap (별도 논문 시드)

실제 NPU는 위 셋 중 어느 것도 풀지 않는다 — **dataflow가 강제하는 고정
blocking**을 buffer에 맞게 greedy 사이징할 뿐이다(WS: weight resident, IS:
row-stationary, OS: output resident). 따라서 자연스러운 논문 각도:

> **각 실기의 고정 blocking이 traffic-/latency-optimal 타일에서 얼마나 떨어져
> 있나?** (설계별 efficiency-gap 연구)

FlexNPUSim은 실기 blocking(프로덕션 경로)과 위 세 optimizer를 **같은 워크로드로
동일 조건 비교**할 수 있으므로, 이 gap을 정량화하는 실험 플랫폼이 된다. 세
목적함수는 그 비교의 "이상적 상한" baseline으로 기능한다.

---

_출처: 이 문서는 tile-calc 알고리즘 논의(2026-07-11)에서 채택하지 않은 세
목적함수를 연구 주제로 보존한 것. 프로덕션 결정은 실기 dataflow blocking._
