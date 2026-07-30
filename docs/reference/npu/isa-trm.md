# FlexNPUSim FunctionDescriptor ISA — 참조 매뉴얼

구현자용. `DnnImage` 바이너리, 그 안의 명령 `FunctionDescriptor`, 주소공간, 그리고
compiler·loader·runtime의 역할을 필드 단위로 규정한다. (타입 폭은 구현에서 확정;
여기서는 의미가 우선.)

---

## 1. 실행 모델

세 단계가 세 산출을 만든다.

```
UI(workload)  ─semantic compile─►  FunctionDescriptor(symbolic)  ─loader(=linker)─►  실행가능 DnnImage  ─►  runtime
op+shape+topology   counts 유도          주소 없음                offset 배정                          packet 하나씩
```

- **compiler**: workload × hardware → `counts`(6-수량) 유도. 전역 뷰. (Mode 2는 counts 직접 제공)
- **loader = linker**: symbolic → 주소(offset) 배정. 전역 뷰. 항상 실행. linker script = `hw.json`의 memory map(§3).
- **runtime**: packet 하나 fetch → function마다 latency model 1개 → LmChain. 국소 뷰.

명령 하나(`FunctionDescriptor`) = latency model 인스턴스 하나. packet(여러 function) = LmChain으로 묶인 latency model 여럿.

---

## 2. 바이너리 레이아웃

image = **field로 구성된 struct**(header/descriptor)와 **raw 데이터 section**(weight/input)으로 나뉜다. section은 이미지의 큰 데이터 영역, field는 struct 멤버다.

```
DnnImage (DRAM 한 base에 로드) =
  DnnImageHeader                                              ← field
  code section   : [ LayerPacketHeader + FunctionDescriptor × N ] × num_packets   ← field struct
  weight section : 모든 weight/const 바이트 (contiguous)      ← section
  input section  : 첫 레이어 입력 텐서                        ← section
```

| 구조체/영역 | 종류 | 역할 |
|---|---|---|
| `DnnImageHeader` | field struct | magic, version, num_packets, `weight_section_offset`, `input_section_offset` |
| `LayerPacketHeader` | field struct | `packet_id`, `num_functions`, `packet_size` |
| `FunctionDescriptor` | field struct | 명령 1개(§4) |
| weight section | 데이터 | 모든 weight를 image 끝에 append. descriptor가 offset으로 참조 |
| input section | 데이터 | 망 입력 |

weight를 image 끝 **global section**에 모으는 것은 HW 표준(TensorRT engine·NVDLA
loadable 방식): 모델을 한 덩어리로 DRAM에 로드하고 per-layer offset으로 참조한다.

runtime은 `LayerPacketHeader.num_functions`를 읽어 그만큼 descriptor를 실행한다.

---

## 3. 주소공간과 memory map

주소공간은 하드웨어 사실이므로 `hw.json`의 `address_map`이 정한다(loader의 linker
script). 하드코딩 상수는 그 default다.

### 3.1 memory map (JSON `address_map`)

```json
"address_map": {
  "sram_base": "0x20000000", "sram_size_kb": 1024,    // on-chip GB/SRAM
  "npu_base":  "0x40000000", "npu_size":     65536,   // MMIO 제어 레지스터
  "dram_base": "0x80000000", "dram_size_mb": 2048     // 데이터 (image·weight·activation)
}
```

| 영역 | 용도 |
|---|---|
| SRAM/GB (`sram_base`) | on-chip 데이터 (resident tensor) |
| **MMIO (`npu_base`)** | **제어 전용** — 레지스터(§3.2). 데이터 주소가 침범하지 않음 |
| DRAM (`dram_base`) | image·weight·activation 데이터 |

### 3.2 MMIO 레지스터 (host ↔ NPU 제어)

| offset | 이름 | 역할 |
|---|---|---|
| `0x00` | NPUCR | 제어 (START) |
| `0x04` | NPUSR | 상태 (RO) |
| `0x08/0x0C` | **DNNSA / DNNSA_MSB** | **DNN image base 주소** (host가 씀 → NPU가 여기서 image 읽음) |
| `0x24…` | PERF_CNT | cycles/MACs/bytes (RO) |

host: image를 DRAM에 로드 → `DNNSA`에 base 씀 → `NPUCR` START. 이후 모든 데이터
주소는 offset으로 DRAM/GB에서 해석된다.

### 3.3 operand → 물리주소 해석 (loader가 수행)

| operand 종류 | 물리주소 | 영역 |
|---|---|---|
| producer func 출력 | 그 func의 `out_offset` (loader 배정) | GB or DRAM |
| weight | `DNNSA + weight_section_offset + woff` | DRAM (in-image) |
| net input | `DNNSA + input_section_offset` | DRAM (in-image) |

두 데이터 종류: **in-image**(weight/input) = image base(DNNSA) 상대, **runtime
텐서**(activation/spill) = loader가 live-range로 배정(§9). 어느 것도 MMIO 영역에
들어가지 않는다.

---

## 4. FunctionDescriptor — 필드 정의

두 부분: **core**(opcode 무관, 의미 고정) + **body**(opcode가 의미 결정).

### 4.1 바이너리 레이아웃 (고정 크기)

```c
struct FunctionDescriptor {           // ≈ 160B (8B align)
  // --- core ---
  u32 func_id;                        // 전역 고유 = latency-model id
  u16 opcode;                         // FuncType (§5)
  u8  operand_cnt;                    // 유효 operand_src 수
  u8  shape_ndim;                     // 유효 shape_dim 수
  u32 operand_src[4];                 // operand 출처 (§8 인코딩)
  // --- body (opcode가 해석) ---
  u32 shape_dim[8];                   // 형상 dim (conv 6개가 최대)
  u32 op_params[4];                   // opcode별 immediate
  u32 tile_dim[6];                    // mapping (없으면 compiler 유도)
  // --- counts (runtime 실행) ---
  u32 fetch_operand0, fetch_operand1, write_output;
  u32 n_min, n_max, issue_latency;
  u32 threshold_in, threshold_wt, enable_gran;
  // --- 부가 ---
  u16 preprocess_type; u32 preprocess_p0, preprocess_p1;
  u32 flags;
  u32 out_offset;                     // loader가 배정
};
```

### 4.2 필드 동작 (누가 채우고 누가 읽나)

| 필드 | 채우는 곳 | 읽는 곳 | 동작 |
|---|---|---|---|
| `func_id` | compiler | loader, runtime | 명령 전역 고유 id. operand_src가 이걸 가리켜 DAG edge |
| `opcode` | user/compiler | compiler, functional | 연산 종류. runtime timing은 안 봄 |
| `operand_src[]` | user(topology)/compiler | loader | 각 operand 출처(§8). loader가 offset으로 해석 |
| `shape_dim[]` | user | compiler, functional | operand/output 형상. **의미는 opcode별**(§5). counts 유도·functional·weight size 산출 |
| `op_params[]` | user | compiler, functional | opcode별 immediate (conv: stride/pad/dilation) |
| `tile_dim[]` | user/compiler | runtime | 타일 형상. 없으면 tile_search 유도 |
| `counts`(9필드) | compiler | **runtime** | latency model이 실행하는 timing. §7 |
| `preprocess_*` | user/compiler | runtime | operand 전처리(zero-skip 등). p0=skip_permille |
| `flags` | compiler | runtime | tile enable, writeback required 등 |
| `out_offset` | **loader** | runtime | 이 func 출력의 offset. compile 단계엔 빔 |

runtime은 `counts`·offset·flags만 본다. opcode·shape는 안 본다.

### 4.3 counts (6-수량 + 보조)

| 필드 | 기호 | 동작 |
|---|---|---|
| `fetch_operand0` | F^I | 첫 operand delivery 수. cycle당 도착이 progress를 gate |
| `fetch_operand1` | F^W | 둘째 operand delivery 수. pool/unary는 0 |
| `write_output` | F^O | pass-completion budget. 이만큼 관측되면 compute 완료 |
| `n_min` / `n_max` | n_min/n_max | cycle당 pass-issue 폭 하한/상한(= sustained throughput) |
| `issue_latency` | l | accept→completion 파이프 지연 |
| `threshold_in/wt` | — | compute 시작 전 GB-resident gate |
| `enable_gran` | — | issue-enable 구조(lockstep/independent) |

---

## 5. Opcode 정의

```c
enum FuncType : u16 {
  CONV2D=0, DEPTHWISE=1, MATMUL=2, POOL=3, ACTIVATION=4,
  ELEMENTWISE=5, CONCAT=6, DECONV=7,
  SOFTMAX=8, LAYERNORM=9, REDUCTION=10, GATHER=11,
};
```
(`FC`=MATMUL 특수형, `residual`=ELEMENTWISE(add). CNN/ResNet/U-Net=0–7, Transformer=+2/8/9.)

opcode별 스펙 — `n(o)`=output 하나가 참여하는 unit-operation 수(reduction 길이):

| opcode | operand | n(o) | shape_dim[] | op_params[] |
|---|---|---|---|---|
| `CONV2D` | in, wt | Kh·Kw·Cin | [H,W,Cin,Cout,Kh,Kw] | [stride,pad,dilation] |
| `DEPTHWISE` | in, wt | Kh·Kw | [H,W,C,Kh,Kw] | [stride,pad,dilation] |
| `MATMUL` | in, wt | K | [M,N,K] | [transpose_a,transpose_b] |
| `POOL` | in | Kh·Kw | [H,W,C,Kh,Kw] | [stride,pad,mode] |
| `ACTIVATION` | in | 1 | [len] | [kind] |
| `ELEMENTWISE` | in,in2 | 1 | [len] | [kind(add/mul)] |
| `CONCAT` | in[] | 0 | [len,…] | [axis] |
| `DECONV` | in, wt | Kh·Kw·Cin | [H,W,Cin,Cout,Kh,Kw] | [stride,pad] |
| `SOFTMAX` | in | R | [rows,R] | [axis] |
| `LAYERNORM` | in,γ,β | R | [tokens,R] | [eps_q,affine] |
| `REDUCTION` | in | R | [outer,R] | [axis,kind] |
| `GATHER` | in,idx | 0 | [num_idx,elem_len] | [axis] |

reduce류(SOFTMAX/LAYERNORM/REDUCTION)는 축소축 R이 n(o). F^I=|O|·R, F^O=|O|.

---

## 6. Composition — 복합 연산 인코딩

복합 연산은 **한 LayerPacket 안 여러 FunctionDescriptor**(같은 packet_id → LmChain).
실제 필드값:

### 6.1 Attention (seq=S, d_k=d_v=D)
```
f0 MATMUL(QKᵀ)  operand_src=[Q,K]     shape_dim=[S,S,D]  op_params=[transpose_b=1]  → scores[S,S]
f1 SOFTMAX      operand_src=[f0]      shape_dim=[S,S]    op_params=[axis=1]          → probs[S,S]
f2 MATMUL(PV)   operand_src=[f1,V]    shape_dim=[S,D,S]                              → context[S,D]
```
### 6.2 Residual block (ResNet)
```
f0 CONV2D       operand_src=[x]                # x=block 입력
f1 CONV2D       operand_src=[f0]
f2 ELEMENTWISE  operand_src=[f1, x]            # skip: 둘째가 먼 x 지목(§8)
```
### 6.3 FFN
```
f0 MATMUL operand_src=[x, W1]   f1 ACTIVATION operand_src=[f0] op_params=[gelu]   f2 MATMUL operand_src=[f1, W2]
```
### 6.4 LayerNorm
```
f0 LAYERNORM operand_src=[x,γ,β]  shape_dim=[tokens,features]  op_params=[eps_q,affine=1]
```
### 6.5 CNN conv 블록
```
f0 CONV2D operand_src=[x]   f1 ACTIVATION operand_src=[f0]   f2 POOL operand_src=[f1]
```

규칙: 한 packet = 한 producer에서 시작하는 순차 chain. 분기/skip은 operand_src로 먼
func_id 지목. 3개 초과 입력(concat N-way)은 2-입력 chain으로 분해.

---

## 7. Counts 유도 (semantic compile)

compiler가 opcode·shape·hardware(k/g/s)에서 결정론적으로 계산:

- `|O|` = output element 수 (shape_dim).
- `n(o)` = §5 공식.
- `F^I = Σ_o n(o)`; MAC류 `F^W = F^I`, pool/unary `F^W = 0`.
- `p(o) = ⌈n(o)/k⌉`, `F^O = Σ_o p(o)` (k=`ops_per_pass`).
- `n_max = num_output_lanes · parallel_passes`, `n_min` = active-lane rule, `l` = pipeline depth(systolic k, adder_tree 1+⌈log2 k⌉).

Mode 2는 이 값을 직접 적어 이 단계를 건너뛴다.

---

## 8. operand_src / DAG 인코딩

`operand_src[i]` (u32) 한 필드가 네 종류를 겸한다 (compact ISA식):

| 값 | 의미 | loader 해석 |
|---|---|---|
| `0x0000_0000 – 0x7FFF_FFFF` | **producer func_id** (activation edge) | 그 func `out_offset` |
| `0x8000_0000 \| woff` (bit31=1) | **weight section 참조** | `DNNSA + weight_section_offset + woff`. **size는 shape_dim에서 유도** |
| `0xFFFF_FFFE` | **SRC_NET_INPUT** | `DNNSA + input_section_offset` |
| `0xFFFF_FFFF` | **linear 기본** (직전 func) | 직전 `out_offset` |

이러면 conv-weight·attention-K·layernorm-γ가 같은 필드로 표현된다:
```
conv       : operand_src=[x, 0x8000_0000|woff]        # in=activation, wt=weight section
attention  : operand_src=[Q_fid, K_fid]               # 둘 다 activation
layernorm  : operand_src=[x, 0x8000_0000|γoff, 0x8000_0000|βoff]
```

- **skip** = 직전이 아닌 producer func_id (ResNet/U-Net/attention KV).
- compiler가 func_id 그래프에서 각 tensor의 **live-range**(producer→마지막 consumer)를 계산 → loader가 배치에 사용(§9).

---

## 9. Placement (loader = linker)

주소는 loader가 `hw.json` memory map(§3, linker script)에 따라 배정한다. 절대 물리주소
대신 image-relative offset.

1. **offset 배정**: activation 출력·spill을 live-range 기반 linear-scan으로 배정. tensor
   마지막 consumer 이후 region free→재사용. footprint = 동시에 살아있는 tensor의 peak.
2. **GB vs DRAM**: live footprint가 GB(`sram_size`)에 맞으면 GB offset(on-chip 유지), 아니면 DRAM offset(spill).
3. **물리화·범위 검증**: `physical = base + offset`; `offset + size ≤ dram_size(또는 sram_size)` assert; MMIO 영역 회피.
4. **weight/input**: in-image라 image base(DNNSA) 상대 offset. runtime 텐서만 loader가 새로 배정.

결과를 각 descriptor의 `out_offset`과 operand offset에 baked. U-Net 장거리 skip은
live-range가 길어 그 span 동안 memory를 점유; 나머지는 회전.

---

## 10. 실행 semantics (runtime)

packet 하나 처리 순서:
1. `LayerPacketHeader` fetch → num_functions.
2. function마다 `counts`로 latency model 1개 구동 (opcode·shape 무시).
3. packet 내 function들 = LmChain (same-cycle hand-off; chained ≤ sequential).
4. operand는 baked offset에서 read, output은 `out_offset`에 write.
5. cross-layer overlap: config가 double-buffer면 packet N+1 memory를 packet N compute와
   겹침 — `mem₀ + Σ max(compute_N, mem_{N+1}) + compute_last`. serial은 `Σ max(mem_N,
   compute_N)`. packet-local하게 honor.

runtime 입력은 `counts`·offset·flags뿐.

---

## 11. 구현 순서 (spec → code)

1. **필드 재구성**: `LayerType`+`FunctionType`→`FuncType`(u16); `i1/i2_connect`→
   `operand_src[4]`+`operand_cnt`; `fetch_input/weight`→`fetch_operand0/1`;
   `TensorDescriptor.{h,w,c}`→`shape_dim[8]`+`shape_ndim`; `TileShape`→`tile_dim[6]`;
   `out_offset` 신설. `DNN_IMAGE_VERSION` bump.
2. **operand_src 배열** + global `func_id`(layer마다 reset 제거) + §8 인코딩.
3. **section 재배치**: packet-local `kernel_data` 제거 → image 끝 **global weight/input
   section**, header에 `weight_section_offset`/`input_section_offset`.
4. **memory map JSON화**: `memory_map::` 상수 → `address_map`(base+size) default로 이동.
   DRAM 범위 assert.
5. **loader = linker**: 입력에서 주소 제거(symbolic), bump cursor→live-range allocator.
6. **opcode 추가**: SOFTMAX/LAYERNORM/REDUCTION/GATHER — §7 counts 유도만, runtime 불변.
7. **memory/scheduling 다형성**: peak_bw·double-buffer를 선택 strategy로.

각 단계는 compiler/loader/format 작업이다. runtime의 latency model·LmChain·packet loop는 그대로 쓴다.
