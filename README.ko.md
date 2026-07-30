[English](README.md) | 한국어

# FlexNPUSim

설정 가능한 SystemC 기반 NPU 시스템 시뮬레이터. 바이너리 하나가 하드웨어 스펙(JSON)과
DNN workload(CSV)로 구동되는 NPU + DMA + AXI bus + DRAM 시스템을 시뮬레이션하고,
latency, DRAM traffic, PE utilization, 레이어별 compute-vs-memory 병목 판정을
보고한다 — 네트워크 하나에 수 초가 걸리는 속도라, RTL 시뮬레이션이 닿지 못하는
가속기 × 메모리 시스템 설계 공간을 훑을 수 있다.

**Validation** — 모든 하드웨어 파라미터는 공개 스펙에서 유도했고, 어느 값도
레퍼런스에 맞춰 fitting하지 않았다. compute 모델은 하드웨어 상수만으로 NVDLA 공식
성능 모델을 재현하고(alexnet_conv5 +0.005%, googlenet_conv2 +0.03%), 전체 네트워크
실행은 저자 레퍼런스 시뮬레이터(MIDAP, 세 네트워크 3.4% 이내) 및 발표된 RTL 실측
(NullHop sparse VGG-16, +3.1%)과 일치한다 — 체크포인트 6개, 평균 오차 1.6%,
보정 파라미터 0개. `./validation/run_validation.sh`로 재현하며, 수치별 출처는
`validation/results.md`에 있다.

## Quick start

```sh
cmake -B build -S . && cmake --build build -j$(nproc)

# hardware spec (or full config JSON) + workload CSV; paths resolve from repo root
build/sim/flexnpusim -hw_conf model/hw/npu/nvdla-large.json \
    -network model/sw/cnn/nvdla-large/resnet18.csv -o out/run.csv
# add -report out/report.txt for the per-layer bottleneck table

# or the shorthand runner / the validation gate
python3 tools/run.py --scenario nvdla-large:resnet18
./validation/check_nvdla_anchors.sh
```

입력은 두 가지다: **하드웨어 모델**(`model/hw/npu/*.json` — PE array, buffer,
dataflow, AXI, DRAM)과 **소프트웨어 모델**인 workload CSV
(`model/sw/<class>/<design>/<network>.csv`, 클래스는 `cnn|attention`; 레이어별
oracle factor 모드와 레이어 치수 자동 유도 모드 중 하나로 쓴다). PyTorch 모델은
`src/compiler/pytorch/`로 변환한다. 모든 실행은 저장소 루트에서 한다 — config
내부 경로가 루트 기준이다. 출력은 `out/`(gitignore 대상)에 쌓인다.

## Layout

```
src/
├── systemc/     SystemC cycle-accurate 컴포넌트 (RTL 대응)
│   ├── bus/     AXI transport 라이브러리, arbiter, topology 배선
│   ├── npu/     NPU controller (tile 실행, GB residency)
│   ├── memory/  DRAM 모델 (bank 모델, DRAMsim3 백엔드)
│   ├── dma/  processor/
├── model/       성능·기능 모델
│   ├── latency/   count 기반 latency 모델 (compute 측 추상화)
│   └── function/  OpenCV 기반 기능 모델 (PyTorch 1:1)
├── compiler/    workload 생성·스케줄링
│   ├── frontend/   hw JSON + network CSV 로더, request builder
│   ├── dnn_image/  network → dnn image (descriptor stream) 컴파일러
│   └── pytorch/    PyTorch 모델 → network CSV 파이프라인 (Python)
├── system/      최상위 조립 (flexnpusim_system) + 성능 리포트
└── common/      config 로더, 타입, 로깅, tile descriptor

model/hw/    하드웨어 모델 (JSON): npu/, memory/ (DRAM ini), system/ (예약)
model/sw/    소프트웨어 모델 — 클래스별 workload CSV: cnn/, attention/
examples/    baseline.json — 전체 블록 config 레퍼런스
tools/       run.py — 원커맨드 러너 (--scenario design:network)
validation/  레퍼런스 체크포인트 하니스 (run_validation.sh), 16-config smoke
             테스트, SCALE-Sim 교차 검증, DSE 재생성 매트릭스
tests/       contract 테스트 (ctest), 모듈별 spec-first 재생성
docs/reference/  서브시스템별 기준 레퍼런스 (코드의 실제 동작);
             docs/reference/README.md 또는 docs/adding-a-new-npu.md에서 시작
docs/research/  배경 연구 노트
external/    dramsim3 / systemc / spdlog (자동 fetch), nlohmann, systemc_axi (legacy)
out/         시뮬레이터 출력 (gitignore 대상)
```

## Build

```sh
cmake -B build -S . && cmake --build build -j$(nproc)
```

CMake ≥ 3.16과 C++17 컴파일러가 필요하다. `external/` 아래 체크아웃이 없으면
SystemC 2.3.4, spdlog, DRAMsim3를 자동으로 fetch한다. OpenCV(core/dnn/imgproc)는
선택 사항이며, 있으면 기능 모델이 활성화된다.

## Authors

- 정현우 (Hyunwoo Chung) — 광운대학교
- 이준환 (Joonhwan Yi) — 광운대학교

## Acknowledgment

이 연구는 과학기술정보통신부의 재원으로 정보통신기획평가원의 지원을 받아 수행되었다
(RS-2022-II220998 / 2022-0-00998, 인공지능 반도체 개발을 위한 개방형 개발환경
플랫폼 개발;
2022년도 제1차 정보통신·방송 기술개발사업 및 표준개발지원사업, 과학기술정보통신부
공고 제2021-1022호).

## License

MIT — [LICENSE](LICENSE) 참조.
