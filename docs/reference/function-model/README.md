# Function Model

The function model is the simulator's **value plane**: it computes actual tensor
values for a layer, at OpenCV/PyTorch parity, when a functional run is requested.
It is optional — the timing plane (the [latency model](../latency-model/)) never
calls into it — and it exists for two jobs: reference-output verification
(dumping per-layer OFM files) and **activation-sparsity profiling**, which is the
sole value→count bridge back into timing.

- Source: `src/model/function/`
  - `function_model_if.h` — the `Tensor` type and the `FunctionModelIf`
    interface plus the per-op models (Conv2D, DepthwiseConv2D, Pooling, FC,
    Activation, MatMul, ElementWise)
  - `conv2d.cpp`, `depthwise_conv2d.cpp`, `pooling.cpp`, `fully_connected.cpp`,
    `activation.cpp`, `matmul.cpp`, `elementwise.cpp` — the op implementations
    (OpenCV `cv::dnn` backend for conv/DW/FC)
  - `function_model.cpp`, `function_model_utils.{h,cpp}` — `Tensor` accessors,
    `Tensor`↔`cv::Mat` conversion
  - `layer_function_runner.h` — decode bytes → `Tensor`, run the matching model,
    encode back (dtype per image header)
  - `ofm_file_generator.{h,cpp}` — precompute a network's layer outputs to files
  - `sparsity_profiler.{h,cpp}`, `sparsity_preprocess.{h,cpp}` — measure ReLU
    activation sparsity and patch fetch counts (the value→count bridge)
- Depends on: the [compiler](../compiler/)'s `DnnImage` / `LayerDescriptor` (it
  reads the compiled layers) and OpenCV (`HAVE_OPENCV`; a no-op without it)
- Feeds back into: the compiled image's `fetch_operand0`/`prefetch_in` — the one
  place a value crosses into the timing plane

## Why it exists

Cycle counts are count-driven and value-independent, so the simulator needs no
tensor values to report timing. But two things do need real values: verifying
the modeled dataflow against a reference (are the OFMs right?), and measuring
**data-dependent** sparsity — a zero-skipping NPU like Nullhop fetches fewer
input operands because ReLU zeroed them, and that fraction can only be known by
actually running the activations. The function model supplies both without
contaminating the timing core: values live here, and the only export is a
measured count correction.

## Two abstraction planes — the single bridge

The simulator keeps **timing** (counts, never values) separate from **function**
(values, optional). This subsystem is the function plane. The bridge in the
value→count direction is exactly one path: `SparsityProfiler` runs the network
forward with random inputs + He-initialized weights, applies ReLU after each
layer, and measures each layer's output-zero fraction; `apply_sparsity` then
scales the next layer's `fetch_operand0` by `(1 − input_sparsity)` in the packet
image (`SparsityProfiler::adjust_fi`). After that, the latency model sees only a
smaller integer operand count — it never learns why. Nothing crosses in the
count→value direction.

## Documents

| File | Contents |
|------|----------|
| [`modeling.md`](modeling.md) | the value→count bridge in detail: `SparsityProfiler::profile` (forward pass, He init, ReLU, per-op handling), `adjust_fi`, `apply_sparsity` (in-place packet patch), where it is invoked, and why it is the only value-dependent timing input |

The per-op forward models (`Conv2dModel::forward`, etc.) are thin wrappers over
OpenCV `cv::dnn` / direct loops and are not documented individually; they
compute standard layer semantics and are only exercised through the profiler and
the OFM file generator.

## One-paragraph summary

Each op implements `FunctionModelIf::forward(input, weight) → Tensor` over an
NCHW `float` `Tensor`, computing standard layer semantics (conv/DW/FC via OpenCV
`cv::dnn`). Given a compiled `DnnImage`, `SparsityProfiler::profile` runs the
network forward with a fixed-seed random input and He-initialized weights,
applies ReLU after each Conv/DW/FC, and records every layer's output-zero
fraction. `apply_sparsity` walks the packet image and, for each function,
rewrites `fetch_operand0 ← round(fetch_operand0·(1 − producer_output_sparsity))`
and scales `prefetch_in` by the same ratio — the one measured value that the
count-based timing plane consumes. Without OpenCV the whole subsystem compiles
to no-ops.
</content>
