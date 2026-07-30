# Function Model — Modeling

This subsystem's only non-trivial model is the **value→count bridge**: how a
value-plane measurement (activation sparsity) becomes a count-plane correction
(reduced `fetch_operand0`). The forward ops themselves are standard layer
semantics and are not modeled here beyond "OpenCV/PyTorch parity."

## The abstraction boundary

The [reference root](../README.md)'s "Two abstraction planes" rule: timing
is count-based and never touches tensor values; function is value-based and
optional. A value can influence timing only through **one** measured count
correction. Sparsity is that correction, and this is the only code that crosses
the line.

Why sparsity must be measured and cannot be derived from shapes: a zero-skipping
NPU (Nullhop) skips input operands that are zero. After ReLU, the zero fraction
depends on the actual activation distribution, not on the layer geometry. The
compiler's count formulas (`F^I` etc.) are dense; only running the activations
tells you how many are zero. So the profiler runs the network for values, and
exports a single scalar per layer — the output-zero fraction — that shrinks the
next layer's input operand count.

## Step 1 — measure: `SparsityProfiler::profile`

`profile(image, input_h, input_w, input_c, seed)`
(`sparsity_profiler.cpp:196`) parses the compiled image and runs it forward,
layer by layer, carrying a single `current` tensor:

- **Input** is a fixed-seed uniform `[0,1]` tensor (`random_input_tensor`,
  `sparsity_profiler.cpp:66`) — deterministic per `seed` (default 42), so the
  measurement is reproducible.
- **Weights** are generated on the fly with He initialization
  `N(0, sqrt(2/fan_in))` (`he_init_tensor`, `sparsity_profiler.cpp:45`); no
  trained weights are needed because only the *sparsity pattern* matters, not
  the exact values.
- **Per-op handling** (`sparsity_profiler.cpp:211`):
  - `Conv2D` / `DepthwiseConv2D` → `FunctionModelIf::forward` (OpenCV), then
    **ReLU in place** — ReLU is what creates the zeros the NPU skips.
  - `Pooling` → `simple_pool` (max; for non-negative post-ReLU inputs, max and
    average pooling give identical zero patterns, `sparsity_profiler.cpp:107`).
  - `FullyConnected` → `simple_fc` (on-the-fly He weights) + ReLU.
  - `Activation` → ReLU in place (the activation subtype is not encoded in the
    descriptor; ReLU is the stable default, `sparsity_profiler.cpp:258`).
  - default (`ElementWise`/`MatMul`) → pass through, except a `Preprocess` +
    `ZeroSkipping` function, which applies random zeroing at the descriptor's
    keep ratio `write_output/fetch_operand0` (`sparsity_profiler.cpp:264`).
- **Record**: after each layer, `compute_sparsity(current)` (the zero fraction
  of the tensor, `sparsity_profiler.cpp:24`) is stored as that layer's
  `output_sparsity`.

The result is a `LayerSparsity[]` — one output-zero fraction per layer.

## Step 2 — convert: `adjust_fi`

`adjust_fi(original_fi, input_sparsity)` (`sparsity_profiler.cpp:33`) is the
whole conversion:

```
adjusted_fi = round(original_fi · (1 − input_sparsity))
```

with the guards `input_sparsity ≤ 0 → original_fi` (no change) and
`input_sparsity ≥ 1 → 0` (fully skipped). A layer's *input* sparsity is the
*output* sparsity of its producer — the previous layer's ReLU zeros are this
layer's skippable input.

## Step 3 — patch: `apply_sparsity`

`apply_sparsity(image, cfg)` (`sparsity_preprocess.cpp:16`) profiles the image,
then walks the packet buffer in place (`DnnImageHeader` → each
`LayerPacketHeader` → each `FunctionDescriptor`) and for every function
(`sparsity_preprocess.cpp:43`):

1. `input_sp = sparsity[flat_idx − 1].output_sparsity` — the producer's zeros
   (the first function has no producer, so `input_sp = 0`).
2. `fetch_operand0 ← adjust_fi(fetch_operand0, input_sp)`
   (`sparsity_preprocess.cpp:53`).
3. If the count shrank, scale the prefetch gate by the same ratio
   `prefetch_in ← max(1, round(prefetch_in · adjusted/original))`
   (`sparsity_preprocess.cpp:54`) so the compute-start gate stays proportional.

Only `fetch_operand0` and `prefetch_in` change; `F^W`, `F^O`, `l`, `n_min`,
`n_max` are untouched. The runtime zero-skip ratio is later re-derived as
`write_output/fetch_operand0`, so no separate ratio field is stored
(`sparsity_preprocess.cpp:59`).

## Where it runs, and the OpenCV gate

`flexnpusim_system` calls `apply_sparsity(image, net_cfg)` immediately after the
DNN image is built and **before** the SystemC model is assembled
(`src/system/flexnpusim_system.cpp:389`), guarded by the `-zero_skipping` flag.
From then on the timing plane sees only the reduced integer counts.

The entire subsystem is compiled under `HAVE_OPENCV`. Without OpenCV,
`apply_sparsity` is a no-op (`sparsity_preprocess.cpp:16`) and the dense
compiler-derived `fetch_operand0` stands — the run still produces timing, just
without the sparsity correction.

## Relationship to the static `preprocess_param0` column

The CSV `preprocess_param0` (zero-skip permille, see
[`../compiler/parameters.md`](../compiler/parameters.md)) is the *static*,
author-declared form of the same idea: a fixed keep ratio baked into the
descriptor at compile time. `apply_sparsity` is the *measured* form: it derives
the ratio from an actual forward pass. They target the same `fetch_operand0`
count from opposite ends — one declared, one measured — and are independent
paths.

TODO: the profiler assumes ReLU-family activations; a network using GELU/other
activations would have its sparsity mismeasured (the subtype is not carried in
`LayerDescriptor`). Not an issue for the current Nullhop/VGG validation set.
</content>
