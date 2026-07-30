/**
 * @file test_function_model.cpp
 * @brief Numeric contracts for the functional model (OpenCV-backed,
 *        PyTorch-equivalent semantics) — values checked by hand.
 *
 * Contracts:
 *  1. Conv2D window arithmetic — 3x3 input, 2x2 all-ones kernel,
 *     stride 1, no padding -> each output is its window sum
 *  2. padding semantics — 3x3 all-ones kernel over a 1x1 input with
 *     padding 1 -> the single output sees only the center element
 *  3. MaxPooling — 2x2/stride-1 window maxima
 */

#include "model/function/function_model_if.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace flexnpu_sim;

static Tensor make(uint32_t h, uint32_t w, uint32_t c, uint32_t n,
                   std::initializer_list<float> vals) {
    Tensor t;
    t.height = h; t.width = w; t.channels = c; t.batch = n;
    t.data.assign(vals);
    t.data.resize(t.size(), 0.0f);
    return t;
}

static bool eq(float a, float b) { return std::fabs(a - b) < 1e-5f; }

static void test_conv2d_window_sum() {
    Conv2dConfig cfg;   // stride 1, pad 0
    Conv2dModel conv(cfg);
    const Tensor in = make(3, 3, 1, 1, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    const Tensor k  = make(2, 2, 1, 1, {1, 1, 1, 1});   // batch = kernel count
    Tensor out = conv.forward(in, k);
    assert(out.height == 2 && out.width == 2 && out.channels == 1);
    assert(eq(out.data[0], 12) && eq(out.data[1], 16) &&
           eq(out.data[2], 24) && eq(out.data[3], 28));
}

static void test_conv2d_padding() {
    Conv2dConfig cfg;
    cfg.padding = 1;
    Conv2dModel conv(cfg);
    const Tensor in = make(1, 1, 1, 1, {5});
    const Tensor k  = make(3, 3, 1, 1, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    Tensor out = conv.forward(in, k);
    assert(out.height == 1 && out.width == 1);
    assert(eq(out.data[0], 5) && "zero padding: only the center contributes");
}

static void test_max_pooling() {
    PoolingConfig cfg;
    cfg.type = PoolingType::Max;
    cfg.kernel_size = 2;
    cfg.stride = 1;
    PoolingModel pool(cfg);
    const Tensor in = make(3, 3, 1, 1, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor out = pool.forward(in, Tensor{});
    assert(out.height == 2 && out.width == 2);
    assert(eq(out.data[0], 5) && eq(out.data[1], 6) &&
           eq(out.data[2], 8) && eq(out.data[3], 9));
}

int main() {
    test_conv2d_window_sum();
    test_conv2d_padding();
    test_max_pooling();
    std::cout << "test_function_model: all contracts hold\n";
    return 0;
}
