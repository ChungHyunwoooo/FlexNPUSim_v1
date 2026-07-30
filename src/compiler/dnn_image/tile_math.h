/**
 * @file tile_math.h
 * @brief Per-FuncType tile-size helpers for R9 tile abstraction.
 *
 * Given LayerGeometry + TileShape, computes tile count N_T and per-tile
 * |I_j|, |W_j|, |O_j|. Axis semantics per FuncType follow the
 * sub-plan table (docs/plan/2026-04-24-r9-tile-abstraction.md §2).
 *
 * Pure C++ / header-only. No SystemC.
 */

#pragma once

#include "compiler/dnn_image/dnn_image_format.h"
#include <cstdint>
#include <string>

namespace flexnpu_sim {

// ============================================================================
// LayerGeometry — shared input to Mode A / Mode B compiler API
// ============================================================================

struct LayerGeometry {
    FuncType type = FuncType::Conv2D;

    // Input tensor
    uint32_t input_h = 0;
    uint32_t input_w = 0;
    uint32_t input_c = 0;

    // Kernel (Conv/DW/Pool): kernel_h/kernel_w are receptive field.
    //   Conv2D          : kernel_oc = output channel count
    //   DepthwiseConv2D : kernel_oc = input_c (enforced by validator)
    //   FullyConnected  : kernel_oc = out_features; kernel_h = kernel_w = 1
    //   MatMul          : kernel_h = M, kernel_w = N, kernel_oc = 0 (use K=input_c)
    //   Pooling/Act     : kernel_oc = input_c (pass-through)
    //   ElementWise     : shapes mirror input; kernel_* ignored
    uint32_t kernel_h  = 1;
    uint32_t kernel_w  = 1;
    uint32_t kernel_oc = 0;

    uint32_t stride   = 1;
    uint32_t padding  = 0;
    uint32_t dilation = 1;

    // DRAM addresses
    uint32_t input_addr  = 0;
    uint32_t weight_addr = 0;
    uint32_t output_addr = 0;
};

// ============================================================================
// Output-shape derivation (shared; see dnn_image_compiler.cpp for Conv path)
// ============================================================================

struct OutputShape {
    uint32_t h = 0;
    uint32_t w = 0;
    uint32_t c = 0;  // for MatMul: h=M, w=N, c=1
};

inline OutputShape compute_output_shape(const LayerGeometry& g) {
    OutputShape o{};
    switch (g.type) {
        case FuncType::Conv2D:
        case FuncType::DepthwiseConv2D:
        case FuncType::Pooling: {
            const uint32_t eff_kh = g.kernel_h + (g.kernel_h - 1) * (g.dilation - 1);
            const uint32_t eff_kw = g.kernel_w + (g.kernel_w - 1) * (g.dilation - 1);
            o.h = (g.input_h + 2 * g.padding - eff_kh) / g.stride + 1;
            o.w = (g.input_w + 2 * g.padding - eff_kw) / g.stride + 1;
            if (g.type == FuncType::Conv2D) {
                o.c = g.kernel_oc;
            } else {
                o.c = g.input_c;  // DW / Pooling keep channel count
            }
            break;
        }
        case FuncType::FullyConnected:
            o.h = 1; o.w = 1; o.c = g.kernel_oc;
            break;
        case FuncType::Activation:
        case FuncType::ElementWise:
            o.h = g.input_h; o.w = g.input_w; o.c = g.input_c;
            break;
        case FuncType::MatMul:
            // input_h = M, input_w = K, kernel_w = N (encoding convention)
            o.h = g.input_h; o.w = g.kernel_w; o.c = 1;
            break;
        default:
            o.h = g.input_h; o.w = g.input_w; o.c = g.input_c;
            break;
    }
    return o;
}

// ============================================================================
// Tile validation
// ============================================================================

struct TileAxes {
    // Effective tile sizes after resolving "0 = full axis".
    uint32_t t_r  = 0;
    uint32_t t_c  = 0;
    uint32_t t_oc = 0;
    uint32_t t_ic = 0;   // 0 means "no reduction axis" (Pool/Act/ElementWise)
};

/// Resolve TileShape: zero-axes default to full extent (except t_ic for
/// no-reduction functions which stays 0). Returns empty string on success,
/// error description on validation failure.
inline std::string resolve_tile(const LayerGeometry& g,
                                const OutputShape& o,
                                const TileShape& req,
                                TileAxes& out) {
    out.t_r  = (req.t_r  == 0) ? o.h : req.t_r;
    out.t_c  = (req.t_c  == 0) ? o.w : req.t_c;
    out.t_oc = (req.t_oc == 0) ? o.c : req.t_oc;

    const bool has_reduction =
        (g.type == FuncType::Conv2D ||
         g.type == FuncType::DepthwiseConv2D ||
         g.type == FuncType::FullyConnected ||
         g.type == FuncType::MatMul);

    if (has_reduction) {
        uint32_t full_ic = g.input_c;
        if (g.type == FuncType::MatMul) full_ic = g.input_w;  // K axis
        out.t_ic = (req.t_ic == 0) ? full_ic : req.t_ic;
    } else {
        out.t_ic = 0;
    }

    // R9 constraint: t_kern must equal full kH*kW (partial kernel tiling out of scope).
    const uint32_t full_kern = g.kernel_h * g.kernel_w;
    if (req.t_kern != 0 && req.t_kern != full_kern) {
        return "partial kernel tiling unsupported in R9 (t_kern must equal kH*kW)";
    }

    // DepthwiseConv2D: t_oc == t_ic invariant.
    if (g.type == FuncType::DepthwiseConv2D && out.t_oc != out.t_ic) {
        return "DepthwiseConv2D requires t_oc == t_ic";
    }

    // Bounds.
    if (out.t_r > o.h || out.t_c > o.w || out.t_oc > o.c) {
        return "tile exceeds output extent";
    }
    if (has_reduction) {
        const uint32_t full_ic = (g.type == FuncType::MatMul) ? g.input_w : g.input_c;
        if (out.t_ic > full_ic) return "t_ic exceeds reduction extent";
    }

    // intra_order within known enum (or Custom escape).
    switch (static_cast<TileTraversal>(req.intra_order)) {
        case TileTraversal::IcFirstOcLast:
        case TileTraversal::OcFirstIcLast:
        case TileTraversal::SpatialFirst:
        case TileTraversal::KernelFirst:
        case TileTraversal::Custom:
            break;
        default:
            return "invalid TileShape.intra_order";
    }
    return {};
}

// ============================================================================
// Tile-count and per-tile operand volumes
// ============================================================================

inline uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

struct TileCounts {
    uint32_t n_tiles_r  = 1;
    uint32_t n_tiles_c  = 1;
    uint32_t n_tiles_oc = 1;
    uint32_t n_tiles_ic = 1;  // passes per tile group (reduction axis)
    uint32_t total() const { return n_tiles_r * n_tiles_c * n_tiles_oc * n_tiles_ic; }
};

inline TileCounts compute_tile_counts(const LayerGeometry& g,
                                      const OutputShape& o,
                                      const TileAxes& t) {
    TileCounts c{};
    c.n_tiles_r  = (t.t_r  > 0) ? ceil_div(o.h, t.t_r)  : 1;
    c.n_tiles_c  = (t.t_c  > 0) ? ceil_div(o.w, t.t_c)  : 1;
    c.n_tiles_oc = (t.t_oc > 0) ? ceil_div(o.c, t.t_oc) : 1;
    if (t.t_ic > 0) {
        const uint32_t full_ic = (g.type == FuncType::MatMul) ? g.input_w : g.input_c;
        c.n_tiles_ic = ceil_div(full_ic, t.t_ic);
    } else {
        c.n_tiles_ic = 1;
    }
    return c;
}

struct PerTileVolumes {
    uint64_t input_elems_per_tile  = 0;  ///< |I_j|
    uint64_t weight_elems_per_tile = 0;  ///< |W_j|
    uint64_t output_elems_per_tile = 0;  ///< |O_j|
};

/// Canonical (non-edge) per-tile volume. Edge tiles shrink along partial axes;
/// aggregate traffic derivation in Mode B should sum per-tile volumes properly
/// (helper below).
inline PerTileVolumes per_tile_volumes(const LayerGeometry& g,
                                       const TileAxes& t) {
    PerTileVolumes v{};
    switch (g.type) {
        case FuncType::Conv2D: {
            const uint32_t in_h = t.t_r * g.stride + (g.kernel_h - 1) * g.dilation;
            const uint32_t in_w = t.t_c * g.stride + (g.kernel_w - 1) * g.dilation;
            v.input_elems_per_tile  = 1ull * in_h * in_w * t.t_ic;
            v.weight_elems_per_tile = 1ull * g.kernel_h * g.kernel_w * t.t_ic * t.t_oc;
            v.output_elems_per_tile = 1ull * t.t_r * t.t_c * t.t_oc;
            break;
        }
        case FuncType::DepthwiseConv2D: {
            const uint32_t in_h = t.t_r * g.stride + (g.kernel_h - 1) * g.dilation;
            const uint32_t in_w = t.t_c * g.stride + (g.kernel_w - 1) * g.dilation;
            v.input_elems_per_tile  = 1ull * in_h * in_w * t.t_ic;
            v.weight_elems_per_tile = 1ull * g.kernel_h * g.kernel_w * t.t_oc;  // t_oc==t_ic
            v.output_elems_per_tile = 1ull * t.t_r * t.t_c * t.t_oc;
            break;
        }
        case FuncType::FullyConnected: {
            // (T_R=T_C=1) × T_OC outputs, (T_IC) inputs, (T_IC × T_OC) weights.
            v.input_elems_per_tile  = 1ull * t.t_ic;
            v.weight_elems_per_tile = 1ull * t.t_ic * t.t_oc;
            v.output_elems_per_tile = 1ull * t.t_oc;
            break;
        }
        case FuncType::MatMul: {
            // (T_R=T_M, T_C=T_N, T_IC=T_K). Two operand tensors A(M×K), B(K×N).
            // input_elems_per_tile lumps A; weight_elems_per_tile lumps B.
            v.input_elems_per_tile  = 1ull * t.t_r * t.t_ic;
            v.weight_elems_per_tile = 1ull * t.t_ic * t.t_c;
            v.output_elems_per_tile = 1ull * t.t_r * t.t_c;
            break;
        }
        case FuncType::Pooling: {
            const uint32_t in_h = t.t_r * g.stride + (g.kernel_h - 1) * g.dilation;
            const uint32_t in_w = t.t_c * g.stride + (g.kernel_w - 1) * g.dilation;
            v.input_elems_per_tile  = 1ull * in_h * in_w * t.t_oc;
            v.weight_elems_per_tile = 0;
            v.output_elems_per_tile = 1ull * t.t_r * t.t_c * t.t_oc;
            break;
        }
        case FuncType::Activation: {
            v.input_elems_per_tile  = 1ull * t.t_r * t.t_c * t.t_oc;
            v.weight_elems_per_tile = 0;
            v.output_elems_per_tile = 1ull * t.t_r * t.t_c * t.t_oc;
            break;
        }
        case FuncType::ElementWise: {
            // Two input tensors of output shape; put second into weight slot.
            v.input_elems_per_tile  = 1ull * t.t_r * t.t_c * t.t_oc;
            v.weight_elems_per_tile = 1ull * t.t_r * t.t_c * t.t_oc;
            v.output_elems_per_tile = 1ull * t.t_r * t.t_c * t.t_oc;
            break;
        }
        default:
            // Concat / Preprocess: tile iteration disabled; return zeros.
            break;
    }
    return v;
}

/// Aggregate |O| across all tiles (layer output-element count).
inline uint64_t layer_output_element_count(const OutputShape& o) {
    return 1ull * o.h * o.w * o.c;
}

struct AggregateVolumes {
    uint64_t input_elems  = 0;  ///< Sum_j |I_j| over all tiles
    uint64_t weight_elems = 0;  ///< Sum_j |W_j| over all tiles
};

/// Halo-aware aggregate operand volumes summed across ALL tiles, with edge
/// tiles counted at their true (smaller) extents. This equals the uniform
/// `per_tile_volumes() * TileCounts.total()` product with every rounded
/// `tile_extent * n_tiles` factor replaced by the exact axis size, while the
/// per-tile input halo `(k-1)*dilation` on spatial axes is preserved (added
/// once per spatial tile). Mode-B traffic derivation should use this rather
/// than the canonical single-tile volume, which over-counts partial edge tiles.
inline AggregateVolumes aggregate_operand_volumes(const LayerGeometry& g,
                                                  const OutputShape& o,
                                                  const TileAxes& t) {
    const TileCounts c = compute_tile_counts(g, o, t);
    const uint64_t full_ic = (g.type == FuncType::MatMul) ? g.input_w : g.input_c;
    const uint64_t nr = c.n_tiles_r, nc = c.n_tiles_c;
    const uint64_t noc = c.n_tiles_oc, nic = c.n_tiles_ic;
    AggregateVolumes a{};
    switch (g.type) {
        case FuncType::Conv2D: {
            const uint64_t SH = 1ull * o.h * g.stride + nr * (g.kernel_h - 1) * g.dilation;
            const uint64_t SW = 1ull * o.w * g.stride + nc * (g.kernel_w - 1) * g.dilation;
            a.input_elems  = SH * SW * full_ic * noc;
            a.weight_elems = 1ull * g.kernel_h * g.kernel_w * full_ic * o.c * nr * nc;
            break;
        }
        case FuncType::DepthwiseConv2D: {
            const uint64_t SH = 1ull * o.h * g.stride + nr * (g.kernel_h - 1) * g.dilation;
            const uint64_t SW = 1ull * o.w * g.stride + nc * (g.kernel_w - 1) * g.dilation;
            a.input_elems  = SH * SW * full_ic * noc;
            a.weight_elems = 1ull * g.kernel_h * g.kernel_w * o.c * nr * nc * nic;
            break;
        }
        case FuncType::FullyConnected: {
            a.input_elems  = full_ic * nr * nc * noc;
            a.weight_elems = full_ic * 1ull * o.c * nr * nc;
            break;
        }
        case FuncType::MatMul: {
            a.input_elems  = 1ull * o.h * full_ic * nc * noc;
            a.weight_elems = full_ic * 1ull * o.w * nr * noc;
            break;
        }
        case FuncType::Pooling: {
            const uint64_t SH = 1ull * o.h * g.stride + nr * (g.kernel_h - 1) * g.dilation;
            const uint64_t SW = 1ull * o.w * g.stride + nc * (g.kernel_w - 1) * g.dilation;
            a.input_elems  = SH * SW * o.c * nic;
            a.weight_elems = 0;
            break;
        }
        case FuncType::Activation: {
            a.input_elems  = 1ull * o.h * o.w * o.c * nic;
            a.weight_elems = 0;
            break;
        }
        case FuncType::ElementWise: {
            a.input_elems  = 1ull * o.h * o.w * o.c * nic;
            a.weight_elems = 1ull * o.h * o.w * o.c * nic;
            break;
        }
        default:
            break;
    }
    return a;
}

}  // namespace flexnpu_sim
