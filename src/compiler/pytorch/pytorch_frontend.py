#!/usr/bin/env python3
"""PyTorch frontend: torch.nn.Module -> FlexNPUSim network CSV.

This is the "pytorch" half of "everything controlled via conf + pytorch": a
PyTorch model becomes simulator input. It walks the model with forward hooks to
capture per-layer I/O shapes, then maps supported layers (Conv2d, Linear,
MaxPool2d, AvgPool2d, ReLU) to the network CSV schema consumed by
dnn_image_compiler.

Hardware mapping params (pe_type / k / g / s / tile) are NOT derivable from the
model — they belong to the target NPU. Pass them via `hw` (defaults match a
systolic NVDLA-class array). So: model = workload (shapes), hw = mapping.

Usage:
    from pytorch_frontend import write_network_csv
    write_network_csv(model, (1, 3, 224, 224), "net.csv")
"""

import csv
import operator

import torch
import torch.nn as nn
import torch.fx as _fx
from torch.fx.passes.shape_prop import ShapeProp as _ShapeProp

HEADER = ("layer_id,function_id,layer_name,layer_type,function_type,"
          "i1_connect,i2_connect,input_h,input_w,input_c,kernel_h,kernel_w,"
          "kernel_c,kernel_count,stride,padding,dilation,tile_h,tile_w,pe_type,"
          "k,g,s,psum_kb,preprocess_type,preprocess_param0,preprocess_param1"
          ).split(",")

NO_CONNECT = 4294967295  # 0xFFFFFFFF — "no upstream function"

HW_DEFAULTS = dict(pe_type="systolic", k=16, g=2, s=8,
                   tile_h=16, tile_w=16, psum_kb=0)

# BatchNorm is intentionally NOT hooked: at inference it folds into the
# preceding conv, so it contributes no separate compute layer. Depthwise convs
# (groups == in_channels, MobileNet) map to depthwise_conv2d.
SUPPORTED = (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AvgPool2d,
             nn.AdaptiveAvgPool2d, nn.ReLU, nn.ReLU6)


def trace_layers(model, input_shape):
    """One forward pass with hooks → [(module, in_shape, out_shape), ...]."""
    records = []
    hooks = []

    def make_hook(m):
        def hook(_m, inp, out):
            records.append((m, tuple(inp[0].shape), tuple(out.shape)))
        return hook

    for m in model.modules():
        if isinstance(m, SUPPORTED):
            hooks.append(m.register_forward_hook(make_hook(m)))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_shape))
    for h in hooks:
        h.remove()
    return records


def _row(idx, name, lt, ft, i1, ih, iw, ic, kh, kw, kc, kcnt, stride, pad, hw):
    return [idx, 0, name, lt, ft, i1, NO_CONNECT, ih, iw, ic, kh, kw, kc, kcnt,
            stride, pad, 1, hw["tile_h"], hw["tile_w"], hw["pe_type"],
            hw["k"], hw["g"], hw["s"], hw["psum_kb"], "none", 0, 0]


def _pair(v):
    return v if isinstance(v, int) else v[0]


def to_csv_rows(records, hw=HW_DEFAULTS):
    rows = []
    prev = NO_CONNECT
    for i, (m, ish, osh) in enumerate(records):
        if isinstance(m, nn.Conv2d):
            ic, ih, iw = ish[1], ish[2], ish[3]
            kh, kw = m.kernel_size
            is_dw = m.groups > 1 and m.groups == ic      # depthwise (MobileNet)
            lt = "depthwise_conv2d" if is_dw else "conv2d"
            rows.append(_row(i, f"{'dw' if is_dw else 'conv'}{i}", lt, lt, prev,
                             ih, iw, ic, kh, kw, 1 if is_dw else ic, m.out_channels,
                             m.stride[0], m.padding[0], hw))
        elif isinstance(m, nn.Linear):
            rows.append(_row(i, f"fc{i}", "fully_connected", "fully_connected",
                             prev, 1, 1, m.in_features, 1, 1, m.in_features,
                             m.out_features, 1, 0, hw))
        elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
            ic, ih, iw = ish[1], ish[2], ish[3]
            k = _pair(m.kernel_size)
            st = _pair(m.stride) if m.stride else k
            rows.append(_row(i, f"pool{i}", "pooling", "pooling", prev,
                             ih, iw, ic, k, k, 0, ic, st, 0, hw))
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            ic, ih, iw = ish[1], ish[2], ish[3]
            oh = osh[2] if len(osh) == 4 else 1
            k = max(1, ih // max(1, oh))
            rows.append(_row(i, f"pool{i}", "pooling", "pooling", prev,
                             ih, iw, ic, k, k, 0, ic, k, 0, hw))
        elif isinstance(m, (nn.ReLU, nn.ReLU6)):
            ic, ih, iw = (ish[1], ish[2], ish[3]) if len(ish) == 4 else (ish[1], 1, 1)
            rows.append(_row(i, f"act{i}", "activation", "activation", prev,
                             ih, iw, ic, 1, 1, 0, ic, 1, 0, hw))
        else:
            continue
        prev = i
    return rows


def _fx_shape(node):
    tm = node.meta.get("tensor_meta", None)
    return tuple(tm.shape) if tm is not None else None


def trace_layers_fx(model, input_shape, hw=HW_DEFAULTS):
    """Graph trace (torch.fx) — handles residual/branching (ResNet): a functional
    `add` becomes a "residual" row with real i1/i2 connectivity, `torch.cat`
    becomes "concat". BatchNorm/Flatten/Dropout pass through (folded). Sequential
    models trace to the same rows as the forward-hook path."""
    gm = _fx.symbolic_trace(model)
    _ShapeProp(gm).propagate(torch.zeros(*input_shape))
    modules = dict(model.named_modules())
    rows = []
    resolved = {}  # fx node -> emitted row index (or NO_CONNECT / upstream idx)

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            resolved[node] = NO_CONNECT
        elif node.op == "output":
            continue
        elif node.op == "call_module":
            m = modules[node.target]
            i1 = resolved.get(node.args[0], NO_CONNECT)
            ish = _fx_shape(node.args[0])
            idx = len(rows)
            if isinstance(m, nn.Conv2d):
                ic, ih, iw = ish[1], ish[2], ish[3]
                kh, kw = m.kernel_size
                dw = m.groups > 1 and m.groups == ic
                lt = "depthwise_conv2d" if dw else "conv2d"
                rows.append(_row(idx, f"{'dw' if dw else 'conv'}{idx}", lt, lt, i1,
                                 ih, iw, ic, kh, kw, 1 if dw else ic, m.out_channels,
                                 m.stride[0], m.padding[0], hw))
                resolved[node] = idx
            elif isinstance(m, nn.Linear):
                rows.append(_row(idx, f"fc{idx}", "fully_connected", "fully_connected",
                                 i1, 1, 1, m.in_features, 1, 1, m.in_features,
                                 m.out_features, 1, 0, hw))
                resolved[node] = idx
            elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
                ic, ih, iw = ish[1], ish[2], ish[3]
                k = _pair(m.kernel_size); st = _pair(m.stride) if m.stride else k
                rows.append(_row(idx, f"pool{idx}", "pooling", "pooling", i1,
                                 ih, iw, ic, k, k, 0, ic, st, 0, hw))
                resolved[node] = idx
            elif isinstance(m, nn.AdaptiveAvgPool2d):
                ic, ih, iw = ish[1], ish[2], ish[3]
                osh = _fx_shape(node); oh = osh[2] if osh and len(osh) == 4 else 1
                k = max(1, ih // max(1, oh))
                rows.append(_row(idx, f"pool{idx}", "pooling", "pooling", i1,
                                 ih, iw, ic, k, k, 0, ic, k, 0, hw))
                resolved[node] = idx
            elif isinstance(m, (nn.ReLU, nn.ReLU6)):
                ic, ih, iw = (ish[1], ish[2], ish[3]) if len(ish) == 4 else (ish[1], 1, 1)
                rows.append(_row(idx, f"act{idx}", "activation", "activation", i1,
                                 ih, iw, ic, 1, 1, 0, ic, 1, 0, hw))
                resolved[node] = idx
            else:  # BatchNorm / Dropout / Flatten / Identity -> fold, pass through
                resolved[node] = i1
        else:  # call_function / call_method
            tgt = node.target
            if tgt in (operator.add, torch.add):  # residual skip-add
                i1 = resolved.get(node.args[0], NO_CONNECT)
                i2 = resolved.get(node.args[1], NO_CONNECT)
                ish = _fx_shape(node); ic, ih, iw = ish[1], ish[2], ish[3]
                idx = len(rows)
                r = _row(idx, f"add{idx}", "elementwise", "residual", i1,
                         ih, iw, ic, 1, 1, 0, ic, 1, 0, hw)
                r[6] = i2  # i2_connect = the skip input
                rows.append(r); resolved[node] = idx
            elif tgt is torch.cat:  # concatenation
                ins = node.args[0]
                ins = list(ins) if isinstance(ins, (list, tuple)) else [ins]
                ish = _fx_shape(node); ic, ih, iw = ish[1], ish[2], ish[3]
                idx = len(rows)
                r = _row(idx, f"cat{idx}", "elementwise", "concat",
                         resolved.get(ins[0], NO_CONNECT), ih, iw, ic, 1, 1, 0, ic, 1, 0, hw)
                if len(ins) > 1:
                    r[6] = resolved.get(ins[1], NO_CONNECT)
                rows.append(r); resolved[node] = idx
            else:  # flatten / functional relu / view / getattr -> pass through
                a = node.args[0] if node.args else None
                resolved[node] = resolved.get(a, NO_CONNECT) if a is not None else NO_CONNECT
    return rows


def write_network_csv(model, input_shape, path, hw=HW_DEFAULTS):
    """Trace a model to a network CSV. Uses torch.fx (handles residual/branching);
    falls back to forward-hook tracing for models fx cannot symbolically trace."""
    try:
        rows = trace_layers_fx(model, input_shape, hw)
    except Exception:
        rows = to_csv_rows(trace_layers(model, input_shape), hw)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    # Self-test: a small CNN → CSV, verify shape propagation and schema.
    import sys, tempfile, os

    # MobileNet-style block: standard conv → depthwise → pointwise (each with
    # BatchNorm that folds away) → adaptive global pool → classifier.
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=2, padding=1),
        nn.BatchNorm2d(16), nn.ReLU(),
        nn.Conv2d(16, 16, 3, 1, 1, groups=16),   # depthwise
        nn.BatchNorm2d(16), nn.ReLU6(),
        nn.Conv2d(16, 32, 1, 1, 0),              # pointwise (1x1)
        nn.BatchNorm2d(32), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 10),
    )
    path = os.path.join(tempfile.gettempdir(), "pytorch_frontend_demo.csv")
    rows = write_network_csv(model, (1, 3, 32, 32), path)

    def fail(msg):
        print("FAIL:", msg); sys.exit(1)

    if any(len(r) != len(HEADER) for r in rows):
        fail("row column count != header")
    types = [r[3] for r in rows]
    # BatchNorm folded → no bn rows
    if "batchnorm" in "".join(types) or len([r for r in rows if "bn" in r[2]]):
        fail(f"BatchNorm should be folded away: {types}")
    # depthwise detected; standard conv + pointwise(1x1) both map to conv2d
    if "depthwise_conv2d" not in types:
        fail(f"depthwise not detected: {types}")
    if types.count("conv2d") != 2:
        fail(f"expected 2 conv2d (std + pointwise), got {types}")
    # adaptive avg pool → pooling
    if "pooling" not in types:
        fail(f"adaptive pool not mapped: {types}")
    dw = [r for r in rows if r[3] == "depthwise_conv2d"][0]
    if dw[12] != 1:  # depthwise kernel_c == 1
        fail(f"depthwise kernel_c != 1: {dw}")
    fc = [r for r in rows if r[3] == "fully_connected"][0]
    if fc[13] != 10:
        fail(f"fc out != 10: {fc}")

    print(f"PASS: {len(rows)} layers (depthwise + pointwise + BN-folded + adaptive pool) → {path}")
    for r in rows:
        print("  ", r[2], r[3], "in=%dx%dx%d" % (r[7], r[8], r[9]),
              "k=%dx%d" % (r[10], r[11]), "out=%d" % r[13])
