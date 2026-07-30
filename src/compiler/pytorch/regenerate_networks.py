#!/usr/bin/env python3
"""Regenerate network CSVs from canonical PyTorch model definitions.

The hand-authored CSVs were flagged inaccurate. This regenerates them from the
*standard* architecture definitions in PyTorch — a trusted source — via the
PyTorch frontend. VGG16 / MobileNet-v1 (sequential) and ResNet-18 (residual) are
all captured: the frontend's torch.fx graph trace turns functional skip-adds into
"residual" rows with real connectivity.
"""

import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_frontend import write_network_csv  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # src/compiler/pytorch -> repo root


def make_vgg16(num_classes=1000):
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
           512, 512, 512, "M", 512, 512, 512, "M"]
    layers, inc = [], 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(2, 2))
        else:
            layers += [nn.Conv2d(inc, v, 3, padding=1), nn.ReLU(inplace=True)]
            inc = v
    layers += [nn.Flatten(),
               nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True),
               nn.Linear(4096, 4096), nn.ReLU(True),
               nn.Linear(4096, num_classes)]
    return nn.Sequential(*layers)


def make_mobilenet_v1(num_classes=1000):
    def conv_bn(i, o, s):
        return [nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]

    def conv_dw(i, o, s):
        return [nn.Conv2d(i, i, 3, s, 1, groups=i, bias=False), nn.BatchNorm2d(i), nn.ReLU(True),
                nn.Conv2d(i, o, 1, 1, 0, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]

    L = conv_bn(3, 32, 2)
    for i, o, s in [(32, 64, 1), (64, 128, 2), (128, 128, 1), (128, 256, 2),
                    (256, 256, 1), (256, 512, 2)] + [(512, 512, 1)] * 5 + \
                   [(512, 1024, 2), (1024, 1024, 1)]:
        L += conv_dw(i, o, s)
    L += [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(1024, num_classes)]
    return nn.Sequential(*L)


def make_resnet18(num_classes=1000):
    class BasicBlock(nn.Module):
        def __init__(self, inp, out, stride=1):
            super().__init__()
            self.c1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
            self.b1 = nn.BatchNorm2d(out); self.r1 = nn.ReLU()
            self.c2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
            self.b2 = nn.BatchNorm2d(out); self.r2 = nn.ReLU()
            self.down = None
            if stride != 1 or inp != out:
                self.down = nn.Sequential(nn.Conv2d(inp, out, 1, stride, bias=False),
                                          nn.BatchNorm2d(out))

        def forward(self, x):
            idn = self.down(x) if self.down is not None else x
            y = self.r1(self.b1(self.c1(x)))
            y = self.b2(self.c2(y))
            return self.r2(y + idn)

    class ResNet18(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU(),
                                      nn.MaxPool2d(3, 2, 1))
            self.l1 = nn.Sequential(BasicBlock(64, 64),  BasicBlock(64, 64))
            self.l2 = nn.Sequential(BasicBlock(64, 128, 2),  BasicBlock(128, 128))
            self.l3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
            self.l4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
            self.pool = nn.AdaptiveAvgPool2d((1, 1)); self.flat = nn.Flatten()
            self.fc = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.l4(self.l3(self.l2(self.l1(self.stem(x)))))
            return self.fc(self.flat(self.pool(x)))

    return ResNet18()


if __name__ == "__main__":
    out_dir = os.path.join(_REPO, "model/sw/cnn/regenerated")
    os.makedirs(out_dir, exist_ok=True)

    nets = {
        "vgg16": (make_vgg16(), (1, 3, 224, 224)),
        "mobilenetv1": (make_mobilenet_v1(), (1, 3, 224, 224)),
        "resnet18": (make_resnet18(), (1, 3, 224, 224)),
    }
    for name, (model, shape) in nets.items():
        path = os.path.join(out_dir, f"{name}.csv")
        rows = write_network_csv(model, shape, path)
        convs = [r for r in rows if r[3] in ("conv2d", "depthwise_conv2d")]
        fcs = [r for r in rows if r[3] == "fully_connected"]
        res = [r for r in rows if r[4] == "residual"]
        print(f"{name}: {len(rows)} layers ({len(convs)} conv, {len(fcs)} fc, "
              f"{len(res)} residual) → {path}")

    # VGG16 structural check vs the known architecture / paper Table II.
    vgg_rows = write_network_csv(make_vgg16(), (1, 3, 224, 224),
                                 os.path.join(out_dir, "vgg16.csv"))
    vconv = [r for r in vgg_rows if r[3] == "conv2d"]
    assert len(vconv) == 13, f"VGG16 must have 13 conv, got {len(vconv)}"
    # Conv1_1: 224x224, 3→64
    assert vconv[0][7] == 224 and vconv[0][9] == 3 and vconv[0][13] == 64, vconv[0]
    # Conv2_1: 112x112, 64→128
    assert vconv[2][7] == 112 and vconv[2][9] == 64 and vconv[2][13] == 128, vconv[2]
    # Conv3_1: 56x56, 128→256
    assert vconv[4][7] == 56 and vconv[4][9] == 128 and vconv[4][13] == 256, vconv[4]
    # Conv4_1: 28x28, 256→512
    assert vconv[7][7] == 28 and vconv[7][9] == 256 and vconv[7][13] == 512, vconv[7]
    # Conv5_1: 14x14, 512→512
    assert vconv[10][7] == 14 and vconv[10][9] == 512 and vconv[10][13] == 512, vconv[10]
    print("VGG16 conv shapes match the standard architecture / paper Table II:")
    for i, r in enumerate(vconv):
        print(f"  conv{i+1}: {r[7]}x{r[8]} {r[9]}->{r[13]}")
    print("PASS: networks regenerated from PyTorch definitions (trusted source)")
