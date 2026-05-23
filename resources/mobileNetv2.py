"""
MobileNetV2 adapted for your audio pipeline (ACDNet-style SFEB) with fixes to avoid ~random accuracy:

Key changes vs your previous version:
1) Outputs LOGITS by default (no Softmax). Use CrossEntropyLoss with class indices.
2) Removes permute hack: keeps SFEB channels as channels (MobileNet expects channels-first).
3) Makes SFEB pooling less aggressive (kernel=(1,10)) to avoid collapsing temporal resolution.
4) Disables SE / DropPath / classifier dropout by default (re-enable later once it learns).
5) Uses a stride-1 stem by default to avoid excessive early downsampling on short time axes.

Recommended training setup:
- criterion = nn.CrossEntropyLoss()
- y targets: integer class indices (0..num_classes-1)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

__all__ = ["MobileNetV2Audio", "mobilenetv2_audio"]
#__all__ = ["MobileNetV2", "mobileNetv2"]
#__all__ = ["MobileNetV2", "mobileNetv2"]


# ----------------------------
# Utilities
# ----------------------------

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    def __init__(self, in_ch: int, se_ratio: float = 0.125):
        super().__init__()
        hidden = max(8, int(in_ch * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, hidden, 1, bias=True)
        self.fc2 = nn.Conv2d(hidden, in_ch, 1, bias=True)

    def forward(self, x):
        s = self.pool(x)
        s = self.fc1(s)
        s = F.relu(s, inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class DropPath(nn.Module):
    """Stochastic depth: per-sample path drop."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


# ----------------------------
# Inverted Residual block
# ----------------------------

class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        use_se: bool = False,          # default OFF for stability on small datasets
        drop_path: float = 0.0,        # default OFF
    ):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.identity = (stride == 1 and inp == oup)
        self.use_se = use_se

        layers = []

        # pointwise expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        dw_in = hidden_dim if expand_ratio != 1 else inp

        # depthwise
        layers.extend([
            nn.Conv2d(dw_in, dw_in, 3, stride, 1, groups=dw_in, bias=False),
            nn.BatchNorm2d(dw_in),
            nn.ReLU6(inplace=True),
        ])

        self.se = SqueezeExcite(dw_in) if use_se else nn.Identity()

        # project (linear bottleneck)
        layers.extend([
            nn.Conv2d(dw_in, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        if self.identity:
            out = self.drop_path(out) + x
        return out


# ----------------------------
# MobileNetV2 with audio SFEB (fixed)
# ----------------------------

class MobileNetV2Audio(nn.Module):
    """
    Audio front-end (SFEB):
        - 1x9 and 1x5 convs over time axis
        - moderate pooling to avoid collapsing time dimension
        - outputs (N, C, 1, W) and feeds as channels-first (no permute)

    Backbone:
        - standard MobileNetV2 inverted residual stages
        - defaults: no SE, no DropPath, no classifier dropout (enable later)

    Output:
        - returns logits (default) -> use CrossEntropyLoss
    """
    def __init__(
        self,
        num_classes: int = 11,
        width_mult: float = 1.0,
        round_nearest: int = 8,
        sfeb_out_ch: int = 32,         # SFEB channel count
        sfeb_pool_k: int = 10,         # reduced from 50 to avoid over-pooling
        stem_stride: int = 1,          # use 1 to avoid early collapse
        use_se: bool = False,          # OFF by default
        drop_path_rate: float = 0.0,   # OFF by default
        cls_dropout: float = 0.0,      # OFF by default
    ):
        super().__init__()

        # ----- Audio front-end (SFEB) -----
        conv1, bn1 = self._make_layers(1, 8, (1, 9), (1, 2))
        conv2, bn2 = self._make_layers(8, sfeb_out_ch, (1, 5), (1, 2))
        self.sfeb = nn.Sequential(
            conv1, bn1, nn.ReLU(inplace=True),
            conv2, bn2, nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, sfeb_pool_k))
        )

        # ----- MobileNetV2 settings (t, c, n, s) -----
        cfgs = [
            # t,   c,  n, s
            [1,   16, 1, 1],
            [6,   24, 2, 2],
            [6,   32, 3, 2],
            [6,   64, 4, 2],
            [6,   96, 3, 1],
            [6,  160, 3, 2],
            [6,  320, 1, 1],
        ]

        # stem: IMPORTANT -> in_ch = SFEB output channels (channels-first)
        input_channel = _make_divisible(32 * width_mult, round_nearest)
        features = [conv_3x3_bn(sfeb_out_ch, input_channel, stem_stride)]

        # stochastic depth decay across blocks (if enabled)
        total_blocks = sum(c[2] for c in cfgs)
        b_idx = 0

        for t, c, n, s in cfgs:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                drop = drop_path_rate * float(b_idx) / max(1, total_blocks - 1) if drop_path_rate > 0 else 0.0
                features.append(
                    InvertedResidual(
                        input_channel, output_channel, stride,
                        expand_ratio=t, use_se=use_se, drop_path=drop
                    )
                )
                input_channel = output_channel
                b_idx += 1

        self.features = nn.Sequential(*features)

        # last 1x1 conv
        last_channel = 1280 if width_mult <= 1.0 else _make_divisible(1280 * width_mult, round_nearest)
        self.conv = conv_1x1_bn(input_channel, last_channel)

        # head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=cls_dropout) if cls_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(last_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.sfeb(x)           # (N, Csfeb, 1, W')
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits              # logits -> CrossEntropyLoss

    # --- helpers ---

    def _make_layers(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=0, bias=False):
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
        bn = nn.BatchNorm2d(out_channels)
        return conv, bn

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.zeros_(m.bias)


def mobilenetv2_audio(**kwargs):
    """Factory to match your import style."""
    return MobileNetV2Audio(**kwargs)
