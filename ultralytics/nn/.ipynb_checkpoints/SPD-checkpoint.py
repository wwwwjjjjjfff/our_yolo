import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import autopad

class SPDConv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        c1 = c1 * 4
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.act(self.conv(x))


# class SPDConv(nn.Module):
#     def __init__(self, c1, c2, k=3, s=1):  # 输入c1需自动扩增4倍
#         super().__init__()
#         self.conv = nn.Conv2d(c1 * 4, c2, k, s, padding=autopad(k))  # 关键修正点
#         self.bn = nn.BatchNorm2d(c2)

#     def forward(self, x):
#         x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
#                        x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
#         return self.bn(self.conv(x))



