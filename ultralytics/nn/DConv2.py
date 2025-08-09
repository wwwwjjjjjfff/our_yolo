import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


# 可变形卷积DConv
# By CSDN 小哥谈
class DConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, act=True):
        super(DConv, self).__init__()
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=k, stride=1, padding=p, groups=g, bias=False)
        deformable_groups = 1
        offset_channels = 18
        mask_channels = 9
        self.conv2_offset = nn.Conv2d(c2, deformable_groups * offset_channels, kernel_size=k, stride=s, padding=p)
        self.conv2_mask = nn.Conv2d(c2, deformable_groups * mask_channels, kernel_size=k, stride=s, padding=p)
        # init_mask = torch.Tensor(np.zeros([mask_channels, 3, 3, 3])) + np.array([0.5])
        # self.conv2_mask.weight = torch.nn.Parameter(init_mask)
        self.conv2 = DeformConv2d(c2, c2, kernel_size=k, stride=s, padding=1, bias=True)

        self.bn1 = nn.BatchNorm2d(c2)
        self.act1 = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        offset = self.conv2_offset(x)
        mask = torch.sigmoid(self.conv2_mask(x))
        x = self.act2(self.bn2(self.conv2(x, offset=offset, mask=mask)))

        return x