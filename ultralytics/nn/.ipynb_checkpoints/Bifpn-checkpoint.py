import torch
import torch.nn as nn
import torch.nn.functional as F

class BiFPN(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.first_time = first_time

        # Conv layers
        self.p3_conv = nn.Conv2d(conv_channels[0], num_channels, kernel_size=1, stride=1, padding=0)
        self.p4_conv = nn.Conv2d(conv_channels[1], num_channels, kernel_size=1, stride=1, padding=0)
        self.p5_conv = nn.Conv2d(conv_channels[2], num_channels, kernel_size=1, stride=1, padding=0)

        # Feature scaling layers
        self.p4_downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.p5_downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # Weight layers
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)

        # ReLU
        self.relu = nn.ReLU()

    def forward(self, inputs):
        p3, p4, p5 = inputs

        # Adjust channels
        p3 = self.p3_conv(p3)
        p4 = self.p4_conv(p4)
        p5 = self.p5_conv(p5)

        # Top-down pathway
        p5_upsampled = F.interpolate(p5, scale_factor=2, mode='nearest')
        p4 = self.relu(p4 + p5_upsampled)

        p4_upsampled = F.interpolate(p4, scale_factor=2, mode='nearest')
        p3 = self.relu(p3 + p4_upsampled)

        # Bottom-up pathway
        p3_downsampled = self.p4_downsample(p3)
        p4 = self.relu(p4 + p3_downsampled)

        p4_downsampled = self.p4_downsample(p4)
        p5 = self.relu(p5 + p4_downsampled)

        # Weighted fusion
        p4_w1 = self.relu(self.p4_w1)
        p4_w2 = self.relu(self.p4_w2)
        p5_w1 = self.relu(self.p5_w1)
        p5_w2 = self.relu(self.p5_w2)

        p4 = (p4_w1[0] * p4 + p4_w1[1] * p5_upsampled) / (p4_w1.sum() + self.epsilon)
        p5 = (p5_w1[0] * p5 + p5_w1[1] * p4_downsampled) / (p5_w1.sum() + self.epsilon)

        p3 = (p4_w2[0] * p3 + p4_w2[1] * p4_upsampled + p4_w2[2] * p3) / (p4_w2.sum() + self.epsilon)
        p4 = (p5_w2[0] * p4 + p5_w2[1] * p3_downsampled + p5_w2[2] * p4) / (p5_w2.sum() + self.epsilon)

        return p3, p4, p5