import torch
import torch.nn as nn

class GroupNorm2D(nn.Moudle):
    def __init__(self, channel_num, group_num, eps=1e-5):
        super(GroupNorm2D, self).__init__()
        self.channel_num = channel_num
        self.group_num = group_num
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channel_num, 1, 1))
        self.beta = nn.Parameter(torch.ones(channel_num, 1, 1))

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, self.channel_num / self.group_num, H, W)
        mean = x.mean(dim=2, keep_dims=True)
        std = x.std(dim=2, keep_dims=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta