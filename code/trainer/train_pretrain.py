import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm_x = x.norm(dim=-1, keepdim=True)
        rms = norm_x / (x.size(-1) ** 0.5)
        return self.scale * (x / (rms + self.eps))