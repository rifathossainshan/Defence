import torch
import torch.nn as nn

class ConcatFusion(nn.Module):
    """
    Concatenates branch features and projects them using an MLP.
    Input: list of [B, 128] tensors
    Output: [B, 512]
    """
    def __init__(self, input_dim=128, num_branches=4, output_dim=512):
        super(ConcatFusion, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * num_branches, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, features):
        # features: list of tensors [B, 128]
        x = torch.cat(features, dim=1) # [B, 512]
        return self.mlp(x)

class AverageFusion(nn.Module):
    """
    Averages branch features.
    Input: list of [B, 128] tensors
    Output: [B, 128]
    """
    def __init__(self):
        super(AverageFusion, self).__init__()

    def forward(self, features):
        # features: list of tensors [B, 128]
        x = torch.stack(features, dim=0) # [4, B, 128]
        return torch.mean(x, dim=0) # [B, 128]

class AttentionFusion(nn.Module):
    """
    Placeholder for future advanced fusion strategies.
    """
    def __init__(self):
        super(AttentionFusion, self).__init__()
    
    def forward(self, features):
        raise NotImplementedError("AttentionFusion is a placeholder for a later phase.")
