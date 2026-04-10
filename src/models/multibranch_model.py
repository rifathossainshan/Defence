import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusion_model import ConcatFusion

class ModalityBranchEncoder(nn.Module):
    """
    Lightweight shallow 3D encoder branch for a single MRI modality.
    Input: [B, 1, 128, 128, 128]
    Output: [B, 128]
    """
    def __init__(self, in_channels=1, base_channels=16, feature_dim=128):
        super(ModalityBranchEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # Layer 1: 128 -> 64
            nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            
            # Layer 2: 64 -> 32
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Layer 3: 32 -> 16
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(base_channels * 4, feature_dim)

    def forward(self, x):
        # x: [B, 1, 128, 128, 128]
        feat = self.encoder(x) # [B, 64, 1, 1, 1]
        feat = feat.view(feat.size(0), -1) # [B, 64]
        return self.fc(feat) # [B, 128]

class ProjectionHead(nn.Module):
    """
    MLP Head for contrastive learning.
    512 -> 256 -> 128
    """
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class ReconstructionHead(nn.Module):
    """
    Decodes the fused latent vector back to the original 4-modality volume.
    Input: [B, 512]
    Output: [B, 4, 128, 128, 128]
    """
    def __init__(self, latent_dim=512, out_channels=4):
        super(ReconstructionHead, self).__init__()
        # Initial projection to spatial shape
        self.fc = nn.Linear(latent_dim, 64 * 8 * 8 * 8)
        
        self.decoder = nn.Sequential(
            # 8 -> 16
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # 16 -> 32
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            
            # 32 -> 64
            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            
            # 64 -> 128
            nn.ConvTranspose3d(8, out_channels, kernel_size=4, stride=2, padding=1),
            # Final output skip activation or use Tanh if normalized to [-1,1]
            # Given we use Z-score, we keep it linear or Tanh.
        )

    def forward(self, x):
        # x: [B, 512]
        x = self.fc(x)
        x = x.view(x.size(0), 64, 8, 8, 8) # [B, 64, 8, 8, 8]
        return self.decoder(x) # [B, 4, 128, 128, 128]

class MultiBranchHybridSSLModel(nn.Module):
    """
    Final Main Model: Modality-Aware Hybrid SSL Model.
    - 4 Independent branches for T1, T1ce, T2, FLAIR.
    - Concat Fusion.
    - Projection Head for retrieval embeddings.
    - Reconstruction Head for SSL detail learning.
    """
    def __init__(self, feature_dim=128, fused_dim=512, embedding_dim=128):
        super(MultiBranchHybridSSLModel, self).__init__()
        
        # 1. Individual Branches
        self.branch_t1 = ModalityBranchEncoder(feature_dim=feature_dim)
        self.branch_t1ce = ModalityBranchEncoder(feature_dim=feature_dim)
        self.branch_t2 = ModalityBranchEncoder(feature_dim=feature_dim)
        self.branch_flair = ModalityBranchEncoder(feature_dim=feature_dim)
        
        # 2. Fusion Block
        self.fusion = ConcatFusion(input_dim=feature_dim, num_branches=4, output_dim=fused_dim)
        
        # 3. Heads
        self.projection_head = ProjectionHead(input_dim=fused_dim, output_dim=embedding_dim)
        self.reconstruction_head = ReconstructionHead(latent_dim=fused_dim, out_channels=4)

    def forward(self, x):
        # x: [B, 4, 128, 128, 128] 
        
        # Split modalities
        x_t1 = x[:, 0:1, :, :, :]
        x_t1ce = x[:, 1:2, :, :, :]
        x_t2 = x[:, 2:3, :, :, :]
        x_flair = x[:, 3:4, :, :, :]
        
        # Branch-wise features
        f_t1 = self.branch_t1(x_t1)
        f_t1ce = self.branch_t1ce(x_t1ce)
        f_t2 = self.branch_t2(x_t2)
        f_flair = self.branch_flair(x_flair)
        
        # Fusion
        fused = self.fusion([f_t1, f_t1ce, f_t2, f_flair]) # [B, 512]
        
        # Heads
        z = self.projection_head(fused) # [B, 128]
        recon = self.reconstruction_head(fused) # [B, 4, 128, 128, 128]
        
        return z, recon

    def get_embeddings(self, x):
        """Helper for retrieval inference."""
        z, _ = self.forward(x)
        return z
