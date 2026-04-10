import torch
import torch.nn as nn
from .resnet3d import resnet18_3d
from .projection_head import ProjectionHead

class MRIEncoder(nn.Module):
    """
    Early Fusion 3D ResNet18 Encoder with Projection Head for SSL.
    """
    def __init__(self, in_channels=4, embedding_dim=128):
        super(MRIEncoder, self).__init__()
        # Backbone (Feature Extractor)
        self.backbone = resnet18_3d(in_channels=in_channels)
        
        # Projection Head (for Contrastive Learning)
        self.projection_head = ProjectionHead(input_dim=512, output_dim=embedding_dim)

    def forward(self, x):
        # x shape: [B, 4, 128, 128, 128]
        features = self.backbone(x) # Output shape: [B, 512]
        embeddings = self.projection_head(features) # Output shape: [B, 128]
        return embeddings

    def get_features(self, x):
        """Extract only backbone features (useful for downstream tasks/FAISS)"""
        return self.backbone(x)
