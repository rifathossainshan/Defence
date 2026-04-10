import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLRLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.
    """
    def __init__(self, temperature=0.07):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        # z1, z2 shape: [B, D]
        batch_size = z1.shape[0]
        device = z1.device
        
        # 1. L2 Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 2. Combine all representations: [2B, D]
        representations = torch.cat([z1, z2], dim=0)
        
        # 3. Compute Similarity Matrix (Cosine Similarity): [2B, 2B]
        sim_matrix = torch.matmul(representations, representations.T)
        
        # 4. Scale by temperature
        sim_matrix = sim_matrix / self.temperature
        
        # 5. Mask out self-similarities (diagonal)
        mask = torch.eye(2 * batch_size, device=device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        # 6. Targets for labels
        # For view1[i], positive is view2[i] (which is at index i + batch_size in sim_matrix)
        # For view2[i], positive is view1[i] (which is at index i in sim_matrix)
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size),   # [B...2B-1]
            torch.arange(0, batch_size)               # [0...B-1]
        ], dim=0).to(device)
        
        # 7. Compute Cross Entropy Loss
        loss = self.criterion(sim_matrix, targets)
        
        return loss
