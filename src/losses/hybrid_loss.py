import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSSLLoss(nn.Module):
    """
    Combined loss for Hybrid SSL:
    SimCLR Loss (Contrastive) + MSE Loss (Reconstruction)
    """
    def __init__(self, temperature=0.07, lambda_recon=0.1):
        super(HybridSSLLoss, self).__init__()
        self.temperature = temperature
        self.lambda_recon = lambda_recon
        self.mse_loss = nn.MSELoss()

    def nt_xent_loss(self, z1, z2):
        """
        Standard NT-Xent loss for SimCLR branch.
        """
        batch_size = z1.shape[0]
        # Normalize to unit sphere
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Representations for both views
        representations = torch.cat([z1, z2], dim=0) # [2N, D]
        # Similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T) # [2N, 2N]
        
        # Positive pairs
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0) # [2N]
        
        mask = (~torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)).float()
        
        # Softmax denominator
        exp_sim = torch.exp(similarity_matrix / self.temperature) * mask
        log_prob = positives / self.temperature - torch.log(exp_sim.sum(dim=1))
        
        return -log_prob.mean()

    def forward(self, z1, z2, recon1, recon2, x1, x2):
        # 1. SimCLR Loss
        simclr_l = self.nt_xent_loss(z1, z2)
        
        # 2. Reconstruction Loss (we reconstruct from both views)
        recon_l1 = self.mse_loss(recon1, x1)
        recon_l2 = self.mse_loss(recon2, x2)
        recon_l = (recon_l1 + recon_l2) / 2
        
        # 3. Total Loss
        total_l = simclr_l + self.lambda_recon * recon_l
        
        return total_l, simclr_l, recon_l
