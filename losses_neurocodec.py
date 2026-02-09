import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuroCodecLoss(nn.Module):
    def __init__(self, lambda_cosine=1.0, lambda_l1=0.0):
        super().__init__()
        self.lambda_cosine = lambda_cosine
        self.lambda_l1 = lambda_l1
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
    def forward(self, z_pred, z_target):
        # 1. MSE Loss (Main reconstruction)
        loss_mse = self.mse(z_pred, z_target)
        
        # 2. Cosine Embedding Loss (Directionality)
        # z: (B, 1024, T)
        # Flatten to (B*T, 1024) for cosine sim
        z_pred_flat = z_pred.transpose(1, 2).reshape(-1, z_pred.shape[1])
        z_target_flat = z_target.transpose(1, 2).reshape(-1, z_target.shape[1])
        
        # Target for CosineEmbeddingLoss is 1 (maximize similarity)
        target_ones = torch.ones(z_pred_flat.shape[0], device=z_pred.device)
        
        # maximization of cosine similarity = minimization of (1 - cos)
        loss_cos = nn.functional.cosine_embedding_loss(z_pred_flat, z_target_flat, target_ones)
        
        # 3. L1 Loss (optional, for sparsity/sharpness)
        loss_l1 = 0.0
        if self.lambda_l1 > 0:
            loss_l1 = self.l1(z_pred, z_target)
            
        # Total Loss
        total_loss = loss_mse + (self.lambda_cosine * loss_cos) + (self.lambda_l1 * loss_l1)
        
        return total_loss, {
            "mse": loss_mse.item(),
            "cosine": loss_cos.item(),
            "l1": loss_l1.item() if self.lambda_l1 > 0 else 0.0
        }

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, z_pred, z_target):
        """
        Contrastive Loss for Temporal Alignment.
        z_pred: (B, C, T)
        z_target: (B, C, T)
        """
        # 1. Normalize Vectors (Cosine Sim requires normalization)
        z_pred = F.normalize(z_pred, p=2, dim=1)   # (B, C, T)
        z_target = F.normalize(z_target, p=2, dim=1) # (B, C, T)
        
        # 2. Permute to (B, T, C)
        B, C, T = z_pred.shape
        z_pred = z_pred.transpose(1, 2)   # (B, T, C)
        z_target = z_target.transpose(1, 2) # (B, T, C)
        
        # 3. Compute Similarity Matrix (B, T, T)
        # For each batch element, we compute T x T matrix
        # logits[b, i, j] = sim(pred[b, i], target[b, j]) / temp
        logits = torch.bmm(z_pred, z_target.transpose(1, 2)) / self.temperature
        
        # 4. Create Labels
        # The correct match for Pred[t] is Target[t] (diagonal)
        labels = torch.arange(T, device=z_pred.device).unsqueeze(0).repeat(B, 1) # (B, T)
        
        # 5. Flatten and Compute Loss
        # We classify along the last dimension (Target Time steps)
        loss = self.cross_entropy(logits.view(-1, T), labels.view(-1))
        
        return loss
