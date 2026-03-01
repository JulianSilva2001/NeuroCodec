import torch
import torch.nn as nn


class NeuroCodecLoss(nn.Module):
    def __init__(self, lambda_recon=1.0):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.mse = nn.MSELoss()

    def forward(self, z_pred, z_target):
        """
        z_pred: (B, 1024, T_audio_tokens)
        z_target: (B, 1024, T_audio_tokens)
        """
        loss_recon = self.mse(z_pred, z_target)
        loss_total = self.lambda_recon * loss_recon
        return loss_total, {
            "loss_recon": loss_recon.item()
        }
