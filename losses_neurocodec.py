
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        """
        Computes Pearson Correlation Coefficient loss: 1 - PCC
        Input shapes: (B, T) or (B, 1, T)
        """
        # Flatten to (B, T)
        if pred.dim() > 2:
            pred = pred.view(pred.shape[0], -1)
        if target.dim() > 2:
            target = target.view(target.shape[0], -1)
            
        # Center the data (subtract mean)
        pred_mean = pred - pred.mean(dim=1, keepdim=True)
        target_mean = target - target.mean(dim=1, keepdim=True)
        
        # Normalize
        pred_norm = torch.norm(pred_mean, p=2, dim=1)
        target_norm = torch.norm(target_mean, p=2, dim=1)
        
        # Compute Cosine Similarity between centered vectors (which is PCC)
        # Avoid division by zero
        eps = 1e-8
        correlation = (pred_mean * target_mean).sum(dim=1) / (pred_norm * target_norm + eps)
        
        # Loss = 1 - Mean Correlation
        loss = 1 - correlation.mean()
        
        return loss, correlation.mean()

class EnvelopeMatcher(nn.Module):
    def __init__(self, target_rate=128, audio_rate=44100):
        super().__init__()
        self.target_rate = target_rate
        self.audio_rate = audio_rate
        self.pool_kernel = int(audio_rate / target_rate)
        
    def extract_envelope(self, audio):
        """
        Extracts Amplitude Envelope from Audio and downsamples to Target Rate.
        Audio: (B, 1, T_audio)
        Returns: (B, T_eeg)
        """
        # 1. Absolute value (Rectification)
        abs_audio = torch.abs(audio)
        
        # 2. Average Pooling to downsample
        # Kernel size matches the resampling ratio
        envelope = F.avg_pool1d(abs_audio, kernel_size=self.pool_kernel, stride=self.pool_kernel)
        
        # Squeeze channel dim: (B, 1, T) -> (B, T)
        return envelope.squeeze(1)

class NeuroCodecLoss(nn.Module):
    def __init__(self, lambda_recon=1.0, lambda_env=1.0):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_env = lambda_env
        self.mse = nn.MSELoss()
        self.env_loss = PearsonCorrelationLoss()
        self.env_extractor = EnvelopeMatcher()
        
    def forward(self, z_pred, z_target, env_pred=None, clean_audio=None):
        """
        z_pred: (B, 1024, T_audio_tokens)
        z_target: (B, 1024, T_audio_tokens)
        env_pred: (B, T_eeg) - Predicted Envelope from EEG
        clean_audio: (B, 1, T_raw) - Ground Truth Audio
        """
        # 1. Reconstruction Loss (MSE on Latents)
        loss_recon = self.mse(z_pred, z_target)
        
        # 2. Envelope Loss (PCC on EEG -> Audio Envelope)
        loss_env = torch.tensor(0.0, device=z_pred.device)
        pcc_score = torch.tensor(0.0, device=z_pred.device)
        
        if env_pred is not None and clean_audio is not None:
            # Extract GT Envelope
            with torch.no_grad():
                env_gt = self.env_extractor.extract_envelope(clean_audio)
            
            # Ensure shapes match (handle rounding errors in length)
            min_len = min(env_pred.shape[-1], env_gt.shape[-1])
            env_pred = env_pred[..., :min_len]
            env_gt = env_gt[..., :min_len]
            
            loss_env, pcc_score = self.env_loss(env_pred, env_gt)
            
        # Total Loss
        loss_total = (self.lambda_recon * loss_recon) + (self.lambda_env * loss_env)
        
        return loss_total, {
            "loss_recon": loss_recon.item(),
            "loss_env": loss_env.item(),
            "pcc": pcc_score.item()
        }
