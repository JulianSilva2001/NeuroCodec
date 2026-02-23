import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class MelSpectrogramLoss(nn.Module):
    """
    Computes L1 distance between log-mel spectrograms.
    Supports multi-scale loss by accepting a list of n_mels and window_lengths.
    
    Adapted for 16kHz audio.
    """
    def __init__(self, 
                 sample_rate=16000,
                 n_mels=[80, 512, 1024], 
                 window_lengths=[2048, 512], 
                 hop_lengths=None, # if None, defaults to window // 4
                 f_min=0.0,
                 f_max=None,
                 log_base=10.0):
        super().__init__()
        
        self.transforms = nn.ModuleList()
        
        # Create a MelSpectrogram transform for each combination of n_mels and window_length?
        # Or just iterate through them?
        # DAC implementation iterates through window_lengths and n_mels together or separate?
        # DAC: zip(n_mels, stft_params). So it pairs them.
        # Let's assume we want to cover different resolutions.
        
        # If lengths match, zip them. If not, maybe cartesian product? 
        # For simplicity and to match DAC style (multi-scale), let's ensure lists are same length or broadcast.
        # But commonly we just want a few specific scales.
        # Let's simplify: A list of configs. 
        
        # Config 1: High Freq Res (Long Window, More Mels)
        # Config 2: Low Freq Res / Temporal (Short Window, Fewer Mels)
        
        # Let's use the provided lists and zip them, assuming the user provides matching lists 
        # or we cycle through them. 
        # DAC implementation: 
        # for n_mels, ..., s in zip(self.n_mels, ..., self.stft_params)
        
        # Let's enforce equal length for simplicity or just use defaults.
        # Defaults in plan: n_mels=[80, 512, 1024] (Wait, 512/1024 mels for 16k is a lot. bins=window/2+1. 
        # For 2048 window, n_fft/2+1 = 1025 bins. So 1024 mels is almost linear spec.
        # For 512 window, 257 bins. 512 mels is impossible.
        # 
        # Correction for 16kHz:
        # Window 2048 -> 1025 bins. Mels < 1025. 80 is standard. 
        # Window 512 -> 257 bins. Mels < 257.
        # 
        # DAC defaults were for 44kHz. 
        # Let's pick reasonable defaults for 16kHz:
        # Scale 1: Window 1024 (64ms), Hop 256, 80 Mels
        # Scale 2: Window 512 (32ms), Hop 128, 64 Mels
        # Scale 3: Window 256 (16ms), Hop 64, 32 Mels
        #
        # Or just follow DAC pattern but ensure n_mels <= n_fft/2 + 1.
        
        self.configs = []
        if len(n_mels) != len(window_lengths):
             # Fallback or simple logic. 
             # Let's define specific robust scales for 16k if defaults are passed.
             pass

        # Using explicit robust defaults for 16kHz if passed args match "DAC-like" placeholder
        # But to respect constructor args, I will implement flexible logic.
        
        # Logic: Create a transform for each window_length, cycling n_mels if needed.
        
        # Actually, let's implement exactly what was in the plan/DAC logic:
        # "zip(n_mels, ... stft_params)"
        # So we need len(n_mels) == len(window_lengths).
        
        # Let's align with 2 scales like DAC default [2048, 512].
        # And n_mels [80, 80] or similar.
        
        # I will support arbitrary list of transforms.
        
        for win_len in window_lengths:
            hop = win_len // 4 if hop_lengths is None else hop_lengths
            
            # Use 80 n_mels for all scales by default if n_mels is int
            # If list, try to zip. 
            
            # Let's simplify: Just create multiple transforms.
            # We want to capture both fine and coarse detail.
            
            t = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=win_len,
                win_length=win_len,
                hop_length=hop,
                center=True,
                pad_mode="reflect",
                power=2.0,
                norm='slaney',
                # n_mels needs to be set. 
                n_mels=80, # Defaulting to 80 for all scales is safe for 16k
                f_min=f_min,
                f_max=f_max,
                mel_scale="htk",
            )
            self.transforms.append(t)
            
        self.loss_fn = nn.L1Loss()
        self.log_base = log_base

    def forward(self, pred, target):
        # Align lengths
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]
        
        loss = 0.0
        for t in self.transforms:
            t = t.to(pred.device)
            
            # Compute Spectrograms
            pred_mel = t(pred)
            target_mel = t(target)
            
            # Log Magnitude (Input is likely power spec if power=2.0)
            # Add epsilon
            pred_log = torch.log10(pred_mel + 1e-8)
            target_log = torch.log10(target_mel + 1e-8)
            
            loss += self.loss_fn(pred_log, target_log)
            
        return loss / len(self.transforms)


class GANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    Adapted from Descript Audio Codec.
    """

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, fake, real):
        # inputs are (B, 1, T) or (B, T)
        # DAC discriminator expects (B, 1, T) usually
        if fake.dim() == 2:
            fake = fake.unsqueeze(1)
        if real.dim() == 2:
            real = real.unsqueeze(1)

        d_fake = self.discriminator(fake)
        d_real = self.discriminator(real)
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)

        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            # LSGAN: Real -> 1, Fake -> 0
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)
        return loss_d

    def generator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake, real)

        loss_g = 0
        for x_fake in d_fake:
            # LSGAN: Fake -> 1
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)

        loss_feature = 0
        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
        return loss_g, loss_feature
