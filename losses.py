import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
import torchaudio
import warnings

from audiotools import AudioSignal
from audiotools import STFTParams

import dac.model.discriminator

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
                 hop_lengths=None, # None -> window//4, int -> broadcast, list -> per-scale
                 f_min=0.0,
                 f_max=None,
                 log_base=10.0,
                 norm='slaney',
                 mel_scale='htk'):
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
        
        if not isinstance(n_mels, list):
            n_mels = [n_mels] * len(window_lengths)
            
        if len(n_mels) != len(window_lengths):
            raise ValueError("n_mels list must match window_lengths list length")

        if hop_lengths is None:
            resolved_hops = [w // 4 for w in window_lengths]
        elif isinstance(hop_lengths, int):
            resolved_hops = [hop_lengths] * len(window_lengths)
        elif isinstance(hop_lengths, list):
            if len(hop_lengths) != len(window_lengths):
                raise ValueError("hop_lengths list must match window_lengths list length")
            resolved_hops = hop_lengths
        else:
            raise TypeError("hop_lengths must be None, int, or list")

        for i, win_len in enumerate(window_lengths):
            hop = resolved_hops[i]
            
            # Safety check for n_mels vs n_fft
            # n_fft = win_len
            n_fft = win_len
            max_mels = n_fft // 2 + 1
            current_n_mels = min(n_mels[i], max_mels)
            if current_n_mels != n_mels[i]:
                print(f"Warning: n_mels={n_mels[i]} too large for win_len={win_len}. Clamping to {current_n_mels}.")

            # Keep f_max strictly below Nyquist for numerical robustness.
            nyquist = sample_rate / 2.0
            current_f_max = f_max if f_max is not None else nyquist - 1.0
            current_f_max = min(current_f_max, nyquist - 1.0)

            # Torchaudio can still produce empty mel filters for some scale/config combos.
            # Reduce n_mels until the filterbank has no all-zero bands.
            n_freqs = n_fft // 2 + 1
            while current_n_mels > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    fb = torchaudio.functional.melscale_fbanks(
                        n_freqs=n_freqs,
                        f_min=f_min,
                        f_max=current_f_max,
                        n_mels=current_n_mels,
                        sample_rate=sample_rate,
                        norm=norm,
                        mel_scale=mel_scale,
                    )
                if not (fb.max(dim=0).values == 0).any():
                    break
                current_n_mels -= 1
            if current_n_mels != n_mels[i]:
                print(
                    f"Info: adjusted n_mels for win_len={win_len} "
                    f"from {n_mels[i]} to {current_n_mels} to avoid empty mel filters."
                )
            
            t = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_len,
                hop_length=hop,
                center=True,
                pad_mode="reflect",
                power=2.0,
                norm=norm,
                n_mels=current_n_mels,
                f_min=f_min,
                f_max=current_f_max,
                mel_scale=mel_scale,
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
            # HingeGAN discriminator objective
            loss_d += torch.mean(F.relu(1 - x_real[-1]))
            loss_d += torch.mean(F.relu(1 + x_fake[-1]))
        return loss_d

    def generator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake, real)

        loss_g = 0
        for x_fake in d_fake:
            # HingeGAN generator objective
            loss_g += -torch.mean(x_fake[-1])

        loss_feature = 0

        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
        return loss_g, loss_feature
