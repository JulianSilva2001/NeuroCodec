"""
OnlineNeuroCodec: Extension of NeuroCodec for online/streaming inference.

Adds three NeuroHeed-style components:
  1. SpeakerEncoder  - encodes past extracted speech into an auditory attractor
  2. OnlineNeuroCodecBlock - fuses both neuronal (EEG) and auditory (speaker) attractors
  3. OnlineNeuroCodec - main model with two-pass training support

Training:  Two-Pass strategy (pseudo-extracted speech) + 20% speaker encoder dropout
Inference: Sliding window (2s buffer, 0.5s hop) with inference normalization
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dac
from mamba_ssm import Mamba

from utility.layers import GraphConvolution
from utility.utils import normalize_A, generate_cheby_adj
from utility.utils import ChannelwiseLayerNorm, Conv1D

# Reuse unchanged components from the original NeuroCodec
from models.neurocodec import Chebynet, FlexibleEEGEncoder, PositionalEncoding


# ---------------------------------------------------------------------------
# Speaker Encoder (Auditory Attractor)
# ---------------------------------------------------------------------------

class SpeakerEncoder(nn.Module):
    """
    Encodes a segment of past extracted speech into an auditory attractor.

    Uses the frozen DAC encoder to project past speech into the 1024-dim
    latent space, then linearly projects to hidden_dim for cross-attention.

    Input:  past_speech (B, 1, T_past)  — 0.5 s of previously extracted audio
    Output: (B, T_dac, hidden_dim)      — auditory attractor sequence
    """

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(1024, hidden_dim)
        self.ln   = nn.LayerNorm(hidden_dim)
        self.pos  = PositionalEncoding(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, z_past: torch.Tensor) -> torch.Tensor:
        """
        z_past: (B, 1024, T_dac) — pre-encoded latents of past speech.
        The caller is responsible for encoding; this removes the DAC encode
        from the hot path so training never pays for a redundant encode+decode.
        """
        z = z_past.transpose(1, 2)   # (B, T_dac, 1024)
        z = self.proj(z)             # (B, T_dac, H)
        z = self.ln(z)
        z = self.pos(z)
        z = self.drop(z)
        return z


# ---------------------------------------------------------------------------
# Fusion Block (Cross-Attention + Mamba)
# ---------------------------------------------------------------------------

class OnlineNeuroCodecBlock(nn.Module):
    """
    One fusion layer. The audio features act as Query; the EEG neuronal
    attractor and (optionally) the speaker auditory attractor are
    concatenated to form a unified Key/Value for cross-attention.

    x:          (B, T_audio, H)   audio query
    eeg_kv:     (B, T_eeg,   H)   neuronal attractor  (from EEG encoder)
    speaker_kv: (B, T_spk,   H)   auditory attractor  (from SpeakerEncoder, optional)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads:    int   = 8,
        d_state:    int   = 16,
        d_conv:     int   = 4,
        expand:     int   = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()

        # --- Cross-Attention ---
        self.ln1  = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            batch_first=True, dropout=dropout,
        )
        self.drop1 = nn.Dropout(dropout)

        # --- Mamba SSM (inherently causal) ---
        self.ln2   = nn.LayerNorm(hidden_dim)
        self.mamba = Mamba(
            d_model=hidden_dim, d_state=d_state,
            d_conv=d_conv, expand=expand,
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x:          torch.Tensor,
        eeg_kv:     torch.Tensor,
        speaker_kv: torch.Tensor | None = None,
    ):
        # Merge attractors → unified Key/Value
        if speaker_kv is not None:
            kv = torch.cat([eeg_kv, speaker_kv], dim=1)   # (B, T_eeg+T_spk, H)
        else:
            kv = eeg_kv                                    # (B, T_eeg, H)

        # Cross-Attention with residual
        x_norm   = self.ln1(x)
        attn_out, attn_w = self.attn(query=x_norm, key=kv, value=kv)
        x = x + self.drop1(attn_out)

        # Mamba with residual
        x_norm    = self.ln2(x)
        mamba_out = self.mamba(x_norm)
        x = x + self.drop2(mamba_out)

        return x, attn_w


# ---------------------------------------------------------------------------
# OnlineNeuroCodec — Main Model
# ---------------------------------------------------------------------------

class OnlineNeuroCodec(nn.Module):
    """
    NeuroCodec with online capabilities:
      • SpeakerEncoder feeds past extracted speech as an auditory attractor.
      • forward() accepts an optional past_speech tensor and a
        force_no_speaker flag (used for two-pass training).
      • encode_audio / decode_audio helpers for reuse in training loops.

    Args:
        dac_model_type:      '44khz' (default)
        eeg_in_channels:     128
        hidden_dim:          256
        num_layers:          4
        speaker_dropout_prob: 0.2  (fraction of training steps where
                                    the speaker encoder is zeroed out)
    """

    def __init__(
        self,
        dac_model_type:       str   = '44khz',
        eeg_in_channels:      int   = 128,
        hidden_dim:           int   = 256,
        num_layers:           int   = 4,
        speaker_dropout_prob: float = 0.2,
    ):
        super().__init__()
        self.speaker_dropout_prob = speaker_dropout_prob

        # ── 1. Frozen DAC backbone ──────────────────────────────────────────
        print(f"Loading Frozen DAC ({dac_model_type})...")
        model_path = dac.utils.download(model_type=dac_model_type)
        self.dac = dac.DAC.load(model_path)
        for param in self.dac.parameters():
            param.requires_grad = False
        self.dac.eval()
        print("DAC loaded and frozen.")

        # ── 2. EEG Encoder (neuronal attractor) ────────────────────────────
        self.eeg_encoder = FlexibleEEGEncoder(
            num_electrodes=eeg_in_channels,
            enc_channel=64,
            feature_channel=64,
            norm='ln',
        )

        # ── 3. Speaker Encoder (auditory attractor) ─────────────────────────
        self.speaker_encoder = SpeakerEncoder(hidden_dim=hidden_dim, dropout=0.1)

        # ── 4. Projections ──────────────────────────────────────────────────
        self.audio_proj = nn.Linear(1024, hidden_dim)
        self.eeg_proj   = nn.Linear(64,   hidden_dim)
        self.pos_enc    = PositionalEncoding(hidden_dim, max_len=5000)

        # ── 5. Stacked fusion blocks ─────────────────────────────────────────
        self.layers = nn.ModuleList([
            OnlineNeuroCodecBlock(hidden_dim=hidden_dim, dropout=0.3)
            for _ in range(num_layers)
        ])

        # ── 6. Output head ───────────────────────────────────────────────────
        self.output_proj = nn.Linear(hidden_dim, 1024)

    # ────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────

    def encode_audio(self, audio: torch.Tensor):
        """Encode audio with frozen DAC. Returns (z, codes)."""
        with torch.no_grad():
            z, codes, _, _, _ = self.dac.encode(audio)
        return z, codes

    def decode_audio(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize latents and decode to waveform via DAC."""
        with torch.no_grad():
            z_q  = self.dac.quantizer(z, n_quantizers=9)[0]
            audio = self.dac.decode(z_q)
        return audio

    # ────────────────────────────────────────────────────────────────────────
    # Forward
    # ────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        mixture:            torch.Tensor,
        eeg:                torch.Tensor,
        past_speech:        torch.Tensor | None = None,
        force_no_speaker:   bool = False,
        past_z:             torch.Tensor | None = None,
        z_mix_precomputed:  torch.Tensor | None = None,
    ):
        """
        mixture:           (B, 1,    T_audio)  — mixed speech buffer
        eeg:               (B, 128,  T_eeg)    — EEG window (same duration)
        past_speech:       (B, 1,    T_past)   — raw past audio (inference path only)
        force_no_speaker:  bool                — skip speaker encoder entirely
        past_z:            (B, 1024, T_dac)    — pre-encoded past latents (training path)
                           When provided, the speaker encoder uses these directly
                           and no DAC encode is performed — eliminates the decode→
                           re-encode round trip that dominated training time.
        z_mix_precomputed: (B, 1024, T_dac)    — pre-encoded mixture latents
                           When provided, DAC encode of `mixture` is skipped.
                           Used to share z_mix between Pass-1 and Pass-2 at Hop 0.

        Returns:
            z_pred    (B, 1024, T_dac)   — predicted clean-speech latents
            codes_mix                    — DAC codes of mixture (None if precomputed)
            z_mix     (B, 1024, T_dac)   — mixture latents
            eeg_feat  (B, 64,   T_eeg')  — encoded EEG features
        """

        # 1. Encode mixture with frozen DAC (or reuse precomputed latents)
        if z_mix_precomputed is not None:
            z_mix, codes_mix = z_mix_precomputed, None
        else:
            z_mix, codes_mix = self.encode_audio(mixture)      # (B, 1024, T_dac)

        # 2. Neuronal attractor (EEG → hidden_dim)
        eeg_feat = self.eeg_encoder(eeg)                       # (B, 64, T_eeg')
        x_eeg    = self.eeg_proj(eeg_feat.transpose(1, 2))    # (B, T_eeg', H)

        # 3. Auditory attractor (past → hidden_dim)
        #    Training: past_z provided → no DAC call needed (fast path)
        #    Inference: past_speech provided → encode here (backward-compatible)
        use_speaker = (past_z is not None or past_speech is not None) and (not force_no_speaker)
        if use_speaker:
            if past_z is None:
                # Inference path: encode raw audio to latents
                with torch.no_grad():
                    past_z, _, _, _, _ = self.dac.encode(past_speech)
            x_speaker = self.speaker_encoder(past_z)           # (B, T_spk, H)
        else:
            x_speaker = None

        # 4. Audio features
        scale   = math.sqrt(self.audio_proj.out_features)
        x_audio = self.audio_proj(z_mix.transpose(1, 2))      # (B, T_dac, H)
        x_audio = self.pos_enc(x_audio * scale)
        x_eeg   = self.pos_enc(x_eeg   * scale)
        if x_speaker is not None:
            x_speaker = self.pos_enc(x_speaker * scale)

        # 5. Stacked fusion blocks
        x = x_audio
        for layer in self.layers:
            x, _ = layer(x, x_eeg, x_speaker)

        # 6. Output
        z_pred = self.output_proj(x).transpose(1, 2)          # (B, 1024, T_dac)
        return z_pred, codes_mix, z_mix, eeg_feat


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = OnlineNeuroCodec().to(dev)
    print("OnlineNeuroCodec instantiated.")

    B, T_audio, T_eeg = 1, 87552, 256
    mixture     = torch.randn(B, 1,   T_audio).to(dev)
    eeg         = torch.randn(B, 128, T_eeg  ).to(dev)
    past_speech = torch.randn(B, 1,   22050  ).to(dev)   # 0.5 s

    z_pred, _, z_mix, _ = model(mixture, eeg, past_speech=past_speech)
    print(f"z_pred shape: {z_pred.shape}")   # (1, 1024, T_dac)
    print(f"z_mix  shape: {z_mix.shape}")
