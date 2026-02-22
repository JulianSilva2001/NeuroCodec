"""
Online Sliding-Window Inference for OnlineNeuroCodec.

Simulates the NeuroHeed real-time cocktail-party scenario:

  For each 0.5 s hop of new mixed audio + EEG:
    1.  Append the new frame to a rolling 2 s audio/EEG buffer.
    2.  Run OnlineNeuroCodec with the buffer + past extracted speech
        (auditory attractor from the speaker encoder).
    3.  Take the LAST hop_samples of the decoded output as the new frame.
    4.  Apply Inference Normalization — scale the output hop to match the
        RMS of the corresponding input hop (prevents volume jumps).
    5.  Store as the new past_extracted for the next iteration.

The full output is assembled by concatenating successive output hops.

Usage:
    python inference_online.py \
        --checkpoint checkpoints/neurocodec_online/best_model.pth \
        --root /path/to/data \
        --num_samples 10 \
        --hop_sec 0.5 \
        --output_dir outputs/online
"""

import os
import argparse
import time

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')   # non-interactive — safe on headless servers
import matplotlib.pyplot as plt

from models.neurocodec_online import OnlineNeuroCodec
from dataset_neurocodec import load_NeuroCodecDataset

try:
    from pystoi import stoi as _stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False
    print("WARNING: pystoi not installed — ESTOI will be skipped. pip install pystoi")

try:
    from pesq import pesq as _pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    print("WARNING: pesq not installed  — PESQ will be skipped. pip install pesq")
try:
    from dataset_neurocodec import load_KUL_NeuroCodecDataset
except ImportError:
    load_KUL_NeuroCodecDataset = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_estoi(ref: np.ndarray, deg: np.ndarray, sr: int):
    """Extended STOI in [0, 1]. Returns None if pystoi is not installed."""
    if not HAS_STOI:
        return None
    try:
        return float(_stoi(ref, deg, sr, extended=True))
    except Exception:
        return None


def compute_pesq(ref: np.ndarray, deg: np.ndarray, sr: int):
    """
    PESQ MOS-LQO score (wideband, ~1.0–4.5).
    PESQ only accepts 8 kHz or 16 kHz — audio is resampled if needed.
    Returns None if pesq is not installed or computation fails.
    """
    if not HAS_PESQ:
        return None
    try:
        pesq_sr = 16000
        if sr != pesq_sr:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, pesq_sr)
            ref = resample_poly(ref, pesq_sr // g, sr // g).astype(np.float32)
            deg = resample_poly(deg, pesq_sr // g, sr // g).astype(np.float32)
        return float(_pesq(pesq_sr, ref, deg, 'wb'))
    except Exception:
        return None


def rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Root-mean-square along the last dimension. Returns a scalar tensor."""
    return torch.sqrt(torch.mean(x ** 2) + eps)


def sisdr_np(reference: np.ndarray, estimation: np.ndarray) -> float:
    ref_energy  = np.sum(reference ** 2)
    alpha       = np.dot(reference, estimation) / (ref_energy + 1e-8)
    projection  = alpha * reference
    noise       = estimation - projection
    return float(10 * np.log10(np.sum(projection**2) / (np.sum(noise**2) + 1e-8)))


# ---------------------------------------------------------------------------
# Online Inference Engine
# ---------------------------------------------------------------------------

class OnlineInferenceEngine:
    """
    Stateful sliding-window engine.

    Internal state per utterance:
      audio_buffer — (1, 1,    window_samples)  rolling mixed audio
      eeg_buffer   — (1, 128,  eeg_window)       rolling EEG
      past_z       — (1, 1024, hop_dac)          auditory attractor in latent space
                     Replaces the old audio-domain past_extracted; eliminates the
                     dac.encode(past_speech) call that was redundant on every hop.

    Call process_full_segment() to run an entire pre-recorded utterance in
    online fashion (split into hops internally).  For true real-time use,
    call _reset() once then step() for each arriving frame.
    """

    def __init__(
        self,
        model:        OnlineNeuroCodec,
        device:       torch.device,
        window_sec:   float = 2.0,
        hop_sec:      float = 0.5,
        sr:           int   = 44100,   # 44100 for Cocktail Party, 16000 for KUL
        eeg_sr:       int   = 128,
        eeg_channels: int   = 128,     # 128 for Cocktail Party, 64 for KUL
    ):
        self.model        = model
        self.device       = device
        self.sr           = sr
        self.eeg_sr       = eeg_sr
        self.eeg_channels = eeg_channels

        self.window_samples = int(window_sec * sr)      # e.g. 88200 @ 44.1 kHz
        self.hop_samples    = int(hop_sec    * sr)      # e.g. 22050 @ 44.1 kHz
        self.eeg_window     = int(window_sec * eeg_sr)  # e.g. 256   @ 128 Hz
        self.eeg_hop        = int(hop_sec    * eeg_sr)  # e.g. 64    @ 128 Hz

        # Compute exact DAC latent frames per hop (model stride, not fixed).
        with torch.no_grad():
            _dummy   = torch.zeros(1, 1, self.hop_samples, device=device)
            self.hop_dac = model.encode_audio(_dummy)[0].shape[-1]

        # EMA coefficient for past_z update (0 = replace hard, 1 = never update).
        # 0.7 means: keep 70% of history, blend in 30% of the new hop each step.
        # This prevents a single bad hop from fully poisoning the attractor.
        self.attractor_ema = 0.7

        self._reset()

    def _reset(self):
        """Initialise all buffers to silence (cold start)."""
        self.audio_buffer      = torch.zeros(1, 1,                self.window_samples, device=self.device)
        self.eeg_buffer        = torch.zeros(1, self.eeg_channels, self.eeg_window,    device=self.device)
        self.past_z            = torch.zeros(1, 1024,             self.hop_dac,        device=self.device)
        self.hop_attractor_rms = []   # latent-norm of past_z at the START of each hop
        self.hop_times         = []   # processing time (ms) for each hop

    # ── Single-hop step ────────────────────────────────────────────────────

    def step(
        self,
        new_audio: torch.Tensor,   # (1, 1, hop_samples)
        new_eeg:   torch.Tensor,   # (1, 128, eeg_hop)
    ) -> torch.Tensor:             # returns (1, 1, hop_samples)
        """
        Process one incoming hop and return the corresponding output hop.
        """

        # 0. Record attractor latent-norm BEFORE this step (shows cold-start warmup)
        self.hop_attractor_rms.append(float(self.past_z.pow(2).mean().sqrt()))
        _t0 = time.perf_counter()

        # 1. Slide audio buffer: drop oldest hop, append new frame
        self.audio_buffer = torch.cat(
            [self.audio_buffer[:, :, self.hop_samples:], new_audio], dim=2
        )

        # 2. Slide EEG buffer
        self.eeg_buffer = torch.cat(
            [self.eeg_buffer[:, :, self.eeg_hop:], new_eeg], dim=2
        )

        # 3. Model inference
        # Hop 0: two-pass cold start — mirrors training exactly.
        #   Pass 1: EEG-only (no speaker) → pseudo_past_z
        #   Pass 2: speaker encoder ON with pseudo_past_z + reuse z_mix from Pass 1
        # Hop 1+: single forward with real accumulated past_z.
        is_cold_start = len(self.hop_times) == 0
        with torch.no_grad():
            if is_cold_start:
                z_p1, _, z_mix_h, _ = self.model(
                    self.audio_buffer,
                    self.eeg_buffer,
                    past_speech=None,
                    force_no_speaker=True,
                )
                pseudo_past_z = z_p1[:, :, -self.hop_dac:]
                z_pred, _, _, _ = self.model(
                    self.audio_buffer,
                    self.eeg_buffer,
                    past_speech=None,
                    force_no_speaker=False,
                    past_z=pseudo_past_z,
                    z_mix_precomputed=z_mix_h,
                )
            else:
                z_pred, _, _, _ = self.model(
                    self.audio_buffer,
                    self.eeg_buffer,
                    past_speech=None,
                    force_no_speaker=False,
                    past_z=self.past_z,
                )
            output_full = self.model.decode_audio(z_pred)   # (1, 1, T_out) — required for audio stream

        # 4. Take last hop_samples from decoded output
        T_out = output_full.shape[-1]
        if T_out >= self.hop_samples:
            output_hop = output_full[:, :, -self.hop_samples:]
        else:
            # Safety pad (should not happen with a 2 s buffer)
            output_hop = F.pad(output_full, (self.hop_samples - T_out, 0))

        # 5. Inference Normalization — match output RMS to input RMS (audio stream only)
        # Threshold: only normalize if the model is producing meaningful output
        # (out_rms > 1% of in_rms). During warmup the model may output near-silence;
        # dividing by a tiny out_rms would massively amplify floor noise.
        # Cap at 3.0x to prevent extreme boosts even when the threshold is met.
        in_rms  = rms(new_audio)
        out_rms = rms(output_hop)
        if out_rms > 0.01 * in_rms:
            scale      = torch.clamp(in_rms / out_rms, max=3.0)
            output_hop = output_hop * scale

        # 6. Update auditory attractor in latent space with EMA smoothing.
        #    Using z_pred[:, :, -hop_dac:] directly avoids the old problem where
        #    RMS-normalizing a quiet-input hop would drain attractor energy —
        #    latents are not affected by the audio normalization above.
        self.past_z = (
            self.attractor_ema       * self.past_z
            + (1.0 - self.attractor_ema) * z_pred[:, :, -self.hop_dac:].detach()
        )

        elapsed_ms = (time.perf_counter() - _t0) * 1000
        self.hop_times.append(elapsed_ms)
        hop_idx = len(self.hop_times)
        print(f"  hop {hop_idx:2d} (t={( hop_idx-1)*self.hop_samples/self.sr:.1f}s) | "
              f"process time: {elapsed_ms:6.1f} ms  "
              f"({'✓ real-time' if elapsed_ms < self.hop_samples/self.sr*1000 else '✗ too slow'})")

        return output_hop

    # ── Full-segment convenience wrapper ──────────────────────────────────

    def process_full_segment(
        self,
        noisy_audio: torch.Tensor,   # (1, 1, T_audio)  full utterance
        eeg_signal:  torch.Tensor,   # (1, 128, T_eeg)  full EEG
    ) -> torch.Tensor:               # (1, 1, T_out)
        """
        Split a full utterance into hops and run step() for each,
        then concatenate the output hops into a full waveform.

        Audio and EEG are zero-padded to the nearest multiple of their
        respective hop sizes before processing.
        """
        self._reset()

        T_audio = noisy_audio.shape[-1]
        T_eeg   = eeg_signal.shape[-1]

        # Pad audio to next multiple of hop_samples
        pad_audio = (-T_audio) % self.hop_samples
        if pad_audio > 0:
            noisy_audio = F.pad(noisy_audio, (0, pad_audio))

        # Pad EEG to next multiple of eeg_hop
        pad_eeg = (-T_eeg) % self.eeg_hop
        if pad_eeg > 0:
            eeg_signal = F.pad(eeg_signal, (0, pad_eeg))

        num_hops    = noisy_audio.shape[-1] // self.hop_samples
        output_hops = []

        for i in range(num_hops):
            a0, a1 = i * self.hop_samples,   (i + 1) * self.hop_samples
            e0, e1 = i * self.eeg_hop,        (i + 1) * self.eeg_hop

            new_audio = noisy_audio[:, :, a0:a1]
            new_eeg   = eeg_signal[:,  :, e0:e1]

            output_hop = self.step(new_audio, new_eeg)
            output_hops.append(output_hop)

        # Short crossfade at each hop junction to remove hard-cut clicks
        # (10 ms linear fade-out on trailing edge, fade-in on leading edge)
        fade_len = min(int(0.01 * self.sr), self.hop_samples // 4)
        fade_out = torch.linspace(1.0, 0.0, fade_len, device=self.device)
        fade_in  = torch.linspace(0.0, 1.0, fade_len, device=self.device)
        for i in range(1, len(output_hops)):
            output_hops[i - 1][0, 0, -fade_len:] *= fade_out
            output_hops[i    ][0, 0, :fade_len]   *= fade_in
        output_audio = torch.cat(output_hops, dim=2)

        # Timing summary
        budget_ms = self.hop_samples / self.sr * 1000
        print(f"  ── timing summary: mean={np.mean(self.hop_times):.1f} ms  "
              f"max={max(self.hop_times):.1f} ms  "
              f"budget={budget_ms:.0f} ms/hop  "
              f"real-time factor={np.mean(self.hop_times)/budget_ms:.2f}x")

        # Trim back to original length
        return output_audio[:, :, :T_audio]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_online_sample(
    noisy_np:       np.ndarray,   # (T,)
    clean_np:       np.ndarray,   # (T,)
    output_np:      np.ndarray,   # (T,)
    attractor_rms:  list,         # per-hop attractor RMS (length = num_hops)
    hop_sisdrs:     list,         # per-hop SI-SDR       (length = num_hops)
    sr:             int,
    hop_samples:    int,
    save_path:      str,
):
    """
    4-row figure:
      Row 1: Waveforms (noisy / clean / online output)
      Row 2: Spectrograms
      Row 3: Per-hop SI-SDR timeline  (shows cold-start warmup)
      Row 4: Auditory attractor RMS   (shows past_extracted building up from zero)
    """
    t = np.arange(len(noisy_np)) / sr
    hops = np.arange(len(attractor_rms))
    hop_t = hops * hop_samples / sr   # hop start time in seconds

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    fig.suptitle("Online Inference Visualisation", fontsize=14, fontweight='bold')

    signals = [("Noisy (input)", noisy_np), ("Clean (target)", clean_np), ("Online output", output_np)]
    colors  = ['#e74c3c', '#2ecc71', '#3498db']

    # ── Row 1: Waveforms ──────────────────────────────────────────────────
    for col, (title, sig) in enumerate(signals):
        ax = axes[0, col]
        ax.plot(t, sig, color=colors[col], linewidth=0.4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_xlim([0, t[-1]])
        ax.grid(True, alpha=0.3)

    # ── Row 2: Spectrograms ───────────────────────────────────────────────
    for col, (title, sig) in enumerate(signals):
        ax = axes[1, col]
        ax.specgram(sig, Fs=sr, NFFT=512, noverlap=256, cmap='inferno', scale='dB')
        ax.set_title(f"{title} — spectrogram", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    # ── Row 3: Per-hop SI-SDR (warmup curve) ─────────────────────────────
    ax = axes[2, 0]
    ax.plot(hop_t, hop_sisdrs, marker='o', markersize=3, color='#3498db', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title("Per-hop SI-SDR  (cold-start warmup)", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SI-SDR (dB)")
    ax.grid(True, alpha=0.3)

    # ── Row 3: SI-SDR improvement bar (overall summary) ──────────────────
    ax = axes[2, 1]
    overall_in  = sisdr_np(clean_np, noisy_np)
    overall_out = sisdr_np(clean_np, output_np)
    ax.bar(["Input", "Online output"], [overall_in, overall_out],
           color=['#e74c3c', '#3498db'], edgecolor='black', linewidth=0.8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title(f"Overall SI-SDR  (Δ {overall_out - overall_in:+.2f} dB)", fontsize=10)
    ax.set_ylabel("SI-SDR (dB)")
    ax.grid(True, alpha=0.3, axis='y')

    # ── Row 3: Hide unused subplot ────────────────────────────────────────
    axes[2, 2].set_visible(False)

    # ── Row 4: Auditory attractor RMS (shows online state evolving) ───────
    ax = axes[3, 0]
    ax.plot(hop_t, attractor_rms, marker='s', markersize=3, color='#9b59b6', linewidth=1.5)
    ax.set_title("Auditory attractor norm per hop  (past_z, latent space)", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS")
    ax.grid(True, alpha=0.3)
    ax.annotate("cold start\n(zeros)", xy=(hop_t[0], attractor_rms[0]),
                xytext=(hop_t[min(3, len(hop_t)-1)], max(attractor_rms) * 0.1),
                arrowprops=dict(arrowstyle='->', color='gray'), fontsize=8, color='gray')

    # ── Row 4: Hide unused subplots ───────────────────────────────────────
    axes[3, 1].set_visible(False)
    axes[3, 2].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Evaluation Script
# ---------------------------------------------------------------------------

def evaluate_online(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset-specific config ────────────────────────────────────────────
    if args.dataset == 'kul':
        dac_model_type = '16khz'
        sr             = 16000
        eeg_channels   = args.eeg_channels if args.eeg_channels != 128 else 64
        original_fs    = 16000
    else:
        dac_model_type = '44khz'
        sr             = 44100
        eeg_channels   = args.eeg_channels   # 128 for Cocktail Party

    print(f"Running online inference on {device}  |  Dataset: {args.dataset.upper()}  |  SR={sr}  |  EEG ch={eeg_channels}")

    # ── Load model ─────────────────────────────────────────────────────────
    model = OnlineNeuroCodec(
        dac_model_type=dac_model_type,
        eeg_in_channels=eeg_channels,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print(f"WARNING: checkpoint not found at {args.checkpoint}. Using random weights.")

    model.eval()

    # ── Dataset ────────────────────────────────────────────────────────────
    if args.dataset == 'cocktail':
        test_loader = load_NeuroCodecDataset(
            root=args.root, subset='test', batch_size=1, num_gpus=1,
            shuffle=False,   # deterministic — same samples every run for fair comparison
        )
    elif args.dataset == 'kul':
        if load_KUL_NeuroCodecDataset is None:
            raise ImportError(
                "load_KUL_NeuroCodecDataset not found in dataset_neurocodec.py. "
                "Make sure the server's dataset_neurocodec.py includes the KUL LMDB loader."
            )
        test_loader = load_KUL_NeuroCodecDataset(
            lmdb_path=args.root, subset='test', batch_size=1, num_gpus=1,
            target_fs=sr, original_fs=original_fs,
            shuffle=False,   # deterministic
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    engine = OnlineInferenceEngine(
        model=model, device=device,
        window_sec=2.0,
        hop_sec=args.hop_sec,
        sr=sr,
        eeg_sr=args.eeg_sr,
        eeg_channels=eeg_channels,
    )

    sisdr_online   = []
    sisdr_input    = []
    estoi_online   = []
    estoi_input    = []
    pesq_online    = []
    pesq_input     = []

    for sample_idx, (noisy, eeg, clean) in enumerate(
        tqdm(test_loader, desc="Online inference")
    ):
        if sample_idx >= args.num_samples:
            break

        noisy = noisy.to(device)    # (1, 1, T)
        eeg   = eeg.to(device)      # (1, 128, T_eeg)
        clean = clean.to(device)    # (1, 1, T)

        # ── Online inference ───────────────────────────────────────────────
        output_audio = engine.process_full_segment(noisy, eeg)   # (1, 1, T)

        # ── Metrics ────────────────────────────────────────────────────────
        min_len = min(output_audio.shape[-1], clean.shape[-1])
        out_np  = output_audio[:, :, :min_len].cpu().numpy().squeeze()
        cln_np  = clean[:, :, :min_len].cpu().numpy().squeeze()
        nsy_np  = noisy[:, :, :min_len].cpu().numpy().squeeze()

        sdr_out = sisdr_np(cln_np, out_np)
        sdr_in  = sisdr_np(cln_np, nsy_np)

        estoi_out = compute_estoi(cln_np, out_np, sr)
        estoi_in  = compute_estoi(cln_np, nsy_np, sr)
        pesq_out  = compute_pesq(cln_np, out_np, sr)
        pesq_in   = compute_pesq(cln_np, nsy_np, sr)

        sisdr_online.append(sdr_out)
        sisdr_input.append(sdr_in)
        if estoi_out is not None:
            estoi_online.append(estoi_out)
            estoi_input.append(estoi_in)
        if pesq_out is not None:
            pesq_online.append(pesq_out)
            pesq_input.append(pesq_in)

        # ── Per-hop SI-SDR (shows cold-start warmup) ───────────────────────
        hop_sisdrs = []
        n_hops = len(engine.hop_attractor_rms)
        for h in range(n_hops):
            h0 = h * engine.hop_samples
            h1 = h0 + engine.hop_samples
            if h1 > len(cln_np):
                break
            hop_sisdrs.append(sisdr_np(cln_np[h0:h1], out_np[h0:h1]))

        estoi_str = (f"  ESTOI {estoi_in:.3f} → {estoi_out:.3f}  (Δ {estoi_out - estoi_in:+.3f})"
                     if estoi_out is not None else "")
        pesq_str  = (f"  PESQ {pesq_in:.2f} → {pesq_out:.2f}  (Δ {pesq_out - pesq_in:+.2f})"
                     if pesq_out is not None else "")
        print(
            f"  Sample {sample_idx+1:3d} | "
            f"SI-SDR {sdr_in:+.2f} → {sdr_out:+.2f} dB  (Δ {sdr_out - sdr_in:+.2f} dB)"
            f"{estoi_str}{pesq_str}"
        )

        # ── Save audio ─────────────────────────────────────────────────────
        prefix = os.path.join(args.output_dir, f"sample_{sample_idx+1:03d}")
        sf.write(f"{prefix}_noisy.wav",      nsy_np, sr)
        sf.write(f"{prefix}_clean.wav",      cln_np, sr)
        sf.write(f"{prefix}_online_out.wav", out_np, sr)

        # ── Plot ───────────────────────────────────────────────────────────
        if args.plot:
            plot_online_sample(
                noisy_np      = nsy_np,
                clean_np      = cln_np,
                output_np     = out_np,
                attractor_rms = engine.hop_attractor_rms,
                hop_sisdrs    = hop_sisdrs,
                sr            = sr,
                hop_samples   = engine.hop_samples,
                save_path     = f"{prefix}_online_plot.png",
            )

    if sisdr_online:
        lines = [
            f"\nResults over {len(sisdr_online)} samples:",
            f"  SI-SDR  | Input {np.mean(sisdr_input):.2f} dB  →  Output {np.mean(sisdr_online):.2f} dB"
            f"  (Δ {np.mean(np.array(sisdr_online) - np.array(sisdr_input)):+.2f} dB)",
        ]
        if estoi_online:
            lines.append(
                f"  ESTOI   | Input {np.mean(estoi_input):.3f}      →  Output {np.mean(estoi_online):.3f}"
                f"  (Δ {np.mean(np.array(estoi_online) - np.array(estoi_input)):+.3f})"
            )
        if pesq_online:
            lines.append(
                f"  PESQ    | Input {np.mean(pesq_input):.2f}       →  Output {np.mean(pesq_online):.2f}"
                f"  (Δ {np.mean(np.array(pesq_online) - np.array(pesq_input)):+.2f})"
            )
        print("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online sliding-window inference for OnlineNeuroCodec")

    parser.add_argument("--root",         type=str,   default="/workspace/Dataset/kul_all_subjects.lmdb")
    parser.add_argument("--dataset",      type=str,   default="kul", choices=["cocktail", "kul"],
                        help="cocktail = Cocktail-Party HDF5; kul = KUL LMDB")
    parser.add_argument("--eeg_channels", type=int,   default=128,
                        help="EEG input channels (auto-set to 64 for KUL if left at default 128)")
    parser.add_argument("--eeg_sr",       type=int,   default=128,
                        help="EEG sample rate in Hz (default 128 for both datasets)")
    parser.add_argument("--checkpoint",   type=str,   default="checkpoints/neurocodec_online/best_model.pth")
    parser.add_argument("--output_dir",   type=str,   default="outputs/online_inference")
    parser.add_argument("--num_samples",  type=int,   default=10)
    parser.add_argument("--hop_sec",      type=float, default=0.5,
                        help="Sliding window hop size in seconds (default 0.5 s)")
    parser.add_argument("--hidden_dim",   type=int,   default=256)
    parser.add_argument("--num_layers",   type=int,   default=4)
    parser.add_argument("--gpu",          type=int,   default=0)
    parser.add_argument("--plot",         action="store_true",
                        help="Save per-sample visualisation plots (PNG) to output_dir")

    args = parser.parse_args()
    evaluate_online(args)
