"""
Two-Pass Training for OnlineNeuroCodec.

Implements the NeuroHeed training strategy:

  Pass 1 (no speaker encoder):
      Run the model with past_speech=None → decode output → pseudo_extracted

  Pass 2 (with speaker encoder, 80% of steps):
      Use pseudo_extracted[-hop:] as the speaker encoder input.
      20% of the time, zero it out (speaker encoder dropout) to force
      the model to rely solely on EEG — preparing it for cold-start.

Loss is always computed on the Pass 2 output (MSE on DAC latents).
"""

import os
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from models.neurocodec_online import OnlineNeuroCodec
from losses import MelSpectrogramLoss
from dataset_neurocodec import load_NeuroCodecDataset
try:
    from dataset_neurocodec import load_KUL_NeuroCodecDataset
except ImportError:
    load_KUL_NeuroCodecDataset = None


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def sisdr(reference: np.ndarray, estimation: np.ndarray) -> np.ndarray:
    ref_energy  = np.sum(reference ** 2, axis=-1, keepdims=True)
    alpha       = np.sum(reference * estimation, axis=-1, keepdims=True) / (ref_energy + 1e-8)
    projections = alpha * reference
    noise       = estimation - projections
    return 10 * np.log10(
        np.sum(projections ** 2, axis=-1) / (np.sum(noise ** 2, axis=-1) + 1e-8)
    )


def align_and_sisdr(reference: np.ndarray, estimation: np.ndarray) -> float:
    """
    Align estimation to reference via FFT cross-correlation, then compute SI-SDR.

    Searches lags in ±T//8 (12.5% of the signal length) which is wide enough
    to capture any encoder/decoder latency while avoiding false global-maximum
    peaks from periodic signals.

    Convention for the FFT cross-correlation output c[l]:
      c[l] = sum_t ref[t] * est[t - l]
      l > 0  →  ref leads est (est is delayed)
      l < 0  →  est leads ref (ref is delayed)
    """
    T         = len(reference)
    max_shift = max(1, T // 8)
    N_fft     = T * 2   # zero-pad to avoid circular wrap-around

    ref_f = np.fft.fft(reference, n=N_fft)
    est_f = np.fft.fft(estimation, n=N_fft)
    corr  = np.real(np.fft.ifft(ref_f * np.conj(est_f)))

    # Gather lags in [-max_shift, +max_shift] into one array
    neg   = corr[N_fft - max_shift: N_fft]   # lags -max_shift … -1
    pos   = corr[:max_shift + 1]              # lags  0 … +max_shift
    combined = np.concatenate([neg, pos])     # length 2*max_shift + 1
    lag  = int(np.argmax(np.abs(combined))) - max_shift  # signed lag

    # Trim to aligned region
    if lag > 0:
        ref_al, est_al = reference[lag:],      estimation[:T - lag]
    elif lag < 0:
        ref_al, est_al = reference[:T + lag],  estimation[-lag:]
    else:
        ref_al, est_al = reference, estimation

    # SI-SDR on aligned signals
    ref_energy = np.sum(ref_al ** 2) + 1e-8
    alpha      = np.dot(ref_al, est_al) / ref_energy
    proj       = alpha * ref_al
    noise      = est_al - proj
    return float(10.0 * np.log10(np.sum(proj ** 2) / (np.sum(noise ** 2) + 1e-8)))


# ---------------------------------------------------------------------------
# Two-Pass Forward
# ---------------------------------------------------------------------------

def two_pass_forward(
    model:       OnlineNeuroCodec,
    noisy:       torch.Tensor,    # (B, 1, T_audio)
    eeg:         torch.Tensor,    # (B, 128, T_eeg)
    hop_samples: int,
    device:      torch.device,
):
    """
    Returns:
        z_pred   (B, 1024, T_dac)  — Pass-2 predicted latents (for loss)
        z_mix    (B, 1024, T_dac)  — mixture latents
    """

    # ── Pass 1: EEG-only (no speaker encoder) ─────────────────────────────
    # Gradients are not needed here; we only need the decoded pseudo audio.
    with torch.no_grad():
        z_p1, _, _, _ = model(
            noisy, eeg,
            past_speech=None,
            force_no_speaker=True,
        )
        # Decode Pass-1 latents → waveform  (B, 1, T_audio_out)
        pseudo_audio = model.decode_audio(z_p1)

    # Use the last hop_samples as "past speech" for Pass 2
    pseudo_past = pseudo_audio[:, :, -hop_samples:].detach()   # (B, 1, hop)

    # ── Pass 2: with speaker encoder (+ 20 % dropout) ─────────────────────
    use_speaker = random.random() > model.speaker_dropout_prob

    z_pred, _, z_mix, _ = model(
        noisy, eeg,
        past_speech=pseudo_past if use_speaker else None,
        force_no_speaker=not use_speaker,
    )

    # Diagnostics: RMS of pseudo_past tells us Pass-1 produced real output (not silence)
    pseudo_rms = float(pseudo_past.pow(2).mean().sqrt())

    return z_pred, z_mix, use_speaker, pseudo_rms


# ---------------------------------------------------------------------------
# Sliding-Window Training Step (full online simulation)
# ---------------------------------------------------------------------------

def sliding_window_train_step(
    model:          OnlineNeuroCodec,
    noisy:          torch.Tensor,   # (B, 1, T_audio)
    eeg:            torch.Tensor,   # (B, C, T_eeg)
    clean:          torch.Tensor,   # (B, 1, T_audio) ground-truth clean audio
    z_target_full:  torch.Tensor,   # (B, 1024, T_dac) pre-encoded clean audio
    window_samples: int,
    eeg_window:     int,
    hop_samples:    int,
    eeg_hop:        int,
    hop_dac:        int,            # exact DAC latent frames per hop
    criterion:      nn.Module,      # MSE on latents
    mel_criterion:  nn.Module,      # MelSpectrogramLoss for perceptual guardrail
    device:         torch.device,
    lambda_latent:  float = 1.0,    # weight for latent MSE
    lambda_mel:     float = 0.1,    # weight for Mel perceptual loss
):
    """
    Exactly replicates inference dynamics in training.

    Slides a window_samples buffer through the full segment in hop_samples steps:
      Hop 0:   Two-pass (EEG-only → pseudo_past → speaker forward)
               Mimics cold start — model sees zeros as initial attractor.
      Hop 1+:  Direct forward with real decoded output from previous hop.
               Mimics inference where hop N's output feeds hop N+1.

    Combined loss per hop:
      L_total = lambda_latent * MSE(z_pred, z_target)
              + lambda_mel    * MelSpec(dac.decode(z_pred), clean_hop)

    The Mel term decodes z_pred_last through the *frozen* DAC decoder WITHOUT
    torch.no_grad() and WITHOUT .detach(), so gradients flow:
      mel_loss → dac.decode (frozen weights, grads pass through) → z_pred_last
              → OnlineNeuroCodecBlock / SpeakerEncoder → optimizer

    past_z is .detach()-ed between hops so there is NO backprop-through-time.
    Each hop has an independent gradient graph.
    """
    B            = noisy.shape[0]
    eeg_channels = eeg.shape[1]
    T_dac_total  = z_target_full.shape[-1]

    # Pad audio/EEG/clean to exact multiples of hop size
    pad_audio = (-noisy.shape[-1]) % hop_samples
    if pad_audio > 0:
        noisy = F.pad(noisy, (0, pad_audio))
        clean = F.pad(clean, (0, pad_audio))
    pad_eeg = (-eeg.shape[-1]) % eeg_hop
    if pad_eeg > 0:
        eeg = F.pad(eeg, (0, pad_eeg))

    num_hops = noisy.shape[-1] // hop_samples

    # Cold-start buffers (zeros, identical to inference _reset())
    audio_buf = torch.zeros(B, 1,            window_samples, device=device)
    eeg_buf   = torch.zeros(B, eeg_channels, eeg_window,     device=device)
    # past_z stores the last hop_dac latent frames — replaces decoded audio.
    # This eliminates the decode→re-encode round trip:
    #   old: decode(z_pred_h) → audio → dac.encode(audio) → latents
    #   new: z_pred_h[:, :, -hop_dac:]  (already latents, no DAC call needed)
    past_z    = torch.zeros(B, 1024, hop_dac, device=device)

    total_loss_val = 0.0   # float — for logging only; backward is done per hop
    speaker_steps  = 0
    pseudo_rms     = 0.0

    for h in range(num_hops):
        a0, a1 = h * hop_samples, (h + 1) * hop_samples
        e0, e1 = h * eeg_hop,     (h + 1) * eeg_hop

        # Slide buffers (identical to inference step())
        audio_buf = torch.cat([audio_buf[:, :, hop_samples:], noisy[:, :, a0:a1]], dim=2)
        eeg_buf   = torch.cat([eeg_buf[:,  :, eeg_hop:],     eeg[:,  :, e0:e1]],  dim=2)

        if h == 0:
            # ── Hop 0: Two-pass cold start ─────────────────────────────────
            # Pass 1: EEG-only, no speaker encoder.
            # z_mix_h is captured here so Pass 2 can reuse it — this avoids
            # encoding the same audio_buf a second time (double-encode fix).
            with torch.no_grad():
                z_p1, _, z_mix_h, _ = model(
                    audio_buf, eeg_buf,
                    past_speech=None, force_no_speaker=True,
                )
            # Pseudo-past in latent space — no decode needed.
            # pseudo_rms is a latent-norm proxy (same diagnostic purpose).
            pseudo_past_z = z_p1[:, :, -hop_dac:].detach()
            pseudo_rms    = float(pseudo_past_z.pow(2).mean().sqrt())

            use_speaker = random.random() > model.speaker_dropout_prob
            z_pred_h, _, _, _ = model(
                audio_buf, eeg_buf,
                past_speech=None,
                force_no_speaker=not use_speaker,
                past_z=pseudo_past_z if use_speaker else None,
                z_mix_precomputed=z_mix_h,   # reuse — no second DAC encode
            )
        else:
            # ── Hop 1+: real previous output as attractor ──────────────────
            use_speaker = random.random() > model.speaker_dropout_prob
            z_pred_h, _, _, _ = model(
                audio_buf, eeg_buf,
                past_speech=None,
                force_no_speaker=not use_speaker,
                past_z=past_z if use_speaker else None,
            )

        speaker_steps += int(use_speaker)

        # ── Loss on last hop_dac frames only (what inference keeps) ────────
        d0 = h * hop_dac
        d1 = min(d0 + hop_dac, T_dac_total)
        actual_hop_dac = d1 - d0
        z_target_h  = z_target_full[:, :, d0:d1]
        z_pred_last = z_pred_h[:, :, -actual_hop_dac:]

        # Latent MSE
        mse_loss_h = criterion(z_pred_last, z_target_h)

        # Mel perceptual loss — decode z_pred_last through the frozen DAC decoder.
        # No torch.no_grad() and no .detach() here: gradients from the Mel loss
        # flow backward through the frozen decoder ops into z_pred_last, and from
        # there back into the trainable OnlineNeuroCodecBlock / SpeakerEncoder.
        # The decoder's own weights are already frozen (requires_grad=False),
        # so they don't accumulate gradients — only the latent input does.
        if lambda_mel > 0.0:
            y_hat      = model.dac.decode(z_pred_last)     # (B, 1, T_decoded)
            clean_hop  = clean[:, :, a0:a1]                # (B, 1, hop_samples)
            mel_loss_h = mel_criterion(y_hat, clean_hop)
            hop_loss   = lambda_latent * mse_loss_h + lambda_mel * mel_loss_h
        else:
            hop_loss = mse_loss_h

        # ── Per-hop backward — frees this hop's graph immediately ──────────
        # Accumulating all hops into one tensor before backward would keep
        # every decoder forward pass alive in memory simultaneously (OOM with
        # the DAC decoder's large intermediate activations at N=20 hops).
        # Calling backward here keeps peak memory = 1 hop's graph at a time.
        # Gradients accumulate in .grad buffers across hops; optimizer.zero_grad()
        # must be called BEFORE this function (done in the outer training loop).
        (hop_loss / num_hops).backward()
        total_loss_val += hop_loss.item()

        # ── Update attractor with EMA (detach = no BPTT) ──────────────────
        # α=0.7 matches the inference engine's attractor_ema, eliminating the
        # train-inference mismatch where training used hard replacement (α=0)
        # but inference used smoothed updates (α=0.7).
        past_z = 0.7 * past_z + 0.3 * z_pred_h[:, :, -hop_dac:].detach()

    # Return the per-hop-normalized loss as a plain float for logging.
    # Backward has already been done; the outer loop should NOT call .backward() again.
    return total_loss_val / num_hops, speaker_steps, pseudo_rms, num_hops


# ---------------------------------------------------------------------------
# Validation (single-pass, EEG-only — measures baseline capability)
# ---------------------------------------------------------------------------

def validate(
    model,
    loader,
    criterion,
    mel_criterion,
    device,
    args,
    window_samples: int,
    eeg_window:     int,
    hop_samples:    int,
    eeg_hop:        int,
    hop_dac:        int,
    lambda_latent:  float = 1.0,
    lambda_mel:     float = 0.1,
):
    """
    Sliding-window validation that faithfully mirrors inference dynamics:

      Hop 0  — EEG-only cold start  (no speaker encoder)
      Hop 1+ — speaker encoder ON   (past = decoded output of previous hop)

    Reports three losses:
      val_loss   — mean per-hop loss across all hops  (drives scheduler)
      val_cold   — hop-0 loss only  (cold-start capability)
      val_steady — hop-1+ loss only (steady-state, speaker-encoder quality)

    SI-SDR is computed twice per batch to isolate the speaker encoder's contribution:
      sisdr_speaker — normal mode  (hop-0 EEG-only, hop-1+ speaker encoder ON)
      sisdr_eeg     — ablation     (ALL hops EEG-only, past_z still updated)

    Same audio/EEG/weights for both passes → any SI-SDR gap is purely from the
    speaker encoder. Buffer-warmup effect is identical across both passes.
    """
    model.eval()
    total_loss      = 0.0
    cold_loss       = 0.0
    steady_loss     = 0.0
    cold_hops       = 0
    steady_hops     = 0
    sisdr_scores    = []
    sisdr_eeg_only  = []

    with torch.no_grad():
        for batch_idx, (noisy, eeg, clean) in enumerate(loader):
            noisy = noisy.to(device)
            clean = clean.to(device)
            eeg   = eeg.to(device)

            z_target_full, _ = model.encode_audio(clean)
            T_dac_total  = z_target_full.shape[-1]
            B            = noisy.shape[0]
            eeg_channels = eeg.shape[1]

            # Pad to exact hop multiples (same as training step)
            pad_audio = (-noisy.shape[-1]) % hop_samples
            if pad_audio > 0:
                noisy = F.pad(noisy, (0, pad_audio))
                clean = F.pad(clean, (0, pad_audio))
            pad_eeg = (-eeg.shape[-1]) % eeg_hop
            if pad_eeg > 0:
                eeg = F.pad(eeg, (0, pad_eeg))

            num_hops = noisy.shape[-1] // hop_samples

            # Cold-start buffers — identical to inference _reset()
            audio_buf = torch.zeros(B, 1,            window_samples, device=device)
            eeg_buf   = torch.zeros(B, eeg_channels, eeg_window,     device=device)
            past_z    = torch.zeros(B, 1024,         hop_dac,        device=device)

            batch_loss = 0.0
            z_kept     = []   # per-hop kept latent frames → assembled for SI-SDR

            for h in range(num_hops):
                a0, a1 = h * hop_samples, (h + 1) * hop_samples
                e0, e1 = h * eeg_hop,     (h + 1) * eeg_hop

                # Slide buffers
                audio_buf = torch.cat([audio_buf[:, :, hop_samples:], noisy[:, :, a0:a1]], dim=2)
                eeg_buf   = torch.cat([eeg_buf[:,  :, eeg_hop:],     eeg[:,  :, e0:e1]],  dim=2)

                if h == 0:
                    # ── Cold start: EEG-only ────────────────────────────────
                    z_pred_h, _, _, _ = model(
                        audio_buf, eeg_buf,
                        past_speech=None,
                        force_no_speaker=True,
                    )
                else:
                    # ── Steady state: speaker encoder ON (latent past) ──────
                    z_pred_h, _, _, _ = model(
                        audio_buf, eeg_buf,
                        past_speech=None,
                        force_no_speaker=False,
                        past_z=past_z,
                    )

                # Update latent attractor with EMA — mirrors inference (α=0.7)
                past_z = 0.7 * past_z + 0.3 * z_pred_h[:, :, -hop_dac:]

                # Log past_z norm for first batch only — diagnoses speaker encoder health.
                # Near-zero norms at all hops means the encoder receives no useful signal.
                if batch_idx == 0 and h < 4:
                    tqdm.write(f"  [val diag] hop {h} past_z norm: {past_z.pow(2).mean().sqrt().item():.4f}")

                # Loss on the kept frames (last hop_dac DAC frames)
                d0 = h * hop_dac
                d1 = min(d0 + hop_dac, T_dac_total)
                actual_hop_dac = d1 - d0
                z_target_h  = z_target_full[:, :, d0:d1]
                z_pred_last = z_pred_h[:, :, -actual_hop_dac:]

                # Combined latent MSE + Mel perceptual loss (mirrors training)
                mse_loss_h = criterion(z_pred_last, z_target_h)
                if lambda_mel > 0.0:
                    y_hat_v    = model.dac.decode(z_pred_last)
                    mel_loss_h = mel_criterion(y_hat_v, clean[:, :, a0:a1])
                    hop_loss   = (lambda_latent * mse_loss_h + lambda_mel * mel_loss_h).item()
                else:
                    hop_loss   = mse_loss_h.item()

                batch_loss += hop_loss
                if h == 0:
                    cold_loss  += hop_loss;  cold_hops  += 1
                else:
                    steady_loss += hop_loss; steady_hops += 1

                z_kept.append(z_pred_last)

            total_loss += batch_loss / num_hops

            # ── SI-SDR: normal mode (speaker encoder ON for hops 1+) ───────
            z_pred_full = torch.cat(z_kept, dim=-1)[:, :, :T_dac_total]
            pred_audio  = model.decode_audio(z_pred_full)
            min_len     = min(pred_audio.shape[-1], clean.shape[-1])
            pred_np     = pred_audio[..., :min_len].cpu().numpy().squeeze(1)
            clean_np    = clean[..., :min_len].cpu().numpy().squeeze(1)
            for ref, est in zip(clean_np, pred_np):
                sisdr_scores.append(align_and_sisdr(ref, est))

            # ── SI-SDR: EEG-only ablation (speaker encoder OFF for all hops) ─
            # Identical audio/EEG/weights — only difference is force_no_speaker=True
            # for every hop. Buffer warmup is present equally in both passes, so
            # any SI-SDR gap is purely the speaker encoder's contribution.
            audio_buf_e = torch.zeros(B, 1,            window_samples, device=device)
            eeg_buf_e   = torch.zeros(B, eeg_channels, eeg_window,     device=device)
            z_kept_e    = []
            for h in range(num_hops):
                a0e, a1e = h * hop_samples, (h + 1) * hop_samples
                e0e, e1e = h * eeg_hop,     (h + 1) * eeg_hop
                audio_buf_e = torch.cat([audio_buf_e[:, :, hop_samples:], noisy[:, :, a0e:a1e]], dim=2)
                eeg_buf_e   = torch.cat([eeg_buf_e[:,  :, eeg_hop:],     eeg[:,  :, e0e:e1e]],  dim=2)
                z_pred_e, _, _, _ = model(
                    audio_buf_e, eeg_buf_e,
                    past_speech=None, force_no_speaker=True,
                )
                d0e = h * hop_dac
                d1e = min(d0e + hop_dac, T_dac_total)
                actual_e = d1e - d0e
                z_kept_e.append(z_pred_e[:, :, -actual_e:])

            z_pred_eeg  = torch.cat(z_kept_e, dim=-1)[:, :, :T_dac_total]
            pred_eeg    = model.decode_audio(z_pred_eeg)
            min_len_e   = min(pred_eeg.shape[-1], clean.shape[-1])
            pred_eeg_np = pred_eeg[..., :min_len_e].cpu().numpy().squeeze(1)
            for ref, est in zip(clean_np, pred_eeg_np):
                sisdr_eeg_only.append(align_and_sisdr(ref, est))

            if args.debug and batch_idx > 2:
                break

    n_batches        = max(len(loader), 1)
    mean_loss        = total_loss  / n_batches
    mean_cold        = cold_loss   / max(cold_hops,   1)
    mean_steady      = steady_loss / max(steady_hops, 1)
    mean_sisdr       = float(np.mean(sisdr_scores))   if sisdr_scores   else 0.0
    mean_sisdr_eeg   = float(np.mean(sisdr_eeg_only)) if sisdr_eeg_only else 0.0
    return mean_loss, mean_sisdr, mean_cold, mean_steady, mean_sisdr_eeg


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Training on {device}  |  Dataset: {args.dataset.upper()}")

    # ── Dataset-specific config ────────────────────────────────────────────
    if args.dataset == 'kul':
        dac_model_type  = '16khz'
        sr              = 16000
        eeg_channels    = args.eeg_channels if args.eeg_channels != 128 else 64
        original_fs     = 16000
    else:
        dac_model_type  = '44khz'
        sr              = 44100
        eeg_channels    = args.eeg_channels   # 128 for Cocktail Party

    # Sliding window dimensions
    hop_samples    = int(args.hop_sec    * sr)
    window_samples = int(args.window_sec * sr)
    eeg_hop        = int(args.hop_sec    * args.eeg_sr)
    eeg_window     = int(args.window_sec * args.eeg_sr)
    print(f"SR={sr} Hz  |  EEG channels={eeg_channels}  |  hop={hop_samples} samples  |  window={window_samples} samples")

    # ── Dataset ────────────────────────────────────────────────────────────
    print("Loading dataset...")
    if args.dataset == 'cocktail':
        train_loader = load_NeuroCodecDataset(
            root=args.root, subset='train',
            batch_size=args.batch_size, num_gpus=1,
        )
        val_loader = load_NeuroCodecDataset(
            root=args.root, subset='val',
            batch_size=args.batch_size, num_gpus=1,
        )
    elif args.dataset == 'kul':
        if load_KUL_NeuroCodecDataset is None:
            raise ImportError(
                "load_KUL_NeuroCodecDataset not found in dataset_neurocodec.py. "
                "Make sure the server's dataset_neurocodec.py includes the KUL LMDB loader."
            )
        train_loader = load_KUL_NeuroCodecDataset(
            lmdb_path=args.root, subset='train',
            batch_size=args.batch_size, num_gpus=1,
            target_fs=sr, original_fs=original_fs,
        )
        val_loader = load_KUL_NeuroCodecDataset(
            lmdb_path=args.root, subset='val',
            batch_size=args.batch_size, num_gpus=1,
            target_fs=sr, original_fs=original_fs,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # ── Model ──────────────────────────────────────────────────────────────
    print("Initialising OnlineNeuroCodec...")
    model = OnlineNeuroCodec(
        dac_model_type=dac_model_type,
        eeg_in_channels=eeg_channels,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        speaker_dropout_prob=0.05,
    ).to(device)

    # ── Compute exact DAC frames per hop (model-dependent stride) ──────────
    with torch.no_grad():
        _dummy = torch.zeros(1, 1, hop_samples, device=device)
        hop_dac = model.encode_audio(_dummy)[0].shape[-1]
    print(f"hop_dac = {hop_dac} DAC frames per {args.hop_sec}s hop")

    # ── Optimiser (skip frozen DAC) ─────────────────────────────────────────
    trainable    = [p for p in model.parameters() if p.requires_grad]
    optimizer    = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    criterion    = nn.MSELoss()
    mel_criterion = MelSpectrogramLoss(sample_rate=sr)
    scheduler    = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    start_epoch   = 0

    # ── Auto-resume from latest checkpoint (highest priority) ──────────────
    latest_ckpt  = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")
    latest_model = os.path.join(args.checkpoint_dir, "latest_model.pth")

    if os.path.exists(latest_ckpt):
        # New full-state format — restores optimizer, scheduler, epoch exactly
        print(f"Resuming from full checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        print(f"  Resumed at epoch {start_epoch}  |  best val loss so far: {best_val_loss:.4f}")

    elif os.path.exists(latest_model):
        # Old model-only format — loads weights, uses --start_epoch for epoch counter
        print(f"Resuming weights from: {latest_model}  (optimizer state not available)")
        model.load_state_dict(torch.load(latest_model, map_location=device))
        start_epoch = args.start_epoch
        print(f"  Loaded model weights  |  starting from epoch {start_epoch + 1}")

    # ── Optional: warm-start from offline NeuroCodec (only if NOT resuming) ─
    elif args.pretrained and os.path.exists(args.pretrained):
        print(f"Loading pretrained weights from {args.pretrained} ...")
        saved      = torch.load(args.pretrained, map_location=device)
        model_dict = model.state_dict()
        matched    = {k: v for k, v in saved.items()
                      if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict)
        print(f"  Loaded {len(matched)}/{len(model_dict)} parameter tensors.")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss     = 0.0
        speaker_steps  = 0   # how many batches used the speaker encoder
        pseudo_rms_sum = 0.0  # accumulated RMS of pseudo_past
        log_every      = 50   # print online diagnostics every N batches
        current_lr     = optimizer.param_groups[0]['lr']
        pbar           = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}  [LR {current_lr:.1e}]",
        )

        for batch_idx, (noisy, eeg, clean) in enumerate(pbar):
            noisy = noisy.to(device)   # (B, 1, T_audio)
            clean = clean.to(device)   # (B, 1, T_audio)
            eeg   = eeg.to(device)     # (B, 128, T_eeg)

            # ── Ground-truth DAC latents (encode full segment once) ───────
            with torch.no_grad():
                z_target_full, _ = model.encode_audio(clean)

            # ── Full sliding-window training (exactly mirrors inference) ──
            # Slides a 2s window through the segment in 0.5s hops.
            # Hop 0: cold start two-pass; Hop 1+: real model output as attractor.
            # Loss only on the last hop_dac frames of each prediction.
            # zero_grad is called HERE (before the step) so that per-hop
            # backward calls inside the function accumulate into clean .grad buffers.
            optimizer.zero_grad()
            loss_val, hop_spk, hop_rms, n_hops = sliding_window_train_step(
                model, noisy, eeg, clean, z_target_full,
                window_samples, eeg_window, hop_samples, eeg_hop, hop_dac,
                criterion, mel_criterion, device,
                lambda_latent=args.lambda_latent,
                lambda_mel=args.lambda_mel,
            )
            # backward is already done inside sliding_window_train_step (per hop);
            # loss_val is a plain float — the per-hop-normalised training loss.
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            optimizer.step()

            train_loss     += loss_val
            speaker_steps  += hop_spk
            pseudo_rms_sum += hop_rms

            pbar.set_postfix({"loss": f"{loss_val:.4f}", "hops": n_hops})

            # ── Online diagnostics every log_every steps ──────────────────
            if (batch_idx + 1) % log_every == 0:
                frac    = speaker_steps / (log_every * n_hops) * 100
                avg_rms = pseudo_rms_sum / log_every
                tqdm.write(
                    f"  [online] step {batch_idx+1} | "
                    f"hops/batch={n_hops} | "
                    f"speaker {frac:.0f}% | "
                    f"cold-start RMS {avg_rms:.5f}"
                )
                speaker_steps  = 0
                pseudo_rms_sum = 0.0

            if args.debug and batch_idx > 5:
                break

        # ── Validation ────────────────────────────────────────────────────
        val_loss, val_sisdr, val_cold, val_steady, val_sisdr_eeg = validate(
            model, val_loader, criterion, mel_criterion, device, args,
            window_samples, eeg_window, hop_samples, eeg_hop, hop_dac,
            lambda_latent=args.lambda_latent,
            lambda_mel=args.lambda_mel,
        )
        scheduler.step(val_loss)

        avg_train  = train_loss / max(len(train_loader), 1)
        spk_gain   = val_sisdr - val_sisdr_eeg   # > 0 means speaker encoder is helping
        print(
            f"Epoch {epoch+1:3d} | "
            f"Train {avg_train:.4f} | Val {val_loss:.4f} | "
            f"Cold {val_cold:.4f} | Steady {val_steady:.4f} | "
            f"SI-SDR {val_sisdr:.2f} dB (EEG-only {val_sisdr_eeg:.2f} dB, gain {spk_gain:+.2f} dB)"
        )

        # ── Checkpoints ───────────────────────────────────────────────────
        # Update best_val_loss FIRST so the checkpoint always stores the
        # current (not previous-epoch) value — prevents stale reads on resume.
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # Save full training state so training can be resumed exactly
        torch.save({
            'epoch':         epoch,
            'model':         model.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict(),
            'best_val_loss': best_val_loss,   # always current
        }, os.path.join(args.checkpoint_dir, "latest_checkpoint.pth"))

        # Also save inference-ready model-only weights
        torch.save(model.state_dict(),
                   os.path.join(args.checkpoint_dir, "latest_model.pth"))

        if is_best:
            torch.save(model.state_dict(),
                       os.path.join(args.checkpoint_dir, "best_model.pth"))
            print("  → Saved best model.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sliding-window training for OnlineNeuroCodec")

    parser.add_argument("--root",           type=str,   default="/workspace/Dataset/kul_all_subjects.lmdb")
    parser.add_argument("--dataset",        type=str,   default="kul", choices=["cocktail", "kul"],
                        help="cocktail = Cocktail-Party HDF5 (128-ch EEG, 44.1 kHz); kul = KUL LMDB (64-ch EEG, 16 kHz)")
    parser.add_argument("--eeg_channels",   type=int,   default=128,
                        help="EEG input channels (auto-set to 64 for KUL if left at default 128)")
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=5e-4)
    parser.add_argument("--epochs",         type=int,   default=15)
    parser.add_argument("--start_epoch",    type=int,   default=0,
                        help="Epoch to start from when resuming from a model-only checkpoint (e.g. 10)")
    parser.add_argument("--hidden_dim",     type=int,   default=256)
    parser.add_argument("--num_layers",     type=int,   default=4)
    parser.add_argument("--dropout",        type=float, default=0.1,
                        help="Dropout rate in fusion blocks (default 0.1)")
    parser.add_argument("--gpu",            type=int,   default=0)
    parser.add_argument("--hop_sec",        type=float, default=0.5,
                        help="Hop size in seconds (default 0.5 s)")
    parser.add_argument("--window_sec",     type=float, default=2.0,
                        help="Sliding window size in seconds (default 2.0 s)")
    parser.add_argument("--eeg_sr",         type=int,   default=128,
                        help="EEG sample rate in Hz (default 128 Hz for KUL; 512 for Cocktail Party)")
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints/neurocodec_online")
    parser.add_argument("--lambda_latent",  type=float, default=1.0,
                        help="Weight for latent MSE loss (default 1.0)")
    parser.add_argument("--lambda_mel",     type=float, default=0.1,
                        help="Weight for Mel perceptual loss (default 0.1)")
    parser.add_argument("--pretrained",     type=str,   default="",
                        help="Optional path to offline NeuroCodec checkpoint for warm-start")
    parser.add_argument("--debug",          action="store_true",
                        help="Run a fast debug pass (few batches per epoch)")

    args = parser.parse_args()
    train(args)
