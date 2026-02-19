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
    z_target_full:  torch.Tensor,   # (B, 1024, T_dac) pre-encoded clean audio
    window_samples: int,
    eeg_window:     int,
    hop_samples:    int,
    eeg_hop:        int,
    hop_dac:        int,            # exact DAC latent frames per hop
    criterion:      nn.Module,
    device:         torch.device,
):
    """
    Exactly replicates inference dynamics in training.

    Slides a window_samples buffer through the full segment in hop_samples steps:
      Hop 0:   Two-pass (EEG-only → pseudo_past → speaker forward)
               Mimics cold start — model sees zeros as initial attractor.
      Hop 1+:  Direct forward with real decoded output from previous hop.
               Mimics inference where hop N's output feeds hop N+1.

    Loss is computed ONLY on the last hop_dac frames of each prediction —
    the slice that inference actually keeps (discards the first 1.5s).

    past_speech is .detach()-ed between hops so there is NO backprop-through-time.
    Each hop has an independent gradient graph → memory = num_hops × 1 forward.
    """
    B            = noisy.shape[0]
    eeg_channels = eeg.shape[1]
    T_dac_total  = z_target_full.shape[-1]

    # Pad audio/EEG to exact multiples of hop size
    pad_audio = (-noisy.shape[-1]) % hop_samples
    if pad_audio > 0:
        noisy = F.pad(noisy, (0, pad_audio))
    pad_eeg = (-eeg.shape[-1]) % eeg_hop
    if pad_eeg > 0:
        eeg = F.pad(eeg, (0, pad_eeg))

    num_hops = noisy.shape[-1] // hop_samples

    # Cold-start buffers (zeros, identical to inference _reset())
    audio_buf = torch.zeros(B, 1,            window_samples, device=device)
    eeg_buf   = torch.zeros(B, eeg_channels, eeg_window,     device=device)
    past      = torch.zeros(B, 1,            hop_samples,    device=device)

    total_loss    = 0.0
    speaker_steps = 0
    pseudo_rms    = 0.0

    for h in range(num_hops):
        a0, a1 = h * hop_samples, (h + 1) * hop_samples
        e0, e1 = h * eeg_hop,     (h + 1) * eeg_hop

        # Slide buffers (identical to inference step())
        audio_buf = torch.cat([audio_buf[:, :, hop_samples:], noisy[:, :, a0:a1]], dim=2)
        eeg_buf   = torch.cat([eeg_buf[:,  :, eeg_hop:],     eeg[:,  :, e0:e1]],  dim=2)

        if h == 0:
            # ── Hop 0: Two-pass cold start ─────────────────────────────────
            with torch.no_grad():
                z_p1, _, _, _ = model(
                    audio_buf, eeg_buf,
                    past_speech=None, force_no_speaker=True,
                )
                pseudo_audio = model.decode_audio(z_p1)
            pseudo_past = pseudo_audio[:, :, -hop_samples:].detach()
            pseudo_rms  = float(pseudo_past.pow(2).mean().sqrt())

            use_speaker = random.random() > model.speaker_dropout_prob
            z_pred_h, _, _, _ = model(
                audio_buf, eeg_buf,
                past_speech=pseudo_past if use_speaker else None,
                force_no_speaker=not use_speaker,
            )
        else:
            # ── Hop 1+: real previous output as attractor ──────────────────
            use_speaker = random.random() > model.speaker_dropout_prob
            z_pred_h, _, _, _ = model(
                audio_buf, eeg_buf,
                past_speech=past if use_speaker else None,
                force_no_speaker=not use_speaker,
            )

        speaker_steps += int(use_speaker)

        # ── Loss on last hop_dac frames only (what inference keeps) ────────
        d0 = h * hop_dac
        d1 = min(d0 + hop_dac, T_dac_total)
        actual_hop_dac = d1 - d0
        z_target_h  = z_target_full[:, :, d0:d1]
        z_pred_last = z_pred_h[:, :, -actual_hop_dac:]
        total_loss += criterion(z_pred_last, z_target_h)

        # ── Update attractor for next hop (detach = no BPTT) ───────────────
        with torch.no_grad():
            past = model.decode_audio(z_pred_h)[:, :, -hop_samples:].detach()

    return total_loss, speaker_steps, pseudo_rms, num_hops


# ---------------------------------------------------------------------------
# Validation (single-pass, EEG-only — measures baseline capability)
# ---------------------------------------------------------------------------

def validate(model, loader, criterion, device, args):
    model.eval()
    total_loss  = 0.0
    sisdr_scores = []

    with torch.no_grad():
        for batch_idx, (noisy, eeg, clean) in enumerate(loader):
            noisy = noisy.to(device)
            clean = clean.to(device)
            eeg   = eeg.to(device)

            z_target, _ = model.encode_audio(clean)

            # EEG-only pass (no speaker encoder) for fair baseline validation
            z_pred, _, _, _ = model(
                noisy, eeg,
                past_speech=None,
                force_no_speaker=True,
            )
            loss = criterion(z_pred, z_target)
            total_loss += loss.item()

            # Decode, align via cross-correlation, then compute SI-SDR
            pred_audio = model.decode_audio(z_pred)
            min_len    = min(pred_audio.shape[-1], clean.shape[-1])
            pred_np    = pred_audio[..., :min_len].cpu().numpy().squeeze(1)  # (B, T)
            clean_np   = clean[..., :min_len].cpu().numpy().squeeze(1)       # (B, T)
            for ref, est in zip(clean_np, pred_np):
                sisdr_scores.append(align_and_sisdr(ref, est))

            if args.debug and batch_idx > 2:
                break

    mean_loss   = total_loss / max(len(loader), 1)
    mean_sisdr  = float(np.mean(sisdr_scores)) if sisdr_scores else 0.0
    return mean_loss, mean_sisdr


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
        speaker_dropout_prob=0.2,       # 20 % dropout as per NeuroHeed
    ).to(device)

    # ── Compute exact DAC frames per hop (model-dependent stride) ──────────
    with torch.no_grad():
        _dummy = torch.zeros(1, 1, hop_samples, device=device)
        hop_dac = model.encode_audio(_dummy)[0].shape[-1]
    print(f"hop_dac = {hop_dac} DAC frames per {args.hop_sec}s hop")

    # ── Optimiser (skip frozen DAC) ─────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer  = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    criterion  = nn.MSELoss()
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
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
            loss, hop_spk, hop_rms, n_hops = sliding_window_train_step(
                model, noisy, eeg, z_target_full,
                window_samples, eeg_window, hop_samples, eeg_hop, hop_dac,
                criterion, device,
            )

            # ── Backward ──────────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            optimizer.step()

            train_loss     += loss.item() / n_hops   # normalise for logging
            speaker_steps  += hop_spk
            pseudo_rms_sum += hop_rms

            pbar.set_postfix({"loss": f"{loss.item()/n_hops:.4f}", "hops": n_hops})

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
        val_loss, val_sisdr = validate(model, val_loader, criterion, device, args)
        scheduler.step(val_loss)

        avg_train = train_loss / max(len(train_loader), 1)
        print(
            f"Epoch {epoch+1:3d} | "
            f"Train {avg_train:.4f} | Val {val_loss:.4f} | SI-SDR {val_sisdr:.2f} dB"
        )

        # ── Checkpoints ───────────────────────────────────────────────────
        # Save full training state so training can be resumed exactly
        torch.save({
            'epoch':         epoch,
            'model':         model.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }, os.path.join(args.checkpoint_dir, "latest_checkpoint.pth"))

        # Also save inference-ready model-only weights
        torch.save(model.state_dict(),
                   os.path.join(args.checkpoint_dir, "latest_model.pth"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--epochs",         type=int,   default=15)
    parser.add_argument("--start_epoch",    type=int,   default=0,
                        help="Epoch to start from when resuming from a model-only checkpoint (e.g. 10)")
    parser.add_argument("--hidden_dim",     type=int,   default=256)
    parser.add_argument("--num_layers",     type=int,   default=4)
    parser.add_argument("--gpu",            type=int,   default=0)
    parser.add_argument("--hop_sec",        type=float, default=0.5,
                        help="Hop size in seconds (default 0.5 s)")
    parser.add_argument("--window_sec",     type=float, default=2.0,
                        help="Sliding window size in seconds (default 2.0 s)")
    parser.add_argument("--eeg_sr",         type=int,   default=128,
                        help="EEG sample rate in Hz (default 128 Hz for KUL; 512 for Cocktail Party)")
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints/neurocodec_online")
    parser.add_argument("--pretrained",     type=str,   default="",
                        help="Optional path to offline NeuroCodec checkpoint for warm-start")
    parser.add_argument("--debug",          action="store_true",
                        help="Run a fast debug pass (few batches per epoch)")

    args = parser.parse_args()
    train(args)
