
import os
import argparse
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import dac.model # Added this import

# Local imports
from models.neurocodec import NeuroCodec
from dataset_neurocodec import load_NeuroCodecDataset, load_KUL_NeuroCodecDataset
# Consolidated and updated imports from 'losses' and 'losses_neurocodec'
from losses import MelSpectrogramLoss, GANLoss # Added GANLoss
from losses_neurocodec import NeuroCodecLoss # Kept this as it was in the original code


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_determinism():
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:
        print(f"Warning: could not enable deterministic algorithms: {exc}")


def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def scale_optimizer_lr(optimizer, scale):
    for param_group in optimizer.param_groups:
        param_group["lr"] *= scale


def get_default_config_path(dataset):
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    config_map = {
        "cocktail": "train_cocktail.json",
        "kul": "train_kul.json",
    }
    if dataset not in config_map:
        raise ValueError(f"Unknown dataset for config selection: {dataset}")
    return os.path.join(config_dir, config_map[dataset])


def load_config_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return config


def get_phase2_epochs(args):
    ramp_epochs = max(0, args.phase2_lambda_mel_end - args.phase2_lambda_mel_start + 1)
    return ramp_epochs + args.phase2_hold_epochs


def get_training_phase(epoch, args):
    phase1_end = args.phase1_epochs
    phase2_ramp_epochs = max(0, args.phase2_lambda_mel_end - args.phase2_lambda_mel_start + 1)
    phase2_epochs = get_phase2_epochs(args)
    phase2_end = phase1_end + phase2_epochs
    phase3_end = phase2_end + args.phase3_epochs

    if epoch < phase1_end:
        return {
            "name": "phase1_mse",
            "display_name": "Phase 1 MSE",
            "loss_mode": "mse",
            "lambda_mel": 0.0,
            "use_gan": False,
            "train_generator": True,
            "scheduler_active": False,
            "lr_g": args.phase1_lr,
            "lr_d": args.lr,
            "train_batch_size": args.phase1_batch_size,
        }

    if epoch < phase2_end:
        phase2_offset = epoch - phase1_end
        if phase2_offset < phase2_ramp_epochs:
            lambda_mel = min(
                args.phase2_lambda_mel_start + phase2_offset,
                args.phase2_lambda_mel_end,
            )
            phase2_name = "phase2_mel_ramp"
            phase2_display_name = "Phase 2 Mel Ramp"
        else:
            lambda_mel = args.phase2_lambda_mel_end
            phase2_name = "phase2_mel_hold"
            phase2_display_name = "Phase 2 Mel Hold"
        return {
            "name": phase2_name,
            "display_name": phase2_display_name,
            "loss_mode": "full",
            "lambda_mel": float(lambda_mel),
            "use_gan": False,
            "train_generator": True,
            "scheduler_active": False,
            "lr_g": args.lr,
            "lr_d": args.lr,
            "train_batch_size": args.phase2_batch_size,
        }

    if epoch < phase3_end:
        return {
            "name": "phase3_gan_warmup",
            "display_name": "Phase 3 GAN Warmup",
            "loss_mode": "full",
            "lambda_mel": float(args.phase2_lambda_mel_end),
            "use_gan": True,
            "train_generator": False,
            "scheduler_active": False,
            "lr_g": args.phase3_lr,
            "lr_d": args.phase3_lr,
            "train_batch_size": args.phase3_batch_size,
        }

    return {
        "name": "phase4_full",
        "display_name": "Phase 4 Full Training",
        "loss_mode": "full",
        "lambda_mel": float(args.phase2_lambda_mel_end),
        "use_gan": True,
        "train_generator": True,
        "scheduler_active": True,
        "lr_g": args.phase4_lr,
        "lr_d": args.phase4_d_lr,
        "train_batch_size": args.batch_size,
    }


def build_loaders(args, target_fs, train_batch_size=None, val_batch_size=None):
    if train_batch_size is None:
        train_batch_size = args.batch_size
    if val_batch_size is None:
        val_batch_size = args.batch_size

    if args.dataset == 'cocktail':
        train_loader = load_NeuroCodecDataset(
            root=args.root,
            subset='train',
            batch_size=train_batch_size,
            num_gpus=1
        )
        val_loader = load_NeuroCodecDataset(
            root=args.root,
            subset='val',
            batch_size=val_batch_size,
            num_gpus=1
        )
    elif args.dataset == 'kul':
        train_loader = load_KUL_NeuroCodecDataset(
            lmdb_path=args.root,
            subset='train',
            batch_size=train_batch_size,
            num_gpus=1,
            target_fs=target_fs,
            original_fs=16000
        )
        val_loader = load_KUL_NeuroCodecDataset(
            lmdb_path=args.root,
            subset='val',
            batch_size=val_batch_size,
            num_gpus=1,
            target_fs=target_fs,
            original_fs=16000
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return train_loader, val_loader


def build_checkpoint(model, discriminator, optimizer_g, optimizer_d, scheduler_g, scheduler_d, epoch, best_val_sisdr, last_val_sisdr):
    checkpoint = {
        "epoch": epoch,
        "best_val_sisdr": best_val_sisdr,
        "last_val_sisdr": last_val_sisdr,
        "model_state_dict": model.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "scheduler_g_state_dict": scheduler_g.state_dict(),
        "scheduler_d_state_dict": scheduler_d.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return checkpoint


def restore_checkpoint(checkpoint, model, discriminator, optimizer_g, optimizer_d, scheduler_g, scheduler_d):
    start_epoch = 0
    best_val_sisdr = float("-inf")
    last_val_sisdr = None

    if not isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint, strict=False)
        return start_epoch, best_val_sisdr, last_val_sisdr

    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        model_state = checkpoint["state_dict"]
    elif "model" in checkpoint:
        model_state = checkpoint["model"]
    else:
        model_state = checkpoint

    incompatible = model.load_state_dict(model_state, strict=False)
    if incompatible.missing_keys:
        print(f"Warning: Missing keys ({len(incompatible.missing_keys)}).")
    if incompatible.unexpected_keys:
        print(f"Warning: Unexpected keys ({len(incompatible.unexpected_keys)}): {incompatible.unexpected_keys}")

    if "discriminator_state_dict" in checkpoint:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    if "optimizer_g_state_dict" in checkpoint:
        optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    if "optimizer_d_state_dict" in checkpoint:
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
    if "scheduler_g_state_dict" in checkpoint:
        scheduler_g.load_state_dict(checkpoint["scheduler_g_state_dict"])
    if "scheduler_d_state_dict" in checkpoint:
        scheduler_d.load_state_dict(checkpoint["scheduler_d_state_dict"])

    if "torch_rng_state" in checkpoint:
        try:
            if isinstance(checkpoint["torch_rng_state"], torch.Tensor) and checkpoint["torch_rng_state"].dtype == torch.uint8:
                torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            else:
                print("Warning: skipping invalid torch RNG state in checkpoint.")
        except Exception as e:
            print(f"Warning: could not restore torch RNG state: {e}")

    if "numpy_rng_state" in checkpoint:
        try:
            np.random.set_state(checkpoint["numpy_rng_state"])
        except Exception as e:
            print(f"Warning: could not restore NumPy RNG state: {e}")

    if "python_rng_state" in checkpoint:
        try:
            random.setstate(checkpoint["python_rng_state"])
        except Exception as e:
            print(f"Warning: could not restore Python RNG state: {e}")

    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        try:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        except Exception as e:
            print(f"Warning: could not restore CUDA RNG state: {e}")


    start_epoch = checkpoint.get("epoch", -1) + 1
    if "best_val_sisdr" in checkpoint:
        best_val_sisdr = checkpoint["best_val_sisdr"]
    else:
        best_val_sisdr = float("-inf")
        if "best_val_loss" in checkpoint:
            print("Warning: checkpoint has best_val_loss but no best_val_sisdr; resetting best SI-SDR tracking.")
    last_val_sisdr = checkpoint.get("last_val_sisdr")
    return start_epoch, best_val_sisdr, last_val_sisdr

def sisdr(reference, estimation):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]
    Returns:
        SI-SDR
    """
    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)
    
    # This is to avoid zero energy
    # reference_energy[reference_energy == 0] = 1e-8

    # Optimal scaling factor
    alpha = np.sum(reference * estimation, axis=-1, keepdims=True) / (reference_energy + 1e-8)
    
    # Projection
    projections = alpha * reference
    
    # Noise
    noise = estimation - projections
    
    projections_energy = np.sum(projections ** 2, axis=-1)
    noise_energy = np.sum(noise ** 2, axis=-1)
    
    si_sdr_val = 10 * np.log10(projections_energy / (noise_energy + 1e-8))
    
    return si_sdr_val

def train(args):
    # 1. Setup Device & output dir
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print(f"Training on {device}...")
    print(f"Backbone: {args.backbone.upper()}")
    print(f"Dataset: {args.dataset.upper()}")
    


    # 2. Dataset
    print("Loading Dataset...")
    if args.dataset == 'kul':
        args.eeg_channels = 64 # Force 64 for KUL unless specified otherwise? No, respect arg but default is 128.
        # Check if user overrode default 128
        # Argparse doesn't tell us if it was default or user-specified easily without a separate flag.
        # But we can just warn.
        if args.eeg_channels == 128:
             print("Info: KUL dataset selected, defaulting EEG channels to 64 (overriding 128).")
             args.eeg_channels = 64
        
        # Audio Configuration for KUL
        dac_model_type = '16khz'
        target_fs = 16000
    else:
        dac_model_type = '44khz'
        target_fs = 44100

    train_loader, val_loader = build_loaders(args, target_fs)

    model = NeuroCodec(
        dac_model_type=dac_model_type,
        eeg_in_channels=args.eeg_channels,
        hidden_dim=args.hidden_dim, # e.g. 256
        num_layers=args.num_layers,  # e.g. 4
        backbone=args.backbone,
        activation=args.activation,
        dropout=args.dropout
    ).to(device)
    criterion = NeuroCodecLoss(lambda_recon=1.0).to(device)
    # Use paper full multi-scale mel schedule for Cocktail (44.1kHz).
    if args.dataset == 'cocktail':
        mel_window_lengths = [32, 64, 128, 256, 512, 1024, 2048]
        # Fixed bins from empirical torchaudio-valid limits for sr=44.1k/f_max=20k.
        mel_n_mels = [5, 8, 15, 29, 57, 112, 222]
        mel_hop_lengths = [w // 4 for w in mel_window_lengths]
        mel_fmax = 20000.0
    else:
        mel_window_lengths = [1024, 512, 256, 128]
        mel_n_mels = [80, 80, 32, 16]
        mel_hop_lengths = [w // 4 for w in mel_window_lengths]
        mel_fmax = 4000.0

    mel_loss_fn = MelSpectrogramLoss(
        sample_rate=target_fs,
        window_lengths=mel_window_lengths,
        n_mels=mel_n_mels,
        hop_lengths=mel_hop_lengths,
        f_max=mel_fmax
    ).to(device)
    
    # GAN Setup
    discriminator = dac.model.Discriminator(sample_rate=target_fs).to(device)
    gan_loss_fn = GANLoss(discriminator).to(device)
    
    # Optimizers
    # Generator Optimizer (NeuroCodec)
    optimizer_g = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.9))
    scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, mode='max', factor=0.5, patience=5
    )
    
    # Discriminator Optimizer
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.9))
    scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_d, mode='max', factor=0.5, patience=5
    )
    
    # 5. Training Loop
    start_epoch = 0
    best_val_sisdr = float('-inf')
    last_val_sisdr = None

    # 5.1 Load Checkpoint if Exists (Resume Training)
    latest_checkpoint = os.path.join(args.checkpoint_dir, "latest_model.pth")
    if os.path.exists(latest_checkpoint):
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            start_epoch, best_val_sisdr, last_val_sisdr = restore_checkpoint(
                checkpoint,
                model,
                discriminator,
                optimizer_g,
                optimizer_d,
                scheduler_g,
                scheduler_d,
            )
            print(f"Checkpoint loaded. Resuming at epoch {start_epoch + 1}.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")
    else:
        print("No existing checkpoint found. Starting from scratch.")

    phase2_epochs = get_phase2_epochs(args)
    min_full_training_epoch = args.phase1_epochs + phase2_epochs + args.phase3_epochs + 1
    print(
        "Training schedule | "
        f"Phase 1: MSE-only for {args.phase1_epochs} epochs @ lr={args.phase1_lr:.1e} | "
        f"Phase 2: mel ramp/hold for {phase2_epochs} epochs "
        f"(lambda_mel {args.phase2_lambda_mel_start}->{args.phase2_lambda_mel_end}, "
        f"hold {args.phase2_hold_epochs} epochs) | "
        f"Phase 3: GAN warmup for {args.phase3_epochs} epochs @ lr={args.phase3_lr:.1e} | "
        f"Phase 4: full training @ lr_g={args.phase4_lr:.1e}, lr_d={args.phase4_d_lr:.1e} "
        "with LR halving on val SI-SDR drop"
    )
    print(
        f"Batch schedule | Phase 1 train batch size: {args.phase1_batch_size} | "
        f"Phase 2 train batch size: {args.phase2_batch_size} | "
        f"Phase 3 train batch size: {args.phase3_batch_size} | "
        f"Phase 4 train batch size: {args.batch_size} | "
        f"Validation batch size: {args.batch_size}"
    )
    if args.epochs < min_full_training_epoch:
        print(
            f"Warning: epochs={args.epochs} only covers phases 1-3/early transition. "
            f"Use at least {min_full_training_epoch} epochs to enter Phase 4."
        )
    
    active_train_batch_size = None
    prev_phase_name = None
    for epoch in range(start_epoch, args.epochs):
        model.train()
        discriminator.train()
        total_loss = 0
        total_g_loss = 0
        total_d_loss = 0

        phase_cfg = get_training_phase(epoch, args)
        use_gan = phase_cfg["use_gan"]
        train_generator_epoch = phase_cfg["train_generator"]
        loss_mode = phase_cfg["loss_mode"]
        lambda_mel = phase_cfg["lambda_mel"]
        train_batch_size = phase_cfg["train_batch_size"]
        entering_new_phase = phase_cfg["name"] != prev_phase_name
        if (not phase_cfg["scheduler_active"]) or entering_new_phase:
            set_optimizer_lr(optimizer_g, phase_cfg["lr_g"])
            set_optimizer_lr(optimizer_d, phase_cfg["lr_d"])

        if active_train_batch_size != train_batch_size:
            train_loader, _ = build_loaders(args, target_fs, train_batch_size=train_batch_size, val_batch_size=args.batch_size)
            active_train_batch_size = train_batch_size
            print(f"Epoch {epoch+1}: rebuilding train loader with batch size {train_batch_size}")

        current_lr_g = optimizer_g.param_groups[0]['lr']
        current_lr_d = optimizer_d.param_groups[0]['lr']
        pbar = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch+1}/{args.epochs} | {phase_cfg['display_name']} "
                f"[BS: {train_batch_size}, LR_G: {current_lr_g:.1e}, LR_D: {current_lr_d:.1e}, MEL: {lambda_mel:.1f}]"
            )
        )
        
        for batch_idx, (noisy, eeg, clean) in enumerate(pbar):
            noisy = noisy.to(device)
            clean = clean.to(device)
            eeg = eeg.to(device)

            if args.noise_cue:
                # Replace EEG with Gaussian Noise matching statistics
                eeg_mean = eeg.mean()
                eeg_std = eeg.std()
                eeg = torch.randn_like(eeg) * eeg_std + eeg_mean
                
            # --- Generator Forward ---
            # 1. Encode Target (for Recon Loss)
            with torch.no_grad():
                z_target, _, _, _, _ = model.dac.encode(clean)
            
            # 2. Model Forward
            z_pred, _, _, _, _ = model(noisy, eeg)
            
            # --- MSE-only mode: skip decoder entirely ---
            if loss_mode == 'mse':
                train_generator = True
                use_gan = False
                optimizer_g.zero_grad()
                recon_loss, loss_dict = criterion(z_pred, z_target)
                loss_dict['loss_mel'] = 0.0
                loss_dict['loss_adv'] = 0.0
                loss_dict['loss_feat'] = 0.0
                recon_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_g.step()
                total_loss += recon_loss.item()
                log_recon_loss = recon_loss.item()
                d_loss = torch.tensor(0.0)
            else:
                # --- Full mode: decode for GAN / Mel Loss ---
                # DAC Quantizer
                z_q, _, _, _, _ = model.dac.quantizer(z_pred, n_quantizers=9)
                
                # Decode
                pred_audio = model.dac.decode(z_q)
                
                # Align lengths
                min_len = min(pred_audio.shape[-1], clean.shape[-1])
                pred_audio = pred_audio[..., :min_len]
                clean_aligned = clean[..., :min_len]
                
                # --- Discriminator Step ---
                if use_gan:
                    optimizer_d.zero_grad()
                    d_loss = gan_loss_fn.discriminator_loss(pred_audio.detach(), clean_aligned)
                    d_loss.backward()
                    optimizer_d.step()
                    total_d_loss += d_loss.item()
                else:
                    d_loss = torch.tensor(0.0)

                # --- Generator Step ---
                train_generator = train_generator_epoch
                
                optimizer_g.zero_grad()
                
                if train_generator:
                    # 1. Reconstruction Losses (MSE on latents)
                    recon_loss, loss_dict = criterion(z_pred, z_target)
                    
                    # 2. Mel Spectrogram Loss
                    if lambda_mel > 0:
                        mel_loss = mel_loss_fn(pred_audio, clean_aligned)
                        recon_loss += lambda_mel * mel_loss
                        loss_dict['loss_mel'] = mel_loss.item()
                    else:
                        loss_dict['loss_mel'] = 0.0
                    
                    # 3. GAN Generator Loss
                    if use_gan:
                        g_loss_adv, g_loss_feat = gan_loss_fn.generator_loss(pred_audio, clean_aligned)
                        gan_term = args.lambda_gan * g_loss_adv + args.lambda_feat * g_loss_feat
                        recon_loss += gan_term
                        loss_dict['loss_adv'] = g_loss_adv.item()
                        loss_dict['loss_feat'] = g_loss_feat.item()
                        total_g_loss += g_loss_adv.item() + g_loss_feat.item() 
                    
                    recon_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer_g.step()
                    
                    total_loss += recon_loss.item()
                    log_recon_loss = recon_loss.item()
                else:
                    # Generator Frozen: Calculate loss for logging but don't backward
                    with torch.no_grad():
                         recon_loss, loss_dict = criterion(z_pred, z_target)
                         log_recon_loss = recon_loss.item()
                         loss_dict['loss_mel'] = 0.0
                         if use_gan:
                            g_loss_adv, g_loss_feat = gan_loss_fn.generator_loss(pred_audio, clean_aligned)
                            loss_dict['loss_adv'] = g_loss_adv.item()
                            loss_dict['loss_feat'] = g_loss_feat.item()
                         else:
                            loss_dict['loss_adv'] = 0.0
                            loss_dict['loss_feat'] = 0.0
            
            # Logging
            postfix = {
                'loss': f"{log_recon_loss:.4f}", 
                'mse': f"{loss_dict['loss_recon']:.4f}",
                'phase': phase_cfg['name'],
            }
            if not train_generator:
                postfix['WARMUP'] = "D_ONLY"
                # 'env': f"{loss_dict.get('loss_env', 0):.4f}", # Removed env
            if 'loss_mel' in loss_dict:
                postfix['mel'] = f"{loss_dict['loss_mel']:.4f}"
            if use_gan:
                postfix['d_loss'] = f"{d_loss.item():.4f}"
                postfix['g_adv'] = f"{loss_dict['loss_adv']:.4f}"
                postfix['g_feat'] = f"{loss_dict['loss_feat']:.4f}"
                
            pbar.set_postfix(postfix)
            

            
            # Optional: Overfit check (break early)
            if args.debug and batch_idx > 5:
                break
                
        
        # Validation
        if (epoch + 1) % args.val_interval == 0:
            val_loss, val_sisdr, val_estoi = validate(model, val_loader, criterion, device, args)
            
            if phase_cfg["scheduler_active"] and last_val_sisdr is not None and val_sisdr < last_val_sisdr:
                scale_optimizer_lr(optimizer_g, 0.5)
                scale_optimizer_lr(optimizer_d, 0.5)
                print(
                    f"Val SI-SDR dropped from {last_val_sisdr:.2f} to {val_sisdr:.2f}. "
                    f"Halved LR_G to {optimizer_g.param_groups[0]['lr']:.1e} and "
                    f"LR_D to {optimizer_d.param_groups[0]['lr']:.1e}."
                )
            last_val_sisdr = val_sisdr
            
            print(
                f"Epoch {epoch+1} | {phase_cfg['display_name']} | "
                f"Train Loss: {total_loss/len(train_loader):.4f} | "
                f"Val Loss: {val_loss:.4f} | Val SI-SDR: {val_sisdr:.2f} dB | "
                f"Val ESTOI: {val_estoi:.4f} | lambda_mel: {lambda_mel:.1f}"
            )
            

            
            # Save Checkpoint (Best)
            if val_sisdr > best_val_sisdr:
                best_val_sisdr = val_sisdr
                torch.save(
                    build_checkpoint(
                        model,
                        discriminator,
                        optimizer_g,
                        optimizer_d,
                        scheduler_g,
                        scheduler_d,
                        epoch,
                        best_val_sisdr,
                        last_val_sisdr,
                    ),
                    os.path.join(args.checkpoint_dir, "best_model.pth")
                )
                print("Saved Best Model (by Val SI-SDR).")

        else:
            print(
                f"Epoch {epoch+1} | {phase_cfg['display_name']} | "
                f"Train Loss: {total_loss/len(train_loader):.4f} | "
                f"Validation Skipped | lambda_mel: {lambda_mel:.1f}"
            )

        
        # Save Latest Checkpoint (Every Epoch)
        torch.save(
            build_checkpoint(
                model,
                discriminator,
                optimizer_g,
                optimizer_d,
                scheduler_g,
                scheduler_d,
                epoch,
                best_val_sisdr,
                last_val_sisdr,
            ),
            os.path.join(args.checkpoint_dir, "latest_model.pth")
        )
        prev_phase_name = phase_cfg["name"]

def validate(model, loader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    sisdr_scores = []
    estoi_scores = []
    
    # Import for metrics
    from scipy import signal
    stoi_fn = None
    if getattr(args, 'estoi', False):
        try:
            from pystoi import stoi as _stoi
            stoi_fn = _stoi
        except ImportError:
            print("Warning: pystoi not installed, ESTOI will be skipped.")
    
    target_fs = 16000 if args.dataset == 'kul' else 44100

    with torch.no_grad():
        val_pbar = tqdm(loader, desc="Validation")
        for batch_idx, (noisy, eeg, clean) in enumerate(val_pbar):
            # ... (Existing Loading & Forward) ...
            noisy = noisy.to(device)
            clean = clean.to(device)
            eeg = eeg.to(device)
            
            if args.noise_cue:
                # Replace EEG with Gaussian Noise matching statistics
                eeg_mean = eeg.mean()
                eeg_std = eeg.std()
                eeg = torch.randn_like(eeg) * eeg_std + eeg_mean
            
            if args.val_batches > 0 and batch_idx >= args.val_batches:
                break
            
            # 1. Encode Target
            z_target, _, _, _, _ = model.dac.encode(clean)
            
            # 2. Model Forward (Z Pred)
            z_pred, _, _, _, _ = model(noisy, eeg)
            
            # 3. Loss
            loss, _ = criterion(z_pred, z_target)
            total_loss += loss.item()
            
            # 4. Skip decoding if MSE-only mode
            if args.loss == 'mse':
                # No audio decoding needed — report zero metrics
                continue
            
            # 5. Neural Decoding & SI-SDR (full mode only)
            z_q = model.dac.quantizer(z_pred, n_quantizers=9)[0]
            pred_audio = model.dac.decode(z_q)
            
            # SI-SDR Calculation
            min_len = min(pred_audio.shape[-1], clean.shape[-1])
            pred_audio = pred_audio[..., :min_len]
            clean_ref = clean[..., :min_len]
            
            pred_np = pred_audio.cpu().numpy().squeeze(1)
            clean_np = clean_ref.cpu().numpy().squeeze(1)
            
            # Handle Batch Dim if needed
            if pred_np.ndim == 1:
                pred_np = pred_np[np.newaxis, :]
                clean_np = clean_np[np.newaxis, :]
            
            batch_sisdr = []
            batch_estoi = []
            
            for b in range(pred_np.shape[0]):
                p = pred_np[b]
                c = clean_np[b]
                
                # Align
                correlation = signal.correlate(c, p, mode='full')
                lags = signal.correlation_lags(c.size, p.size, mode='full')
                lag = lags[np.argmax(correlation)]
                
                if lag > 0:
                    p_aligned = np.roll(p, shift=lag)
                    p_aligned[:lag] = 0
                elif lag < 0:
                    p_aligned = np.roll(p, shift=lag)
                    p_aligned[lag:] = 0
                else:
                    p_aligned = p
                
                # SI-SDR
                score = sisdr(c, p_aligned)
                batch_sisdr.append(score)
                
                # ESTOI
                if stoi_fn is not None:
                    try:
                        e_val = stoi_fn(c, p_aligned, target_fs, extended=True)
                        batch_estoi.append(e_val)
                    except Exception:
                        pass
            
            sisdr_scores.extend(batch_sisdr)
            estoi_scores.extend(batch_estoi)

            # Update progress bar
            running_loss = total_loss / (batch_idx + 1)
            running_sisdr = np.mean(sisdr_scores) if len(sisdr_scores) > 0 else 0.0
            val_pbar.set_postfix({'loss': f'{running_loss:.4f}', 'sisdr': f'{running_sisdr:.2f}'})

            if args.debug and batch_idx > 2:
                break
                
    num_batches = max(batch_idx + 1, 1) if 'batch_idx' in dir() else len(loader)
    mean_loss = total_loss / num_batches
    mean_sisdr = np.mean(sisdr_scores) if len(sisdr_scores) > 0 else 0.0
    mean_estoi = np.mean(estoi_scores) if len(estoi_scores) > 0 else 0.0
    
    return mean_loss, mean_sisdr, mean_estoi

if __name__ == "__main__":
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument('--dataset', type=str, default='cocktail', choices=['cocktail', 'kul'])
    bootstrap_parser.add_argument('--config', type=str, default=None, help='Path to a dataset-specific JSON config file')
    bootstrap_args, _ = bootstrap_parser.parse_known_args()

    config_path = bootstrap_args.config or get_default_config_path(bootstrap_args.dataset)
    config_defaults = load_config_file(config_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path, help='Path to a dataset-specific JSON config file')
    parser.add_argument('--root', type=str, default='/home/avishka/isuranga/TSE/data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=256) 
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoint_dir', type=str, default='/home/avishka/isuranga/TSE/NeuroCodec/checkpoints/CP/mse')
    parser.add_argument('--debug', action='store_true', help="Run fast debug mode")
    parser.add_argument('--dataset', type=str, default='cocktail', choices=['cocktail', 'kul'], help='Dataset to use')
    parser.add_argument('--eeg_channels', type=int, default=128, help='Number of EEG channels (128 for Cocktail, 64 for KUL)')
    
    parser.add_argument('--evaluate', action='store_true', help="Run validation only")
    parser.add_argument('--noise_cue', action='store_true', help="Use random noise instead of EEG during validation")
    
    parser.add_argument('--backbone', type=str, default='mamba', choices=['mamba', 'transformer'], help='Backbone architecture')
    parser.add_argument('--activation', type=str, default='gelu', choices=['gelu', 'snake', 'relu'], help='Activation function (transformer only)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate used in EEG encoder and fusion blocks')
    parser.add_argument('--val_interval', type=int, default=1, help="Validation interval in epochs (default: 1)")
    parser.add_argument('--lambda_mel', type=float, default=13, help="Weight for Mel Spectrogram Loss")
    parser.add_argument('--mel_start_epoch', type=int, default=0, help="Epoch to start applying Mel Loss")
    parser.add_argument('--val_batches', type=int, default=0, help="Limit number of validation batches (0 = full)")
    parser.add_argument('--estoi', action='store_true', help="Compute ESTOI during validation (slow, disabled by default)")
    parser.add_argument('--seed', type=int, default=42, help='Global random seed for reproducible training and evaluation')
    parser.add_argument('--phase1_epochs', type=int, default=8, help='Phase 1 duration: latent MSE-only training epochs')
    parser.add_argument('--phase1_lr', type=float, default=1e-4, help='Phase 1 generator learning rate')
    parser.add_argument('--phase1_batch_size', type=int, default=64, help='Phase 1 training batch size')
    parser.add_argument('--phase2_lambda_mel_start', type=int, default=2, help='Starting lambda_mel value for phase 2 ramp')
    parser.add_argument('--phase2_lambda_mel_end', type=int, default=15, help='Final lambda_mel value for phase 2 ramp and later phases')
    parser.add_argument('--phase2_hold_epochs', type=int, default=1, help='Extra phase 2 epochs to keep lambda_mel fixed at the final value')
    parser.add_argument('--phase2_batch_size', type=int, default=16, help='Phase 2 training batch size')
    parser.add_argument('--phase3_epochs', type=int, default=1, help='Phase 3 duration: discriminator warmup epochs')
    parser.add_argument('--phase3_batch_size', type=int, default=8, help='Phase 3 training batch size')
    parser.add_argument('--phase3_lr', type=float, default=1e-5, help='Phase 3 learning rate')
    parser.add_argument('--phase4_lr', type=float, default=1e-5, help='Phase 4 initial learning rate before scheduler updates')
    parser.add_argument('--phase4_d_lr', type=float, default=1e-5, help='Phase 4 discriminator learning rate before scheduler updates')
    
    parser.add_argument('--lambda_gan', type=float, default=0.5, help="Weight for GAN Adversarial Loss")
    parser.add_argument('--lambda_feat', type=float, default=1, help="Weight for GAN Feature Matching Loss")
    parser.add_argument('--gan_start_epoch', type=int, default=0, help="Epoch to start GAN training")
    parser.add_argument('--disc_warmup_epochs', type=int, default=0, help="Number of epochs to freeze Generator for Discriminator warmup")
    parser.add_argument('--loss', type=str, default='full', choices=['mse', 'full'], help="Loss mode: 'mse' = latent MSE only (no decoder), 'full' = MSE + Mel + GAN (requires decoder)")
    
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    print(f"Loaded config: {args.config}")
    
    # Set seed / deterministic behavior
    set_global_seed(args.seed)
    configure_determinism()
    
    if args.evaluate:
        # Evaluate Only Mode
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"Evaluating on {device}...")
        print(f"Backbone: {args.backbone.upper()}")
        print(f"Dataset: {args.dataset.upper()}")
        
        # Load Data
        target_fs = 16000 if args.dataset == 'kul' else 44100
        if args.dataset == 'cocktail':
             val_loader = load_NeuroCodecDataset(root=args.root, subset='val', batch_size=args.batch_size, num_gpus=1)
        elif args.dataset == 'kul':
             val_loader = load_KUL_NeuroCodecDataset(
                lmdb_path=args.root, 
                subset='val', 
                batch_size=args.batch_size, 
                num_gpus=1, 
                target_fs=target_fs,
                original_fs=16000 # Correct FS
             )
             # args.eeg_channels should be set by user or we trust default?
             # If user didn't set, default is 128 (wrong for KUL).
             # We should probably force it here if it's default?
             if args.eeg_channels == 128 and args.dataset == 'kul':
                 print("Warning: Dataset is KUL but eeg_channels is 128. Assuming user wants 64 (Autofix).")
                 args.eeg_channels = 64
        
        # Determine DAC type
        dac_type = '16khz' if args.dataset == 'kul' else '44khz'
        
        # Load Model
        model = NeuroCodec(
            dac_model_type=dac_type,
            eeg_in_channels=args.eeg_channels,
            hidden_dim=args.hidden_dim, 
            num_layers=args.num_layers,
            backbone=args.backbone,
            dropout=args.dropout
        ).to(device)
        
        checkpoint_path = args.checkpoint_dir if args.checkpoint_dir.endswith('.pth') else os.path.join(args.checkpoint_dir, "best_model.pth")
        
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"No checkpoint found at {checkpoint_path}! Running with random weights.")
            
        # Use NeuroCodecLoss for compatibility with validate() function signature
        criterion = NeuroCodecLoss(lambda_recon=1.0).to(device)
        
        val_loss, val_sisdr, val_estoi = validate(model, val_loader, criterion, device, args)
        print(f"Validation Result | Loss: {val_loss:.4f} | SI-SDR: {val_sisdr:.2f} dB | ESTOI: {val_estoi:.4f}")
        
    else:
        train(args)
