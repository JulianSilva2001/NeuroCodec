
import os
import json
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np
import dac.model  # Added this import
import dac.nn.layers

# Monkey-patch: replace JIT-compiled Snake with plain Python version
# (JIT snake is incompatible with torch.utils.checkpoint — produces
#  inconsistent internal tensor metadata on recomputation)
def _snake_no_jit(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x

dac.nn.layers.snake = _snake_no_jit
# Also patch the class method so existing instances use the new function
dac.nn.layers.Snake1d.forward = lambda self, x: _snake_no_jit(x, self.alpha)

# Local imports
from models.neurocodec import NeuroCodec
from dataset_neurocodec import load_NeuroCodecDataset, load_KUL_NeuroCodecDataset
# Consolidated and updated imports from 'losses' and 'losses_neurocodec'
from losses import MelSpectrogramLoss, GANLoss  # Added GANLoss
from losses_neurocodec import NeuroCodecLoss  # Kept this as it was in the original code

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
    
    # Initialize WandB
    if not args.debug:
        wandb.init(project="NeuroCodec", config=vars(args))

    # 2. Dataset
    print("Loading Dataset...")
    if args.dataset == 'kul':
        args.eeg_channels = 64
        if args.eeg_channels == 128:
             print("Info: KUL dataset selected, defaulting EEG channels to 64 (overriding 128).")
             args.eeg_channels = 64
        
        # Audio Configuration for KUL
        dac_model_type = '16khz'
        target_fs = 16000
    else:
        dac_model_type = '44khz'
        target_fs = 44100

    # Initial batch size comes from phase 1 default (will be adjusted on resume)
    init_batch_size = args.batch_size  # placeholder; overridden below after phase is known
    if args.dataset == 'cocktail':
         train_loader = load_NeuroCodecDataset(
            root=args.root, 
            subset='train', 
            batch_size=init_batch_size,
            num_gpus=1
        )
         val_loader = load_NeuroCodecDataset(
            root=args.root, 
            subset='val', 
            batch_size=init_batch_size, 
            num_gpus=1
        )
    elif args.dataset == 'kul':
         train_loader = load_KUL_NeuroCodecDataset(
            lmdb_path=args.root,
            subset='train',
            batch_size=init_batch_size,
            num_gpus=1,
            target_fs=target_fs,
            original_fs=16000
         )
         val_loader = load_KUL_NeuroCodecDataset(
            lmdb_path=args.root,
            subset='val',
            batch_size=init_batch_size,
            num_gpus=1,
            target_fs=target_fs,
            original_fs=16000
         )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    model = NeuroCodec(
        dac_model_type=dac_model_type,
        eeg_in_channels=args.eeg_channels,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        backbone=args.backbone,
        activation=args.activation
    ).to(device)
    
    if not args.debug:
        wandb.watch(model, log="all", log_freq=100)
    
    # 3. Loss Functions
    criterion = NeuroCodecLoss(lambda_recon=1.0, lambda_env=0.0).to(device)
    mel_loss_fn = MelSpectrogramLoss(
        sample_rate=16000,
        window_lengths=[1024, 512, 256, 128],
        n_mels=[80, 80, 32, 16],
        f_max=4000
    ).to(device)
    
    # GAN Setup
    discriminator = dac.model.Discriminator(sample_rate=16000).to(device)
    gan_loss_fn = GANLoss(discriminator).to(device)
    
    # 4. Optimizers
    optimizer_g = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.9))
    
    # Scheduler for Phase 6 (created later when entering phase 6)
    scheduler_g = None
    scheduler_d = None
    
    # =========================================================================
    # MULTI-PHASE STATE MACHINE
    # =========================================================================
    # Phase 1: MSE only                                           (batch_size=64)
    # Phase 2: MSE + Mel (LR -> 1e-4, lambda_mel ramps 2->13)    (batch_size=16)
    # Phase 3: MSE + Mel (lambda_mel=13, LR -> 5e-5), plateau    (batch_size=16)
    # Phase 4: Freeze generator, train discriminator 4 epochs     (batch_size=8)
    # Phase 5: Unfreeze, ramp GAN lambdas                         (batch_size=8)
    # Phase 6: Hold all lambdas, ReduceLROnPlateau scheduler       (batch_size=8)
    # =========================================================================
    
    PHASE_BATCH_SIZE = {1: 64, 2: 8, 3: 8, 4: 8, 5: 8, 6: 8}
    
    def rebuild_loaders(bs):
        """Rebuild train/val DataLoaders with a new batch size."""
        nonlocal train_loader, val_loader
        print(f"  → Rebuilding DataLoaders with batch_size={bs}")
        if args.dataset == 'cocktail':
            train_loader = load_NeuroCodecDataset(
                root=args.root, subset='train', batch_size=bs, num_gpus=1
            )
            val_loader = load_NeuroCodecDataset(
                root=args.root, subset='val', batch_size=bs, num_gpus=1
            )
        elif args.dataset == 'kul':
            train_loader = load_KUL_NeuroCodecDataset(
                lmdb_path=args.root, subset='train', batch_size=bs,
                num_gpus=1, target_fs=target_fs, original_fs=16000
            )
            val_loader = load_KUL_NeuroCodecDataset(
                lmdb_path=args.root, subset='val', batch_size=bs,
                num_gpus=1, target_fs=target_fs, original_fs=16000
            )
    
    phase = 1
    phase_epoch = 0          # epoch counter within current phase
    lambda_mel = 0.0
    lambda_gan = 0.0
    lambda_feat = 0.0
    best_sisdr = -float('inf')
    sisdr_patience_counter = 0
    best_val_loss = float('inf')
    start_epoch = 0
    
    # =========================================================================
    # HELPER: Set LR for all param groups
    # =========================================================================
    def set_lr(optimizer, lr):
        for pg in optimizer.param_groups:
            pg['lr'] = lr
    
    # =========================================================================
    # HELPER: Check SI-SDR plateau
    # =========================================================================
    def check_sisdr_plateau(current_sisdr):
        nonlocal best_sisdr, sisdr_patience_counter
        if current_sisdr > best_sisdr:
            best_sisdr = current_sisdr
            sisdr_patience_counter = 0
            return False  # Not plateaued
        else:
            sisdr_patience_counter += 1
            if sisdr_patience_counter >= 3:
                return True  # Plateaued
            return False
    
    # 5. Load checkpoint if exists (resume training)
    checkpoint_path = os.path.join(args.checkpoint_dir, "training_state.pth")
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            if ckpt.get('discriminator_state_dict'):
                discriminator.load_state_dict(ckpt['discriminator_state_dict'])
            if ckpt.get('optimizer_g_state_dict'):
                optimizer_g.load_state_dict(ckpt['optimizer_g_state_dict'])
            if ckpt.get('optimizer_d_state_dict'):
                optimizer_d.load_state_dict(ckpt['optimizer_d_state_dict'])
            phase = ckpt.get('phase', 1)
            phase_epoch = ckpt.get('phase_epoch', 0)
            lambda_mel = ckpt.get('lambda_mel', 0.0)
            lambda_gan = ckpt.get('lambda_gan', 0.0)
            lambda_feat = ckpt.get('lambda_feat', 0.0)
            best_sisdr = ckpt.get('best_sisdr', -float('inf'))
            sisdr_patience_counter = ckpt.get('sisdr_patience_counter', 0)
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            start_epoch = ckpt.get('epoch', 0) + 1
            
            # Recreate schedulers if in phase 6
            if phase == 6:
                scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_g, mode='min', factor=0.5, patience=5
                )
                scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_d, mode='min', factor=0.5, patience=5
                )
                if 'scheduler_g_state_dict' in ckpt:
                    scheduler_g.load_state_dict(ckpt['scheduler_g_state_dict'])
                if 'scheduler_d_state_dict' in ckpt:
                    scheduler_d.load_state_dict(ckpt['scheduler_d_state_dict'])
            
            print(f"Resumed at epoch {start_epoch}, Phase {phase}, phase_epoch {phase_epoch}")
            print(f"  lambda_mel={lambda_mel:.1f}, lambda_gan={lambda_gan:.2f}, lambda_feat={lambda_feat:.2f}")
            print(f"  best_sisdr={best_sisdr:.2f}, patience={sisdr_patience_counter}")
            
            # Set correct LR for the resumed phase
            if phase == 2:
                set_lr(optimizer_g, 1e-4)
                print(f"  LR_G set to 1e-4 for Phase 2")
            elif phase >= 3 and phase <= 5:
                set_lr(optimizer_g, 5e-5)
                print(f"  LR_G set to 5e-5 for Phase {phase}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
            phase = 1
    else:
        # Also try loading a legacy checkpoint (just model weights)
        legacy_ckpt = os.path.join(args.checkpoint_dir, "latest_model.pth")
        if os.path.exists(legacy_ckpt):
            print(f"Found legacy checkpoint: {legacy_ckpt}")
            try:
                model.load_state_dict(torch.load(legacy_ckpt, map_location=device))
                print("Legacy model weights loaded. Starting from Phase 1, epoch 0.")
            except Exception as e:
                print(f"Failed to load legacy checkpoint: {e}. Starting from scratch.")
        else:
            print("No existing checkpoint found. Starting from scratch.")
    
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    # Rebuild DataLoaders with the correct batch size for the current phase
    current_bs = PHASE_BATCH_SIZE.get(phase, 8)
    rebuild_loaders(current_bs)
    
    print(f"\n{'='*60}")
    print(f"Starting training at Phase {phase} (batch_size={current_bs})")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs):
        # =================================================================
        # PHASE-SPECIFIC SETUP (at start of each epoch)
        # =================================================================
        
        # Determine what to train this epoch based on phase
        train_generator = (phase != 4)
        train_discriminator = (phase >= 4)
        use_mel = (phase >= 2)
        use_gan = (phase >= 5)
        
        # Phase 4: freeze generator
        if phase == 4:
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            discriminator.train()
        else:
            model.train()
            for p in model.parameters():
                p.requires_grad_(True)
            if train_discriminator:
                discriminator.train()
        
        current_lr_g = optimizer_g.param_groups[0]['lr']
        current_lr_d = optimizer_d.param_groups[0]['lr']
        
        phase_str = f"Phase {phase}"
        desc = f"Epoch {epoch+1}/{args.epochs} [{phase_str}] [LR_G: {current_lr_g:.1e}"
        if train_discriminator:
            desc += f", LR_D: {current_lr_d:.1e}"
        desc += f", λ_mel={lambda_mel:.1f}"
        if use_gan:
            desc += f", λ_gan={lambda_gan:.2f}, λ_feat={lambda_feat:.2f}"
        desc += "]"
        
        pbar = tqdm(train_loader, desc=desc)
        
        total_loss = 0
        total_g_loss = 0
        total_d_loss = 0
        
        for batch_idx, (noisy, eeg, clean) in enumerate(pbar):
            noisy = noisy.to(device)
            clean = clean.to(device)
            eeg = eeg.to(device)

            if args.noise_cue:
                eeg_mean = eeg.mean()
                eeg_std = eeg.std()
                eeg = torch.randn_like(eeg) * eeg_std + eeg_mean
                
            # --- Generator Forward ---
            with torch.no_grad():
                z_target, _, _, _, _ = model.dac.encode(clean)
            
            if train_generator:
                z_pred, _, _, _, _, env_pred = model(noisy, eeg)
            else:
                with torch.no_grad():
                    z_pred, _, _, _, _, env_pred = model(noisy, eeg)
            
            # Quantize & decode for mel/GAN losses
            if use_mel or use_gan or train_discriminator:
                torch.cuda.empty_cache()
                z_q, _, _, _, _ = model.dac.quantizer(z_pred, n_quantizers=9)
                pred_audio = grad_checkpoint(
                    model.dac.decode, z_q, use_reentrant=False
                )
                
                min_len = min(pred_audio.shape[-1], clean.shape[-1])
                pred_audio = pred_audio[..., :min_len]
                clean_aligned = clean[..., :min_len]
            
            # --- Discriminator Step ---
            if train_discriminator:
                optimizer_d.zero_grad()
                d_loss = gan_loss_fn.discriminator_loss(pred_audio.detach(), clean_aligned)
                d_loss.backward()
                optimizer_d.step()
                total_d_loss += d_loss.item()
            else:
                d_loss = torch.tensor(0.0)

            # --- Generator Step ---
            if train_generator:
                optimizer_g.zero_grad()
                
                # 1. MSE Reconstruction Loss (always on)
                recon_loss, loss_dict = criterion(z_pred, z_target, env_pred, clean)
                
                # 2. Mel Spectrogram Loss
                if use_mel and lambda_mel > 0:
                    mel_loss = mel_loss_fn(pred_audio, clean_aligned)
                    mel_loss = torch.clamp(mel_loss, max=100.0)  # Prevent explosion
                    recon_loss += lambda_mel * mel_loss
                    loss_dict['loss_mel'] = mel_loss.item()
                else:
                    loss_dict['loss_mel'] = 0.0
                
                # 3. GAN Generator Loss
                if use_gan and lambda_gan > 0:
                    g_loss_adv, g_loss_feat = gan_loss_fn.generator_loss(pred_audio, clean_aligned)
                    gan_term = lambda_gan * g_loss_adv + lambda_feat * g_loss_feat
                    recon_loss += gan_term
                    loss_dict['loss_adv'] = g_loss_adv.item()
                    loss_dict['loss_feat'] = g_loss_feat.item()
                    total_g_loss += g_loss_adv.item() + g_loss_feat.item()
                
                # Guard: skip batch if loss is NaN/Inf/exploded
                if torch.isnan(recon_loss) or torch.isinf(recon_loss) or recon_loss.item() > 1e6:
                    print(f"  ⚠ Skipping batch {batch_idx}: loss={recon_loss.item():.2e}")
                    optimizer_g.zero_grad()
                    continue
                
                recon_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_g.step()
                
                total_loss += recon_loss.item()
                log_recon_loss = recon_loss.item()
            else:
                # Phase 4: generator frozen, just log
                with torch.no_grad():
                    recon_loss, loss_dict = criterion(z_pred, z_target, env_pred, clean)
                    log_recon_loss = recon_loss.item()
                    loss_dict['loss_mel'] = 0.0
            
            # Logging
            postfix = {
                'phase': phase,
                'loss': f"{log_recon_loss:.4f}",
                'mse': f"{loss_dict['loss_recon']:.4f}",
            }
            if use_mel:
                postfix['mel'] = f"{loss_dict['loss_mel']:.4f}"
                postfix['λm'] = f"{lambda_mel:.0f}"
            if train_discriminator:
                postfix['d_loss'] = f"{d_loss.item():.4f}"
            if use_gan:
                postfix['g_adv'] = f"{loss_dict.get('loss_adv', 0):.4f}"
                postfix['g_feat'] = f"{loss_dict.get('loss_feat', 0):.4f}"
            if phase == 4:
                postfix['DISC_ONLY'] = f"{phase_epoch+1}/4"
                
            pbar.set_postfix(postfix)
            
            if not args.debug:
                log_dict = {
                    "train_loss": log_recon_loss,
                    "train_loss_recon": loss_dict['loss_recon'],
                    "lr_g": current_lr_g,
                    "lr_d": current_lr_d,
                    "phase": phase,
                    "lambda_mel": lambda_mel,
                    "lambda_gan": lambda_gan,
                    "lambda_feat": lambda_feat,
                }
                if use_mel:
                    log_dict['train_loss_mel'] = loss_dict['loss_mel']
                if train_discriminator:
                    log_dict['train_loss_d'] = d_loss.item()
                if use_gan:
                    log_dict['train_loss_adv'] = loss_dict.get('loss_adv', 0.0)
                    log_dict['train_loss_feat'] = loss_dict.get('loss_feat', 0.0)
                    
                wandb.log(log_dict)
            
            if args.debug and batch_idx > 5:
                break
        
        # =================================================================
        # VALIDATION
        # =================================================================
        val_loss = None
        val_sisdr = None
        
        if (epoch + 1) % args.val_interval == 0:
            val_loss, val_sisdr = validate(model, val_loader, criterion, device, args)
            
            print(f"\nEpoch {epoch+1} [Phase {phase}] | "
                  f"Train Loss: {total_loss/max(len(train_loader),1):.4f} | "
                  f"Val MSE: {val_loss:.4f} | Val SI-SDR: {val_sisdr:.2f} dB")
            
            if not args.debug:
                wandb.log({
                    "val_loss": val_loss,
                    "val_sisdr": val_sisdr,
                    "epoch": epoch + 1
                })
            
            # Log validation results to JSON file
            val_json_path = os.path.join(args.checkpoint_dir, "val_results.json")
            val_entry = {
                "epoch": epoch + 1,
                "phase": phase,
                "val_loss": round(float(val_loss), 6),
                "val_sisdr": round(float(val_sisdr), 4),
                "train_loss": round(total_loss / max(len(train_loader), 1), 6),
                "lambda_mel": lambda_mel,
                "lambda_gan": lambda_gan,
                "lambda_feat": lambda_feat,
                "lr_g": optimizer_g.param_groups[0]['lr'],
                "lr_d": optimizer_d.param_groups[0]['lr'],
                "best_sisdr": round(float(best_sisdr), 4) if best_sisdr != -float('inf') else None,
                "timestamp": datetime.now().isoformat()
            }
            # Load existing or create new
            if os.path.exists(val_json_path):
                with open(val_json_path, 'r') as f:
                    val_history = json.load(f)
            else:
                val_history = []
            val_history.append(val_entry)
            with open(val_json_path, 'w') as f:
                json.dump(val_history, f, indent=2)
            print(f"  ✓ Validation logged to {val_json_path}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pth"))
                print("  ✓ Saved Best Model.")
            
            # Phase 6: step the LR schedulers
            if phase == 6 and scheduler_g is not None:
                scheduler_g.step(val_loss)
                scheduler_d.step(val_loss)
        else:
            print(f"\nEpoch {epoch+1} [Phase {phase}] | "
                  f"Train Loss: {total_loss/max(len(train_loader),1):.4f} | Validation Skipped")
            if not args.debug:
                wandb.log({"epoch": epoch + 1})
        
        # =================================================================
        # PHASE TRANSITIONS
        # =================================================================
        
        # --- Phase 1 → 2: val MSE drops below 8.2 ---
        if phase == 1 and val_loss is not None and val_loss < 8.8:
            phase = 2
            phase_epoch = 0
            lambda_mel = 1.0
            set_lr(optimizer_g, 1e-4)
            best_sisdr = -float('inf')
            sisdr_patience_counter = 0
            rebuild_loaders(PHASE_BATCH_SIZE[2])
            print(f"\n{'='*60}")
            print(f"[PHASE 1 → 2] Val MSE {val_loss:.4f} < 8.8")
            print(f"  → LR set to 1e-4, lambda_mel starting at 2.0")
            print(f"  → batch_size → {PHASE_BATCH_SIZE[2]}")
            print(f"{'='*60}\n")
        
        # --- Phase 2: ramp lambda_mel, then watch SI-SDR ---
        elif phase == 2:
            phase_epoch += 1
            
            if lambda_mel < 13.0:
                lambda_mel = min(lambda_mel + 0.25, 10.0)
                print(f"  [Phase 2] lambda_mel → {lambda_mel:.0f}")
            
            # Once lambda_mel is at 13, start checking SI-SDR plateau
            if lambda_mel >= 13.0 and val_sisdr is not None:
                plateaued = check_sisdr_plateau(val_sisdr)
                print(f"  [Phase 2] SI-SDR: {val_sisdr:.2f} | best: {best_sisdr:.2f} | patience: {sisdr_patience_counter}/3")
                
                if plateaued:
                    phase = 3
                    phase_epoch = 0
                    set_lr(optimizer_g, 5e-5)
                    best_sisdr = -float('inf')  # Reset for phase 3 tracking
                    sisdr_patience_counter = 0
                    # batch_size stays at 16 (same as phase 2), no rebuild needed
                    print(f"\n{'='*60}")
                    print(f"[PHASE 2 → 3] SI-SDR plateaued (patience exhausted)")
                    print(f"  → LR reduced to 5e-5, continuing with lambda_mel=13")
                    print(f"{'='*60}\n")
        
        # --- Phase 3: Continue until SI-SDR plateaus again ---
        elif phase == 3:
            phase_epoch += 1
            
            if val_sisdr is not None:
                plateaued = check_sisdr_plateau(val_sisdr)
                print(f"  [Phase 3] SI-SDR: {val_sisdr:.2f} | best: {best_sisdr:.2f} | patience: {sisdr_patience_counter}/3")
                
                if plateaued:
                    phase = 4
                    phase_epoch = 0
                    rebuild_loaders(PHASE_BATCH_SIZE[4])
                    
                    # Freeze generator
                    print(f"\n{'='*60}")
                    print(f"[PHASE 3 → 4] SI-SDR plateaued again")
                    print(f"  → Freezing generator, training discriminator for 4 epochs")
                    print(f"  → Saving discriminator weights")
                    print(f"  → batch_size → {PHASE_BATCH_SIZE[4]}")
                    print(f"{'='*60}\n")
                    
                    # Save discriminator initial weights
                    torch.save(discriminator.state_dict(),
                               os.path.join(args.checkpoint_dir, "disc_phase4_start.pth"))
        
        # --- Phase 4: Train disc for 4 epochs, then → 5 ---
        elif phase == 4:
            phase_epoch += 1
            
            # Save discriminator weights every epoch in phase 4
            torch.save(discriminator.state_dict(),
                       os.path.join(args.checkpoint_dir, "disc_latest.pth"))
            print(f"  [Phase 4] Discriminator epoch {phase_epoch}/4 — disc weights saved")
            
            if phase_epoch >= 4:
                phase = 5
                phase_epoch = 0
                lambda_gan = 0.5
                lambda_feat = 1.0
                
                # Unfreeze generator
                for p in model.parameters():
                    p.requires_grad_(True)
                
                print(f"\n{'='*60}")
                print(f"[PHASE 4 → 5] 4 disc epochs complete")
                print(f"  → Unfreezing generator")
                print(f"  → lambda_gan=0.5, lambda_feat=1.0 (will ramp)")
                print(f"{'='*60}\n")
        
        # --- Phase 5: Ramp GAN lambdas, then → 6 ---
        elif phase == 5:
            phase_epoch += 1
            
            # Save disc weights every epoch
            torch.save(discriminator.state_dict(),
                       os.path.join(args.checkpoint_dir, "disc_latest.pth"))
            
            if lambda_gan < 1.0:
                lambda_gan = min(lambda_gan + 0.1, 1.0)
            if lambda_feat < 2.0:
                lambda_feat = min(lambda_feat + 0.2, 2.0)
            
            print(f"  [Phase 5] lambda_gan → {lambda_gan:.2f}, lambda_feat → {lambda_feat:.2f}")
            
            # Transition once both are maxed
            if lambda_gan >= 1.0 and lambda_feat >= 2.0:
                phase = 6
                phase_epoch = 0
                
                # Create LR schedulers for final convergence
                scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_g, mode='min', factor=0.5, patience=3
                )
                scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_d, mode='min', factor=0.5, patience=3
                )
                
                print(f"\n{'='*60}")
                print(f"[PHASE 5 → 6] GAN lambdas fully ramped")
                print(f"  → Holding lambda_gan=1.0, lambda_feat=2.0, lambda_mel=13")
                print(f"  → Using ReduceLROnPlateau scheduler for convergence")
                print(f"{'='*60}\n")
        
        # --- Phase 6: Final convergence (scheduler handles LR) ---
        elif phase == 6:
            phase_epoch += 1
            
            # Save disc weights every epoch
            torch.save(discriminator.state_dict(),
                       os.path.join(args.checkpoint_dir, "disc_latest.pth"))
            
            print(f"  [Phase 6] Converging — LR_G: {optimizer_g.param_groups[0]['lr']:.1e}, "
                  f"LR_D: {optimizer_d.param_groups[0]['lr']:.1e}")
        
        # =================================================================
        # SAVE FULL TRAINING STATE (every epoch)
        # =================================================================
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'phase': phase,
            'phase_epoch': phase_epoch,
            'lambda_mel': lambda_mel,
            'lambda_gan': lambda_gan,
            'lambda_feat': lambda_feat,
            'best_sisdr': best_sisdr,
            'sisdr_patience_counter': sisdr_patience_counter,
            'best_val_loss': best_val_loss,
        }
        if scheduler_g is not None:
            save_dict['scheduler_g_state_dict'] = scheduler_g.state_dict()
        if scheduler_d is not None:
            save_dict['scheduler_d_state_dict'] = scheduler_d.state_dict()
        
        torch.save(save_dict, os.path.join(args.checkpoint_dir, "training_state.pth"))
        
        # Also save model-only checkpoint for easy loading
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "latest_model.pth"))

def validate(model, loader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    sisdr_scores = []
    
    from scipy import signal
    
    val_pbar = tqdm(loader, desc="Validating")
    with torch.no_grad():
        for batch_idx, (noisy, eeg, clean) in enumerate(val_pbar):
            noisy = noisy.to(device)
            clean = clean.to(device)
            eeg = eeg.to(device)
            
            if args.noise_cue:
                eeg_mean = eeg.mean()
                eeg_std = eeg.std()
                eeg = torch.randn_like(eeg) * eeg_std + eeg_mean
            
            if args.val_batches > 0 and batch_idx >= args.val_batches:
                break
            
            # 1. Encode Target
            z_target, _, _, _, _ = model.dac.encode(clean)
            
            # 2. Model Forward (Z Pred)
            z_pred, _, _, _, _, env_pred = model(noisy, eeg)
            
            # 3. Loss (MSE on latents)
            loss, _ = criterion(z_pred, z_target, env_pred, clean)
            total_loss += loss.item()
            
            # 4. Decode for SI-SDR
            z_q = model.dac.quantizer(z_pred, n_quantizers=9)[0]
            pred_audio = model.dac.decode(z_q)
            
            # SI-SDR Calculation
            min_len = min(pred_audio.shape[-1], clean.shape[-1])
            pred_audio = pred_audio[..., :min_len]
            clean_ref = clean[..., :min_len]
            
            pred_np = pred_audio.cpu().numpy().squeeze(1)
            clean_np = clean_ref.cpu().numpy().squeeze(1)
            
            if pred_np.ndim == 1:
                pred_np = pred_np[np.newaxis, :]
                clean_np = clean_np[np.newaxis, :]
            
            batch_sisdr = []
            
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
            
            sisdr_scores.extend(batch_sisdr)
            
            # Update progress bar with running stats
            running_loss = total_loss / (batch_idx + 1)
            running_sisdr = np.mean(sisdr_scores) if sisdr_scores else 0.0
            val_pbar.set_postfix({
                'loss': f"{running_loss:.4f}",
                'sisdr': f"{running_sisdr:.2f}"
            })

            if args.debug and batch_idx > 2:
                break
                
    mean_loss = total_loss / max(len(loader), 1)
    mean_sisdr = np.mean(sisdr_scores) if len(sisdr_scores) > 0 else 0.0
    
    return mean_loss, mean_sisdr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/workspace/KUL-mix/KUL_eeg/kul_all_subjects.lmdb')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate (used in Phase 1)")
    parser.add_argument('--epochs', type=int, default=200, help="Max total epochs across all phases")
    parser.add_argument('--hidden_dim', type=int, default=256) 
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/neurocodec/KUL/mamba-GAN')
    parser.add_argument('--debug', action='store_true', help="Run fast debug mode")
    parser.add_argument('--dataset', type=str, default='kul', choices=['cocktail', 'kul'], help='Dataset to use')
    parser.add_argument('--eeg_channels', type=int, default=64, help='Number of EEG channels (128 for Cocktail, 64 for KUL)')
    
    parser.add_argument('--evaluate', action='store_true', help="Run validation only")
    parser.add_argument('--noise_cue', action='store_true', help="Use random noise instead of EEG during validation")
    
    parser.add_argument('--backbone', type=str, default='mamba', choices=['mamba', 'transformer'], help='Backbone architecture')
    parser.add_argument('--activation', type=str, default='gelu', choices=['gelu', 'snake', 'relu'], help='Activation function (transformer only)')
    parser.add_argument('--val_interval', type=int, default=2, help="Validation interval in epochs (default: 2)")
    parser.add_argument('--val_batches', type=int, default=0, help="Limit number of validation batches (0 = full)")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(42)
    
    if args.evaluate:
        # Evaluate Only Mode
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"Evaluating on {device}...")
        print(f"Backbone: {args.backbone.upper()}")
        print(f"Dataset: {args.dataset.upper()}")
        
        # Load Data
        if args.dataset == 'cocktail':
             val_loader = load_NeuroCodecDataset(root=args.root, subset='val', batch_size=args.batch_size, num_gpus=1)
        elif args.dataset == 'kul':
             val_loader = load_KUL_NeuroCodecDataset(
                lmdb_path=args.root, 
                subset='val', 
                batch_size=args.batch_size, 
                num_gpus=1, 
                target_fs=16000,
                original_fs=16000
             )
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
            backbone=args.backbone
        ).to(device)
        
        checkpoint_path = args.checkpoint_dir if args.checkpoint_dir.endswith('.pth') else os.path.join(args.checkpoint_dir, "best_model.pth")
        
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint {checkpoint_path}...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"No checkpoint found at {checkpoint_path}! Running with random weights.")
            
        criterion = NeuroCodecLoss(lambda_recon=1.0, lambda_env=0.0).to(device)
        
        val_loss, val_sisdr = validate(model, val_loader, criterion, device, args)
        print(f"Validation Result | Loss: {val_loss:.4f} | SI-SDR: {val_sisdr:.2f} dB")
        
    else:
        train(args)
