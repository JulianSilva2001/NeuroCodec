
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import wandb
import numpy as np
import scipy.signal
import torch.nn.functional as F
from losses.mel_loss import MelSpectrogramLoss
from losses.gan_loss import GANLoss
from models.discriminator import Discriminator

# Local imports
from models.neurocodec import NeuroCodec
from dataset_neurocodec import load_NeuroCodecDataset, load_KUL_NeuroCodecDataset
from losses_neurocodec import NeuroCodecLoss

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

def align_signals(ref, est):
    """
    Aligns est to ref using cross-correlation (FFT based).
    Returns aligned_ref, aligned_est (both truncated to common length).
    """
    if ref.ndim == 1:
        ref = ref[np.newaxis, :]
    if est.ndim == 1:
        est = est[np.newaxis, :]
        
    # Assume distinct samples in batch if batched. But validation loop does item-wise?
    # Actually validation loop passes (B, T) to sisdr? No, line 279 squeeze(1).
    # If B>1, sisdr handles it (axis=-1).
    # But alignment needs to be per-sample for batch > 1.
    # The current validation loop iterates batches. 
    # SISDR function takes (B, T).
    # We need to loop over batch to align each sample individually.
    
    # Simple implementation: Iterate batch
    B, T = ref.shape
    aligned_ref_list = []
    aligned_est_list = []
    
    for i in range(B):
        r = ref[i]
        e = est[i]
        
        correlation = scipy.signal.fftconvolve(r, e[::-1], mode='full')
        lag = np.argmax(correlation) - (len(e) - 1)
        
        if lag > 0:
            # Est is ahead of Ref (shifted left) -> Est starts later? 
            # If lag positive: Ref[lag:] aligns with Est[:-lag]
            # Wait, implementation details matter.
            r_aligned = r[lag:]
            e_aligned = e[:len(e)-lag]
        elif lag < 0:
            r_aligned = r[:len(r)+lag] 
            e_aligned = e[-lag:]
        else:
            r_aligned = r
            e_aligned = e
            
        # Ensure lengths match exactly
        min_l = min(len(r_aligned), len(e_aligned))
        aligned_ref_list.append(r_aligned[:min_l])
        aligned_est_list.append(e_aligned[:min_l])
        
    # Pad to same length to return batch? Or return list?
    # SISDR expects numpy array.
    # Since lag varies, lengths vary. SISDR can't take ragged array efficiently without padding.
    # But usually eval scores are mean. We can compute SISDR per sample here.
    return aligned_ref_list, aligned_est_list

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

    if args.dataset == 'cocktail':
         train_loader = load_NeuroCodecDataset(
            root=args.root, 
            subset='train', 
            batch_size=args.batch_size,
            num_gpus=1
        )
         val_loader = load_NeuroCodecDataset(
            root=args.root, 
            subset='val', 
            batch_size=args.batch_size, 
            num_gpus=1
        )
    elif args.dataset == 'kul':
         train_loader = load_KUL_NeuroCodecDataset(
            lmdb_path=args.root, # Root should be LMDB path for KUL
            subset='train',
            batch_size=args.batch_size,
            num_gpus=1,
            target_fs=target_fs
         )
         val_loader = load_KUL_NeuroCodecDataset(
            lmdb_path=args.root,
            subset='val',
            batch_size=args.batch_size,
            num_gpus=1,
            target_fs=target_fs
         )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # 3. Model
    print(f"Initializing Model (DAC: {dac_model_type})...")
    model = NeuroCodec(
        dac_model_type=dac_model_type,
        eeg_in_channels=args.eeg_channels,
        hidden_dim=args.hidden_dim, # e.g. 256
        num_layers=args.num_layers,  # e.g. 4
        backbone=args.backbone,
        activation=args.activation,
        normalize_latents=args.normalize_latents
    ).to(device)
    
    if not args.debug:
        wandb.watch(model, log="all", log_freq=100)
    
    # 3.1 Load Checkpoint if Exists (Resume Training)
    latest_checkpoint = os.path.join(args.checkpoint_dir, "latest_model.pth")
    if os.path.exists(latest_checkpoint):
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        try:
             # Use weights_only=False due to warnings but consider safer alternative if needed
             checkpoint = torch.load(latest_checkpoint, map_location=device)
             model.load_state_dict(checkpoint)
             print("Checkpoint loaded successfully.")
        except Exception as e:
             print(f"Failed to load checkpoint: {e}. Starting from scratch.")
    else:
        print("No existing checkpoint found. Starting from scratch.")
    
    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-2) 
    
    # Updated: Transformer Ablation (No Envelope Loss per user request)
    # lambda_env=0.0 removes PCC loss
    criterion = NeuroCodecLoss(lambda_recon=args.lambda_recon, lambda_env=args.lambda_env).to(device)
    
    # Initialize Mel Loss
    if args.lambda_mel > 0:
        print(f"Initializing Mel Reconstruction Loss (lambda={args.lambda_mel})...")
        criterion_mel = MelSpectrogramLoss(
            n_mels=[32, 64, 128, 256], 
            window_lengths=[32, 64, 128, 256, 512, 1024, 2048], 
            loss_fn=torch.nn.L1Loss(),
            clamp_eps=1e-5,
            mag_weight=0.0, 
            log_weight=1.0,
            pow=1.0, 
            weight=1.0
        ).to(device)
    else:
        criterion_mel = None
        
    # Discriminator Setup
    if args.lambda_adv > 0:
        print("Initializing Discriminator (MPD + MSD + MRD)...")
        discriminator = Discriminator(sample_rate=target_fs).to(device)
        criterion_gan = GANLoss(discriminator).to(device)
        opt_disc = optim.AdamW(discriminator.parameters(), lr=args.lr, weight_decay=5e-2)
    else:
        discriminator = None
        criterion_gan = None
        opt_disc = None
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 5. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [LR: {current_lr:.1e}]")
        
        for batch_idx, (noisy, eeg, clean) in enumerate(pbar):
            noisy = noisy.to(device)
            clean = clean.to(device)
            eeg = eeg.to(device)
            
            # A. Get Target Latents (Ground Truth Z)
            with torch.no_grad():
                z_target, _, _, _, _ = model.dac.encode(clean)
                
            # B. Forward Pass
            # Updated to unpack env_pred
            z_pred, _, _, _, _, env_pred = model(noisy, eeg)
            
            # C. Compute Loss
            # Updated to pass env_pred and clean audio
            loss, loss_dict = criterion(z_pred, z_target, env_pred, clean)
            
            # Mel Spectrogram Loss (Optional)
            if args.lambda_mel > 0 or args.lambda_adv > 0:
                # Decode predicted latents to audio
                # z_pred: (B, 1024, T) -> (B, 1024, T)
                # Warning: DAC decode consumes memory.
                pred_audio = model.dac.decode(z_pred)
                
                if args.lambda_mel > 0:
                    # Calculate Mel Loss against Ground Truth Clean Audio
                    loss_mel = criterion_mel(pred_audio, clean, sample_rate=target_fs)
                    loss_dict['mel'] = loss_mel.item()
                    loss += args.lambda_mel * loss_mel

            # Adversarial Training (GAN)
            if args.lambda_adv > 0:
                # 1. Train Discriminator
                # Detach generator output to stop backprop to generator
                opt_disc.zero_grad()
                loss_d = criterion_gan.discriminator_loss(pred_audio.detach(), clean)
                loss_d.backward()
                opt_disc.step()
                loss_dict['d_loss'] = loss_d.item()
                
                # 2. Train Generator (Adversarial Loss + Feature Matching)
                loss_g, loss_feat = criterion_gan.generator_loss(pred_audio, clean)
                loss_dict['g_loss'] = loss_g.item()
                loss_dict['g_feat'] = loss_feat.item()
                
                loss += args.lambda_adv * loss_g + args.lambda_feat * loss_feat

            # D. Backend
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping (Prevent Explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            postfix_dict = {
                'loss': f"{loss.item():.4f}", 
                'mse': f"{loss_dict['loss_recon']:.4f}",
                'env': f"{loss_dict['loss_env']:.4f}",
                'pcc': f"{loss_dict['pcc']:.4f}",
                'mel': f"{loss_dict.get('mel', 0.0):.4f}",
                'adv': f"{loss_dict.get('g_loss', 0.0):.4f}" 
            }
            pbar.set_postfix(postfix_dict)
            
            if not args.debug and batch_idx % 100 == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "train_loss_recon": loss_dict['loss_recon'],
                    "train_loss_env": loss_dict['loss_env'],
                    "train_pcc": loss_dict['pcc'],
                    "lr": current_lr
                })
            
            # Optional: Overfit check (break early)
            if args.debug and batch_idx > 5:
                break
                
        # Validation
        val_loss, val_sisdr = validate(model, val_loader, criterion, device, args)
        
        # Step Scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val SI-SDR: {val_sisdr:.2f} dB")
        
        if not args.debug:
            wandb.log({
                "val_loss": val_loss,
                "val_sisdr": val_sisdr,
                "epoch": epoch + 1
            })
        
        # Save Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pth"))
            print("Saved Best Model.")
            
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "latest_model.pth"))

def validate(model, loader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    sisdr_scores = []
    
    with torch.no_grad():
        for batch_idx, (noisy, eeg, clean) in enumerate(loader):
            # ... (Existing Loading & Forward) ...
            noisy = noisy.to(device)
            clean = clean.to(device)
            eeg = eeg.to(device)
            
            if args.noise_cue:
                # Replace EEG with Gaussian Noise matching statistics
                eeg_mean = eeg.mean()
                eeg_std = eeg.std()
                eeg = torch.randn_like(eeg) * eeg_std + eeg_mean
            
            # 1. Encode Target
            z_target, _, _, _, _ = model.dac.encode(clean)
            
            # Normalize target if model uses normalization
            if args.normalize_latents:
                z_target = F.normalize(z_target, p=2, dim=1)
            
            # 2. Model Forward (Z Pred)
            z_pred, _, _, _, _, env_pred = model(noisy, eeg)
            
            # 3. Loss
            loss, _ = criterion(z_pred, z_target, env_pred, clean)
            total_loss += loss.item()
            
            # 4. Neural Decoding & SI-SDR
            # z_pred is (B, 1024, T)
            # We need to quantize it and then decode?
            # Or just decode directly if DAC supports unquantized Z decoding?
            # Usually dac.decode(z) works on quantized Z. 
            # Ideally we run it through quantizer to get discrete codes then decode.
            # z_q, codes, _ = model.dac.quantizer(z_pred, n_quantizers=9) # Check API
            
            # DAC's Quantizer.from_latents(z) returns 5 values
            z_q = model.dac.quantizer(z_pred, n_quantizers=9)[0]
            # return self.quantize(z) which returns z_q, codes, latents
            # But we might need exactly 9 layers.
            
            # Let's try direct decode first, assuming z_pred is close enough.
            # But real inference handles quantization.
            # model.dac.decode(z_q)
            pred_audio = model.dac.decode(z_q)
            
            # SI-SDR Calculation
            min_len = min(pred_audio.shape[-1], clean.shape[-1])
            pred_audio = pred_audio[..., :min_len]
            clean_ref = clean[..., :min_len]
            
            pred_np = pred_audio.cpu().numpy().squeeze(1)
            clean_np = clean_ref.cpu().numpy().squeeze(1)
            
            # Align Signals before SI-SDR
            # This handles the varying lags.
            aligned_refs, aligned_ests = align_signals(clean_np, pred_np)
            
            # Compute SI-SDR per sample
            batch_scores = []
            for r, e in zip(aligned_refs, aligned_ests):
                # Expand dims for sisdr function expectation [..., T]
                score = sisdr(r[np.newaxis, :], e[np.newaxis, :])
                batch_scores.append(score.item())
                
            sisdr_scores.extend(batch_scores)

            if args.debug and batch_idx > 2:
                break
                
    mean_loss = total_loss / len(loader)
    mean_sisdr = np.mean(sisdr_scores) if len(sisdr_scores) > 0 else 0.0
    
    return mean_loss, mean_sisdr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help="Path to config file (optional)")
    
    # Default arguments (can be overridden by config file)
    parser.add_argument('--root', type=str, default='/home/jaliya/eeg_speech/navindu/data/Cocktail_Party/Normalized-1')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/neurocodec/KUL/mamba')
    parser.add_argument('--debug', action='store_true', help="Run fast debug mode")
    parser.add_argument('--dataset', type=str, default='cocktail', choices=['cocktail', 'kul'], help='Dataset to use')
    parser.add_argument('--eeg_channels', type=int, default=128, help='Number of EEG channels (128 for Cocktail, 64 for KUL)')
    
    parser.add_argument('--evaluate', action='store_true', help="Run validation only")
    parser.add_argument('--noise_cue', action='store_true', help="Use random noise instead of EEG during validation")
    
    parser.add_argument('--backbone', type=str, default='mamba', choices=['mamba', 'transformer'], help='Backbone architecture')
    parser.add_argument('--activation', type=str, default='prelu', choices=['prelu', 'snake'], help="Activation function")
    parser.add_argument('--normalize_latents', type=bool, default=False, help="L2-Normalize Latents")
    
    # Loss Weights
    parser.add_argument('--lambda_recon', type=float, default=1.0, help="Weight for Reconstruction Loss")
    parser.add_argument('--lambda_env', type=float, default=0.0, help="Weight for PCC Loss")
    parser.add_argument('--lambda_mel', type=float, default=0.0, help="Weight for Mel Reconstruction Loss")
    parser.add_argument('--lambda_adv', type=float, default=0.0, help="Weight for Adversarial Loss")
    parser.add_argument('--lambda_feat', type=float, default=0.0, help="Weight for Feature Matching Loss")
    
    # Parse initial args to check for config file
    temp_args, _ = parser.parse_known_args()
    
    if temp_args.config:
        import yaml
        with open(temp_args.config, 'r') as f:
            config = yaml.safe_load(f)
            parser.set_defaults(**config)
            print(f"Loaded config from {temp_args.config}")
            
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(42)
    
    if args.evaluate:
        # Evaluate Only Mode
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"Evaluating on {device}...")
        print(f"Backbone: {args.backbone.upper()}")
        print(f"Dataset: {args.dataset.upper()}")
        
        # Dataset Configuration (Mirrors train function)
        if args.dataset == 'kul':
            if args.eeg_channels == 128:
                 print("Info: KUL dataset selected, defaulting EEG channels to 64 (overriding 128).")
                 args.eeg_channels = 64
            
            # Audio Configuration for KUL
            dac_model_type = '16khz'
            target_fs = 16000
        else:
            dac_model_type = '44khz'
            target_fs = 44100
        
        # Load Data
        if args.dataset == 'cocktail':
             val_loader = load_NeuroCodecDataset(root=args.root, subset='val', batch_size=args.batch_size, num_gpus=1)
        elif args.dataset == 'kul':
             val_loader = load_KUL_NeuroCodecDataset(lmdb_path=args.root, subset='val', batch_size=args.batch_size, num_gpus=1, target_fs=target_fs)

        # Load Model
        print(f"Initializing Model (DAC: {dac_model_type})...")
        model = NeuroCodec(
            dac_model_type=dac_model_type,
            eeg_in_channels=args.eeg_channels,
            hidden_dim=args.hidden_dim, 
            num_layers=args.num_layers,
            backbone=args.backbone,
            activation=args.activation,
            normalize_latents=args.normalize_latents
        ).to(device)
        
        checkpoint_path = args.checkpoint_dir if args.checkpoint_dir.endswith('.pth') else os.path.join(args.checkpoint_dir, "best_model.pth")
        
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint {checkpoint_path}...")
            # Use weights_only=False to avoid future warnings if safe, or handle pickle security
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"No checkpoint found at {checkpoint_path}! Running with random weights.")
            
        # Use NeuroCodecLoss for compatibility with validate() function signature
        criterion = NeuroCodecLoss(lambda_recon=args.lambda_recon, lambda_env=args.lambda_env).to(device)
        
        val_loss, val_sisdr = validate(model, val_loader, criterion, device, args)
        print(f"Validation Result | Loss: {val_loss:.4f} | SI-SDR: {val_sisdr:.2f} dB")
        
    else:
        train(args)
