
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np

# Local imports
from models.neurocodec import NeuroCodec
from dataset_neurocodec import load_NeuroCodecDataset, load_KUL_NeuroCodecDataset
from losses_neurocodec import NeuroCodecLoss

def sisdr(reference, estimation):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) with proper
    mean removal and length alignment.

    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]
    Returns:
        SI-SDR per sample (same leading dims as inputs)
    """
    reference = np.asarray(reference)
    estimation = np.asarray(estimation)

    # Time align (safeguard if caller didn't already crop)
    min_len = min(reference.shape[-1], estimation.shape[-1])
    reference = reference[..., :min_len]
    estimation = estimation[..., :min_len]

    # Remove DC offset (true scale-invariant version)
    reference = reference - np.mean(reference, axis=-1, keepdims=True)
    estimation = estimation - np.mean(estimation, axis=-1, keepdims=True)

    ref_energy = np.sum(reference ** 2, axis=-1, keepdims=True) + 1e-8

    # Optimal scaling factor
    alpha = np.sum(reference * estimation, axis=-1, keepdims=True) / ref_energy

    e_true = alpha * reference
    e_noise = estimation - e_true

    si_sdr_val = 10 * np.log10(
        np.sum(e_true ** 2, axis=-1) / (np.sum(e_noise ** 2, axis=-1) + 1e-8)
    )

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
        args.eeg_channels = 64 # Force 64 for KUL unless specified otherwise? No, respect arg but default is 128.
        # Check if user overrode default 128
        # Argparse doesn't tell us if it was default or user-specified easily without a separate flag.
        # But we can just warn.
        if args.eeg_channels == 128:
             print("Info: KUL dataset selected, defaulting EEG channels to 64 (overriding 128).")
             args.eeg_channels = 64

        # Validate LMDB path; fall back to local default if obvious mismatch is given
        if not os.path.exists(args.root):
            fallback_lmdb = os.path.join(os.path.dirname(__file__), 'kul_mixcsv_16k.lmdb')
            if os.path.exists(fallback_lmdb):
                print(f"Warning: KUL LMDB not found at '{args.root}'. Falling back to '{fallback_lmdb}'.")
                args.root = fallback_lmdb
            else:
                raise FileNotFoundError(
                    f"KUL LMDB path '{args.root}' does not exist. "
                    "Point --root to your KUL LMDB file (e.g., kul_mixcsv_16k.lmdb)."
                )
        
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
            num_gpus=1,
            fraction=args.fraction
        )
         val_loader = load_NeuroCodecDataset(
            root=args.root, 
            subset='val', 
            batch_size=args.batch_size, 
            num_gpus=1,
            fraction=args.fraction
        )
    elif args.dataset == 'kul':
         train_loader = load_KUL_NeuroCodecDataset(
            lmdb_path=args.root, # Root should be LMDB path for KUL
            subset='train',
            batch_size=args.batch_size,
            num_gpus=1,
            target_fs=target_fs,
            fraction=args.fraction
         )
         val_loader = load_KUL_NeuroCodecDataset(
            lmdb_path=args.root,
            subset='val',
            batch_size=args.batch_size,
            num_gpus=1,
            target_fs=target_fs,
            fraction=args.fraction
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
        backbone=args.backbone
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
    criterion = NeuroCodecLoss(lambda_recon=1.0, lambda_env=0.0).to(device)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 5. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [LR: {current_lr:.1e}]")
        
        for batch_idx, batch in enumerate(pbar):
            # Support loaders that return extra length/index tensors (KUL collate_fn returns 5-6 items)
            noisy, eeg, clean = batch[0], batch[1], batch[2]
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
            
            # D. Backend
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping (Prevent Explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'mse': f"{loss_dict['loss_recon']:.4f}",
                'env': f"{loss_dict['loss_env']:.4f}",
                'pcc': f"{loss_dict['pcc']:.4f}"
            })
            
            if not args.debug:
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
        for batch_idx, batch in enumerate(loader):
            noisy, eeg, clean = batch[0], batch[1], batch[2]
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
            
            scores = sisdr(clean_np, pred_np)
            sisdr_scores.extend(scores)

            if args.debug and batch_idx > 2:
                break
                
    mean_loss = total_loss / len(loader)
    mean_sisdr = np.mean(sisdr_scores) if len(sisdr_scores) > 0 else 0.0
    
    return mean_loss, mean_sisdr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/jaliya/eeg_speech/navindu/data/Cocktail_Party/Normalized-1')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/neurocodec/KUL/mamba')
    parser.add_argument('--debug', action='store_true', help="Run fast debug mode")
    parser.add_argument('--dataset', type=str, default='kul', choices=['cocktail', 'kul'], help='Dataset to use')
    parser.add_argument('--eeg_channels', type=int, default=128, help='Number of EEG channels (128 for Cocktail, 64 for KUL)')
    parser.add_argument('--fraction', type=float, default=1.0, help='Use a fraction of the dataset for quick experiments (0,1]')
    
    parser.add_argument('--evaluate', action='store_true', help="Run validation only")
    parser.add_argument('--noise_cue', action='store_true', help="Use random noise instead of EEG during validation")
    
    parser.add_argument('--backbone', type=str, default='mamba', choices=['mamba', 'transformer'], help='Backbone architecture')
    
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
             val_loader = load_NeuroCodecDataset(root=args.root, subset='val', batch_size=args.batch_size, num_gpus=1, fraction=args.fraction)
        elif args.dataset == 'kul':
             val_loader = load_KUL_NeuroCodecDataset(lmdb_path=args.root, subset='val', batch_size=args.batch_size, num_gpus=1, fraction=args.fraction)
             # args.eeg_channels should be set by user or we trust default?
             # If user didn't set, default is 128 (wrong for KUL).
             # We should probably force it here if it's default?
             if args.eeg_channels == 128 and args.dataset == 'kul':
                 print("Warning: Dataset is KUL but eeg_channels is 128. Assuming user wants 64 (Autofix).")
                 args.eeg_channels = 64
        
        # Match DAC to dataset (KUL=16k, Cocktail=44.1k)
        dac_model_type = '16khz' if args.dataset == 'kul' else '44khz'

        # Load Model
        model = NeuroCodec(
            dac_model_type=dac_model_type,
            eeg_in_channels=args.eeg_channels,
            hidden_dim=args.hidden_dim, 
            num_layers=args.num_layers,
            backbone=args.backbone
        ).to(device)
        
        checkpoint_path = args.checkpoint_dir if args.checkpoint_dir.endswith('.pth') else os.path.join(args.checkpoint_dir, "best_model.pth")
        
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint {checkpoint_path}...")
            # Use weights_only=False to avoid future warnings if safe, or handle pickle security
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"No checkpoint found at {checkpoint_path}! Running with random weights.")
            
        # Use NeuroCodecLoss for compatibility with validate() function signature
        criterion = NeuroCodecLoss(lambda_recon=1.0, lambda_env=0.0).to(device)
        
        val_loss, val_sisdr = validate(model, val_loader, criterion, device, args)
        print(f"Validation Result | Loss: {val_loss:.4f} | SI-SDR: {val_sisdr:.2f} dB")
        
    else:
        train(args)
