
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
from dataset_neurocodec import load_NeuroCodecDataset

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

    # 2. Dataset
    print("Loading Dataset...")
    train_loader = load_NeuroCodecDataset(
        root=args.root, 
        subset='train', 
        batch_size=args.batch_size,
        num_gpus=1 # Simple single GPU for now
    )
    val_loader = load_NeuroCodecDataset(
        root=args.root, 
        subset='val', 
        batch_size=args.batch_size, # validation batch size can be same
        num_gpus=1
    )

    # 3. Model
    print("Initializing Model...")
    model = NeuroCodec(
        dac_model_type='44khz',
        eeg_in_channels=128,
        hidden_dim=args.hidden_dim, # e.g. 256
        num_layers=args.num_layers  # e.g. 4
    ).to(device)
    
    # 4. Optimizer
    # 4. Optimizer
    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.MSELoss()
    
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
            z_pred, _, _, _, _ = model(noisy, eeg)
            
            # C. Compute Loss (MSE)
            loss = criterion(z_pred, z_target)
            
            # D. Backend
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Optional: Overfit check (break early)
            if args.debug and batch_idx > 5:
                break
                
        # Validation
        val_loss, val_sisdr = validate(model, val_loader, criterion, device, args)
        
        # Step Scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val SI-SDR: {val_sisdr:.2f} dB")
        
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
            
            # 2. Model Forward (Z Pred)
            z_pred, _, _, _, _ = model(noisy, eeg)
            
            # 3. Loss (MSE)
            loss = criterion(z_pred, z_target)
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
    parser.add_argument('--root', type=str, default='/home/jaliya/eeg_speech/navindu/data/Cocktail_Party/Normalized/2s/eeg/new')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/neurocodec')
    parser.add_argument('--debug', action='store_true', help="Run fast debug mode")
    parser.add_argument('--evaluate', action='store_true', help="Run validation only")
    parser.add_argument('--noise_cue', action='store_true', help="Use random noise instead of EEG during validation")
    
    args = parser.parse_args()
    
    if args.evaluate:
        # Evaluate Only Mode
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"Evaluating on {device}...")
        
        # Load Data
        val_loader = load_NeuroCodecDataset(root=args.root, subset='val', batch_size=args.batch_size, num_gpus=1)
        
        # Load Model
        model = NeuroCodec(dac_model_type='44khz', hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
        
        checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint {checkpoint_path}...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print("No checkpoint found! Running with random weights (Sanity Check).")
            
        criterion = nn.MSELoss()
        
        val_loss, val_sisdr = validate(model, val_loader, criterion, device, args)
        print(f"Validation Result | Loss: {val_loss:.4f} | SI-SDR: {val_sisdr:.2f} dB")
        
    else:
        train(args)
