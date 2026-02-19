
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np
import dac.model # Added this import

# Local imports
from models.neurocodec import NeuroCodec
from dataset_neurocodec import load_NeuroCodecDataset, load_KUL_NeuroCodecDataset
# Consolidated and updated imports from 'losses' and 'losses_neurocodec'
from losses import MelSpectrogramLoss, GANLoss # Added GANLoss
from losses_neurocodec import NeuroCodecLoss # Kept this as it was in the original code

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
            target_fs=target_fs,
            original_fs=16000 # Correct FS
         )
         val_loader = load_KUL_NeuroCodecDataset(
            lmdb_path=args.root,
            subset='val',
            batch_size=args.batch_size,
            num_gpus=1,
            target_fs=target_fs,
            original_fs=16000 # Correct FS
         )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    model = NeuroCodec(
        dac_model_type=dac_model_type,
        eeg_in_channels=args.eeg_channels,
        hidden_dim=args.hidden_dim, # e.g. 256
        num_layers=args.num_layers,  # e.g. 4
        backbone=args.backbone,
        activation=args.activation
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
    mel_loss_fn = MelSpectrogramLoss(
        sample_rate=16000,
        window_lengths=[1024, 512, 256, 128],
        n_mels=[80, 80, 32, 16],
        f_max=4000
    ).to(device)
    
    # GAN Setup
    discriminator = dac.model.Discriminator(sample_rate=16000).to(device)
    gan_loss_fn = GANLoss(discriminator).to(device)
    
    # Optimizers
    # Generator Optimizer (NeuroCodec)
    optimizer_g = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.9))
    scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, mode='min', factor=0.5, patience=5
    )
    
    # Discriminator Optimizer
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.9))
    scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_d, mode='min', factor=0.5, patience=5
    )
    
    # 5. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        discriminator.train()
        total_loss = 0
        total_g_loss = 0
        total_d_loss = 0
        
        # Determine when to start GAN training
        use_gan = epoch >= args.gan_start_epoch
        
        current_lr_g = optimizer_g.param_groups[0]['lr']
        current_lr_d = optimizer_d.param_groups[0]['lr']
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [LR_G: {current_lr_g:.1e}, LR_D: {current_lr_d:.1e}]")
        
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
            z_pred, _, _, _, _, env_pred = model(noisy, eeg)
            
            # Decode for GAN / Mel Loss (Using Gumbel Softmax or Straight Through in Quantizer inside DAC?)
            # DAC encode returns z, codes, latents, etc. 
            # We need differentiable audio for GAN.
            # model.dac.decode(z_pred) should be differentiable if z_pred is differentiable.
            # But z_pred usually goes through quantizer.
            # Let's assume model(noisy, eeg) returns 'z_pred' which is the predicted latent.
            # We need to pass this through DAC's quantizer (straight-through) and decoder.
            
            # DAC Quantizer
            # The quantizer returns (z_q, codes, latents, bandwidth, commit_loss)
            z_q, _, _, _, _ = model.dac.quantizer(z_pred, n_quantizers=9) # Use 9 quantizers as in validate
            
            # Decode
            pred_audio = model.dac.decode(z_q)
            
            # Align lengths (pred_audio might be slightly longer/shorter due to padding/striding)
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
            # Update Generator only if not in warmup
            train_generator = epoch >= args.disc_warmup_epochs
            
            optimizer_g.zero_grad()
            
            if train_generator:
                # 1. Reconstruction Losses (MSE, Env, PCC)
                recon_loss, loss_dict = criterion(z_pred, z_target, env_pred, clean)
                
                # 2. Mel Spectrogram Loss
                if epoch >= args.mel_start_epoch:
                    mel_loss = mel_loss_fn(pred_audio, clean_aligned)
                    recon_loss += args.lambda_mel * mel_loss
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
                # Gradient Clipping (Prevent Explosion) for Generator
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_g.step()
                
                total_loss += recon_loss.item()
                log_recon_loss = recon_loss.item()
            else:
                # Generator Frozen: Calculate loss for logging but don't backward
                with torch.no_grad():
                     recon_loss, loss_dict = criterion(z_pred, z_target, env_pred, clean)
                     log_recon_loss = recon_loss.item()
                     loss_dict['loss_mel'] = 0.0
                     if use_gan: # Log what G loss would be
                        g_loss_adv, g_loss_feat = gan_loss_fn.generator_loss(pred_audio, clean_aligned)
                        loss_dict['loss_adv'] = g_loss_adv.item()
                        loss_dict['loss_feat'] = g_loss_feat.item()
            
            # Logging
            postfix = {
                'loss': f"{log_recon_loss:.4f}", 
                'mse': f"{loss_dict['loss_recon']:.4f}",
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
            
            if not args.debug:
                log_dict = {
                    "train_loss": recon_loss.item(),
                    "train_loss_recon": loss_dict['loss_recon'],
                    "train_loss_env": loss_dict.get('loss_env', 0.0), # Use get
                    "train_pcc": loss_dict.get('pcc', 0.0), # Use get
                    "lr_g": current_lr_g,
                    "lr_d": current_lr_d
                }
                if 'loss_mel' in loss_dict:
                    log_dict['train_loss_mel'] = loss_dict['loss_mel']
                    log_dict['lambda_mel'] = args.lambda_mel
                
                if use_gan:
                    log_dict['train_loss_adv'] = loss_dict['loss_adv']
                    log_dict['train_loss_feat'] = loss_dict['loss_feat']
                    log_dict['train_loss_d'] = d_loss.item()
                    
                wandb.log(log_dict)
            
            # Optional: Overfit check (break early)
            if args.debug and batch_idx > 5:
                break
                
        
        # Validation
        if (epoch + 1) % args.val_interval == 0:
            val_loss, val_sisdr, val_estoi = validate(model, train_loader, criterion, device, args)
            
            # Step Schedulers
            scheduler_g.step(val_loss)
            scheduler_d.step(val_loss)
            
            print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val SI-SDR: {val_sisdr:.2f} dB | Val ESTOI: {val_estoi:.4f}")
            
            if not args.debug:
                wandb.log({
                    "val_loss": val_loss,
                    "val_sisdr": val_sisdr,
                    "val_estoi": val_estoi,
                    "epoch": epoch + 1
                })
            
            # Save Checkpoint (Best)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pth"))
                print("Saved Best Model.")

        else:
            print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Validation Skipped")
            if not args.debug:
                wandb.log({
                    "epoch": epoch + 1
                })
        
        # Save Latest Checkpoint (Every Epoch)
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "latest_model.pth"))

def validate(model, loader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    sisdr_scores = []
    estoi_scores = []
    
    # Import for metrics
    from scipy import signal
    try:
        from pystoi import stoi
    except ImportError:
        stoi = None
    estoi_scores = []
    
    # Import for metrics
    from scipy import signal
    try:
        from pystoi import stoi
    except ImportError:
        stoi = None
    
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
            
            if args.val_batches > 0 and batch_idx >= args.val_batches:
                break
            
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
                if stoi is not None:
                    # KUL: 16000, Cocktail: 44100
                    if args.dataset == 'kul': fs = 16000
                    else: fs = 44100
                    
                    try:
                        e_val = stoi(c, p_aligned, fs, extended=True)
                        batch_estoi.append(e_val)
                    except Exception:
                        pass
            
            sisdr_scores.extend(batch_sisdr)
            estoi_scores.extend(batch_estoi)

            if args.debug and batch_idx > 2:
                break
                
    mean_loss = total_loss / len(loader)
    mean_sisdr = np.mean(sisdr_scores) if len(sisdr_scores) > 0 else 0.0
    mean_estoi = np.mean(estoi_scores) if len(estoi_scores) > 0 else 0.0
    
    return mean_loss, mean_sisdr, mean_estoi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/workspace/Dataset/kul_all_subjects.lmdb')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
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
    parser.add_argument('--val_interval', type=int, default=2, help="Validation interval in epochs (default: 1)")
    parser.add_argument('--lambda_mel', type=float, default=13.0, help="Weight for Mel Spectrogram Loss")
    parser.add_argument('--mel_start_epoch', type=int, default=0, help="Epoch to start applying Mel Loss")
    parser.add_argument('--val_batches', type=int, default=0, help="Limit number of validation batches (0 = full)")
    
    parser.add_argument('--lambda_gan', type=float, default=1.0, help="Weight for GAN Adversarial Loss")
    parser.add_argument('--lambda_feat', type=float, default=2.0, help="Weight for GAN Feature Matching Loss")
    parser.add_argument('--gan_start_epoch', type=int, default=0, help="Epoch to start GAN training")
    parser.add_argument('--disc_warmup_epochs', type=int, default=3, help="Number of epochs to freeze Generator for Discriminator warmup")
    
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
                target_fs=16000, # Hardcoded or use args
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
        
        val_loss, val_sisdr, val_estoi = validate(model, val_loader, criterion, device, args)
        print(f"Validation Result | Loss: {val_loss:.4f} | SI-SDR: {val_sisdr:.2f} dB | ESTOI: {val_estoi:.4f}")
        
    else:
        train(args)
