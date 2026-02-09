
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.neurocodec import NeuroCodec
from dataset_neurocodec import load_NeuroCodecDataset

def visualize(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Visualizing on {device}...")
    
    # 1. Load Model
    model = NeuroCodec(dac_model_type='44khz').to(device)
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        try:
            model.load_state_dict(checkpoint, strict=False)
            print("Loaded checkpoint (strict=False).")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            print("Using random initialization.")
        
    model.eval()
    
    # 2. Load Data
    loader = load_NeuroCodecDataset(root=args.root, subset='test', batch_size=1)
    noisy, eeg, clean = next(iter(loader))
    noisy = noisy.to(device)
    eeg = eeg.to(device)
    
    # 3. Forward Pass
    print("Running Forward Pass...")
    with torch.no_grad():
        # returns: z_pred, codes_mix, z_mix, eeg_feat, attn_weights
        z_pred, codes, z, eeg_feat, attn = model(noisy, eeg)
        
    print(f"\nSHAPES CHECK:")
    print(f"  Input EEG:      {eeg.shape}  (Batch, 128, Time)")
    print(f"  Encoded EEG:    {eeg_feat.shape} (Batch, 64,  Time_Latent)")
    print(f"  Attn Weights:   {attn.shape} (Batch, Time_Audio, Time_EEG)")
    print(f"  Z Prediction:   {z_pred.shape} (Batch, 1024, Time_Audio)")
        
    # 4. Process for Plotting
    # EEG Features: (B, 64, T_eeg) -> (64, T_eeg)
    eeg_map = eeg_feat[0].cpu().numpy()
    
    # Attention: (B, T_audio, T_eeg)
    attn_map = attn[0].cpu().numpy()
    
    # Z Prediction (B, 1024, T) -> (1024, T)
    z_pred_map = z_pred[0].cpu().numpy()
    
    # Z Ground Truth
    # Need to get Z Target again from clean
    with torch.no_grad():
        z_target, _, _, _, _ = model.dac.encode(clean.to(device))
    z_gt_map = z_target[0].cpu().numpy()
    
    # 5. Plotting
    output_dir = "results/NeuroCodec_cos5_clip/Internals"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 16))
    
    limit = min(attn_map.shape[0], 200) # First 200 audio frames
    
    # 0. Raw EEG
    plt.subplot(4, 1, 1)
    # Raw EEG (B, 128, T) -> (128, T)
    raw_eeg_map = eeg[0].cpu().numpy()
    sns.heatmap(raw_eeg_map, cmap='viridis', cbar=True)
    plt.title("Raw EEG Input (First channel check)")
    plt.ylabel("Channel (128)")
    plt.xlabel("Time")

    # A. EEG Features
    plt.subplot(4, 1, 2)
    sns.heatmap(eeg_map, cmap='viridis', cbar=True)
    plt.title("EEG Features (Input to Cross-Attn Key/Value)")
    plt.ylabel("Channel (64)")
    plt.xlabel("Time (Downsampled)")
    
    # B. Attention Map
    plt.subplot(4, 1, 3)
    sns.heatmap(attn_map[:limit, :], cmap='magma', cbar=True)
    plt.title(f"Cross-Attention Weights (First {limit} Audio Frames)")
    plt.ylabel("Audio Time (Query)")
    plt.xlabel("EEG Time (Key)")
    
    # C. Z-Prediction
    # D. Histograms of Values
    plt.subplot(4, 2, 7)
    sns.histplot(z_gt_map.flatten(), color='blue', label='GT', stat='density', alpha=0.5, bins=50)
    sns.histplot(z_pred_map.flatten(), color='red', label='Pred', stat='density', alpha=0.5, bins=50)
    plt.title("Z-Value Distribution (Checking for Collapse)")
    plt.legend()
    
    # E. Cosine Similarity per Frame
    from sklearn.metrics.pairwise import cosine_similarity
    # (1024, T) -> (T, 1024)
    sim = torch.nn.functional.cosine_similarity(torch.from_numpy(z_gt_map), torch.from_numpy(z_pred_map), dim=0)
    
    plt.subplot(4, 2, 8)
    plt.plot(sim.numpy())
    plt.title(f"Frame-wise Cosine Similarity (Mean: {sim.mean():.4f})")
    plt.ylim(-1, 1)
    plt.grid(True, alpha=0.3)
    
    print("\nDEBUG STATS:")
    print(f"Raw EEG   - Global Std: {raw_eeg_map.std():.4f}, Temporal Std: {np.mean(raw_eeg_map.std(axis=-1)):.4f}")
    print(f"EEG Feat  - Global Std: {eeg_map.std():.4f}, Temporal Std: {np.mean(eeg_map.std(axis=-1)):.4f}")
    print(f"Z GT      - Mean: {z_gt_map.mean():.4f}, Std: {z_gt_map.std():.4f}")
    print(f"Z Pred    - Mean: {z_pred_map.mean():.4f}, Std: {z_pred_map.std():.4f}")
    print(f"Cosine Sim- Mean: {sim.mean():.4f}")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/internals_plot.png")
    print(f"Saved plot to {output_dir}/internals_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/jaliya/eeg_speech/navindu/data/Cocktail_Party/Normalized-2/2s/eeg/new')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/neurocodec_cos5/latest_model.pth')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    visualize(args)
