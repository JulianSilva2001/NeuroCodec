
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.neurocodec import NeuroCodec
from dataset_neurocodec import load_NeuroCodecDataset, load_KUL_NeuroCodecDataset

def visualize(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Visualizing on {device}...")
    
    # Configure for Dataset
    if args.dataset == 'kul':
        dac_model_type = '16khz'
        target_fs = 16000
        eeg_channels = 64
    else:
        dac_model_type = '44khz'
        target_fs = 44100
        eeg_channels = 128
        
    # 1. Load Model
    # Note: If checkpoint was trained on 44k, this will fail if we force 16k.
    # But user wants to visualize KUL data.
    # If using a pre-trained model (trained on Cocktail), it expects 128ch EEG and 44k Audio.
    # If checking KUL *data* only, we don't need a model?
    # Actually, visualize() runs the model.
    # If we just want to see data, we should just plot inputs.
    # But assuming we want to see model INTERNALS (Attn, Z), we need a compatible model.
    # I will assume we are using a model compatible with the dataset (or random weights if just checking data flow).
    
    print(f"Using DAC: {dac_model_type}, EEG Channels: {eeg_channels}")
    model = NeuroCodec(dac_model_type=dac_model_type, eeg_in_channels=eeg_channels, hidden_dim=args.hidden_dim).to(device)
    
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        try:
            model.load_state_dict(checkpoint, strict=False)
            print("Loaded checkpoint (strict=False).")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            print("Using random initialization (Visualizing inputs/untrained outputs).")
        
    model.eval()
    
    # 2. Load Data
    print(f"Loading {args.subset} set ({args.dataset})...")
    if args.dataset == 'kul':
        loader = load_KUL_NeuroCodecDataset(
            lmdb_path=args.root, 
            subset=args.subset, 
            batch_size=1,
            num_gpus=1,
            target_fs=target_fs
        )
    else:
        loader = load_NeuroCodecDataset(root=args.root, subset=args.subset, batch_size=1)
    iterator = iter(loader)
    
    output_dir = f"results/NeuroCodec_Internals/{args.dataset}_{args.subset}"
    os.makedirs(output_dir, exist_ok=True)
    
    for sample_idx in range(args.num_samples):
        print(f"\n--- Processing Sample {sample_idx+1}/{args.num_samples} ---")
        try:
            noisy, eeg, clean = next(iterator)
        except StopIteration:
            print("End of dataset reached.")
            break
            
        noisy = noisy.to(device)
        eeg = eeg.to(device)
        
        if args.noise_eeg:
            print("WARNING: Replacing Real EEG with Random Gaussian Noise!")
            eeg = torch.randn_like(eeg) * 350.0  # Approx std of clipped EEG
            
        clean = clean.to(device)
        
        # 3. Forward Pass
        with torch.no_grad():
            output = model(noisy, eeg)
            if len(output) == 6:
                z_pred, codes, z_mix, eeg_feat, attn, env_pred = output
            else:
                z_pred, codes, z_mix, eeg_feat, attn = output
                env_pred = torch.zeros(eeg_feat.shape[0], eeg_feat.shape[-1]).to(device) # Dummy
            
        print(f"\nSHAPES CHECK:")
        print(f"  Input EEG:      {eeg.shape}  (Batch, 128, Time)")
        print(f"  Encoded EEG:    {eeg_feat.shape} (Batch, 64,  Time_Latent)")
        print(f"  Attn Weights:   {attn.shape} (Batch, Time_Audio, Time_EEG)")
        print(f"  Z Prediction:   {z_pred.shape} (Batch, 1024, Time_Audio)")
        print(f"  Env Prediction: {env_pred.shape} (Batch, Time_Latent)")
            
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
        plt.figure(figsize=(15, 24)) # Increased height for 6 rows
        # Calculate Time Axis
        # Audio is available as 'noisy' and 'clean' tensor (B, 1, T)
        # Convert to numpy
        clean_np = clean[0, 0].cpu().numpy()
        noisy_np = noisy[0, 0].cpu().numpy()
        
        sr = target_fs # Dynamic variable from earlier
        duration = clean_np.shape[-1] / sr
        time_audio = np.linspace(0, duration, clean_np.shape[-1])
        
        limit = min(attn_map.shape[0], 200) # First 200 audio frames
        
        # 0. Raw EEG (Subplot 1)
        plt.subplot(6, 1, 1)
        raw_eeg_map = eeg[0].cpu().numpy()
        # Use extent to map x-axis to seconds: [0, duration, 0, eeg_channels]
        # Note: imshow/heatmap origin is usually top-left or bottom-left. 
        # sns.heatmap doesn't support extent easily, use imshow for precision alignment.
        plt.imshow(raw_eeg_map, aspect='auto', origin='lower', extent=[0, duration, 0, eeg_channels], cmap='viridis')
        plt.title(f"Raw EEG Input")
        plt.ylabel("Channel")
        plt.xlabel("Time (s)")
        plt.xlim(0, duration)

        # 1. Input Audio (Subplot 2 Left)
        plt.subplot(6, 2, 3)
        plt.plot(time_audio, clean_np, label='Clean', alpha=0.9, linewidth=0.5)
        plt.plot(time_audio, noisy_np, label='Noisy', alpha=0.5, linewidth=0.5)
        plt.title("Input Audio (Clean vs Noisy)")
        plt.legend(loc='upper right')
        plt.xlim(0, duration)
        plt.xlabel("Time (s)")

        # 2. Output Audio (Subplot 2 Right - Aligned)
        plt.subplot(6, 2, 4)
        
        # Decode Prediction to Audio
        with torch.no_grad():
            z_q = model.dac.quantizer(z_pred, n_quantizers=9)[0]
            pred_audio = model.dac.decode(z_q)
        pred_np = pred_audio[0, 0].cpu().numpy()
        
        # Align Prediction (Correlation)
        import scipy.signal
        clean_centered = clean_np - np.mean(clean_np)
        pred_centered = pred_np - np.mean(pred_np)
        
        corr = scipy.signal.correlate(clean_centered, pred_centered, mode='full', method='fft')
        lags = scipy.signal.correlation_lags(len(clean_centered), len(pred_centered), mode='full')
        lag = lags[np.argmax(corr)]
        
        # Shift Pred to align with Clean
        shift = -lag
        if shift > 0:
             pred_aligned = pred_np[shift:]
             clean_aligned = clean_np[:len(pred_aligned)]
             time_aligned = time_audio[:len(pred_aligned)]
        else:
             pred_aligned = pred_np[:len(pred_np)+shift]
             clean_aligned = clean_np[-shift: -shift+len(pred_aligned)]
             time_aligned = time_audio[:len(pred_aligned)]

        # Secondary trim if lengths differ slightly after shift
        min_len = min(len(clean_aligned), len(pred_aligned))
        clean_aligned = clean_aligned[:min_len]
        pred_aligned = pred_aligned[:min_len]
        time_aligned = time_aligned[:min_len]
        
        plt.plot(time_aligned, clean_aligned, label='Target', alpha=0.9, linewidth=0.5, color='blue')
        plt.plot(time_aligned, pred_aligned, label=f'Pred (Lag:{lag})', alpha=0.7, linewidth=0.5, linestyle='--', color='red')
        plt.title("Output Audio (Target vs Pred Aligned)")
        plt.legend(loc='upper right')
        plt.xlim(0, time_aligned[-1] if len(time_aligned)>0 else duration)
        plt.xlabel("Time (s)")
        
        # 2. Spectrogram (Subplot 3)
        plt.subplot(6, 1, 3)
        # specgram plots time on x if Fs is given
        Pxx, freqs, bins, im = plt.specgram(clean_np, NFFT=1024, Fs=sr, noverlap=512, cmap='inferno')
        plt.title("Clean Audio Spectrogram")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.xlim(0, duration)
        
        # 3. Attention Map (Subplot 4)
        plt.subplot(6, 1, 4)
        # Attn: Audio Time vs EEG Time
        # Extent: [0, duration (EEG), 0, duration (Audio)]
        plt.imshow(attn_map[:limit, :], aspect='auto', origin='lower', 
                   extent=[0, duration, 0, duration * (limit/attn_map.shape[0])], 
                   cmap='magma')
        plt.title(f"Cross-Attention Weights")
        plt.ylabel("Audio Time (s)")
        plt.xlabel("EEG Time (s)")
        # Note: xlim might need adjustment if EEG and Audio lengths differ slightly in mapping
        
        # 4. EEG Features (Subplot 5 - Split row)
        plt.subplot(6, 2, 9) # Moved way down to fit 4 rows of full-width plots?
        # Wait, if 1, 2, 3, 4 are full width (6, 1, x), they take up 4 rows.
        # Remaining rows: 5 and 6.
        # Subplot indices for 6x2 grid:
        # Row 1: 1, 2
        # Row 2: 3, 4
        # Row 3: 5, 6
        # Row 4: 7, 8
        # Row 5: 9, 10
        # Row 6: 11, 12
        
        # If I use (6, 1, 1), it spans (1,2).
        # (6, 1, 2) spans (3,4).
        # (6, 1, 3) spans (5,6).
        # (6, 1, 4) spans (7,8).
        
        # So:
        # P1 (Raw EEG): Row 1
        # P2 (Waveform): Row 2
        # P3 (Spectrogram): Row 3
        # P4 (Attention): Row 4
        
        # Remaining: Row 5 (9,10) and Row 6 (11,12).
        
        # P5 (EEG Feat) -> Subplot 9 (Row 5 Left)
        plt.subplot(6, 2, 9)
        sns.heatmap(eeg_map, cmap='viridis', cbar=True)
        plt.title("EEG Features")
        
        # P6 (Cosine Sim) -> Subplot 10 (Row 5 Right)
        # P6 (Cosine Sim) -> Subplot 10 (Row 5 Right)
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Align Z-features for Cosine Sim
        # z_gt_map: (1024, T)
        # z_pred_map: (1024, T)
        
        # We can re-use the lag found for audio, OR re-calculate for Z (safer)
        # Cross-corr on mean activity
        z_gt_mean = z_gt_map.mean(axis=0)
        z_pred_mean = z_pred_map.mean(axis=0)
        
        corr_z = scipy.signal.correlate(z_gt_mean - z_gt_mean.mean(), z_pred_mean - z_pred_mean.mean(), mode='full')
        lags_z = scipy.signal.correlation_lags(len(z_gt_mean), len(z_pred_mean), mode='full')
        lag_z = lags_z[np.argmax(corr_z)]
        
        shift_z = -lag_z
        if shift_z > 0:
             z_pred_aligned = z_pred_map[:, shift_z:]
             z_gt_aligned = z_gt_map[:, :z_pred_aligned.shape[1]]
        else:
             z_pred_aligned = z_pred_map[:, :z_pred_map.shape[1]+shift_z]
             z_gt_aligned = z_gt_map[:, -shift_z: -shift_z+z_pred_aligned.shape[1]]
             
        # Crop to same length
        min_len_z = min(z_pred_aligned.shape[1], z_gt_aligned.shape[1])
        z_pred_aligned = z_pred_aligned[:, :min_len_z]
        z_gt_aligned = z_gt_aligned[:, :min_len_z]
        
        sim = torch.nn.functional.cosine_similarity(torch.from_numpy(z_gt_aligned), torch.from_numpy(z_pred_aligned), dim=0)
        plt.subplot(6, 2, 10)
        plt.plot(sim.numpy())
        plt.title(f"Cosine Sim (Mean: {sim.mean():.4f}, Lag: {lag_z})")
        plt.ylim(-1, 1)
        plt.grid(True, alpha=0.3)
    
        # F. InfoNCE Similarity Matrix (Time x Time)
        # F. Envelope Prediction
        from losses_neurocodec import EnvelopeMatcher
        env_matcher = EnvelopeMatcher(target_rate=128, audio_rate=44100)
        
        # Pred Envelope (B, T_eeg) -> (T_eeg)
        env_pred_map = env_pred[0].cpu().numpy()
        
        # GT Envelope (from clean audio)
        with torch.no_grad():
             env_gt = env_matcher.extract_envelope(clean.to(device))
        env_gt_map = env_gt[0].cpu().numpy()
        
        # Align lengths
        min_len_env = min(len(env_pred_map), len(env_gt_map))
        
    
        # F. Z-Values (Condensed to Row 6)
        plt.subplot(6, 2, 11)
        sns.heatmap(z_gt_map[:100, :limit], cmap='coolwarm', center=0, cbar=False)
        plt.title("Z Ground Truth (Subset)")
        
        plt.subplot(6, 2, 12)
        sns.heatmap(z_pred_map[:100, :limit], cmap='coolwarm', center=0, cbar=False)
        plt.title("Z Prediction (Subset)")
        
        # We dropped Envelope and Histogram to make space for full rows.
        # This tradeoff is acceptable for the user's focus on EEG-Speech alignment.
        
        # G. (Metrics/Debug are printed below)
        # We filled 1-12.
        
        print("\nDEBUG STATS:")
        print(f"Raw EEG   - Global Std: {raw_eeg_map.std():.4f}, Temporal Std: {np.mean(raw_eeg_map.std(axis=-1)):.4f}")
        print(f"EEG Feat  - Global Std: {eeg_map.std():.4f}, Temporal Std: {np.mean(eeg_map.std(axis=-1)):.4f}")
        print(f"Z GT      - Mean: {z_gt_map.mean():.4f}, Std: {z_gt_map.std():.4f}")
        print(f"Z Pred    - Mean: {z_pred_map.mean():.4f}, Std: {z_pred_map.std():.4f}")
        print(f"Cosine Sim- Mean: {sim.mean():.4f}")
        
        savename = f"{output_dir}/internals_plot_sample_{sample_idx}.png"
        plt.tight_layout()
        plt.savefig(savename)
        print(f"Saved plot to {savename}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/jaliya/eeg_speech/navindu/data/Cocktail_Party/Normalized-without-bad-components')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/neurocodec_v4_infonce/latest_model.pth')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden dimension (default: 256)")
    parser.add_argument('--subset', type=str, default='train', help="Dataset subset (train, val, test)")
    parser.add_argument('--noise_eeg', action='store_true', help="Use random noise instead of real EEG")
    parser.add_argument('--num_samples', type=int, default=5, help="Number of samples to visualize")
    parser.add_argument('--dataset', type=str, default='cocktail', choices=['cocktail', 'kul'], help='Dataset to use')
    
    args = parser.parse_args()
    
    # Run visualization loop
    visualize(args)
