
import os
import argparse
import torch
import time
from tqdm import tqdm
import numpy as np
import scipy.signal
import torchaudio
from models.neurocodec import NeuroCodec
from dataset_neurocodec import load_NeuroCodecDataset, load_KUL_NeuroCodecDataset

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
    alpha = np.sum(reference * estimation, axis=-1, keepdims=True) / (reference_energy + 1e-8)
    projections = alpha * reference
    noise = estimation - projections
    projections_energy = np.sum(projections ** 2, axis=-1)
    noise_energy = np.sum(noise ** 2, axis=-1)
    si_sdr_val = 10 * np.log10(projections_energy / (noise_energy + 1e-8))
    return si_sdr_val

def inference(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Inference on {device}...")
    
    # Configure for Dataset
    if args.dataset == 'kul':
        dac_model_type = '16khz'
        target_fs = 16000
        eeg_channels = 64
        print(f"Info: Using KUL configuration (DAC: 16khz, EEG: 64ch)")
    else:
        dac_model_type = '44khz'
        target_fs = 44100
        eeg_channels = 128
        
    # 1. Load Model
    print(f"Loading Model from {args.checkpoint}...")
    model = NeuroCodec(
        dac_model_type=dac_model_type, 
        eeg_in_channels=eeg_channels, 
        hidden_dim=args.hidden_dim, 
        num_layers=args.num_layers,
        backbone=args.backbone,
        activation=args.activation
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Support both plain state_dict checkpoints and wrapped training checkpoints.
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']

    # Handle DataParallel/DDP checkpoints saved with a 'module.' prefix.
    if isinstance(checkpoint, dict) and any(k.startswith('module.') for k in checkpoint.keys()):
        checkpoint = {k.replace('module.', '', 1): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint)
    model.eval()
    print("Model Loaded.")
    
    # 2. Load Data
    if args.dataset == 'kul':
        val_loader = load_KUL_NeuroCodecDataset(
            lmdb_path=args.root, 
            subset=args.subset, 
            batch_size=1,            
            num_gpus=1,
            target_fs=target_fs,
            original_fs=16000,
            shuffle=args.shuffle
        )
    else:
        val_loader = load_NeuroCodecDataset(
            root=args.root, 
            subset=args.subset, 
            batch_size=1, 
            num_gpus=1,
            shuffle=args.shuffle
        )
    
    # 3. Process Multiple Samples
    data_iter = iter(val_loader)
    
    for i in range(args.num_samples):
        print(f"\n--- Processing Sample {i+1}/{args.num_samples} ---")
        try:
            noisy, eeg, clean = next(data_iter)
        except StopIteration:
            print("No more samples in dataset.")
            break
            
        noisy = noisy.to(device)
        eeg = eeg.to(device)
        clean = clean.to(device)
        
        if args.noise_cue:
            # Replace EEG with Noise matching the statistics of the real EEG
            # This ensures we test "Information Content" not "Signal Magnitude"
            eeg_mean = eeg.mean()
            eeg_std = eeg.std()
            eeg = torch.randn_like(eeg) * eeg_std + eeg_mean
            print(f"  [Noise Cue] Replaced EEG with Gaussian Noise (Mean: {eeg_mean:.2f}, Std: {eeg_std:.2f})")
            
        with torch.no_grad():
            # Forward Pass
            start_time = time.time()
            output = model(noisy, eeg)
            end_time = time.time()
            inference_time = end_time - start_time
            
            if isinstance(output, tuple):
                z_pred = output[0]
            else:
                z_pred = output
            
            # Quantize Predicted Z
            z_q = model.dac.quantizer(z_pred, n_quantizers=9)[0]
            
            # Decode to Audio
            pred_audio = model.dac.decode(z_q)
            
        # 4. Save Audio
        output_dir = "results/NeuroCodec_v1/last/Inference"
        if args.noise_cue:
            output_dir = "results/NeuroCodec_v1/last/Inference_NoiseCue"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Trim to shortest length
        min_len = min(pred_audio.shape[-1], clean.shape[-1], noisy.shape[-1])
        pred_audio = pred_audio[..., :min_len]
        clean = clean[..., :min_len]
        noisy = noisy[..., :min_len]
        
        import soundfile as sf
        
        # Ensure numpy format for soundfile
        def to_numpy(t):
            return t.detach().cpu().numpy().squeeze()
            
        sf.write(f"{output_dir}/input_noisy_{i}.wav", to_numpy(noisy), target_fs)
        sf.write(f"{output_dir}/target_clean_{i}.wav", to_numpy(clean), target_fs)
        sf.write(f"{output_dir}/prediction_{i}.wav", to_numpy(pred_audio), target_fs)
        print(f"Saved audio_{i} to {output_dir}")
        
        # 5. Calculate Metrics
        # 5. Calculate Metrics with Alignment
        clean_np = clean.cpu().numpy().squeeze()
        pred_np = pred_audio.cpu().numpy().squeeze()
        noisy_np = noisy.cpu().numpy().squeeze()
        
        # Helper for alignment
        def align_signals(ref, est):
            if ref.ndim == 2: ref = ref[0]
            if est.ndim == 2: est = est[0]
            
            # Cross-correlation alignment
            ref_centered = ref - np.mean(ref)
            est_centered = est - np.mean(est)
            
            # Fast correlation using FFT
            corr = scipy.signal.correlate(ref_centered, est_centered, mode='full', method='fft')
            lags = scipy.signal.correlation_lags(len(ref), len(est), mode='full')
            lag = lags[np.argmax(corr)]
            
            if lag < 0:
                est_aligned = est[-lag:]
                ref_aligned = ref[:len(est_aligned)]
            else:
                est_aligned = est[:len(ref)-lag]
                ref_aligned = ref[lag:lag+len(est_aligned)]
                
            # Truncate to match
            min_len = min(len(ref_aligned), len(est_aligned))
            ref_aligned = ref_aligned[:min_len]
            est_aligned = est_aligned[:min_len]
            
            return ref_aligned, est_aligned, lag

        # Align Model Output
        clean_aligned_pred, pred_aligned, lag_pred = align_signals(clean_np, pred_np)
        si_sdr_pred = sisdr(clean_aligned_pred, pred_aligned)
        
        # Align Oracle (DAC Reconstruction)
        with torch.no_grad():
            z_gt, _, _, _, _ = model.dac.encode(clean)
            clean_recon = model.dac.decode(z_gt)
            clean_recon_np = clean_recon.cpu().numpy().squeeze()
            
        clean_aligned_oracle, oracle_aligned, lag_oracle = align_signals(clean_np, clean_recon_np)
        
        # 6. Other Metrics (on Aligned signals)
        
        # Band-limited Filtering (0-4kHz) for Metrics
        def lowpass_filter(audio, cutoff=4000, fs=16000, order=5):
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
            # Use filtfilt for zero-phase filtering
            filtered = scipy.signal.filtfilt(b, a, audio)
            return filtered

        # Filter original clean/noisy for input metrics
        noisy_filt = lowpass_filter(noisy_np, fs=target_fs)
        clean_filt = lowpass_filter(clean_np, fs=target_fs)
        
        # Filter aligned prediction/clean for output metrics
        pred_aligned_filt = lowpass_filter(pred_aligned, fs=target_fs)
        clean_aligned_pred_filt = lowpass_filter(clean_aligned_pred, fs=target_fs)
        
        # Filter oracle
        oracle_aligned_filt = lowpass_filter(oracle_aligned, fs=target_fs)
        clean_aligned_oracle_filt = lowpass_filter(clean_aligned_oracle, fs=target_fs)

        # SI-SDR Input (Filtered)
        min_len_in = min(len(clean_filt), len(noisy_filt))
        si_sdr_orig = sisdr(clean_filt[:min_len_in], noisy_filt[:min_len_in])
        
        # ESTOI
        estoi_val = -1
        try:
            from pystoi import stoi
            # Use filtered signals for ESTOI as well? Yes, user requested 4kHz limit.
            estoi_val = stoi(clean_aligned_pred_filt, pred_aligned_filt, target_fs, extended=True)
        except ImportError:
            pass
        except Exception as e:
            print(f"ESTOI Error: {e}")
        
        # Calculate Output Metrics (Filtered)
        si_sdr_pred = sisdr(clean_aligned_pred_filt, pred_aligned_filt)
        si_sdr_oracle = sisdr(clean_aligned_oracle_filt, oracle_aligned_filt)

        print(f"Metrics Sample {i} (0-4kHz Band-limited):")
        print(f"  Inference Time:   {inference_time:.4f}s")
        print(f"  Input Duration:   {noisy.shape[-1] / target_fs:.2f}s ({noisy.shape[-1]} samples)")
        print(f"  Input SI-SDR:     {si_sdr_orig:.2f} dB")
        print(f"  Oracle SI-SDR:    {si_sdr_oracle:.2f} dB (Lag: {lag_oracle})")
        print(f"  Output SI-SDR:    {si_sdr_pred:.2f} dB (Lag: {lag_pred})")
        print(f"  Output ESTOI:     {estoi_val:.4f}")
        print(f"  Improvement:      {si_sdr_pred - si_sdr_orig:.2f} dB")
        
        
        # 6.5 Calculate Filtered Interferer
        # Re-calc matches lengths of filtered signals
        min_len_filt = min(len(noisy_filt), len(clean_filt))
        interferer_filt = noisy_filt[:min_len_filt] - clean_filt[:min_len_filt]

        # 7. Plotting (Using Filtered Signals)
        import matplotlib.pyplot as plt
        import librosa.display
        
        fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        
        def plot_spec(y, title, ax_idx):
            D = librosa.stft(y, n_fft=1024, hop_length=256)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img = librosa.display.specshow(S_db, y_axis='linear', x_axis='time', sr=target_fs, 
                                         hop_length=256, fmax=4000, ax=ax[ax_idx]) # fmax=4000 matches filter
            ax[ax_idx].set_title(title)
            ax[ax_idx].set_ylim(0, 4000) # Zoom in to 4kHz
            return img

        # Use filtered signals
        plot_spec(noisy_filt[:min_len_filt], "Mixture (0-4kHz)", 0)
        plot_spec(clean_filt[:min_len_filt], "Target (0-4kHz)", 1)
        plot_spec(interferer_filt, "Interferer (0-4kHz)", 2)
        plot_spec(pred_aligned_filt, "Reconstruction (0-4kHz)", 3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/inference_plot_{i}.png")
        plt.close()
        print(f"Saved plot_{i} to {output_dir}")
        
        # 8. Save Filtered Audio
        # Save aligned filtered prediction
        sf.write(f"{output_dir}/audio_{i}_pred_filtered.wav", pred_aligned_filt, target_fs)
        # Save aligned filtered clean for comparison
        sf.write(f"{output_dir}/audio_{i}_clean_filtered.wav", clean_aligned_pred_filt, target_fs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/jaliya/eeg_speech/navindu/data/Apr-1/lmdb')
    parser.add_argument('--checkpoint', type=str, default='/home/jaliya/eeg_speech/shaveen/NeuroCodec/checkpoints/neurocodec_KUL/best_model.pth')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--subset', type=str, default='val', help="Dataset subset to use (train, val, test)")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of samples to process")
    parser.add_argument('--noise_cue', action='store_true', help="Use random noise instead of EEG as input")
    parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden dimension of the model (default: 128)")
    parser.add_argument('--use_fast_bss', action='store_true', default=True, help="Use fast_bss_eval for SIR-SDR")
    
    parser.add_argument('--dataset', type=str, default='kul', choices=['cocktail', 'kul'], help='Dataset to use')
    parser.add_argument('--eeg_channels', type=int, default=64, help='Number of EEG channels (128 for Cocktail, 64 for KUL)')
    parser.add_argument('--backbone', type=str, default='mamba', choices=['mamba', 'transformer'], help='Backbone architecture')
    parser.add_argument('--activation', type=str, default='gelu', choices=['gelu', 'snake', 'relu'], help='Activation function (transformer only)')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle the dataset to pick random samples")
    
    args = parser.parse_args()
    
    inference(args)
