
import os
import argparse
import torch
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
    model = NeuroCodec(dac_model_type=dac_model_type, eeg_in_channels=eeg_channels, hidden_dim=args.hidden_dim).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
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
            output = model(noisy, eeg)
            if isinstance(output, tuple):
                z_pred = output[0]
            else:
                z_pred = output
            
            # Quantize Predicted Z
            z_q = model.dac.quantizer(z_pred, n_quantizers=9)[0]
            
            # Decode to Audio
            pred_audio = model.dac.decode(z_q)
            
        # 4. Save Audio
        output_dir = "results/NeuroCodec_v4_infonce_best/Inference"
        if args.noise_cue:
            output_dir = "results/NeuroCodec_v4_infonce_best/Inference_NoiseCue"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Trim to shortest length
        min_len = min(pred_audio.shape[-1], clean.shape[-1], noisy.shape[-1])
        pred_audio = pred_audio[..., :min_len]
        clean = clean[..., :min_len]
        noisy = noisy[..., :min_len]
        
        torchaudio.save(f"{output_dir}/input_noisy_{i}.wav", noisy.cpu().squeeze(0), target_fs)
        torchaudio.save(f"{output_dir}/target_clean_{i}.wav", clean.cpu().squeeze(0), target_fs)
        torchaudio.save(f"{output_dir}/prediction_{i}.wav", pred_audio.cpu().squeeze(0), target_fs)
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
        si_sdr_oracle = sisdr(clean_aligned_oracle, oracle_aligned)
        
        # 6. Other Metrics (on Aligned signals)
        # SI-SDR Input
        min_len_in = min(len(clean_np), len(noisy_np))
        si_sdr_orig = sisdr(clean_np[:min_len_in], noisy_np[:min_len_in])
        
        print(f"Metrics Sample {i}:")
        print(f"  Input SI-SDR:     {si_sdr_orig:.2f} dB")
        print(f"  Oracle SI-SDR:    {si_sdr_oracle:.2f} dB (Lag: {lag_oracle})")
        print(f"  Output SI-SDR:    {si_sdr_pred:.2f} dB (Lag: {lag_pred})")
        print(f"  Improvement:      {si_sdr_pred - si_sdr_orig:.2f} dB")
        
        # 7. Plotting (Using Aligned Signals for Prediction)
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 12))
        
        def plot_waveform_spectrogram(audio, title, idx):
            # Waveform
            plt.subplot(3, 2, 2*idx + 1)
            plt.plot(audio)
            plt.title(f"{title} Waveform")
            plt.grid(True, alpha=0.3)
            # plt.ylim(-1, 1) # Auto-scale might be better if levels differ
            
            # Spectrogram
            plt.subplot(3, 2, 2*idx + 2)
            spec = torch.stft(torch.from_numpy(audio), n_fft=1024, hop_length=256, return_complex=True)
            spec_mag = torch.abs(spec)
            spec_db = 20 * torch.log10(spec_mag + 1e-8)
            plt.imshow(spec_db.numpy(), aspect='auto', origin='lower', cmap='inferno', vmin=-100, vmax=20)
            plt.title(f"{title} Spectrogram")
            plt.colorbar(format='%+2.0f dB')
            
        plot_waveform_spectrogram(noisy_np, "Input (Noisy)", 0)
        plot_waveform_spectrogram(clean_np, "Target (Clean)", 1)
        plot_waveform_spectrogram(pred_aligned, f"Prediction (Aligned, Lag:{lag_pred})", 2)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/inference_plot_{i}.png")
        plt.close() # Close figure to free memory
        print(f"Saved plot_{i} to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/jaliya/eeg_speech/navindu/data/Cocktail_Party/Normalized-without-bad-components')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/neurocodec/d2/mamba/best_model.pth')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--subset', type=str, default='val', help="Dataset subset to use (train, val, test)")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of samples to process")
    parser.add_argument('--noise_cue', action='store_true', help="Use random noise instead of EEG as input")
    parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden dimension of the model (default: 128)")
    parser.add_argument('--use_fast_bss', action='store_true', default=True, help="Use fast_bss_eval for SIR-SDR")
    
    parser.add_argument('--dataset', type=str, default='cocktail', choices=['cocktail', 'kul'], help='Dataset to use')
    parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle the dataset to pick random samples")
    
    args = parser.parse_args()
    
    inference(args)
