
import torchaudio
import torch
import matplotlib.pyplot as plt
import os
import numpy as np

try:
    import dac
    from dac.utils import load_model
except ImportError:
    print("Error: descript-audio-codec not installed.")
    exit(1)

def sisdr(ref, est):
    """
    Calculates SI-SDR.
    Args:
        ref (np.ndarray): Reference signal.
        est (np.ndarray): Estimated signal.
    Returns:
        float: SI-SDR value in dB.
    """
    # Ensure signals are 1D
    ref = ref.squeeze()
    est = est.squeeze()

    # Calculate projection of est onto ref
    s_target = np.sum(ref * est) * ref / np.sum(ref ** 2)
    
    # Calculate noise component
    e_noise = est - s_target

    # Calculate energies
    projections_energy = np.sum(s_target ** 2, axis=-1)
    noise_energy = np.sum(e_noise ** 2, axis=-1)
    
    return 10 * np.log10(projections_energy / (noise_energy + 1e-8))

def align_signal(ref, est):
    """
    Aligns estimation to reference using cross-correlation.
    Returns aligned_est.
    """
    # Use scipy for cross-correlation
    from scipy import signal
    correlation = signal.correlate(ref, est, mode='full')
    lags = signal.correlation_lags(ref.size, est.size, mode='full')
    lag = lags[np.argmax(correlation)]
    
    if lag > 0:
        # Est is ahead of Ref, roll forward (pad left)
        # aligned = np.pad(est, (lag, 0))[:ref.size] # Simple padding
        # Better: Roll
        aligned = np.roll(est, shift=lag)
        # Zero out the wrapped part if using roll, or just pad/slice
        aligned[:lag] = 0 
    elif lag < 0:
        # Est is behind Ref, roll backward
        aligned = np.roll(est, shift=lag)
        aligned[lag:] = 0
    else:
        aligned = est
        
    return aligned

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load DAC
    model_path = dac.utils.download(model_type="44khz")
    model = dac.DAC.load(model_path)
    model.to(device)
    model.eval()
    print("DAC Model Loaded (44.1kHz).")

    # 2. Load Dataset (Validation)
    import sys
    sys.path.append(os.getcwd())
    from dataset_neurocodec import load_NeuroCodecDataset
    
    root = '/home/jaliya/eeg_speech/navindu/data/Cocktail_Party/Normalized-2/2s/eeg/new'
    val_loader = load_NeuroCodecDataset(root, 'val', batch_size=1, num_gpus=1)
    
    print("Dataset Loaded. Evaluating first 10 samples...")
    
    metrics = {'44k': [], '44k_fast': []}
    
    for i, (noisy, eeg, clean) in enumerate(val_loader):
        if i >= 10: break
        
        clean = clean.to(device)
        
        # 3. Encode & Decode
        with torch.no_grad():
            z, _, _, _, _ = model.encode(clean)
            recon = model.decode(z)
            
        # Align lengths
        min_len = min(clean.shape[-1], recon.shape[-1])
        clean = clean[..., :min_len]
        recon = recon[..., :min_len]
        
        # 4. Compute Metrics
        clean_np = clean.cpu().numpy().squeeze()
        recon_np = recon.cpu().numpy().squeeze()
        
        # A. Raw SI-SDR (Custom)
        sdr_raw = sisdr(clean_np, recon_np)
        metrics['44k'].append(sdr_raw)

        # B. fast_bss_eval SDR (Stabilized)
        try:
            from fast_bss_eval import sdr
            # sdr returns (B,) tensor. Inputs (B, T).
            # We have (1, T) or need to add batch dim.
            clean_t = clean # (1, 1, T) or (1, T)
            recon_t = recon # (1, 1, T) or (1, T)
            
            if clean_t.ndim == 3: clean_t = clean_t.squeeze(1) # (B, T)
            if recon_t.ndim == 3: recon_t = recon_t.squeeze(1) # (B, T)

            # Check for load_diag or similar via introspection if needed, or just standard call
            # The library typically uses `sdr(ref, est, load_diag=...)`
            sdr_fast = sdr(clean_t, recon_t, load_diag=1e-5).item()
            metrics['44k_fast'].append(sdr_fast)
            
            print(f"Sample {i}: Raw={sdr_raw:.2f}dB | FastBSS={sdr_fast:.2f}dB")
            
        except ImportError:
            print(f"Sample {i}: Raw={sdr_raw:.2f}dB | FastBSS=N/A")
        
    # Average
    print("\n--- Average Oracle Performance (DAC Reconstruction) ---")
    print(f"Raw 44.1kHz:     {np.mean(metrics['44k']):.2f} dB")
    if metrics['44k_fast']:
        print(f"FastBSS 44.1kHz: {np.mean(metrics['44k_fast']):.2f} dB")

if __name__ == "__main__":
    main()
