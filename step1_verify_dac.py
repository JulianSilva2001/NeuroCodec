
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
    import argparse
    parser = argparse.ArgumentParser(description="Verify DAC reconstruction on different datasets.")
    parser.add_argument('--dataset', type=str, default='cocktail', choices=['cocktail', 'kul'], help='Dataset to use')
    parser.add_argument('--root', type=str, default=None, help='Path to dataset root/LMDB')
    parser.add_argument('--model_type', type=str, default=None, choices=['44khz', '16khz', '24khz'], help='DAC model type (default: 44khz for cocktail, 16khz for kul)')
    parser.add_argument('--output_dir', type=str, default='output_dac_verify', help='Directory to save samples')
    parser.add_argument('--num_save', type=int, default=5, help='Number of samples to save audio for')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    
    parser.add_argument('--original_fs', type=int, default=16000, help='Original sampling rate of the KUL dataset (default: 16000)')

    args = parser.parse_args()

    # Defaults logic
    if args.dataset == 'cocktail':
        if args.root is None:
            # Let's keep the user's workflow simple. 
            # If they want KUL, they say --dataset kul.
            # If they say nothing, it defaults to cocktail.
            # But the USER requested "edit ... to work for the current dataset which is 16khz".
            # The current dataset is KUL.
            # So maybe default should be KUL or flexible.
            # Let's check the file content again... 
            # Original file had: root = '/workspace/Dataset/kul_all_subjects.lmdb' in line 83?
            # Wait, line 83 in previous view was: `root = '/workspace/Dataset/kul_all_subjects.lmdb'`
            # But line 84 used `load_NeuroCodecDataset`.
            # `load_NeuroCodecDataset` expects H5 files, not LMDB.
            # So the file was likely already broken or pointing to wrong path?
            # Or `kul_all_subjects.lmdb` IS the path they want to use now.
            pass
        if args.model_type is None:
            args.model_type = '44khz'
        target_fs = 44100
    elif args.dataset == 'kul':
        if args.root is None:
             args.root = '/workspace/Dataset/kul_all_subjects.lmdb'
        if args.model_type is None:
            args.model_type = '16khz'
        target_fs = 16000
    
    # Override root if user didn't specify and we want to be safe?
    # Actually, let's just use what arg provides.

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Root: {args.root}")
    print(f"Model: {args.model_type} (Target FS: {target_fs}) (Original FS: {args.original_fs})")

    # 1. Load DAC
    print(f"Loading DAC {args.model_type}...")
    model_path = dac.utils.download(model_type=args.model_type)
    model = dac.DAC.load(model_path)
    model.to(device)
    model.eval()
    print("DAC Model Loaded.")

    # 2. Load Dataset
    import sys
    sys.path.append(os.getcwd())
    from dataset_neurocodec import load_NeuroCodecDataset, load_KUL_NeuroCodecDataset
    
    print("Loading Dataset...")
    if args.dataset == 'cocktail':
        # Provide a default valid path if none
        if args.root is None: args.root = '/home/jaliya/eeg_speech/navindu/data/Cocktail_Party/Normalized-1' 
        val_loader = load_NeuroCodecDataset(args.root, 'val', batch_size=1, num_gpus=1)
    else:
        val_loader = load_KUL_NeuroCodecDataset(args.root, subset='val', batch_size=1, num_gpus=1, target_fs=target_fs, original_fs=args.original_fs)
    
    print("Dataset Loaded. Evaluating...")
    
    metrics = {'sdr_raw': [], 'sdr_fast': [], 'estoi': []}
    
    max_samples = 10
    
    for i, data in enumerate(val_loader):
        
        # Unpack data (handle differences)
        if len(data) == 3:
            noisy, eeg, clean = data
        else:
            print(f"Warning: Unexpected data len {len(data)}")
            continue

        if i >= max_samples: break
        
        clean = clean.to(device)
        
        # 3. Encode & Decode
        with torch.no_grad():
            # DAC Encode expects (B, 1, T) usually? 
            # If clean is (B, T), unsqueeze.
            if clean.ndim == 2:
                clean_in = clean.unsqueeze(1)
            else:
                clean_in = clean

            z, _, _, _, _ = model.encode(clean_in)
            recon = model.decode(z)
            
        # Align lengths
        min_len = min(clean_in.shape[-1], recon.shape[-1])
        clean_aligned = clean_in[..., :min_len]
        recon_aligned = recon[..., :min_len]

        clean_np = clean_aligned.cpu().numpy().squeeze()
        recon_np = recon_aligned.cpu().numpy().squeeze()
        
        # A. Signal Alignment
        # Calculate lag
        from scipy import signal
        correlation = signal.correlate(clean_np, recon_np, mode='full')
        lags = signal.correlation_lags(clean_np.size, recon_np.size, mode='full')
        lag = lags[np.argmax(correlation)]
        print(f"  Lag: {lag}")

        if lag > 0:
            # recon is ahead of clean (or vice versa depending on definition)
            # If lag is positive, clean is shifted right relative to recon?
            # xcorr(clean, recon). Max at lag L means clean[t] matches recon[t+L]?
            # Let's align recon to clean.
            # If lag > 0, we should shift recon by lag? 
            # Actually, let's just use the aligned versions.
            recon_aligned = np.roll(recon_np, shift=lag)
            recon_aligned[:lag] = 0 # Zero out padding
            clean_aligned = clean_np
        elif lag < 0:
            recon_aligned = np.roll(recon_np, shift=lag)
            recon_aligned[lag:] = 0
            clean_aligned = clean_np
        else:
            recon_aligned = recon_np
            clean_aligned = clean_np
            
        # Recalculate metrics on aligned
        sdr_aligned = sisdr(clean_aligned, recon_aligned)
        
        # C. ESTOI
        estoi_val = -1
        try:
            from pystoi import stoi
            # stoi(clean, den, fs, extended=False)
            estoi_val = stoi(clean_aligned, recon_aligned, target_fs, extended=True)
            metrics['estoi'].append(estoi_val)
        except ImportError:
            pass
        except Exception as e:
             print(f"ESTOI Error: {e}")
        
        # B. Raw SI-SDR (Custom)
        sdr_raw = sisdr(clean_np, recon_np) # Keep original for comparison
        metrics['sdr_raw'].append(sdr_aligned) # Store aligned as main metric now? Or separate? 
        # User wants "after properly matching", so let's log aligned.
        
        # B. fast_bss_eval SDR (Stabilized)
        sdr_fast = -999
        try:
            from fast_bss_eval import sdr
            # sdr expects (B, T)
            c_t = torch.from_numpy(clean_aligned).float()
            r_t = torch.from_numpy(recon_aligned).float()
            if c_t.ndim == 1: c_t = c_t.unsqueeze(0)
            if r_t.ndim == 1: r_t = r_t.unsqueeze(0)
            
            sdr_fast = sdr(c_t, r_t, load_diag=1e-5).item()
            metrics['sdr_fast'].append(sdr_fast)
        except ImportError:
            pass
            
        print(f"Sample {i}: Raw={sdr_raw:.2f}dB | Aligned={sdr_aligned:.2f}dB | ESTOI={estoi_val:.4f} | FastBSS={sdr_fast:.2f}dB")
        
        # 5. Save Audio (Input & Output)
        if i < args.num_save:
            from scipy.io import wavfile
            # Use wavfile or sf
            import soundfile as sf
            
            # clean_np is likely (T,) from squeeze() earlier at line 178
            # But let's be safe.
            def to_numpy_audio(t):
                if isinstance(t, torch.Tensor): t = t.cpu().numpy()
                if t.ndim == 2: 
                     # (C, T) -> (T, C)
                     t = t.T
                return t
                
            c_save = to_numpy_audio(clean_aligned)
            r_save = to_numpy_audio(recon_aligned)
            
            path_clean = os.path.join(args.output_dir, f"sample_{i}_clean.wav")
            path_recon = os.path.join(args.output_dir, f"sample_{i}_recon.wav")
            
            sf.write(path_clean, c_save, target_fs)
            sf.write(path_recon, r_save, target_fs)
            print(f"  Saved {path_clean} & {path_recon}")
        
    # Average
    print(f"\n--- Average Performance ({args.dataset.upper()} @ {target_fs}Hz) ---")
    print(f"Raw SI-SDR:     {np.mean(metrics['sdr_raw']):.2f} dB")
    if metrics['estoi']:
        print(f"ESTOI:          {np.mean(metrics['estoi']):.4f}")
    if metrics['sdr_fast']:
        print(f"FastBSS SI-SDR: {np.mean(metrics['sdr_fast']):.2f} dB")

if __name__ == "__main__":
    main()
