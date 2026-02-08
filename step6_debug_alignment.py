
import h5py
import numpy as np
import scipy.signal
import torch
import torchaudio

def cross_correlation(x, y):
    # Normalize
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    
    # Compute correlation
    corr = scipy.signal.correlate(x, y, mode='full')
    lags = scipy.signal.correlation_lags(len(x), len(y), mode='full')
    
    peak_idx = np.argmax(np.abs(corr))
    peak_lag = lags[peak_idx]
    peak_val = corr[peak_idx] / len(x) # Normalized correlation coefficient approx
    
    return peak_lag, peak_val

def check_alignment():
    print("Checking Data Alignment between Noisy and Clean H5 files...")
    
    root = "/home/jaliya/eeg_speech/navindu/data/Cocktail_Party/Normalized/2s/eeg/new"
    noisy_path = f"{root}/noisy_train.h5"
    clean_path = f"{root}/clean_train.h5"
    
    try:
        f_n = h5py.File(noisy_path, 'r')
        f_c = h5py.File(clean_path, 'r')
        
        noisy_d = f_n['noisy_train=']
        clean_d = f_c['clean_train=']
        
        print(f"Total Samples: {len(noisy_d)}")
        
        # Check first 5 samples
        for i in range(5):
            print(f"\nSample {i}:")
            # Shapes are (T, 1) based on previous inspection
            n = noisy_d[i].squeeze()
            c = clean_d[i].squeeze()
            
            # Simple check: Do lengths match?
            if len(n) != len(c):
                print(f"  Length Mismatch! Noisy={len(n)}, Clean={len(c)}")
                continue
            
            # Cross Correlation
            lag, val = cross_correlation(n, c)
            print(f"  Lag: {lag}, Correlation: {val:.4f}")
            
            if abs(lag) < 5 and val > 0.1:
                print("  -> ALIGNED OK")
            else:
                print("  -> POSSIBLE MISALIGNMENT")
                
        f_n.close()
        f_c.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_alignment()
