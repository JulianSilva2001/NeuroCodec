
import h5py
import numpy as np

file_path = "/home/jaliya/eeg_speech/navindu/data/Cocktail_Party/Normalized/2s/eeg/new/eegs_test.h5"

try:
    with h5py.File(file_path, 'r') as f:
        print(f"Keys: {list(f.keys())}")
        # Usually keys are 'data' or similar, or the file is the dataset itself?
        # Let's check typical keys from previous context or just iterate
        for key in f.keys():
            data = f[key]
            print(f"Key: {key}, Shape: {data.shape}, Dtype: {data.dtype}")
            
            # Infer SR
            # Assuming 2 seconds per sample based on path
            duration = 2.0
            if len(data.shape) > 1:
                samples = data.shape[-1] # Usually (N, T) or (N, C, T)
                sr_inferred = samples / duration
                print(f"  -> Inferred Sample Rate (assuming 2s): {sr_inferred} Hz")
                
except Exception as e:
    print(f"Error: {e}")
