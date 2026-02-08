import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def visualize_mat_eeg(file_path):
    print(f"Opening {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist!")
        return

    try:
        mat = scipy.io.loadmat(file_path)
    except Exception as e:
        print(f"Failed to load .mat: {e}")
        return
        
    print(f"Keys: {list(mat.keys())}")
    
    # Find data key (exclude __header__, __version__, __globals__)
    data_keys = [k for k in mat.keys() if not k.startswith('__')]
    print(f"Data Keys: {data_keys}")
    
    for key in data_keys:
        data = mat[key]
        if isinstance(data, np.ndarray):
            print(f"\nAnalyzing Key: '{key}'")
            print(f"Shape: {data.shape}")
            
            # Assume (Channels, Time) or (Time, Channels)
            # Usually EEG is (Channels, Time) in Matlab? Or (Time, Channels)?
            # Let's check dimensions.
            dims = data.shape
            
            # Heuristic: dimension with 128 or 64 is channels.
            # If both are large, assume (Time, Channels)?
            # For now, let's just calc std on both axes.
            
            std0 = np.std(data, axis=0).mean()
            std1 = np.std(data, axis=1).mean()
            print(f"Std along axis 0: {std0:.6f}")
            print(f"Std along axis 1: {std1:.6f}")
            
            # Simple Plot (first 5 channels)
            plt.figure(figsize=(12, 6))
            if dims[0] < dims[1]: # (Channels, Time)
                t_axis = np.arange(dims[1])
                for i in range(min(5, dims[0])):
                    plt.plot(t_axis, data[i, :], label=f"Ch {i}")
            else: # (Time, Channels)
                t_axis = np.arange(dims[0])
                for i in range(min(5, dims[1])):
                    plt.plot(t_axis, data[:, i], label=f"Ch {i}")
            
            plt.title(f"Waveforms from {key}")
            plt.legend()
            output_path = f"results/mat_check_{key}.png"
            plt.savefig(output_path)
            print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    path = '/home/jaliya/eeg_speech/navindu/data/Cocktail_Party/EEG/Subject1/Subject1_Run1.mat'
    visualize_mat_eeg(path)
