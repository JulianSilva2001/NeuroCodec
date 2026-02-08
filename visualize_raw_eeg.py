import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_raw_eeg(file_path, group_key, sample_idx=0, num_channels=5):
    print(f"Opening {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist!")
        return

    with h5py.File(file_path, 'r') as f:
        print(f"Keys: {list(f.keys())}")
        
        if group_key not in f:
            print(f"Error: Key {group_key} not found. Available: {list(f.keys())}")
            # Try to guess
            group_key = list(f.keys())[0]
            print(f"Using first key: {group_key}")

        data = f[group_key]
        print(f"Dataset Shape: {data.shape}")
        
        # Load Sample
        # Shape: (N, Time, Channels) usually (N, 256, 128)
        eeg_sample = data[sample_idx]
        print(f"Sample {sample_idx} Shape: {eeg_sample.shape}")
        
        # Scale to Microvolts for visualization (same as dataloader fix)
        eeg_sample = eeg_sample * 1e6
        print(f"Scaled to uV.")

        # Check Statistics
        # Expect (Time, Channels)
        # Calculate Std across Time (axis 0)
        temporal_std = np.std(eeg_sample, axis=0)
        print(f"Temporal Std (Per Channel) - Mean: {np.mean(temporal_std):.6f}")
        print(f"Temporal Std (Per Channel) - Min: {np.min(temporal_std):.6f}")
        print(f"Temporal Std (Per Channel) - Max: {np.max(temporal_std):.6f}")
        
        # Plotting
        plt.figure(figsize=(15, 10))
        
        # 1. Heatmap (Transposed to Channels x Time for convention)
        plt.subplot(2, 1, 1)
        # (Time, Channels) -> (Channels, Time)
        eeg_T = eeg_sample.T 
        # Normalize for visualization if needed, but let's see raw first
        plt.imshow(eeg_T, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(label='Amplitude (uV)')
        plt.title(f"Raw EEG Heatmap (Sample {sample_idx}) - Shape {eeg_T.shape} (uV)")
        plt.xlabel("Time")
        plt.ylabel("Channels")
        
        # 2. Waveforms
        plt.subplot(2, 1, 2)
        time_axis = np.arange(eeg_sample.shape[0])
        for i in range(min(num_channels, eeg_sample.shape[1])):
            plt.plot(time_axis, eeg_sample[:, i], label=f"Ch {i}")
            
        plt.title(f"First {num_channels} Channels Waveforms (uV)")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        output_path = "results/raw_eeg_check.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    # Path found previously
    path = '/home/jaliya/eeg_speech/navindu/data/Cocktail_Party/Normalized-2/2s/eeg/new/eegs_train.h5'
    key = 'eegs_train='
    visualize_raw_eeg(path, key)
