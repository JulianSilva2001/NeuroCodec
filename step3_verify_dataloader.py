
import torch
import sys
import os
sys.path.append(os.getcwd())

from dataset_neurocodec import load_NeuroCodecDataset

def main():
    # Use dummy root as path is hardcoded in dataset class
    root = "/dummy/path" 
    batch_size = 2
    
    print("Loading NeuroCodec Dataloader (Test Split)...")
    dataloader = load_NeuroCodecDataset(root, 'test', batch_size)
    
    for i, (noisy, eeg, clean) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Noisy Audio Shape: {noisy.shape}")
        print(f"  Clean Audio Shape: {clean.shape}")
        print(f"  EEG Shape:         {eeg.shape}")
        
        # Verify Sample Rate implications
        # Audio length 87552 @ 44.1k = 1.98s
        # EEG length 256 @ 128Hz = 2.0s
        # They are close enough.
        
        if noisy.shape[-1] != 87552:
            print("WARNING: Audio length mismatch! Expected ~87552")
            
        if eeg.shape[-1] != 256:
            print("WARNING: EEG length mismatch! Expected 256")
            
        break # Only check one batch

if __name__ == "__main__":
    main()
