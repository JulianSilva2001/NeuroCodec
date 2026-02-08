
import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from models.neurocodec import NeuroCodec

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Instantiate Model
    print("Instantiating NeuroCodec...")
    model = NeuroCodec(dac_model_type='44khz').to(device)
    
    # Verify Frozen
    dac_frozen = all(not p.requires_grad for p in model.dac.parameters())
    print(f"DAC Frozen: {dac_frozen}")
    
    if not dac_frozen:
        print("ERROR: DAC is not frozen!")
        exit(1)
        
    # Dummy Inputs
    # Audio: 1 sec at 44.1kHz
    mixture = torch.randn(1, 1, 44100).to(device)
    # EEG: 1 sec at 128Hz
    eeg = torch.randn(1, 128, 128).to(device)
    
    print(f"Input Mixture: {mixture.shape}")
    print(f"Input EEG: {eeg.shape}")
    
    # Forward
    outputs = model(mixture, eeg)
    
    z_mix = outputs["z_mix"]
    eeg_feat = outputs["eeg_feat"]
    
    print(f"Z_mix Shape: {z_mix.shape}") # (B, 1024, T_frames)
    print(f"EEG_feat Shape: {eeg_feat.shape}") # (B, 64, T_eeg_frames)
    
    print("Step 2 Verification Successful: Model Skeleton works.")

if __name__ == "__main__":
    main()
