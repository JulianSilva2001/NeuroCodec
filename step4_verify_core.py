
import torch
import sys
import os

sys.path.append(os.getcwd())

from models.neurocodec import NeuroCodec

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Step 4 Verification on {device}...")
    
    # Instantiate
    model = NeuroCodec(dac_model_type='44khz').to(device)
    model.eval()
    
    # Dummy Input
    # Audio: 1 second ~ 44100 samples
    mixture = torch.randn(2, 1, 44100).to(device)
    # EEG: 1 second ~ 128 samples
    eeg = torch.randn(2, 128, 128).to(device)
    
    print("Running Forward Pass...")
    try:
        logits, codes_mix, z_mix, eeg_feat = model(mixture, eeg)
        
        print("\nSuccess!")
        print(f"Logits Shape: {logits.shape}")
        # Expected: (B, 9, 1024, T)
        # T should be 86 (512 stride for 44100)
        
        print(f"Codes Mix Shape: {codes_mix.shape}")
        print(f"EEG Feat Shape: {eeg_feat.shape}")
        
        # Verify T matching
        T_logits = logits.shape[-1]
        T_codes = codes_mix.shape[-1]
        
        if T_logits == T_codes:
            print("Time Dimension Matches! (T={})".format(T_logits))
        else:
            print(f"WARNING: Time Dimension Mismatch! Logits={T_logits}, Codes={T_codes}")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
