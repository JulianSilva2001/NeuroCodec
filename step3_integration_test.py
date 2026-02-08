
import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from dataset_neurocodec import load_NeuroCodecDataset
from models.neurocodec import NeuroCodec

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Integration Test on {device}...")

    # 1. Load Data
    print("1. Loading Real Data...")
    # Use the hardcoded path in the dataset class, so 'root' arg is dummy
    dataloader = load_NeuroCodecDataset(root="/dummy", subset='test', batch_size=1)
    
    # Get one batch
    noisy, eeg, clean = next(iter(dataloader))
    noisy = noisy.to(device)
    eeg = eeg.to(device)
    clean = clean.to(device)
    
    print(f"   Data Loaded:")
    print(f"   - Noisy: {noisy.shape}")
    print(f"   - EEG:   {eeg.shape}")
    print(f"   - Clean: {clean.shape}")

    # 2. Initialize Model
    print("\n2. Initializing NeuroCodec...")
    model = NeuroCodec(dac_model_type='44khz').to(device)
    model.eval()

    # 3. Forward Pass (Encoders)
    print("\n3. Running Forward Pass (Encoders Only)...")
    try:
        with torch.no_grad():
            outputs = model(noisy, eeg)
            
        print("   Success! Model processed the data.")
        
        # 4. Inspect Outputs
        z_mix = outputs['z_mix']
        codes_mix = outputs['codes_mix']
        eeg_feat = outputs['eeg_feat']
        
        print(f"\n4. Output Shapes:")
        print(f"   - DAC Latents (z): {z_mix.shape}")
        print(f"   - DAC Codes:       {codes_mix.shape}")
        print(f"   - EEG Features:    {eeg_feat.shape}")
        
        # Check alignment logic
        # DAC 44.1kHz -> ~86Hz frame rate (stride 512?)
        # 87552 / 512 = 171 frames?
        # Let's see what DAC gives.
        
        T_audio_frames = z_mix.shape[-1]
        T_eeg_frames = eeg_feat.shape[-1]
        
        print(f"\n   Time Alignment Check:")
        print(f"   - Audio Frames: {T_audio_frames}")
        print(f"   - EEG Frames:   {T_eeg_frames}")
        
        ratio = T_audio_frames / T_eeg_frames
        print(f"   - Ratio (Audio/EEG): {ratio:.2f}")
        
        print("\nAll systems go for Training Loop implementation.")
        
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
