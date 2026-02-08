
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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    audio_path = "/home/jaliya/eeg_speech/navindu/data/Cocktail_Party/Dichotic/20000L_JourneyR_run_1.wav"
    output_dir = "results/NeuroCodec/Step1"
    os.makedirs(output_dir, exist_ok=True)
    
    output_wav = os.path.join(output_dir, "step1_recon.wav")
    output_plot = os.path.join(output_dir, "step1_visualization.png")
    output_latents = os.path.join(output_dir, "step1_latents.png")

    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        return

    # 1. Load Model
    # Download/Load the 44.1kHz model
    model_path = dac.utils.download(model_type="44khz")
    model = dac.DAC.load(model_path)
    model.to(device)
    model.eval()
    print("DAC Model Loaded (44.1kHz).")

    # 2. Load Audio
    # Resample if needed
    wav, sr = torchaudio.load(audio_path)
    print(f"Loaded Audio: {wav.shape}, SR: {sr}")
    
    # Process only first 5 seconds to be quick/clean
    if wav.shape[1] > sr * 5:
        wav = wav[:, :sr*5]
        print("Trimmed to 5 seconds.")

    if sr != 44100:
        resampler = torchaudio.transforms.Resample(sr, 44100)
        wav = resampler(wav)
        print(f"Resampled to 44100Hz: {wav.shape}")

    # Ensure Mono
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
        print(f"Converted to Mono: {wav.shape}")
    
    wav = wav.unsqueeze(0).to(device) # (1, 1, T)

    # 3. Encode
    with torch.no_grad():
        # model.encode returns named tuple or tuple
        z, codes, latents, _, _ = model.encode(wav)
    
    print(f"Encoded Codes Shape: {codes.shape}") 
    # (B, NumQuantizers, Frames)

    # 4. Decode
    with torch.no_grad():
        recon_wav = model.decode(z)
    
    print(f"Reconstructed Audio Shape: {recon_wav.shape}")

    # Save Output
    recon_cpu = recon_wav.squeeze(0).cpu() # (C, T)
    torchaudio.save(output_wav, recon_cpu, 44100)
    print(f"Saved reconstruction to {output_wav}")

    # 5. Visualize
    # Plot Waveform Comparison
    wav_cpu = wav.squeeze(0).cpu().numpy()
    recon_numpy = recon_cpu.numpy()

    # Take first channel for plotting
    orig = wav_cpu[0]
    recon = recon_numpy[0]
    
    # Align lengths for plotting if needed
    min_len = min(len(orig), len(recon))
    orig = orig[:min_len]
    recon = recon[:min_len]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    
    ax1.plot(orig)
    ax1.set_title("Original Waveform (First 5s)")
    
    ax2.plot(recon)
    ax2.set_title("Reconstructed Waveform (DAC 44.1kHz)")
    
    ax3.specgram(orig, Fs=44100, NFFT=1024, noverlap=512)
    ax3.set_title("Spectrogram (Original)")

    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Saved visualization to {output_plot}")
    
    # 6. Visualize Latents (Codes)
    # Shape: (1, 9, 430)
    codes_np = codes.squeeze(0).cpu().numpy() # (9, 430)
    
    plt.figure(figsize=(12, 4))
    plt.imshow(codes_np, aspect='auto', cmap='tab20', interpolation='nearest', origin='lower')
    plt.title("DAC Latent Codes (9 Codebooks x 430 Time Steps)")
    plt.ylabel("Codebook Index (0-8)")
    plt.xlabel("Time Step (Frame)")
    plt.colorbar(label="Code Index")
    plt.savefig(output_latents)
    print(f"Saved latent visualization to {output_latents}")

if __name__ == "__main__":
    main()
