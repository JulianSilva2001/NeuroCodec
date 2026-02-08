import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import torch.nn as nn
from scipy.spatial.distance import cdist

# Add current directory to sys.path
sys.path.append(os.getcwd())

from M3ANET import M3ANET
from dataset import cock_tail

def main():
    # Configuration
    config_path = 'configs/M3ANET.json'
    # Default to 500.pkl as user changed it in previous step, but let's stick to 115000.pkl 
    # OR better yet, check what works. The user switched to 500.pkl previously.
    # I will use 115000.pkl as per original request, unless it fails.
    checkpoint_path = '/home/jaliya/eeg_speech/Julian/M3ANet-main/exp/M3ANET/checkpoint/500.pkl' 
    # Note: User changed line 17 of previous script to 500.pkl. I should probably respect that.
    
    dataset_root = '/media/datasets/AAD_enhance' 
    subject_id = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")

    # Load Config
    with open(config_path) as f:
        config_data = f.read()
    config = json.loads(config_data)
    network_config = config["network_config"]

    # Initialize Model
    print("Initializing model...")
    model = M3ANET(L1=network_config["L1"], L2=network_config["L2"], L3=network_config["L3"], L4=network_config["L4"], 
                   enc_channel=network_config["enc_channel"], feature_channel=network_config["feature_channel"],
                   encoder_kernel_size=network_config["encoder_kernel_size"], layers=network_config["layers"], 
                   rnn_type=network_config["rnn_type"], norm=network_config["norm"], K=network_config["K"], 
                   dropout=network_config["dropout"], bidirectional=network_config["bidirectional"],
                   CMCA_kernel=network_config["kernel"], CMCA_layer_num=network_config["CMCA_layer_num"])
    
    model = model.to(device)

    # Load Checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("Model loaded.")

    # Hook to capture Fusion Inputs
    features = {}
    def fusion_hook(module, input, output):
        # input is tuple (audio, spike)
        # audio shape: (Batch, Channel, Audio_len)
        # spike shape: (Batch, Channel, EEG_len) - Note: In forward it might be padded already?
        # Let's see M3ANET.py line 376: spike = F.pad(spike...) happens INSIDE DPRNN but BEFORE fusion.
        # Wait, DPRNN.forward:
        # spike = F.pad(spike...)
        # input = self.fusion(input, spike)
        # So 'spike' entering fusion IS ALREADY PADDED to length 1624.
        
        # NOTE: If we want to answer "Does it learn alignment" we should probably look at the UNPADDED spike?
        # But 'spike' passed to fusion is padded.
        # If I want to compare 1624 Audio vs 270 EEG, I should probably unpad it myself or slice it.
        # Or I can just look at the full 1624x1624 matrix and expect zeros/noise in the padded region.
        
        audio, spike = input
        features['audio'] = audio.detach()
        features['spike'] = spike.detach()

    handle = model.DPRNN.fusion.register_forward_hook(fusion_hook)

    # Load Data
    print("Loading data...")
    dataset = cock_tail(root=dataset_root, mode='test', subject=subject_id)
    # Get one sample
    noisy, eeg, clean = dataset[0]
    
    # Preprocess dimensions
    noisy = noisy.unsqueeze(0).to(device) # (1, 1, T)
    eeg = eeg.unsqueeze(0).to(device)     # (1, 128, T)
    
    # Forward Pass
    print("Running forward pass...")
    with torch.no_grad():
        _ = model(noisy, eeg)
        
    handle.remove()
    
    audio_feat = features['audio'].squeeze().cpu().numpy() # (64, 1624)
    spike_feat = features['spike'].squeeze().cpu().numpy() # (64, 1624)
    
    print(f"Captured Audio Feat: {audio_feat.shape}")
    print(f"Captured Spike Feat: {spike_feat.shape}")
    
    # IMPORTANT: The spike feature was padded. We know the original length is ~270.
    # The padding is zeros.
    # Let's crop spike to its non-zero region to see the "true" temporal correspondence.
    # Or just visualize the whole thing and let the user see the padding.
    # Visualizing the whole thing is safer to understand what the model sees.
    
    # Compute Cosine Similarity Matrix
    # We compare column vectors (time steps).
    # Audio Matrix A: (64, T_a)
    # Spike Matrix S: (64, T_s)
    
    # Norm columns
    EPS = 1e-8
    audio_norm = audio_feat / (np.linalg.norm(audio_feat, axis=0, keepdims=True) + EPS)
    spike_norm = spike_feat / (np.linalg.norm(spike_feat, axis=0, keepdims=True) + EPS)
    
    # Sim = A.T @ S  -> (T_a, T_s)
    similarity = np.dot(audio_norm.T, spike_norm) # (1624, 1624)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarity, aspect='auto', cmap='RdBu_r', origin='lower', vmin=-1, vmax=1)
    
    ax.set_title("Latent Temporal Correspondence (Cosine Similarity)")
    ax.set_ylabel("Audio Time (0-1624)")
    ax.set_xlabel("EEG Time (0-1624)")
    plt.colorbar(im, ax=ax)
    
    # Add annotation for expected real EEG length
    ax.axvline(x=270, color='k', linestyle='--', linewidth=1, label='Original EEG Boundary (~270)')
    ax.legend()
    
    plt.savefig('temporal_correspondence.png')
    print("Saved temporal_correspondence.png")

if __name__ == '__main__':
    main()
