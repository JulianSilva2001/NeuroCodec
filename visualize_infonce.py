import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import torch.nn as nn
import torch.nn.functional as F

# Add current directory to sys.path
sys.path.append(os.getcwd())

from M3ANET import M3ANET
from dataset import cock_tail

def main():
    # Configuration
    config_path = 'configs/M3ANET.json'
    checkpoint_path = '/home/jaliya/eeg_speech/Julian/M3ANet-main/exp/M3ANET/checkpoint/500.pkl' 
    # Sticking to 500.pkl since user might have preferred it, or I can switch back. 
    # Let's use 115000.pkl to be consistent with original request unless user insists.
    # Actually user changed previous script to 500.pkl. I will respect that. 
    # Wait, the tool output showed "Loading checkpoint 500.pkl" so the file modification persisted.
    # I should write this script to use 500.pkl if that's what is current, or 115000.pkl.
    # I will use 500.pkl to be safe given recent context.
    
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
        # output, pad_spike, mamba_enc
        output, pad_spike, mamba_enc = model(noisy, eeg)
        
    # Shapes:
    # pad_spike: (B, 103936)
    # mamba_enc: (B, 103936)
    
    # Reshape back to (64, 1624)
    # Since B=1, we just reshape the vector.
    
    eeg_feat = pad_spike.view(64, 1624).cpu().numpy()
    audio_feat = mamba_enc.view(64, 1624).cpu().numpy()
    
    print(f"EEG Feature Shape: {eeg_feat.shape}")
    print(f"Audio Feature (Mamba) Shape: {audio_feat.shape}")
    
    # 1. Cosine Similarity Matrix (Temporal)
    # Compare Time Steps [Columns]
    # Sim[i, j] = Cosine(Audio[:, i], EEG[:, j])
    
    EPS = 1e-8
    eeg_norm = eeg_feat / (np.linalg.norm(eeg_feat, axis=0, keepdims=True) + EPS)
    audio_norm = audio_feat / (np.linalg.norm(audio_feat, axis=0, keepdims=True) + EPS)
    
    similarity_matrix = np.dot(audio_norm.T, eeg_norm) # (1624, 1624)
    
    # Plot Similarity Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    # We expect high values in (0..270) on both axes if perfectly aligned time-wise
    # But latent alignment might be different.
    im = ax.imshow(similarity_matrix, aspect='auto', cmap='RdBu_r', origin='lower', vmin=-1, vmax=1)
    
    ax.set_title("InfoNCE Alignment (Audio VSSS vs EEG Padded)")
    ax.set_ylabel("Audio Time (0-1624)")
    ax.set_xlabel("EEG Time (0-1624)")
    ax.axvline(x=270, color='k', linestyle='--', linewidth=1, label='EEG Boundary')
    ax.legend()
    plt.colorbar(im, ax=ax)
    
    plt.savefig('infonce_alignment_matrix.png')
    print("Saved infonce_alignment_matrix.png")
    
    # 2. Local Time Correlation (Diagonal)
    # Compute correlation between Audio[:, t] and EEG[:, t] for t in 0..1624
    # Ideally should be high for t < 270 and low after.
    
    correlations = []
    for t in range(1624):
        u = audio_norm[:, t]
        v = eeg_norm[:, t]
        # Since u, v are unit vectors, dot product is cosine similarity
        sim = np.dot(u, v)
        correlations.append(sim)
        
    correlations = np.array(correlations)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(correlations, label='Cosine Similarity')
    ax.set_title("Time-Step Alignment Score (Audio vs EEG)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cosine Similarity")
    ax.axvline(x=270, color='r', linestyle='--', label='EEG Boundary (270)')
    ax.legend()
    
    # Calculate avg sim in region
    avg_active = np.mean(correlations[:270])
    avg_silent = np.mean(correlations[270:])
    
    print(f"Average Similarity (0-270): {avg_active:.4f}")
    print(f"Average Similarity (270-end): {avg_silent:.4f}")
    ax.text(100, 0.8, f"Avg: {avg_active:.2f}", color='green', fontweight='bold')
    ax.text(1000, 0.8, f"Avg: {avg_silent:.2f}", color='red')
    
    plt.savefig('infonce_local_correlation.png')
    print("Saved infonce_local_correlation.png")

if __name__ == '__main__':
    main()
