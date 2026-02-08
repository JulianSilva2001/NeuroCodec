import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import torch.nn as nn
from scipy.signal import hilbert, resample

# Add current directory to sys.path
sys.path.append(os.getcwd())

from M3ANET import M3ANET
from dataset import cock_tail
from utility.utils import normalize_A

def main():
    # Configuration
    config_path = 'configs/M3ANET.json'
    checkpoint_path = '/home/jaliya/eeg_speech/Julian/M3ANet-main/exp/M3ANET/checkpoint/500.pkl'
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

    # 1. Visualize Adjacency Matrix
    print("Visualizing Adjacency Matrix...")
    A = model.spike_encoder.A.detach()
    L = normalize_A(A).cpu().numpy()
    A = A.cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    im1 = axes[0].imshow(A, aspect='auto', cmap='viridis', origin='lower')
    axes[0].set_title("Learned Adjacency Matrix A")
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(L, aspect='auto', cmap='viridis', origin='lower')
    axes[1].set_title("Normalized Laplacian L")
    plt.colorbar(im2, ax=axes[1])
    
    plt.savefig('learned_adjacency.png')
    print(f"Adjacency visualization saved to learned_adjacency.png")
    plt.close()

    # 2. Envelope Correlation Analysis
    print("Running Envelope Correlation Analysis...")
    
    # Load Data
    dataset = cock_tail(root=dataset_root, mode='test', subject=subject_id)
    # Get one sample
    noisy, eeg, clean = dataset[0]
    
    # Preprocess dimensions
    noisy = noisy.unsqueeze(0).to(device) # (1, 1, T)
    eeg_in = eeg.unsqueeze(0).to(device)     # (1, 128, T)
    
    # Forward pass to get embeddings
    with torch.no_grad():
        # M3ANET logic:
        # enc_output_spike = self.spike_encoder(spike_input)
        enc_output = model.spike_encoder(eeg_in) # (1, 64, 270)
        
    embeddings = enc_output.squeeze().cpu().numpy() # (64, 270)
    
    # Process Audio Envelopes
    clean_audio = clean.squeeze().cpu().numpy() # (T,)
    noisy_audio = noisy.squeeze().cpu().numpy() # (T,)
    noise_only = noisy_audio - clean_audio
    
    def get_envelope(offset_signal):
        return np.abs(hilbert(offset_signal))
        
    clean_env = get_envelope(clean_audio)
    noise_env = get_envelope(noise_only)
    
    # Resample envelopes to match embedding length (270)
    target_len = embeddings.shape[1]
    clean_env_resampled = resample(clean_env, target_len)
    noise_env_resampled = resample(noise_env, target_len)
    
    # Compute Correlations
    corrs_clean = []
    corrs_noise = []
    
    for i in range(embeddings.shape[0]):
        feat = embeddings[i, :]
        # Pearson correlation
        r_clean = np.corrcoef(feat, clean_env_resampled)[0, 1]
        r_noise = np.corrcoef(feat, noise_env_resampled)[0, 1]
        corrs_clean.append(r_clean)
        corrs_noise.append(r_noise)
        
    corrs_clean = np.array(corrs_clean)
    corrs_noise = np.array(corrs_noise)
    
    avg_clean = np.mean(np.abs(corrs_clean))
    avg_noise = np.mean(np.abs(corrs_noise))
    print(f"Average Abs Correlation with Speech Envelope: {avg_clean:.4f}")
    print(f"Average Abs Correlation with Noise Envelope: {avg_noise:.4f}")

    # Plot Correlations
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = np.arange(len(corrs_clean))
    width = 0.35
    
    ax.bar(indices - width/2, np.abs(corrs_clean), width, label='Speech Envelope')
    ax.bar(indices + width/2, np.abs(corrs_noise), width, label='Noise Envelope')
    
    ax.set_xlabel('Feature Channel Index')
    ax.set_ylabel('Absolute Correlation')
    ax.set_title('EEG Feature Correlation with Envelopes')
    ax.legend()
    
    plt.savefig('envelope_correlation.png')
    print("Correlation plot saved to envelope_correlation.png")
    plt.close()
    
    # Visual Tracking Plot
    # Identify top 3 channels correlated with Speech
    top_indices = np.argsort(np.abs(corrs_clean))[-3:][::-1]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    for i, idx in enumerate(top_indices):
        ax = axes[i]
        feat = embeddings[idx, :]
        
        # Normalize for plotting
        feat_norm = (feat - feat.mean()) / feat.std()
        env_norm = (clean_env_resampled - clean_env_resampled.mean()) / clean_env_resampled.std()
        
        ax.plot(feat_norm, label=f'EEG Feature {idx}')
        ax.plot(env_norm, label='Speech Envelope', linestyle='--', alpha=0.7)
        ax.set_title(f"Channel {idx}: Corr = {corrs_clean[idx]:.3f}")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig('feature_tracking.png')
    print("Feature tracking plot saved to feature_tracking.png")
    plt.close()

if __name__ == '__main__':
    main()
