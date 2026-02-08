import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import torch.nn as nn
from collections import OrderedDict

# Add current directory to sys.path
sys.path.append(os.getcwd())

from M3ANET import M3ANET
from dataset import cock_tail

def main():
    # Configuration
    config_path = 'configs/M3ANET.json'
    # Default to 115000.pkl as per plan, user can modify if needed
    checkpoint_path = '/home/jaliya/eeg_speech/Julian/M3ANet-main/exp/M3ANET/checkpoint/500.pkl'
    dataset_root = '/media/datasets/AAD_enhance' 
    subject_id = 2
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

    # Hook Registration for Attention Maps
    attention_maps = OrderedDict()
    hooks = []

    def get_attention_hook(name):
        def hook(module, input, output):
            # output of ScaledDotProductAttention is (output, attn)
            # attn shape: (batch, n_head, len_q, len_k)
            _, attn = output
            attention_maps[name] = attn.detach()
        return hook

    # The fusion block is in model.DPRNN.fusion
    # It has audio_encoder (list) and spike_encoder (list)
    # Inside each item (ConvCrossAttention), there is .attention (ScaledDotProductAttention)
    
    fusion_block = model.DPRNN.fusion
    num_layers = len(fusion_block.audio_encoder)
    print(f"Detected {num_layers} CMCA layers/blocks.")

    for i in range(num_layers):
        # Audio Encoder: Audio Q attends to EEG K,V ?? Or EEG Q attends to Audio K,V?
        # Let's check models.py ConvCrossAttention call in MultiLayerCrossAttention.forward:
        # out_audio = self.audio_encoder[i](out_spike, out_audio, out_audio)
        # Definition: forward(q, k, v)
        # So q=out_spike, k=out_audio, v=out_audio. 
        # This means EEG (Spike) is the Query, Audio is Key/Value.
        # This computes attention OF EEG ON Audio. (EEG-to-Audio)
        
        # Wait, let's re-read models.py in the thought process/memory.
        # Line 238: out_audio = self.audio_encoder[i](out_spike, out_audio, out_audio)
        # Line 239: out_spike = self.spike_encoder[i](out_audio, out_spike, out_spike)
        
        # Checking ConvCrossAttention forward(q, k, v):
        # q is projected to qs, k to ks. attn = q @ k.T
        # Attention map is (len_q, len_k).
        
        # So for audio_encoder[i]:
        # q=out_spike (EEG), k=out_audio (Audio).
        # Attention map size: (EEG_len, Audio_len).
        # This represents "How much each EEG timepoint attends to each Audio timepoint".
        # This is EEG-to-Audio Attention.
        
        hooks.append(fusion_block.audio_encoder[i].attention.register_forward_hook(
            get_attention_hook(f"Layer_{i}_EEG_Query_Audio_Key")
        ))
        
        # For spike_encoder[i]:
        # q=out_audio (Audio), k=out_spike (EEG).
        # Attention map size: (Audio_len, EEG_len).
        # This represents "How much each Audio timepoint attends to each EEG timepoint".
        # This is Audio-to-EEG Attention (Alignment).
        
        hooks.append(fusion_block.spike_encoder[i].attention.register_forward_hook(
            get_attention_hook(f"Layer_{i}_Audio_Query_EEG_Key")
        ))

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
        
    # Remove hooks
    for h in hooks:
        h.remove()
        
    print(f"Captured {len(attention_maps)} attention maps.")

    # Visualization
    # Group by type
    audio_to_eeg_maps = {k: v for k, v in attention_maps.items() if "Audio_Query_EEG_Key" in k}
    eeg_to_audio_maps = {k: v for k, v in attention_maps.items() if "EEG_Query_Audio_Key" in k}

    # Plot Audio-to-EEG (Alignment)
    # These show for each Audio timepoint, which EEG timepoints are relevant.
    # Expectation: Diagonal structure (Audio at time T aligns with EEG at time T)
    
    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 5 * num_layers))
    if num_layers == 1: axes = [axes]
    
    for i in range(num_layers):
        name = f"Layer_{i}_Audio_Query_EEG_Key"
        attn = audio_to_eeg_maps[name] # Shape: (1, 1, Audio_len, EEG_len)
        # Squeeze batch and head
        attn_np = attn.squeeze().cpu().numpy() # (Audio_len, EEG_len)
        
        ax = axes[i]
        im = ax.imshow(attn_np, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f"Layer {i}: Audio Query -> EEG Key (Alignment)")
        ax.set_ylabel("Audio Time (Queries)")
        ax.set_xlabel("EEG Time (Keys)")
        plt.colorbar(im, ax=ax)
        
    plt.tight_layout()
    plt.savefig('alignment_maps_audio_to_eeg.png')
    print("Saved alignment_maps_audio_to_eeg.png")
    
    # Plot EEG-to-Audio
    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 5 * num_layers))
    if num_layers == 1: axes = [axes]
    
    for i in range(num_layers):
        name = f"Layer_{i}_EEG_Query_Audio_Key"
        attn = eeg_to_audio_maps[name] # Shape: (1, 1, EEG_len, Audio_len)
        attn_np = attn.squeeze().cpu().numpy()
        
        ax = axes[i]
        im = ax.imshow(attn_np, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f"Layer {i}: EEG Query -> Audio Key")
        ax.set_ylabel("EEG Time (Queries)")
        ax.set_xlabel("Audio Time (Keys)")
        plt.colorbar(im, ax=ax)
        
    plt.tight_layout()
    plt.savefig('alignment_maps_eeg_to_audio.png')
    print("Saved alignment_maps_eeg_to_audio.png")

if __name__ == '__main__':
    main()
