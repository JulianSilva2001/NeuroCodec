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
    checkpoint_path = '/home/jaliya/eeg_speech/Julian/M3ANet-main/exp/M3ANET/checkpoint/115000.pkl'
    # Use the path from dataset.py logic, but we need to pass something to init
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
    print(f"Dataset length: {len(dataset)}")
    
    # Get one sample
    noisy, eeg_real, clean = dataset[0]
    
    # Preprocess sample
    noisy = noisy.unsqueeze(0).to(device) # (1, 1, T)
    eeg_real = eeg_real.unsqueeze(0).to(device)     # (1, 128, T)
    clean = clean.unsqueeze(0).to(device) # (1, 1, T)
    
    # Generate Noise Input
    print("Generating noise input...")
    # Gaussian noise with same mean/std as real EEG roughly, or just standard normal
    # Let's use similar statistics to real EEG to make it a fair comparison amplitude-wise
    eeg_mean = eeg_real.mean()
    eeg_std = eeg_real.std()
    eeg_noise = torch.randn_like(eeg_real) * eeg_std + eeg_mean
    
    inputs = {
        'Real EEG': eeg_real,
        'Noise': eeg_noise
    }

    # Helper to capture activations
    def capture_activations(input_eeg, input_noisy):
        activations = OrderedDict()
        hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        # Register hooks
        model_part = model.spike_encoder
        
        hooks.append(model_part.BN1.register_forward_hook(get_activation('0_BN1')))
        hooks.append(model_part.layer1.register_forward_hook(get_activation('1_Chebynet')))
        hooks.append(model_part.projection.register_forward_hook(get_activation('2_Projection')))
        
        sequential_block = model_part.eeg_encoder
        hooks.append(sequential_block[0].register_forward_hook(get_activation('3_ChannelwiseLayerNorm')))
        hooks.append(sequential_block[1].register_forward_hook(get_activation('4_Conv1D_1')))
        hooks.append(sequential_block[2].register_forward_hook(get_activation('5_ResBlock_1')))
        hooks.append(sequential_block[3].register_forward_hook(get_activation('6_ResBlock_2')))
        hooks.append(sequential_block[4].register_forward_hook(get_activation('7_ResBlock_3')))
        hooks.append(sequential_block[5].register_forward_hook(get_activation('8_Conv1D_2')))

        # Run forward
        with torch.no_grad():
             # M3ANET(input, spike_input)
            _ = model(input_noisy, input_eeg)
            
        # Remove hooks
        for h in hooks:
            h.remove()
            
        return activations

    # Run for both inputs
    print("Running forward pass for Real EEG...")
    activations_real = capture_activations(inputs['Real EEG'], noisy)
    
    print("Running forward pass for Noise Input...")
    activations_noise = capture_activations(inputs['Noise'], noisy)

    print("Generating comparison visualization...")
    
    # Visualization
    # We want 2 columns: Real, Noise
    # Rows: Input + Layers
    
    layer_names = list(activations_real.keys())
    num_rows = len(layer_names) + 1 # +1 for input
    num_cols = 2
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))
    
    # 1. Plot Inputs
    input_data = [inputs['Real EEG'], inputs['Noise']]
    titles = ['Real EEG', 'Noise Input']
    
    for col in range(num_cols):
        ax = axes[0, col]
        data = input_data[col].squeeze().cpu().numpy()
        im = ax.imshow(data, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f"Input: {titles[col]}")
        ax.set_ylabel("Channels")
        ax.set_xlabel("Time")
        plt.colorbar(im, ax=ax)
        
    # 2. Plot Layers
    for i, name in enumerate(layer_names):
        row = i + 1
        
        # Real
        ax = axes[row, 0]
        act = activations_real[name]
        if act.dim() == 3: act_np = act[0].cpu().numpy()
        elif act.dim() == 2: act_np = act.cpu().numpy()
        else: act_np = np.zeros((10, 10)) # Fallback
        
        im = ax.imshow(act_np, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f"Real - {name} {act_np.shape}")
        ax.set_ylabel("Ch/Feats")
        ax.set_xlabel("Time")
        plt.colorbar(im, ax=ax)
        
        # Noise
        ax = axes[row, 1]
        act = activations_noise[name]
        if act.dim() == 3: act_np = act[0].cpu().numpy()
        elif act.dim() == 2: act_np = act.cpu().numpy()
        else: act_np = np.zeros((10, 10))
        
        im = ax.imshow(act_np, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f"Noise - {name} {act_np.shape}")
        ax.set_ylabel("Ch/Feats")
        ax.set_xlabel("Time")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    save_path_png = 'eeg_encoder_comparison.png'
    save_path_pdf = 'eeg_encoder_comparison.pdf'
    plt.savefig(save_path_png)
    plt.savefig(save_path_pdf)
    print(f"Comparison saved to {os.path.abspath(save_path_png)} and {os.path.abspath(save_path_pdf)}")

if __name__ == '__main__':
    main()
