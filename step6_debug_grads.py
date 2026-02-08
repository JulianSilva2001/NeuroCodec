
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from models.neurocodec import NeuroCodec
from dataset_neurocodec import load_NeuroCodecDataset

def check_grads(args):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checking Gradients on {device}...")

    # Load small data
    dataloader = load_NeuroCodecDataset(root=args.root, subset='train', batch_size=2)
    
    model = NeuroCodec(dac_model_type='44khz').to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3) # Aggressive LR
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    # Get one batch
    noisy, eeg, clean = next(iter(dataloader))
    noisy, eeg, clean = noisy.to(device), eeg.to(device), clean.to(device)
    
    # Target
    with torch.no_grad():
        _, target_codes, _, _, _ = model.dac.encode(clean)
        
    logits, _, _, _ = model(noisy, eeg)
    
    loss = 0
    for k in range(logits.shape[1]):
        loss += criterion(logits[:, k, :, :], target_codes[:, k, :])
        
    print(f"Initial Loss: {loss.item()}")
    
    optimizer.zero_grad()
    loss.backward()
    
    print("\nGradient Norms:")
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            if param_norm > 1.0: # Print only significant ones
                print(f"  {name}: {param_norm:.4f}")
        else:
            if param.requires_grad:
                print(f"  {name}: NO GRAD (WARNING)")
                
    total_norm = total_norm ** 0.5
    print(f"\nTotal Gradient Norm: {total_norm:.4f}")
    
    if total_norm < 1e-4:
        print("CRITICAL: Gradients are Vanishing!")
    else:
        print("Gradients look OK.")
        
    optimizer.step()
    
    # Check if loss decreases
    logits, _, _, _ = model(noisy, eeg)
    loss_new = 0
    for k in range(logits.shape[1]):
        loss_new += criterion(logits[:, k, :, :], target_codes[:, k, :])
        
    print(f"Loss after 1 step: {loss_new.item()} (Delta: {loss.item() - loss_new.item()})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/jaliya/eeg_speech/navindu/data/Cocktail_Party/Normalized/2s/eeg/new')
    args = parser.parse_args()
    check_grads(args)
