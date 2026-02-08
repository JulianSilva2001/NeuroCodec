import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from M3ANET import M3ANET
from dataset import cock_tail
from tools.calculate_intelligibility import find_intel

def main():
    parser = argparse.ArgumentParser(description="M3ANet Validation Evaluation")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint (.pkl)')
    parser.add_argument('--dataset_root', type=str, default='/media/datasets/AAD_enhance', help='Root directory (default matches usage in dataset.py)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    
    args = parser.parse_args()

    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading model from {args.model_path}")
    try:
        model = M3ANET().to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load Data (Validation Split)
    print("Loading Validation Data...")
    try:
        # mode='val' attempts to load val h5 files defined in dataset.py
        val_data = cock_tail(args.dataset_root, 'val')
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print(f"Validation set loaded. Total samples: {len(val_data)}")
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return

    fs = 14700 # Sampling rate
    si_sdr_scores = []

    print("Starting Evaluation...")
    with torch.no_grad():
        for i, (noisy, eeg, clean) in enumerate(tqdm(val_loader)):
            try:
                noisy, eeg, clean = noisy.to(device), eeg.to(device), clean.to(device)
                
                # Input Chunking (If needed, similar to test.py)
                # test.py assumes batch_size=1 mostly for the loop logic, but let's stick to matching test.py
                # test.py does: noisy = torch.cat(torch.split(noisy, 29184, dim=2), dim=0)
                # This splits along time dimension if it's too long? 
                # Or reshapes? 29184 is the window size.
                
                # If batch_size > 1, this logic might need adjustment if using cat(..., dim=0).
                # The existing code in test.py seems to handle batch_size=1 specifically for the concatenation.
                # If we want to support batch_size > 1, we should be careful. 
                # For safety and consistency with trained model input expectations, forcing batch_size=1 logic inside loop.
                # Actually, if batch_size is 1, dim=0 is batch (1). split on dim=2 (time).
                
                # Process each item in batch if batch_size > 1
                for b in range(noisy.shape[0]):
                    noisy_sample = noisy[b:b+1] # Keep dims (1, C, T)
                    eeg_sample = eeg[b:b+1]
                    clean_sample = clean[b:b+1]

                    # Prepare input
                    # noisy_sample is (1, 1, T). Split on time (dim 2)
                    noisy_in = torch.cat(torch.split(noisy_sample, 29184, dim=2), dim=0) # (Chunks, 1, 29184)
                    eeg_in = torch.cat(torch.split(eeg_sample, 29184, dim=2), dim=0)     # (Chunks, 128, 29184)

                    # Model Forward
                    pred, _, _ = model(noisy_in, eeg_in) # (Chunks, 1, 29184) (?)

                    # Reconstruct
                    pred = torch.cat(torch.split(pred, 1, dim=0), dim=2) # (1, 1, TotalTime)
                    
                    # Convert to numpy for metric calculation
                    pred_np = pred.squeeze().unsqueeze(0).cpu().detach().numpy() # (1, T)
                    clean_np = clean_sample.squeeze().unsqueeze(0).cpu().detach().numpy() # (1, T)

                    # Calculate SI-SDR
                    score = find_intel(clean_np, pred_np, metric='si-sdr', fs=fs)
                    si_sdr_scores.append(score)

            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue

    if len(si_sdr_scores) > 0:
        mean_si_sdr = np.mean(si_sdr_scores)
        print("\n==================================")
        print(f"Validation Set Evaluation Complete")
        print(f"Samples Evaluated: {len(si_sdr_scores)}")
        print(f"Mean SI-SDR: {mean_si_sdr:.4f} dB")
        print("==================================")
    else:
        print("No scores calculated.")

if __name__ == "__main__":
    main()
