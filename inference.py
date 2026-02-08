import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import warnings

# Suppress potential warnings for cleaner output
warnings.filterwarnings("ignore")

# Add current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from M3ANET import M3ANET
from dataset import cock_tail
from tools.plotting import save_wav, one_plot_test
from tools.calculate_intelligibility import find_intel

def main():
    parser = argparse.ArgumentParser(description="M3ANet Inference Script")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint (.pkl)')
    parser.add_argument('--dataset_root', type=str, default='/media/datasets/AAD_enhance', help='Root directory of the dataset (default matches test.py config)')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Directory to save results')
    parser.add_argument('--subject', type=int, default=1, help='Subject ID to test (default: 1)')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to process (default: 5)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use (default: 0)')
    parser.add_argument('--validate', action='store_true', help='Calculate and print metrics (SI-SDR, STOI, PESQ)')
    parser.add_argument('--noise_cue', action='store_true', help='Use random noise instead of real EEG as cue')
    
    args = parser.parse_args()

    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

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
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load Data
    print(f"Loading data for subject {args.subject}...")
    try:
        # Note: dataset.py relies on hardcoded paths inside. 
        # We pass args.dataset_root mostly to satisfy the init signature if needed, or if it uses it.
        test_data = cock_tail(args.dataset_root, 'test', subject=args.subject)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        print(f"Data loaded. Total samples available: {len(test_data)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check your dataset paths in dataset.py")
        return

    print(f"Starting inference on {min(len(test_loader), args.num_samples)} samples...")
    
    fs = 14700 # Sampling rate from M3ANET.py (L1 calculation)

    # Metrics storage
    metrics_acc = {'si-sdr': [], 'stoi': [], 'pesq': []}

    with torch.no_grad():
        for batch_idx, (noisy, eeg, clean) in enumerate(tqdm(test_loader, total=min(len(test_loader), args.num_samples))):
            if batch_idx >= args.num_samples:
                break
            
            try:
                noisy, eeg, clean = noisy.to(device), eeg.to(device), clean.to(device)

                if args.noise_cue:
                    # Generate noise with same statistics as the current EEG batch
                    mean = eeg.mean()
                    std = eeg.std()
                    eeg = torch.randn_like(eeg) * std + mean
                
                # Keep originals for saving/plotting (move to CPU/Numpy)
                noisy_snd = noisy.squeeze().unsqueeze(0).cpu().detach().numpy()
                clean_snd = clean.squeeze().unsqueeze(0).cpu().detach().numpy()
                
                # Chunking / Reshaping as per test.py
                # This ensures input dimensions match what the model expects (likely 2s window)
                noisy_in = torch.cat(torch.split(noisy, 29184, dim=2), dim=0)
                eeg_in = torch.cat(torch.split(eeg, 29184, dim=2), dim=0)

                # Inference
                pred, _, _ = model(noisy_in, eeg_in)
                
                # Reconstruct
                pred = torch.cat(torch.split(pred, 1, dim=0), dim=2)
                pred = pred.squeeze().unsqueeze(0).cpu().detach().numpy()
                
                # Validation / Metrics
                if args.validate:
                    for m in metrics_acc.keys():
                        try:
                            # clean_snd and pred are numpy arrays (1, T) or similar
                            # find_intel expects shapes to match and be (1, T)?
                            # Let's ensure they are 1D or (1, T) as needed. 
                            # test.py passes 'clean' and 'pred' which are (1, T) numpy arrays.
                            val = find_intel(clean_snd, pred, metric=m, fs=fs)
                            metrics_acc[m].append(val)
                        except Exception as e_metric:
                            print(f"Warning: Failed to calc {m}: {e_metric}")

                # Define paths
                fig_path = os.path.join(
                    args.output_dir,
                    f'prediction_b{batch_idx}_s{args.subject}.png'
                )
                
                # Visualization
                # Visualization
                plot_title = f'Subject {args.subject} - Sample {batch_idx}'
                if args.noise_cue:
                    plot_title += ' (Noise Cue)'
                
                one_plot_test(pred, clean_snd, noisy_snd, plot_title, fig_path)
                
                # Audio Saving
                # save_wav saves 3 files: clean, noisy, and prediction
                gen_type = 'test_noise' if args.noise_cue else 'test'
                save_wav(pred, noisy_snd, clean_snd, 'inference', batch_idx, fs, args.output_dir, subject=args.subject, generator_type=gen_type)
            
            except Exception as e:
                print(f"Error processing sample {batch_idx}: {e}")
                continue

    if args.validate and len(metrics_acc['si-sdr']) > 0:
        print("\n=== Validation Results ===")
        for m, vals in metrics_acc.items():
            avg_val = np.mean(vals)
            print(f"Mean {m.upper()}: {avg_val:.4f}")
        print("==========================")
    
    print(f"Inference completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
