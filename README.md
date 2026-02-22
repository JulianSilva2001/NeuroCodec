# NeuroCodec: Generative EEG-to-Audio Decoding

**NeuroCodec** is a generative framework for Brain-Assisted Target Speaker Extraction (TSE).
Unlike traditional masking approaches (e.g., DPRNN, M3ANet), NeuroCodec treats speech extraction as a **conditional generation task**.

It leverages a **Frozen Descript Audio Codec (DAC)** to represent audio as discrete tokens and uses a **Causal Mamba Decoder** to predict the target speech tokens from a noisy mixture, guided by EEG signals.

![Architecture](overall.jpg)

---

## Key Features

- **Generative Approach**: Predicts clean speech latents directly (Latent-to-Latent) rather than masking.
- **Frozen Backbone**: Uses the high-fidelity `descript-audio-codec` (44.1kHz / 16kHz) — no codec training required.
- **Causal Streaming**: Processes audio in causal hops; the model never looks ahead, enabling real-time use.
- **Two-Pass Speaker Encoder**: First pass extracts a pseudo-clean reference (no speaker cue); second pass conditions on it for refinement. 20% speaker-dropout forces EEG-only cold-start robustness.
- **Combined Perceptual Loss**: MSE on DAC latents + multi-scale MelSpectrogram L1 loss for audio-space supervision.
- **Speaker Encoder Ablation**: Validation runs in both normal and EEG-only modes each epoch, reporting the SI-SDR gain from the speaker encoder.
- **High-Res EEG**: Processes 128-channel EEG at 128Hz without downsampling.

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+ with CUDA
- `ffmpeg`

### Setup
```bash
git clone https://github.com/JulianSilva2001/NeuroCodec.git
cd NeuroCodec

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install descript-audio-codec
pip install mamba-ssm causal-conv1d
pip install h5py scipy matplotlib tqdm wandb
```

---

## Data Preparation

The model expects HDF5 or LMDB files containing:
- **Audio**: 16kHz (Noisy Mixture & Clean Target)
- **EEG**: 128Hz, 128 channels
- EEG is automatically scaled from Volts to Microvolts in the dataloader

Supported datasets: `kul` (KUL LMDB) and `cocktail` (HDF5).

Expected HDF5 path structure (configurable via `--root`):
```
data/
  train/
    noisy_train.h5
    clean_train.h5
    eegs_train.h5
  test/
    noisy_test.h5
    clean_test.h5
    eegs_test.h5
```

---

## Usage

### Online (Causal Streaming) Training

The primary training script is `train_online.py`, which implements the causal sliding-window two-pass strategy.

```bash
python train_online.py \
  --root /path/to/dataset \
  --dataset kul \
  --batch_size 16 \
  --lr 5e-4 \
  --epochs 30 \
  --hidden_dim 256 \
  --num_layers 4 \
  --hop_sec 0.5 \
  --window_sec 2.0 \
  --lambda_latent 1.0 \
  --lambda_mel 0.1 \
  --gpu 0
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--root` | `/workspace/...` | Path to dataset root |
| `--dataset` | `kul` | Dataset format: `kul` or `cocktail` |
| `--batch_size` | `64` | Training batch size |
| `--lr` | `5e-4` | Initial learning rate |
| `--epochs` | `15` | Number of training epochs |
| `--hop_sec` | `0.5` | Causal hop size in seconds |
| `--window_sec` | `2.0` | Context window size in seconds |
| `--lambda_latent` | `1.0` | Weight for latent MSE loss |
| `--lambda_mel` | `0.1` | Weight for MelSpectrogram perceptual loss |
| `--pretrained` | `""` | Path to checkpoint to resume from |
| `--checkpoint_dir` | `checkpoints/neurocodec_online` | Where to save checkpoints |
| `--hidden_dim` | `256` | Mamba hidden dimension |
| `--num_layers` | `4` | Number of Mamba layers |
| `--eeg_channels` | `128` | Number of EEG channels |
| `--debug` | flag | Run with a small subset for quick iteration |

Checkpoints are saved to `--checkpoint_dir` as `best_model.pth` and `last_model.pth`.

#### Training Loss

The total loss per hop is:

```
L_total = lambda_latent * MSE(z_pred, z_target) + lambda_mel * MelSpec(decode(z_pred), clean_hop)
```

- **MSE** on DAC latents: mathematical alignment in codec space.
- **MelSpectrogram L1** (multi-scale, log-mel): perceptual supervision in audio space. Gradients backpropagate through the frozen DAC decoder into the trainable model without updating decoder weights.
- Set `--lambda_mel 0.0` to disable MelSpec loss entirely and fall back to MSE-only (zero overhead — the decoder is not called).

Memory note: the decoder forward graph for each hop is freed immediately via per-hop `.backward()`, keeping peak VRAM to one hop's graph regardless of sequence length.

#### Resuming from a Checkpoint

```bash
python train_online.py --pretrained checkpoints/neurocodec_online/last_model.pth [... other args]
```

> **Note on learning rate when resuming**: The optimizer state (including LR) is restored from the checkpoint. If you want to change the LR for a resumed run, patch the checkpoint before restarting:
> ```python
> import torch
> ckpt = torch.load("checkpoints/neurocodec_online/last_model.pth", map_location="cpu")
> for pg in ckpt['optimizer']['param_groups']:
>     pg['lr'] = 5e-4   # your new LR
> ckpt['scheduler']['num_bad_epochs'] = 0
> ckpt['scheduler']['best'] = ckpt['best_val_loss']
> torch.save(ckpt, "checkpoints/neurocodec_online/last_model.pth")
> ```

#### Training Output Format

```
Epoch  21 | Train 0.0312 | Val 0.0289 | Cold 0.0301 | Steady 0.0271 | SI-SDR 8.43 dB (EEG-only 7.91 dB, gain +0.52 dB)
```

- **Cold**: validation loss on the first hop (no prior context)
- **Steady**: validation loss on later hops (context accumulated)
- **SI-SDR**: full-sequence SI-SDR with speaker encoder active
- **EEG-only**: SI-SDR when speaker encoder is disabled (all hops use `force_no_speaker=True`)
- **gain**: `SI-SDR − EEG-only SI-SDR` — positive values confirm the speaker encoder is contributing

---

### Offline Baseline Training

To train the original offline (batch) model:

```bash
python train_neurocodec.py \
  --root /path/to/data \
  --batch_size 16 \
  --lr 1e-3 \
  --epochs 50 \
  --gpu 0
```

Checkpoints are saved to `checkpoints/neurocodec/`.

---

### Inference

Generate separated audio from the test set:

```bash
python inference_neurocodec.py \
  --checkpoint checkpoints/neurocodec_online/best_model.pth \
  --root /path/to/data \
  --num_samples 5 \
  --gpu 0
```

Outputs saved to `results/NeuroCodec/Inference/`:
- `input_noisy_*.wav`: mixed audio input
- `prediction_*.wav`: NeuroCodec separated speech
- `target_clean_*.wav`: ground truth target
- `inference_plot_*.png`: spectrogram comparisons

---

### Ablation: EEG Dependency Check

To verify the model is using EEG rather than performing blind separation, replace EEG with random noise matching channel statistics:

```bash
python train_neurocodec.py --evaluate --noise_cue --gpu 0
```

Expected result: significant SI-SDR drop (−5 to −7 dB), confirming EEG dependency.

The online training script reports an equivalent in-training ablation each epoch via the **EEG-only gain** metric (see output format above).

---

## Project Structure

```
NeuroCodec/
├── train_online.py          # Online (causal streaming) training — primary script
├── train_neurocodec.py      # Offline baseline training
├── inference_neurocodec.py  # Inference / audio generation
├── losses.py                # MelSpectrogramLoss + other losses
├── dataset_neurocodec.py    # HDF5 / LMDB dataloaders
├── models/
│   ├── neurocodec_online.py # OnlineNeuroCodec (two-pass, causal Mamba)
│   └── ...
├── kernels/                 # Custom CUDA kernels (selective scan)
└── checkpoints/
    └── neurocodec_online/   # Saved checkpoints
```

---

## Acknowledgements

Based on the **M3ANet** architecture and the **NeuroHeed** two-pass speaker extraction strategy.
Uses the [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) as a frozen speech tokenizer.
