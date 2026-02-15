# NeuroCodec: Generative EEG-to-Audio Decoding

**NeuroCodec** is a generative framework for Brain-Assisted Target Speaker Extraction (TSE).
Unlike traditional masking approaches (e.g., DPRNN, M3ANet), NeuroCodec treats speech extraction as a **conditional generation task**.

It leverages a **Frozen Descript Audio Codec (DAC)** to represent audio as discrete tokens and uses a **Causal Mamba Decoder** to predict the target speech tokens from a mixture, guided by EEG signals.

![Architecture](overall.jpg)


## 🧠 Key Features
-   **Generative Approach**: Predicts clean speech tokens directly (Latent-to-Latent).
-   **Frozen Backbone**: Uses the high-fidelity `descript-audio-codec` (44.1kHz).
-   **Efficient Decoder**: Causal Mamba layers for fast, streaming-compatible generation.
-   **High-Res EEG**: Processes 128Hz EEG without downsampling.
-   **Snake Activations**: Replaces ReLU/PReLU with the Snake function for improved waveform fidelity.

## 🎵 Acoustic Guide (New)
The model includes advanced training objectives inspired by the Descript Audio Codec (DAC) paper to ensure high-fidelity audio reconstruction:

1.  **L2-Normalized Latents (Cosine Similarity)**: Forces the model to predict the *direction* of latent vectors rather than magnitude, stabilizing training.
2.  **Multi-Scale Mel-Reconstruction Loss**: Guides the model using the actual audio waveform (decoded from latents) to improve perceptual quality.
3.  **Adversarial Feedback (GAN)**: Uses Multi-Period (MPD), Multi-Scale (MSD), and Multi-Band (MRD) discriminators as external critics to eliminate artifacts and produce realistic speech.
    - *Configurable via `lambda_adv` in `configs/neurocodec.yaml`.*

## 🛠️ Installation

### Prerequisites
-   Python 3.8+
-   PyTorch 1.12+ (CUDA recommended)
-   `ffmpeg` (for audio handling)

### Setup
```bash
# Clone the repository
git clone https://github.com/JulianSilva2001/NeuroCodec.git
cd NeuroCodec

# Install dependencies
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install descript-audio-codec
pip install mamba-ssm causal-conv1d
pip install h5py scipy matplotlib tqdm wandb
```

## 📂 Data Preparation
The model expects HDF5 files containing:
-   **Audio**: 44.1kHz (Noisy Mixture & Clean Target).
-   **EEG**: 128Hz (128 Channels).
-   **Note**: The dataloader handles scaling EEG from Volts to Microvolts automatically.

Expected path structure (configurable via `--root`):
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

## 🚀 Usage

### 1. Training
To train the NeuroCodec model (using the configuration file):
```bash
python train_neurocodec.py --config configs/neurocodec.yaml
```
Key configuration options in `configs/neurocodec.yaml`:
-   `activation`: `'snake'` (Recommended)
-   `normalize_latents`: `true` (Enable Cosine Similarity)
-   `lambda_mel`: `15.0` (Enable Acoustic Guide)
-   `lambda_adv`: `1.0` (Enable Discriminators)
-   `batch_size`: `8` (Adjust for VRAM)

Checkpoints will be saved in `checkpoints/neurocodec/KUL/mod/`.

### 2. Inference & Generation
To generate audio from the test set:
```bash
python inference_neurocodec.py \
  --checkpoint checkpoints/neurocodec/best_model.pth \
  --root /path/to/data \
  --num_samples 5 \
  --gpu 0
```
**Outputs** (saved to `results/NeuroCodec/Inference/`):
-   `input_noisy_*.wav`: The mixed audio input.
-   `prediction_*.wav`: The *NeuroCodec* separated speech.
-   `target_clean_*.wav`: The ground truth target.
-   `inference_plot_*.png`: Spectrogram comparisons.

### 3. Evaluation
To run validation metrics on the full dataset:
```bash
python train_neurocodec.py --evaluate --gpu 0
```
*Note: This computes the Loss and SI-SDR on the validation split.*

### 4. Ablation Studies (Noise Cue)
To check if the model is actually using EEG (and not just performing blind separation), run with the `--noise_cue` flag. This replaces the EEG with random noise matching the channel statistics.
```bash
python train_neurocodec.py --evaluate --noise_cue --gpu 0
```
-   **Expected Result**: A significant drop in performance (e.g., -5dB to -7dB), confirming EEG dependency.

## 📊 Results and Checkpoints
-   **Current Best Model**: Trained for 50 epochs on Normalized-2 dataset.
-   **Codecs**: Uses DAC 44kHz backbone.



