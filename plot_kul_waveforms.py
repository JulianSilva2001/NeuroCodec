import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset_neurocodec import load_KUL_NeuroCodecDataset


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_channels(value):
    channels = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        channels.append(int(item))
    if not channels:
        raise ValueError("At least one EEG channel must be provided.")
    return channels


def to_numpy_1d(tensor):
    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu()
    arr = np.asarray(tensor).squeeze()
    return arr.astype(np.float32)


def plot_sample(mix, clean, eeg, subject, index, channels, target_fs, output_path):
    mix_np = to_numpy_1d(mix)
    clean_np = to_numpy_1d(clean)
    eeg_np = np.asarray(eeg.detach().cpu()).squeeze().astype(np.float32)

    if eeg_np.ndim != 2:
        raise ValueError(f"Expected EEG shape (channels, time), got {eeg_np.shape}")

    duration = len(clean_np) / float(target_fs)
    audio_time = np.arange(len(clean_np)) / float(target_fs)
    eeg_fs = eeg_np.shape[-1] / duration if duration > 0 else 1.0
    eeg_time = np.arange(eeg_np.shape[-1]) / eeg_fs

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=False,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.6]},
    )

    audio_len = min(len(audio_time), len(mix_np), len(clean_np))
    interferer_np = mix_np[:audio_len] - clean_np[:audio_len]
    axes[0].plot(audio_time[:audio_len], mix_np[:audio_len], color="#7E57C2", linewidth=0.9, label="Mixture")
    axes[0].plot(audio_time[:audio_len], interferer_np, color="#E53935", linewidth=0.9, alpha=0.85, label="Interferer ~= mix - target")
    axes[0].set_title(f"KUL sample {index} | subject: {subject}")
    axes[0].set_ylabel("Mix amp.")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(audio_time[:audio_len], clean_np[:audio_len], color="#1E88E5", linewidth=1.0)
    axes[1].set_title("Target audio")
    axes[1].set_ylabel("Target amp.")
    axes[1].grid(alpha=0.25)

    valid_channels = [ch for ch in channels if 0 <= ch < eeg_np.shape[0]]
    if len(valid_channels) != len(channels):
        invalid = sorted(set(channels) - set(valid_channels))
        print(f"Warning: skipped invalid EEG channels for sample {index}: {invalid}")

    if not valid_channels:
        raise ValueError(f"No valid EEG channels for EEG shape {eeg_np.shape}")

    spacing = 0.0
    normalized_channels = []
    for ch in valid_channels:
        channel = eeg_np[ch]
        channel = channel - channel.mean()
        scale = channel.std() + 1e-8
        normalized_channels.append(channel / scale)
    spacing = max(4.0, max(np.max(np.abs(ch)) for ch in normalized_channels) * 1.8)

    for row, (ch, channel) in enumerate(zip(valid_channels, normalized_channels)):
        offset = (len(valid_channels) - row - 1) * spacing
        axes[2].plot(eeg_time, channel + offset, linewidth=0.9, label=f"Ch {ch}")
        axes[2].text(eeg_time[-1] + 0.02 * duration, offset, f"Ch {ch}", va="center", fontsize=9)

    axes[2].set_title("Selected EEG channels, z-scored and vertically offset")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("EEG channels")
    axes[2].set_yticks([])
    axes[2].grid(alpha=0.25)
    axes[2].set_xlim(0, max(duration, eeg_time[-1] if len(eeg_time) else duration))

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    default_config = os.path.join(os.path.dirname(__file__), "configs", "train_kul.json")

    parser = argparse.ArgumentParser(description="Plot KUL mixture, target, and selected EEG channel waveforms.")
    parser.add_argument("--config", type=str, default=default_config, help="Path to KUL training config JSON.")
    parser.add_argument("--root", type=str, default=None, help="Override KUL LMDB path.")
    parser.add_argument("--subset", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to plot.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to plot.")
    parser.add_argument("--channels", type=str, default="0,8,16,24,32,40,48,56", help="Comma-separated EEG channel indices.")
    parser.add_argument("--target_fs", type=int, default=16000, help="Audio sampling rate used by the KUL loader.")
    parser.add_argument("--original_fs", type=int, default=16000, help="Original audio sampling rate stored in the LMDB.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle samples before plotting.")
    parser.add_argument("--output_dir", type=str, default="results/kul_waveforms", help="Directory to save plots.")
    args = parser.parse_args()

    config = load_config(args.config)
    lmdb_path = args.root or config["root"]
    channels = parse_channels(args.channels)

    loader = load_KUL_NeuroCodecDataset(
        lmdb_path=lmdb_path,
        subset=args.subset,
        batch_size=1,
        num_gpus=1,
        target_fs=args.target_fs,
        original_fs=args.original_fs,
        shuffle=args.shuffle,
        return_subject=True,
        return_index=True,
    )

    for sample_idx, batch in enumerate(loader):
        if sample_idx >= args.num_samples:
            break

        mix, eeg, clean, subject, index = batch
        subject_value = subject[0] if isinstance(subject, (list, tuple)) else subject
        if torch.is_tensor(index):
            index_value = int(index.squeeze().item())
        else:
            index_value = int(index[0]) if isinstance(index, (list, tuple)) else int(index)

        output_path = os.path.join(args.output_dir, f"{args.subset}_sample_{sample_idx:03d}_idx_{index_value}.png")
        plot_sample(
            mix=mix[0],
            clean=clean[0],
            eeg=eeg[0],
            subject=subject_value,
            index=index_value,
            channels=channels,
            target_fs=args.target_fs,
            output_path=output_path,
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
