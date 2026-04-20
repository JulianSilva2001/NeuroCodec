"""Shared utilities for NeuroCodec explainability experiments.

This module is intentionally standalone and evaluation-focused so it can be
used without modifying training code.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import signal, stats

from dataset_neurocodec import load_KUL_NeuroCodecDataset, load_NeuroCodecDataset
from models.neurocodec import NeuroCodec


HIGHER_IS_BETTER_METRICS = {"si_sdr", "stoi", "estoi", "pesq"}
LOWER_IS_BETTER_METRICS = {"latent_mse"}

EEG_BANDS_HZ = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass
class ModelRuntimeConfig:
    dataset: str
    target_fs: int
    eeg_channels: int
    dac_model_type: str


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_csv_floats(text: str) -> List[float]:
    if not text.strip():
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text: str) -> List[str]:
    if not text.strip():
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def infer_runtime_config(dataset: str, eeg_channels_override: Optional[int] = None) -> ModelRuntimeConfig:
    if dataset == "kul":
        return ModelRuntimeConfig(
            dataset="kul",
            target_fs=16000,
            eeg_channels=eeg_channels_override or 64,
            dac_model_type="16khz",
        )
    return ModelRuntimeConfig(
        dataset="cocktail",
        target_fs=44100,
        eeg_channels=eeg_channels_override or 128,
        dac_model_type="44khz",
    )


def resolve_device(gpu: int) -> torch.device:
    return torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")


def load_model_from_checkpoint(
    checkpoint_path: str,
    runtime: ModelRuntimeConfig,
    hidden_dim: int,
    num_layers: int,
    backbone: str,
    activation: str,
    device: torch.device,
) -> NeuroCodec:
    model = NeuroCodec(
        dac_model_type=runtime.dac_model_type,
        eeg_in_channels=runtime.eeg_channels,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        backbone=backbone,
        activation=activation,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and any(k.startswith("module.") for k in checkpoint):
        checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


def build_dataloader(
    dataset: str,
    root: str,
    subset: str,
    batch_size: int,
    shuffle: bool,
    target_fs: int,
    original_fs: int,
) -> torch.utils.data.DataLoader:
    if dataset == "kul":
        return load_KUL_NeuroCodecDataset(
            lmdb_path=root,
            subset=subset,
            batch_size=batch_size,
            num_gpus=1,
            target_fs=target_fs,
            original_fs=original_fs,
            shuffle=shuffle,
        )
    return load_NeuroCodecDataset(
        root=root,
        subset=subset,
        batch_size=batch_size,
        num_gpus=1,
        shuffle=shuffle,
    )


def parse_batch(
    batch: Any,
    device: torch.device,
) -> Dict[str, Optional[torch.Tensor]]:
    if not isinstance(batch, (list, tuple)):
        raise TypeError("Batch must be tuple/list. Adapt parse_batch for dict-style batches.")
    if len(batch) < 3:
        raise ValueError("Expected at least (mixture, eeg, target) in each batch.")

    mixture = batch[0].to(device)
    eeg = batch[1].to(device)
    target = batch[2].to(device)
    unattended = batch[3].to(device) if len(batch) >= 4 and torch.is_tensor(batch[3]) else None

    if mixture.ndim == 2:
        mixture = mixture.unsqueeze(1)
    if target.ndim == 2:
        target = target.unsqueeze(1)

    if eeg.ndim != 3:
        raise ValueError(f"Expected EEG tensor with 3 dims (B,C,T), got shape {tuple(eeg.shape)}")

    if mixture.ndim != 3 or target.ndim != 3:
        raise ValueError(
            f"Audio tensors must be (B,1,T). mixture={tuple(mixture.shape)}, target={tuple(target.shape)}"
        )

    return {
        "mixture": mixture.float(),
        "eeg": eeg.float(),
        "target": target.float(),
        "unattended": unattended.float() if unattended is not None else None,
    }


def trim_to_common_length(*tensors: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:
    valid = [t for t in tensors if t is not None]
    min_len = min(t.shape[-1] for t in valid)
    out = []
    for t in tensors:
        out.append(t[..., :min_len] if t is not None else None)
    return out


def sisdr_np(reference: np.ndarray, estimation: np.ndarray) -> float:
    reference = np.asarray(reference)
    estimation = np.asarray(estimation)
    if reference.ndim != 1 or estimation.ndim != 1:
        raise ValueError("sisdr_np expects 1D arrays.")

    reference_energy = np.sum(reference ** 2) + 1e-8
    alpha = np.sum(reference * estimation) / reference_energy
    projection = alpha * reference
    noise = estimation - projection
    value = 10.0 * np.log10((np.sum(projection ** 2) + 1e-8) / (np.sum(noise ** 2) + 1e-8))
    return float(value)


def align_1d_signals(reference: np.ndarray, estimate: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    reference = np.asarray(reference).squeeze()
    estimate = np.asarray(estimate).squeeze()

    corr = signal.correlate(reference - reference.mean(), estimate - estimate.mean(), mode="full", method="fft")
    lags = signal.correlation_lags(len(reference), len(estimate), mode="full")
    lag = int(lags[np.argmax(corr)])

    if lag < 0:
        est_aligned = estimate[-lag:]
        ref_aligned = reference[: len(est_aligned)]
    else:
        est_aligned = estimate[: max(0, len(reference) - lag)]
        ref_aligned = reference[lag : lag + len(est_aligned)]

    min_len = min(len(ref_aligned), len(est_aligned))
    return ref_aligned[:min_len], est_aligned[:min_len], lag


def _safe_metric_stoi(clean: np.ndarray, pred: np.ndarray, sr: int, extended: bool) -> float:
    try:
        from pystoi import stoi

        return float(stoi(clean, pred, sr, extended=extended))
    except Exception:
        return float("nan")


def _safe_metric_pesq(clean: np.ndarray, pred: np.ndarray, sr: int) -> float:
    try:
        from pesq import pesq

        mode = "wb" if sr >= 16000 else "nb"
        pesq_sr = 16000 if mode == "wb" else 8000
        if sr != pesq_sr:
            clean_rs = signal.resample_poly(clean, pesq_sr, sr)
            pred_rs = signal.resample_poly(pred, pesq_sr, sr)
        else:
            clean_rs = clean
            pred_rs = pred
        min_len = min(len(clean_rs), len(pred_rs))
        return float(pesq(pesq_sr, clean_rs[:min_len], pred_rs[:min_len], mode))
    except Exception:
        return float("nan")


def run_model_once(
    model: NeuroCodec,
    mixture: torch.Tensor,
    eeg: torch.Tensor,
    target: Optional[torch.Tensor] = None,
) -> Dict[str, Optional[torch.Tensor]]:
    with torch.no_grad():
        out = model(mixture, eeg)

        if isinstance(out, tuple):
            z_pred = out[0]
            last_attn = out[4] if len(out) > 4 else None
        else:
            z_pred = out
            last_attn = None

        pred_audio: Optional[torch.Tensor]
        if z_pred.ndim == 3 and z_pred.shape[1] != 1:
            z_q = model.dac.quantizer(z_pred, n_quantizers=9)[0]
            pred_audio = model.dac.decode(z_q)
        else:
            pred_audio = z_pred

        target_latent = None
        if target is not None and target.ndim == 3:
            with torch.no_grad():
                target_latent, _, _, _, _ = model.dac.encode(target)

    return {
        "pred_audio": pred_audio,
        "pred_latent": z_pred,
        "target_latent": target_latent,
        "attention": last_attn,
    }


def compute_batch_metrics(
    mixture: torch.Tensor,
    target: torch.Tensor,
    pred_audio: torch.Tensor,
    pred_latent: Optional[torch.Tensor],
    target_latent: Optional[torch.Tensor],
    sr: int,
    metric_names: Sequence[str],
    align_signals: bool = True,
) -> List[Dict[str, float]]:
    mixture, target, pred_audio = trim_to_common_length(mixture, target, pred_audio)
    metric_names = [m.lower() for m in metric_names]

    batch_size = pred_audio.shape[0]
    rows: List[Dict[str, float]] = []

    for b in range(batch_size):
        mix_np = mixture[b, 0].detach().cpu().numpy()
        tgt_np = target[b, 0].detach().cpu().numpy()
        pred_np = pred_audio[b, 0].detach().cpu().numpy()

        if align_signals:
            tgt_aligned, pred_aligned, lag = align_1d_signals(tgt_np, pred_np)
        else:
            min_len = min(len(tgt_np), len(pred_np))
            tgt_aligned, pred_aligned, lag = tgt_np[:min_len], pred_np[:min_len], 0

        row: Dict[str, float] = {"lag": float(lag)}

        if "si_sdr" in metric_names:
            row["si_sdr"] = sisdr_np(tgt_aligned, pred_aligned)
            mix_len = min(len(mix_np), len(tgt_aligned))
            row["si_sdr_input"] = sisdr_np(tgt_aligned[:mix_len], mix_np[:mix_len])
            row["si_sdr_improvement"] = row["si_sdr"] - row["si_sdr_input"]

        if "stoi" in metric_names:
            row["stoi"] = _safe_metric_stoi(tgt_aligned, pred_aligned, sr=sr, extended=False)

        if "estoi" in metric_names:
            row["estoi"] = _safe_metric_stoi(tgt_aligned, pred_aligned, sr=sr, extended=True)

        if "pesq" in metric_names:
            row["pesq"] = _safe_metric_pesq(tgt_aligned, pred_aligned, sr=sr)

        if "latent_mse" in metric_names and pred_latent is not None and target_latent is not None:
            pred_z = pred_latent[b]
            tgt_z = target_latent[b]
            z_len = min(pred_z.shape[-1], tgt_z.shape[-1])
            row["latent_mse"] = float(F.mse_loss(pred_z[..., :z_len], tgt_z[..., :z_len]).item())

        rows.append(row)

    return rows


def summarize_metrics(df: pd.DataFrame, group_cols: Sequence[str], metric_cols: Sequence[str]) -> pd.DataFrame:
    grouped = df.groupby(list(group_cols), dropna=False)
    summary = grouped[list(metric_cols)].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
    return summary


def run_paired_tests(
    df: pd.DataFrame,
    baseline_condition: str,
    compare_conditions: Sequence[str],
    metric_cols: Sequence[str],
    sample_id_col: str = "sample_id",
    condition_col: str = "condition",
) -> pd.DataFrame:
    rows = []
    base_df = df[df[condition_col] == baseline_condition]

    for cond in compare_conditions:
        cond_df = df[df[condition_col] == cond]
        merged = base_df[[sample_id_col] + list(metric_cols)].merge(
            cond_df[[sample_id_col] + list(metric_cols)], on=sample_id_col, suffixes=("_base", "_cond")
        )

        for metric in metric_cols:
            a = merged[f"{metric}_base"].to_numpy()
            b = merged[f"{metric}_cond"].to_numpy()
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() < 3:
                t_p = np.nan
                w_p = np.nan
            else:
                _, t_p = stats.ttest_rel(a[valid], b[valid], nan_policy="omit")
                try:
                    _, w_p = stats.wilcoxon(a[valid], b[valid])
                except Exception:
                    w_p = np.nan

            rows.append(
                {
                    "baseline": baseline_condition,
                    "condition": cond,
                    "metric": metric,
                    "n": int(valid.sum()),
                    "ttest_pvalue": float(t_p) if np.isfinite(t_p) else np.nan,
                    "wilcoxon_pvalue": float(w_p) if np.isfinite(w_p) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def save_dataframe(df: pd.DataFrame, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    df.to_csv(out_path, index=False)


def plot_metric_bar(
    summary_df: pd.DataFrame,
    metric: str,
    condition_col: str,
    out_path: str,
    title: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    m_col = f"{metric}_mean"
    s_col = f"{metric}_std"
    if m_col not in summary_df.columns:
        return

    x = np.arange(len(summary_df))
    labels = summary_df[condition_col].astype(str).tolist()
    y = summary_df[m_col].to_numpy(dtype=float)
    yerr = summary_df[s_col].to_numpy(dtype=float) if s_col in summary_df.columns else None

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, y, yerr=yerr, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by condition")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()

    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def zero_eeg(eeg: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(eeg)


def shuffle_eeg_across_batch(eeg: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    bsz = eeg.shape[0]
    if bsz <= 1:
        return eeg.clone()
    # torch.randperm requires generator and output device to match.
    # Keep deterministic behavior with CPU generators by sampling on CPU,
    # then moving indices to EEG device.
    if generator is not None and eeg.device.type != "cpu":
        perm = torch.randperm(bsz, generator=generator, device="cpu").to(eeg.device)
    else:
        perm = torch.randperm(bsz, generator=generator, device=eeg.device)
    return eeg[perm]


def shuffle_eeg_time_within_sample(eeg: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    out = eeg.clone()
    bsz, _, t = out.shape
    for i in range(bsz):
        # Same generator/device compatibility guard as batch shuffle.
        if generator is not None and eeg.device.type != "cpu":
            perm = torch.randperm(t, generator=generator, device="cpu").to(eeg.device)
        else:
            perm = torch.randperm(t, generator=generator, device=eeg.device)
        out[i] = out[i, :, perm]
    return out


def shift_eeg(eeg: torch.Tensor, shift_samples: int, pad_mode: str = "zero") -> torch.Tensor:
    if shift_samples == 0:
        return eeg.clone()

    out = torch.zeros_like(eeg)
    t = eeg.shape[-1]

    if abs(shift_samples) >= t:
        if pad_mode == "circular":
            return torch.roll(eeg, shifts=shift_samples, dims=-1)
        return out

    if pad_mode == "circular":
        return torch.roll(eeg, shifts=shift_samples, dims=-1)

    if shift_samples > 0:
        out[..., shift_samples:] = eeg[..., : t - shift_samples]
    else:
        s = abs(shift_samples)
        out[..., : t - s] = eeg[..., s:]
    return out


def get_eeg_channel_occluded(
    eeg: torch.Tensor,
    channel_idx: int,
    fill_mode: str = "zero",
) -> torch.Tensor:
    out = eeg.clone()
    if fill_mode == "zero":
        out[:, channel_idx, :] = 0.0
    elif fill_mode == "mean":
        channel_mean = out[:, channel_idx, :].mean(dim=-1, keepdim=True)
        out[:, channel_idx, :] = channel_mean
    else:
        raise ValueError(f"Unknown fill_mode: {fill_mode}")
    return out


def _design_butter_band(fs: float, low: float, high: float, btype: str, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 0.999)

    if btype in ("bandpass", "bandstop"):
        return signal.butter(order, [low_n, high_n], btype=btype, output="sos")
    raise ValueError(f"Unsupported btype: {btype}")


def filter_eeg_band(
    eeg: torch.Tensor,
    fs: float,
    band: Tuple[float, float],
    mode: str = "bandstop",
    order: int = 4,
) -> torch.Tensor:
    if mode not in {"bandstop", "bandpass"}:
        raise ValueError("mode must be 'bandstop' or 'bandpass'")

    low, high = band
    sos = _design_butter_band(fs, low, high, btype=mode, order=order)

    eeg_np = eeg.detach().cpu().numpy()
    bsz, channels, t = eeg_np.shape
    flat = eeg_np.reshape(bsz * channels, t)

    filtered = signal.sosfiltfilt(sos, flat, axis=-1)
    filtered = filtered.reshape(bsz, channels, t)
    # scipy filtering can return arrays with negative strides, which torch
    # cannot wrap directly. Ensure contiguous positive-stride memory.
    filtered = np.ascontiguousarray(filtered)

    return torch.from_numpy(filtered).to(eeg.device, dtype=eeg.dtype)


def restore_attention_module_forwards(model: NeuroCodec) -> None:
    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "attn", None)
        if attn is not None and hasattr(attn, "_orig_forward_xai"):
            attn.forward = attn._orig_forward_xai
            delattr(attn, "_orig_forward_xai")


def enable_per_head_attention(model: NeuroCodec) -> None:
    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "attn", None)
        if attn is None or hasattr(attn, "_orig_forward_xai"):
            continue

        attn._orig_forward_xai = attn.forward

        def wrapped_forward(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            _orig: Callable = attn._orig_forward_xai,
            **kwargs: Any,
        ):
            kwargs.setdefault("need_weights", True)
            kwargs.setdefault("average_attn_weights", False)
            return _orig(query, key, value, **kwargs)

        attn.forward = wrapped_forward


class AttentionRecorder:
    def __init__(self, model: NeuroCodec, layer_indices: Sequence[int]):
        self.model = model
        self.layer_indices = list(layer_indices)
        self.handles: List[Any] = []
        self.cache: Dict[int, List[torch.Tensor]] = {idx: [] for idx in self.layer_indices}

    def _make_hook(self, layer_idx: int):
        def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any):
            if isinstance(output, tuple) and len(output) > 1 and torch.is_tensor(output[1]):
                self.cache[layer_idx].append(output[1].detach().cpu())

        return hook

    def register(self) -> None:
        self.clear()
        for idx in self.layer_indices:
            handle = self.model.layers[idx].attn.register_forward_hook(self._make_hook(idx))
            self.handles.append(handle)

    def clear(self) -> None:
        for k in self.cache:
            self.cache[k] = []

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def attention_to_eeg_importance(attn: torch.Tensor) -> torch.Tensor:
    if attn.ndim == 4:
        return attn.mean(dim=1).mean(dim=1)
    if attn.ndim == 3:
        return attn.mean(dim=1)
    raise ValueError(f"Unexpected attention shape: {tuple(attn.shape)}")


def resize_2d_map(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    out = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return out.squeeze(0).squeeze(0).numpy()


def save_attention_heatmap(attn_map: np.ndarray, out_path: str, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_map, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("EEG tokens")
    ax.set_ylabel("Audio tokens")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def extract_speech_envelope(audio: np.ndarray, sr: int, lowpass_hz: float = 8.0, filt_order: int = 4) -> np.ndarray:
    x = np.asarray(audio).squeeze().astype(np.float64)
    analytic = signal.hilbert(x)
    env = np.abs(analytic)

    nyq = 0.5 * sr
    cutoff = min(lowpass_hz / nyq, 0.99)
    sos = signal.butter(filt_order, cutoff, btype="low", output="sos")
    env_smooth = signal.sosfiltfilt(sos, env)
    return env_smooth.astype(np.float32)


def resample_1d_to_length(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x).squeeze()
    if len(x) == length:
        return x.astype(np.float32)
    if len(x) <= 1 or length <= 1:
        return np.zeros(length, dtype=np.float32)
    out = signal.resample(x, length)
    return out.astype(np.float32)


def zscore_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    std = x.std()
    if std < 1e-8:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y) or len(x) < 3:
        return np.nan
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return float(stats.pearsonr(x, y)[0])


def compute_directional_drop(metric: str, baseline: float, perturbed: float) -> float:
    if metric in HIGHER_IS_BETTER_METRICS:
        return baseline - perturbed
    if metric in LOWER_IS_BETTER_METRICS:
        return perturbed - baseline
    return baseline - perturbed
