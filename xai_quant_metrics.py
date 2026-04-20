"""Minimal quantitative XAI metrics for per-sample analysis.

This module adds 3 simple metrics with plots:
1) Attention concentration (entropy)
2) EEG vs attention correlation
3) Speech envelope vs attention temporal alignment
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal
from scipy.stats import pearsonr, spearmanr


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_attention_3d(attn: Any) -> np.ndarray:
    """Return attention as [heads, audio_tokens, eeg_tokens]."""
    arr = _to_numpy(attn).astype(np.float32)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected attention with 2 or 3 dims, got {arr.shape}")


def interpolate_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if target_len <= 1:
        return x[:1].copy() if x.size else np.zeros((1,), dtype=np.float32)
    if x.size == 0:
        return np.zeros((target_len,), dtype=np.float32)
    if x.size == target_len:
        return x.copy()
    src = np.linspace(0.0, 1.0, num=x.size, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32)


def safe_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("safe_corr expects same-length arrays")
    if x.size < 2:
        return float("nan"), float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan")
    return float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0])


def compute_attention_concentration(attention_maps: Sequence[Any]) -> Dict[str, Dict[str, float]]:
    """Compute entropy/max/top10% mass per layer from attention maps."""
    results: Dict[str, Dict[str, float]] = {}
    for layer_idx, layer_attn in enumerate(attention_maps):
        attn_3d = _ensure_attention_3d(layer_attn)
        attn_2d = attn_3d.mean(axis=0)  # average heads -> [A, E]

        flat = attn_2d.reshape(-1).astype(np.float64)
        mass = float(np.sum(flat))
        if mass <= 0:
            p = np.full_like(flat, 1.0 / max(flat.size, 1), dtype=np.float64)
        else:
            p = flat / mass

        entropy = float(-np.sum(p * np.log(p + 1e-8)))
        max_val = float(np.max(attn_2d)) if attn_2d.size else float("nan")

        k = max(1, int(np.ceil(0.1 * p.size)))
        top_mass = float(np.sort(p)[-k:].sum())

        results[f"layer_{layer_idx}"] = {
            "entropy": entropy,
            "max": max_val,
            "top10_mass": top_mass,
        }
    return results


def summarize_eeg_features(eeg_features: Any) -> np.ndarray:
    """EEG summary: mean(abs(eeg_features), axis=channels)."""
    eeg = _to_numpy(eeg_features).astype(np.float32)
    if eeg.ndim != 2:
        raise ValueError(f"Expected eeg_features [channels, time], got {eeg.shape}")
    return np.mean(np.abs(eeg), axis=0)


def _audio_timeline_from_attn(attn_2d: np.ndarray) -> np.ndarray:
    """Audio-token timeline; avoid flat summaries from strict row-normalized attention."""
    series = attn_2d.mean(axis=1)
    if float(np.std(series)) < 1e-8:
        series = attn_2d.max(axis=1)
    return series.astype(np.float32)


def _eeg_timeline_from_attn(attn_2d: np.ndarray) -> np.ndarray:
    """EEG-token timeline for EEG-vs-attention correlation."""
    series = attn_2d.mean(axis=0)
    if float(np.std(series)) < 1e-8:
        series = attn_2d.max(axis=0)
    return series.astype(np.float32)


def summarize_attention_over_time(
    attention_maps: Sequence[Any],
    target_len: int | None = None,
    mode: str = "audio",
) -> np.ndarray:
    """Build a 1D attention timeline.

    mode="audio": timeline over audio tokens (for speech-alignment metric)
    mode="eeg": timeline over EEG tokens (for EEG-attention correlation)
    """
    if not attention_maps:
        raise ValueError("attention_maps is empty")

    if mode not in {"audio", "eeg"}:
        raise ValueError(f"Unsupported mode: {mode}")

    layer_series = []
    for layer_attn in attention_maps:
        attn_3d = _ensure_attention_3d(layer_attn)
        attn_2d = attn_3d.mean(axis=0)       # [A, E]
        series = _audio_timeline_from_attn(attn_2d) if mode == "audio" else _eeg_timeline_from_attn(attn_2d)
        layer_series.append(series.astype(np.float32))

    max_len = max(s.shape[0] for s in layer_series)
    common = np.stack([interpolate_to_length(s, max_len) for s in layer_series], axis=0)
    merged = np.mean(common, axis=0)

    if target_len is not None:
        return interpolate_to_length(merged, target_len)
    return merged


def extract_speech_envelope(audio_target: Any, sample_rate: int, smooth_ms: float = 25.0) -> np.ndarray:
    """Speech envelope via Hilbert magnitude + moving-average smoothing."""
    x = _to_numpy(audio_target).astype(np.float32).reshape(-1)
    if x.size == 0:
        return x

    env = np.abs(signal.hilbert(x)).astype(np.float32)

    win = max(1, int(round(sample_rate * smooth_ms / 1000.0)))
    if win > 1:
        kernel = np.ones((win,), dtype=np.float32) / float(win)
        env = np.convolve(env, kernel, mode="same").astype(np.float32)

    return env


def plot_attention_concentration(entropy_by_layer: Dict[str, Dict[str, float]], out_path: str) -> None:
    layers = list(entropy_by_layer.keys())
    entropy_vals = [entropy_by_layer[k]["entropy"] for k in layers]
    max_vals = [entropy_by_layer[k]["max"] for k in layers]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(layers, entropy_vals, color="#0ea5a3")
    axes[0].set_title("Attention Entropy per Layer")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Entropy")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(layers, max_vals, color="#2563eb")
    axes[1].set_title("Max Attention per Layer")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Max value")
    axes[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_eeg_attention_relationship(
    eeg_summary: np.ndarray,
    attn_summary: np.ndarray,
    pearson: float,
    spearman: float,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    t = np.arange(eeg_summary.shape[0])
    axes[0].plot(t, eeg_summary, label="EEG summary", linewidth=1.2)
    axes[0].plot(t, attn_summary, label="Attention summary", linewidth=1.2)
    axes[0].set_title("EEG vs Attention (time)")
    axes[0].set_xlabel("Time index")
    axes[0].set_ylabel("Normalized value")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(eeg_summary, attn_summary, s=9, alpha=0.6)
    axes[1].set_title(f"Scatter (r={pearson:.3f}, rho={spearman:.3f})")
    axes[1].set_xlabel("EEG summary")
    axes[1].set_ylabel("Attention summary")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_speech_attention_alignment(
    speech_env: np.ndarray,
    attn_summary: np.ndarray,
    corr: float,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    t = np.arange(speech_env.shape[0])

    ax.plot(t, speech_env, label="Speech envelope", linewidth=1.1, color="#7c3aed")
    ax.plot(t, attn_summary, label="Attention summary", linewidth=1.1, color="#f97316")

    peaks, _ = signal.find_peaks(speech_env, distance=max(1, speech_env.size // 30))
    if peaks.size > 0:
        ax.scatter(peaks, speech_env[peaks], s=14, color="#4d7c0f", alpha=0.8, label="Envelope peaks")

    ax.set_title(f"Speech vs Attention Alignment (r={corr:.3f})")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Normalized value")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def run_xai_metrics(sample_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Compute and plot all 3 quantitative XAI metrics for one sample."""
    os.makedirs(output_dir, exist_ok=True)

    attention_maps = sample_data["attention_maps"]
    eeg_features = sample_data["eeg_features"]
    audio_target = sample_data["audio_target"]
    sample_rate = int(sample_data["sample_rate"])

    entropy = compute_attention_concentration(attention_maps)

    attn_summary_audio = summarize_attention_over_time(attention_maps, mode="audio")
    attn_summary_eeg = summarize_attention_over_time(attention_maps, mode="eeg")
    eeg_summary = summarize_eeg_features(eeg_features)

    corr_len = max(attn_summary_eeg.shape[0], eeg_summary.shape[0])
    attn_corr = interpolate_to_length(attn_summary_eeg, corr_len)
    eeg_corr = interpolate_to_length(eeg_summary, corr_len)
    eeg_corr_n = _normalize_01(eeg_corr)
    attn_corr_n = _normalize_01(attn_corr)
    pearson, spearman = safe_corr(eeg_corr, attn_corr)

    speech_env = extract_speech_envelope(audio_target, sample_rate)
    speech_len = max(speech_env.shape[0], attn_summary_audio.shape[0])
    speech_env_a = interpolate_to_length(speech_env, speech_len)
    attn_speech_a = interpolate_to_length(attn_summary_audio, speech_len)
    speech_env_n = _normalize_01(speech_env_a)
    attn_speech_n = _normalize_01(attn_speech_a)
    speech_corr, _ = safe_corr(speech_env_a, attn_speech_a)

    plot_attention_concentration(
        entropy,
        os.path.join(output_dir, "xai_attention_concentration.png"),
    )
    plot_eeg_attention_relationship(
        eeg_corr_n,
        attn_corr_n,
        pearson,
        spearman,
        os.path.join(output_dir, "xai_eeg_attention_relationship.png"),
    )
    plot_speech_attention_alignment(
        speech_env_n,
        attn_speech_n,
        speech_corr,
        os.path.join(output_dir, "xai_speech_attention_alignment.png"),
    )

    results: Dict[str, Any] = {
        "entropy": entropy,
        "eeg_attention_corr": {
            "pearson": float(pearson),
            "spearman": float(spearman),
        },
        "speech_attention_corr": float(speech_corr),
    }

    with open(os.path.join(output_dir, "xai_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    # Example usage with dummy data (replace with your pipeline tensors)
    rng = np.random.default_rng(42)
    dummy_attention = [
        rng.random((4, 120, 64), dtype=np.float32),
        rng.random((4, 120, 64), dtype=np.float32),
        rng.random((4, 120, 64), dtype=np.float32),
        rng.random((4, 120, 64), dtype=np.float32),
    ]
    dummy_eeg = rng.normal(0, 1, size=(64, 512)).astype(np.float32)
    dummy_audio = rng.normal(0, 0.1, size=(16000,)).astype(np.float32)

    out = "results/xai/quant_metrics_dummy"
    res = run_xai_metrics(
        {
            "attention_maps": dummy_attention,
            "eeg_features": dummy_eeg,
            "audio_target": dummy_audio,
            "sample_rate": 16000,
        },
        output_dir=out,
    )
    print("Saved dummy quantitative XAI outputs to:", out)
    print(json.dumps(res, indent=2))
