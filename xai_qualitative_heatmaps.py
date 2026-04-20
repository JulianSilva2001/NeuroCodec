"""Standalone qualitative heatmap test for NeuroCodec on KUL.

This script is intentionally separate from existing XAI runs.
For each sample, it saves:
1. Spectrogram comparison (0-4kHz): mixture, target, reconstruction
2. DAC encoder latent heatmap
3. EEG encoder feature heatmap
4. Per CM-S3 block attention heatmap
5. Per CM-S3 block activation heatmap

No averaging across samples is performed.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal
import soundfile as sf
from tqdm import tqdm

from xai_utils import (
    align_1d_signals,
    build_dataloader,
    ensure_dir,
    infer_runtime_config,
    load_model_from_checkpoint,
    parse_batch,
    resolve_device,
    set_seed,
    _safe_metric_pesq,
    _safe_metric_stoi,
)
from xai_quant_metrics import run_xai_metrics


def to_numpy_1d(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().squeeze().astype(np.float32)


def plot_stft_panel(ax: plt.Axes, audio: np.ndarray, sr: int, title: str, fmax_hz: float) -> None:
    n_fft = 1024
    hop = 256
    f, t, z = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft, boundary=None)
    mag = np.abs(z)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-8))

    keep = f <= fmax_hz
    f = f[keep]
    mag_db = mag_db[keep, :]

    im = ax.pcolormesh(t, f, mag_db, shading="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0.0, fmax_hz)
    plt.colorbar(im, ax=ax, pad=0.01, label="Magnitude (dB)")


def save_spectrogram_comparison(
    mixture: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    sr: int,
    out_path: str,
    fmax_hz: float,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plot_stft_panel(axes[0], mixture, sr, "(a) Input mixture", fmax_hz)
    plot_stft_panel(axes[1], target, sr, "(b) Ground-truth target speaker", fmax_hz)
    plot_stft_panel(axes[2], pred, sr, "(c) NeuroCodec reconstruction", fmax_hz)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_audio_clip(audio: np.ndarray, sr: int, out_path: str) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    sf.write(out_path, audio, sr)


def save_heatmap(
    arr_2d: np.ndarray,
    out_path: str,
    title: str,
    x_label: str,
    y_label: str,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(arr_2d, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.colorbar(im, ax=ax, pad=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _extract_attn_sample(attn_batch: torch.Tensor, sample_idx: int) -> np.ndarray:
    # Supports (B, T_audio, T_eeg) or (B, H, T_audio, T_eeg).
    arr = attn_batch.detach().cpu().numpy()
    if arr.ndim == 4:
        # Average heads for a stable per-layer map.
        return arr[sample_idx].mean(axis=0)
    if arr.ndim == 3:
        return arr[sample_idx]
    raise ValueError(f"Unexpected attention shape: {tuple(arr.shape)}")


def register_intermediate_hooks(model: torch.nn.Module) -> Tuple[Dict[str, Dict[int, List[torch.Tensor]]], List[Any]]:
    cache: Dict[str, Dict[int, List[torch.Tensor]]] = {
        "cm_s3_attn": {},
        "cm_s3_act": {},
    }
    handles: List[Any] = []

    for idx, layer in enumerate(model.layers):
        cache["cm_s3_attn"][idx] = []
        cache["cm_s3_act"][idx] = []

        def make_attn_hook(layer_idx: int):
            def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                if isinstance(output, tuple) and len(output) > 1 and torch.is_tensor(output[1]):
                    cache["cm_s3_attn"][layer_idx].append(output[1].detach().cpu())

            return hook

        def make_layer_hook(layer_idx: int):
            def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                if isinstance(output, tuple) and len(output) > 0 and torch.is_tensor(output[0]):
                    cache["cm_s3_act"][layer_idx].append(output[0].detach().cpu())

            return hook

        handles.append(layer.attn.register_forward_hook(make_attn_hook(idx)))
        handles.append(layer.register_forward_hook(make_layer_hook(idx)))

    return cache, handles


def clear_cache(cache: Dict[str, Dict[int, List[torch.Tensor]]]) -> None:
    for key in cache:
        for idx in cache[key]:
            cache[key][idx] = []


def remove_handles(handles: List[Any]) -> None:
    for h in handles:
        h.remove()


def write_sample_notes(path: str, info: Dict[str, Any]) -> None:
    xai = info.get("xai_metrics", {})
    eeg_attn = xai.get("eeg_attention_corr", {})
    lines = [
        "Qualitative heatmap test summary",
        "",
        f"sample_id: {info['sample_id']}",
        f"subset: {info['subset']}",
        f"dataset: {info['dataset']}",
        f"target_fs: {info['target_fs']}",
        f"pesq: {info['pesq']:.4f}",
        f"stoi: {info['stoi']:.4f}",
        f"estoi: {info['estoi']:.4f}",
        f"eeg_attention_pearson: {eeg_attn.get('pearson', float('nan'))}",
        f"eeg_attention_spearman: {eeg_attn.get('spearman', float('nan'))}",
        f"speech_attention_corr: {xai.get('speech_attention_corr', float('nan'))}",
        "",
        "Saved artifacts:",
        "- spectrogram_comparison_0_4kHz.png: qualitative audio comparison",
        "- mixture.wav / target.wav / prediction.wav: playable audio clips",
        "- dac_encoder_latent_heatmap.png: DAC latent channels vs time tokens",
        "- eeg_encoder_feature_heatmap.png: EEG feature channels vs time",
        "- cm_s3_layer*_attention_heatmap.png: per-layer cross-attention map",
        "- cm_s3_layer*_activation_heatmap.png: per-layer hidden activation energy",
        "- xai_attention_concentration.png: entropy and max attention per layer",
        "- xai_eeg_attention_relationship.png: EEG vs attention line+scatter",
        "- xai_speech_attention_alignment.png: speech envelope vs attention overlay",
        "- xai_metrics.json: quantitative XAI values for this sample",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _read_json_if_exists(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
                return {}
        with open(path, "r", encoding="utf-8") as f:
                return json.load(f)


def _discover_sample_dirs(output_dir: str) -> List[str]:
        names = []
        for name in os.listdir(output_dir):
                full = os.path.join(output_dir, name)
                if name.startswith("sample_") and os.path.isdir(full):
                        names.append(name)
        return sorted(names)


def _write_text(path: str, text: str) -> None:
        ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
                f.write(text)


def _html_shell(title: str, body: str) -> str:
        return f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>{html.escape(title)}</title>
    <style>
        :root {{
            --bg: #f7f8fa;
            --card: #ffffff;
            --ink: #0f172a;
            --muted: #475569;
            --accent: #0ea5a3;
            --line: #dbe2ea;
        }}
        body {{
            margin: 0;
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            background: var(--bg);
            color: var(--ink);
        }}
        .wrap {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 24px;
        }}
        h1, h2, h3 {{ margin: 0 0 12px 0; }}
        .card {{
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
        }}
        .muted {{ color: var(--muted); }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 14px;
        }}
        .thumb {{
            width: 100%;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #fff;
        }}
        .row {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
        .pill {{
            font-size: 12px;
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 4px 10px;
            background: #eef9f8;
            color: var(--accent);
            font-weight: 600;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--line);
            text-align: left;
            vertical-align: top;
        }}
        th {{
            background: #f1f5f9;
        }}
        audio {{
            display: block;
        }}
        a {{ color: var(--accent); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="wrap">
        {body}
    </div>
</body>
</html>
"""


def _build_sample_page(report_dir: str, sample_name: str, metadata: Dict[str, Any], num_layers: int) -> None:
    sample_rel = f"../{sample_name}"
    xai = metadata.get("xai_metrics", {})
    eeg_attn = xai.get("eeg_attention_corr", {})
    rows = [
        ("sample_id", metadata.get("sample_id", "")),
        ("dataset", metadata.get("dataset", "")),
        ("subset", metadata.get("subset", "")),
        ("target_fs", metadata.get("target_fs", "")),
        ("pesq", metadata.get("pesq", "")),
        ("stoi", metadata.get("stoi", "")),
        ("estoi", metadata.get("estoi", "")),
        ("eeg_attn_pearson", eeg_attn.get("pearson", "")),
        ("eeg_attn_spearman", eeg_attn.get("spearman", "")),
        ("speech_attn_corr", xai.get("speech_attention_corr", "")),
    ]

    table_rows = "\n".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows
    )

    block_cards = []
    for layer_idx in range(num_layers):
        attn_png = f"{sample_rel}/cm_s3_layer{layer_idx}_attention_heatmap.png"
        act_png = f"{sample_rel}/cm_s3_layer{layer_idx}_activation_heatmap.png"
        block_cards.append(
            f"""
            <div class=\"card\">
                <h3>CM-S3 Layer {layer_idx}</h3>
                <div class=\"grid\">
                    <div>
                        <p class=\"muted\">Cross-attention heatmap</p>
                        <img class=\"thumb\" src=\"{attn_png}\" alt=\"layer {layer_idx} attention\">
                    </div>
                    <div>
                        <p class=\"muted\">Hidden activation heatmap</p>
                        <img class=\"thumb\" src=\"{act_png}\" alt=\"layer {layer_idx} activation\">
                    </div>
                </div>
            </div>
            """
        )

    body = f"""
    <div class=\"card\">
        <div class=\"row\">
            <a href=\"index.html\">Back to index</a>
            <span class=\"pill\">{html.escape(sample_name)}</span>
        </div>
        <h1>Qualitative Heatmap Report: {html.escape(sample_name)}</h1>
        <p class=\"muted\">Per-sample visual analysis with no averaging.</p>
        <table>{table_rows}</table>
    </div>

    <div class=\"grid\">
        <div class=\"card\">
            <h2>Listen: Mixture</h2>
            <audio controls src=\"{sample_rel}/mixture.wav\" style=\"width: 100%;\"></audio>
        </div>
        <div class=\"card\">
            <h2>Listen: Target</h2>
            <audio controls src=\"{sample_rel}/target.wav\" style=\"width: 100%;\"></audio>
        </div>
        <div class=\"card\">
            <h2>Listen: NeuroCodec Reconstruction</h2>
            <audio controls src=\"{sample_rel}/prediction.wav\" style=\"width: 100%;\"></audio>
        </div>
    </div>


    <div class=\"card\">
        <h2>Spectrogram Comparison (0-4kHz)</h2>
        <img class=\"thumb\" src=\"{sample_rel}/spectrogram_comparison_0_4kHz.png\" alt=\"spectrogram comparison\">
    </div>

    <div class=\"grid\">
        <div class=\"card\">
            <h2>DAC Encoder</h2>
            <img class=\"thumb\" src=\"{sample_rel}/dac_encoder_latent_heatmap.png\" alt=\"dac latent heatmap\">
        </div>
        <div class=\"card\">
            <h2>EEG Encoder</h2>
            <img class=\"thumb\" src=\"{sample_rel}/eeg_encoder_feature_heatmap.png\" alt=\"eeg encoder heatmap\">
        </div>
    </div>


    {''.join(block_cards)}
    """

    _write_text(os.path.join(report_dir, f"{sample_name}.html"), _html_shell(f"Heatmap Report {sample_name}", body))


def build_html_report(output_dir: str, report_dir_name: str, title: str) -> str:
        report_dir = os.path.join(output_dir, report_dir_name)
        ensure_dir(report_dir)

        sample_dirs = _discover_sample_dirs(output_dir)
        if not sample_dirs:
                raise RuntimeError(f"No sample_* directories found in {output_dir}")

        num_layers_detected = 0
        index_rows = []

        for sample_name in sample_dirs:
                sample_path = os.path.join(output_dir, sample_name)
                meta = _read_json_if_exists(os.path.join(sample_path, "metadata.json"))
                num_layers = int(meta.get("num_layers", 0))
                num_layers_detected = max(num_layers_detected, num_layers)

                _build_sample_page(report_dir, sample_name, meta, num_layers)

                index_rows.append(
                        f"""
                        <tr>
                            <td><a href=\"{sample_name}.html\">{html.escape(sample_name)}</a></td>
                            <td>{html.escape(str(meta.get('pesq', '')))}</td>
                            <td>{html.escape(str(meta.get('stoi', '')))}</td>
                            <td>{html.escape(str(meta.get('estoi', '')))}</td>
                            <td>{html.escape(str(meta.get('subset', '')))}</td>
                        </tr>
                        """
                )

        body = f"""
        <div class=\"card\">
            <h1>{html.escape(title)}</h1>
            <p class=\"muted\">Report folder: {html.escape(report_dir)}</p>
            <div class=\"row\">
                <span class=\"pill\">samples: {len(sample_dirs)}</span>
                <span class=\"pill\">cm-s3 layers: {num_layers_detected}</span>
            </div>
        </div>

        <div class=\"card\">
            <h2>Samples</h2>
            <table>
                <tr><th>Sample</th><th>PESQ</th><th>STOI</th><th>ESTOI</th><th>Subset</th></tr>
                {''.join(index_rows)}
            </table>
        </div>
        """

        index_path = os.path.join(report_dir, "index.html")
        _write_text(index_path, _html_shell(title, body))
        return index_path


def run_qualitative_test(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    runtime = infer_runtime_config(args.dataset, eeg_channels_override=args.eeg_channels)
    device = resolve_device(args.gpu)

    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        runtime=runtime,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        backbone=args.backbone,
        activation=args.activation,
        device=device,
    )

    loader = build_dataloader(
        dataset=runtime.dataset,
        root=args.root,
        subset=args.subset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        target_fs=runtime.target_fs,
        original_fs=args.original_fs,
    )

    cache, handles = register_intermediate_hooks(model)

    sample_count = 0
    pbar = tqdm(loader, desc="Qualitative heatmap test", dynamic_ncols=True)

    try:
        for batch in pbar:
            if args.max_samples > 0 and sample_count >= args.max_samples:
                break

            parsed = parse_batch(batch, device=device)
            mixture = parsed["mixture"]
            eeg = parsed["eeg"]
            target = parsed["target"]

            clear_cache(cache)
            with torch.no_grad():
                z_pred, _codes_mix, z_mix, eeg_feat, _last_attn, _envelope_pred = model(mixture, eeg)
                z_q = model.dac.quantizer(z_pred, n_quantizers=args.n_quantizers)[0]
                pred_audio = model.dac.decode(z_q)

            min_len = min(mixture.shape[-1], target.shape[-1], pred_audio.shape[-1])
            mixture = mixture[..., :min_len]
            target = target[..., :min_len]
            pred_audio = pred_audio[..., :min_len]

            bsz = mixture.shape[0]
            for i in range(bsz):
                if args.max_samples > 0 and sample_count >= args.max_samples:
                    break

                sample_dir = os.path.join(args.output_dir, f"sample_{sample_count:05d}")
                ensure_dir(sample_dir)

                mix_np = to_numpy_1d(mixture[i, 0])
                tgt_np = to_numpy_1d(target[i, 0])
                pred_np = to_numpy_1d(pred_audio[i, 0])

                tgt_aligned, pred_aligned, lag = align_1d_signals(tgt_np, pred_np)
                pesq = _safe_metric_pesq(tgt_aligned, pred_aligned, runtime.target_fs)
                stoi = _safe_metric_stoi(tgt_aligned, pred_aligned, runtime.target_fs, extended=False)
                estoi = _safe_metric_stoi(tgt_aligned, pred_aligned, runtime.target_fs, extended=True)

                save_spectrogram_comparison(
                    mixture=mix_np,
                    target=tgt_np,
                    pred=pred_np,
                    sr=runtime.target_fs,
                    out_path=os.path.join(sample_dir, "spectrogram_comparison_0_4kHz.png"),
                    fmax_hz=args.fmax_hz,
                )

                save_audio_clip(mix_np, runtime.target_fs, os.path.join(sample_dir, "mixture.wav"))
                save_audio_clip(tgt_np, runtime.target_fs, os.path.join(sample_dir, "target.wav"))
                save_audio_clip(pred_np, runtime.target_fs, os.path.join(sample_dir, "prediction.wav"))

                dac_latent = z_mix[i].detach().cpu().numpy()  # (1024, T)
                np.save(os.path.join(sample_dir, "dac_encoder_latent.npy"), dac_latent)
                save_heatmap(
                    arr_2d=dac_latent,
                    out_path=os.path.join(sample_dir, "dac_encoder_latent_heatmap.png"),
                    title="DAC encoder latent (channels x tokens)",
                    x_label="Time token index",
                    y_label="DAC latent channel",
                    cmap="magma",
                )

                eeg_encoded = eeg_feat[i].detach().cpu().numpy()  # (64, T)
                np.save(os.path.join(sample_dir, "eeg_encoder_feature.npy"), eeg_encoded)
                save_heatmap(
                    arr_2d=eeg_encoded,
                    out_path=os.path.join(sample_dir, "eeg_encoder_feature_heatmap.png"),
                    title="EEG encoder feature map (channels x time)",
                    x_label="EEG time index",
                    y_label="EEG feature channel",
                    cmap="viridis",
                )

                attn_maps_for_metrics: List[np.ndarray] = []
                for layer_idx in range(len(model.layers)):
                    attn_list = cache["cm_s3_attn"][layer_idx]
                    act_list = cache["cm_s3_act"][layer_idx]

                    if attn_list:
                        raw_attn = attn_list[-1].detach().cpu().numpy()
                        if raw_attn.ndim == 4:
                            attn_maps_for_metrics.append(raw_attn[i])
                        elif raw_attn.ndim == 3:
                            attn_maps_for_metrics.append(raw_attn[i][None, ...])

                        attn_map = _extract_attn_sample(attn_list[-1], i)
                        np.save(os.path.join(sample_dir, f"cm_s3_layer{layer_idx}_attention.npy"), attn_map)
                        save_heatmap(
                            arr_2d=attn_map,
                            out_path=os.path.join(sample_dir, f"cm_s3_layer{layer_idx}_attention_heatmap.png"),
                            title=f"CM-S3 layer {layer_idx} cross-attention",
                            x_label="EEG token index",
                            y_label="Audio token index",
                            cmap="magma",
                        )

                    if act_list:
                        # (B, T, H) -> (H, T) for channel-vs-time visualization
                        act = act_list[-1][i].detach().cpu().numpy().T
                        np.save(os.path.join(sample_dir, f"cm_s3_layer{layer_idx}_activation.npy"), act)
                        save_heatmap(
                            arr_2d=act,
                            out_path=os.path.join(sample_dir, f"cm_s3_layer{layer_idx}_activation_heatmap.png"),
                            title=f"CM-S3 layer {layer_idx} hidden activation",
                            x_label="Audio token index",
                            y_label="Hidden channel",
                            cmap="plasma",
                        )

                if attn_maps_for_metrics:
                    xai_metrics = run_xai_metrics(
                        {
                            "attention_maps": attn_maps_for_metrics,
                            "eeg_features": eeg_encoded,
                            "audio_target": tgt_np,
                            "sample_rate": runtime.target_fs,
                        },
                        output_dir=sample_dir,
                    )
                else:
                    xai_metrics = {
                        "entropy": {},
                        "eeg_attention_corr": {"pearson": float("nan"), "spearman": float("nan")},
                        "speech_attention_corr": float("nan"),
                    }

                metadata = {
                    "sample_id": sample_count,
                    "subset": args.subset,
                    "dataset": runtime.dataset,
                    "target_fs": runtime.target_fs,
                    "pesq": float(pesq),
                    "stoi": float(stoi),
                    "estoi": float(estoi),
                    "num_layers": len(model.layers),
                    "fmax_hz": args.fmax_hz,
                    "xai_metrics": xai_metrics,
                }
                with open(os.path.join(sample_dir, "metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
                write_sample_notes(os.path.join(sample_dir, "README.txt"), metadata)

                sample_count += 1

            if args.max_samples > 0 and sample_count >= args.max_samples:
                break

    finally:
        remove_handles(handles)

    print("\n[Done] Qualitative heatmap test complete")
    print(f"Output directory: {args.output_dir}")
    print(f"Saved samples: {sample_count}")

    if args.build_html_report:
        report_index = build_html_report(
            output_dir=args.output_dir,
            report_dir_name=args.report_dir_name,
            title=args.report_title,
        )
        print(f"HTML report index: {report_index}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone qualitative heatmap test for NeuroCodec")

    p.add_argument("--root", type=str, default="/home/jaliya/eeg_speech/navindu/data/Apr-1/lmdb")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="/home/jaliya/eeg_speech/shaveen/NeuroCodec/checkpoints/neurocodec_KUL/best_model.pth",
    )
    p.add_argument("--output_dir", type=str, default="results/xai/qualitative_heatmaps")

    p.add_argument("--dataset", type=str, default="kul", choices=["kul", "cocktail"])
    p.add_argument("--subset", type=str, default="val")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--max_samples", type=int, default=10, help="Maximum samples for this qualitative test")

    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--backbone", type=str, default="mamba", choices=["mamba", "transformer"])
    p.add_argument("--activation", type=str, default="gelu", choices=["gelu", "snake", "relu"])
    p.add_argument("--eeg_channels", type=int, default=None)

    p.add_argument("--original_fs", type=int, default=16000)
    p.add_argument("--n_quantizers", type=int, default=9)
    p.add_argument("--fmax_hz", type=float, default=4000.0)
    p.add_argument("--build_html_report", action="store_true", help="Build compact HTML index and per-sample pages")
    p.add_argument("--report_dir_name", type=str, default="report")
    p.add_argument("--report_title", type=str, default="NeuroCodec Qualitative Heatmap Report")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_qualitative_test(args)
