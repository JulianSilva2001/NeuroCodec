"""Run all NeuroCodec XAI experiments and aggregate one paper-ready summary table.

This launcher keeps your training code untouched and only calls evaluation/XAI scripts.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_cmd(cmd: List[str], dry_run: bool = False) -> None:
    print("\n[Run]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def _find_value_cols(df: pd.DataFrame, metric: str) -> Dict[str, Optional[str]]:
    mean_col = f"{metric}_mean" if f"{metric}_mean" in df.columns else None
    std_col = f"{metric}_std" if f"{metric}_std" in df.columns else None
    if mean_col is None and metric in df.columns:
        mean_col = metric
    return {"mean": mean_col, "std": std_col}


def summarize_experiment_1(ablation_dir: str, metric: str) -> pd.DataFrame:
    summary_path = os.path.join(ablation_dir, "ablation_summary.csv")
    sig_path = os.path.join(ablation_dir, "ablation_significance.csv")

    df = csv_if_exists(summary_path)
    if df is None or df.empty:
        return pd.DataFrame()

    cols = _find_value_cols(df, metric)
    if cols["mean"] is None:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "experiment": "exp1_eeg_ablation",
            "item": df["condition"].astype(str),
            "metric": metric,
            "value_mean": df[cols["mean"]],
            "value_std": df[cols["std"]] if cols["std"] else np.nan,
            "notes": "",
        }
    )

    sig_df = csv_if_exists(sig_path)
    if sig_df is not None and not sig_df.empty:
        sig_df = sig_df[sig_df["metric"] == metric]
        if not sig_df.empty:
            p_map = dict(zip(sig_df["condition"].astype(str), sig_df["ttest_pvalue"]))
            out["notes"] = out["item"].map(lambda x: f"ttest_p={p_map.get(x, np.nan):.3g}")

    return out


def summarize_experiment_2(occlusion_dir: str, metric: str, top_k: int) -> pd.DataFrame:
    path = os.path.join(occlusion_dir, "channel_occlusion_importance.csv")
    df = csv_if_exists(path)
    if df is None or df.empty:
        return pd.DataFrame()

    drop_col = f"drop_{metric}"
    if drop_col not in df.columns:
        drop_candidates = [c for c in df.columns if c.startswith("drop_")]
        if not drop_candidates:
            return pd.DataFrame()
        drop_col = drop_candidates[0]

    sort_df = df.sort_values(drop_col, ascending=False).head(top_k).copy()
    name_col = "electrode_name" if "electrode_name" in sort_df.columns else "channel_idx"

    out = pd.DataFrame(
        {
            "experiment": "exp2_channel_occlusion",
            "item": sort_df[name_col].astype(str),
            "metric": drop_col,
            "value_mean": sort_df[drop_col],
            "value_std": np.nan,
            "notes": "top_channel",
        }
    )
    return out


def summarize_experiment_3(band_dir: str, metric: str) -> pd.DataFrame:
    path = os.path.join(band_dir, "band_importance_summary.csv")
    df = csv_if_exists(path)
    if df is None or df.empty:
        return pd.DataFrame()

    cols = _find_value_cols(df, metric)
    if cols["mean"] is None:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "experiment": "exp3_band_importance",
            "item": df["condition"].astype(str),
            "metric": metric,
            "value_mean": df[cols["mean"]],
            "value_std": df[cols["std"]] if cols["std"] else np.nan,
            "notes": df.get("mode", pd.Series([""] * len(df))).astype(str),
        }
    )
    return out


def _attention_entropy(attn_map: np.ndarray) -> float:
    # attn_map is 2D (audio_tokens, eeg_tokens)
    p = attn_map.astype(np.float64)
    p = p - p.min()
    p = p + 1e-12
    p = p / p.sum(axis=1, keepdims=True)
    ent = -(p * np.log(p)).sum(axis=1)
    return float(np.mean(ent))


def summarize_experiment_4(attn_dir: str) -> pd.DataFrame:
    npy_paths = sorted(glob.glob(os.path.join(attn_dir, "layer*_dataset_avg.npy")))
    rows = []
    for p in npy_paths:
        layer_name = os.path.basename(p).replace("_dataset_avg.npy", "")
        arr = np.load(p)
        rows.append(
            {
                "experiment": "exp4_attention_viz",
                "item": layer_name,
                "metric": "attention_entropy",
                "value_mean": _attention_entropy(arr),
                "value_std": np.nan,
                "notes": f"shape={arr.shape}",
            }
        )
        rows.append(
            {
                "experiment": "exp4_attention_viz",
                "item": layer_name,
                "metric": "attention_max",
                "value_mean": float(np.max(arr)),
                "value_std": np.nan,
                "notes": f"shape={arr.shape}",
            }
        )

    return pd.DataFrame(rows)


def summarize_experiment_5(env_dir: str) -> pd.DataFrame:
    path = os.path.join(env_dir, "envelope_consistency_summary.csv")
    df = csv_if_exists(path)
    if df is None or df.empty:
        return pd.DataFrame()

    row = df.iloc[0]
    rows = [
        {
            "experiment": "exp5_env_consistency",
            "item": "attended_corr",
            "metric": "corr_attended_mean",
            "value_mean": row.get("corr_attended_mean", np.nan),
            "value_std": row.get("corr_attended_std", np.nan),
            "notes": "",
        },
        {
            "experiment": "exp5_env_consistency",
            "item": "unattended_corr",
            "metric": "corr_unattended_mean",
            "value_mean": row.get("corr_unattended_mean", np.nan),
            "value_std": row.get("corr_unattended_std", np.nan),
            "notes": "",
        },
        {
            "experiment": "exp5_env_consistency",
            "item": "corr_diff",
            "metric": "corr_diff_mean",
            "value_mean": row.get("corr_diff_mean", np.nan),
            "value_std": row.get("corr_diff_std", np.nan),
            "notes": f"ttest_p={row.get('paired_ttest_pvalue', np.nan):.3g}",
        },
    ]
    return pd.DataFrame(rows)


def aggregate_summary(args: argparse.Namespace) -> str:
    frames = []
    frames.append(summarize_experiment_1(os.path.join(args.output_root, "ablation"), args.primary_metric))
    frames.append(summarize_experiment_2(os.path.join(args.output_root, "channel_occlusion"), args.primary_metric, args.top_k_channels))
    frames.append(summarize_experiment_3(os.path.join(args.output_root, "band_importance"), args.primary_metric))
    frames.append(summarize_experiment_4(os.path.join(args.output_root, "attention_viz")))
    frames.append(summarize_experiment_5(os.path.join(args.output_root, "envelope_consistency")))

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise RuntimeError("No experiment outputs found to aggregate. Did the runs complete?")

    table = pd.concat(frames, ignore_index=True)
    table_path = os.path.join(args.output_root, "paper_summary_table.csv")
    table.to_csv(table_path, index=False)

    compact = table.copy()
    compact["value_mean"] = compact["value_mean"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
    compact["value_std"] = compact["value_std"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    txt_path = os.path.join(args.output_root, "paper_summary_table.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(compact.to_string(index=False))

    return table_path


def main(args: argparse.Namespace) -> None:
    ensure_dir(args.output_root)

    base = [
        "--root",
        args.root,
        "--checkpoint",
        args.checkpoint,
        "--dataset",
        args.dataset,
        "--subset",
        args.subset,
        "--gpu",
        str(args.gpu),
        "--hidden_dim",
        str(args.hidden_dim),
        "--num_layers",
        str(args.num_layers),
        "--backbone",
        args.backbone,
        "--activation",
        args.activation,
        "--batch_size",
        str(args.batch_size),
        "--max_samples",
        str(args.max_samples),
        "--original_fs",
        str(args.original_fs),
    ]

    py = sys.executable

    if not args.skip_exp1:
        cmd = [py, "xai_ablation.py"] + base + [
            "--output_dir",
            os.path.join(args.output_root, "ablation"),
            "--metrics",
            args.metrics,
            "--shift_seconds",
            args.shift_seconds,
            "--eeg_fs",
            str(args.eeg_fs),
            "--shift_pad_mode",
            args.shift_pad_mode,
        ]
        if args.run_significance:
            cmd.append("--run_significance")
        run_cmd(cmd, dry_run=args.dry_run)

    if not args.skip_exp2:
        cmd = [py, "xai_occlusion.py"] + base + [
            "--output_dir",
            os.path.join(args.output_root, "channel_occlusion"),
            "--metrics",
            args.metrics,
            "--primary_metric",
            args.primary_metric,
            "--fill_mode",
            args.occlusion_fill_mode,
        ]
        if args.electrode_names_path:
            cmd += ["--electrode_names_path", args.electrode_names_path]
        run_cmd(cmd, dry_run=args.dry_run)

    if not args.skip_exp3:
        cmd = [py, "xai_band_importance.py"] + base + [
            "--output_dir",
            os.path.join(args.output_root, "band_importance"),
            "--metrics",
            args.metrics,
            "--bands",
            args.bands,
            "--modes",
            args.band_modes,
            "--eeg_fs",
            str(args.eeg_fs),
            "--filter_order",
            str(args.filter_order),
        ]
        if args.run_significance:
            cmd.append("--run_significance")
        run_cmd(cmd, dry_run=args.dry_run)

    if not args.skip_exp4:
        cmd = [py, "xai_attention_viz.py"] + base + [
            "--output_dir",
            os.path.join(args.output_root, "attention_viz"),
            "--layers",
            args.attn_layers,
            "--head_idx",
            str(args.attn_head_idx),
            "--avg_audio_tokens",
            str(args.avg_audio_tokens),
            "--avg_eeg_tokens",
            str(args.avg_eeg_tokens),
        ]
        if args.save_attention_tensors:
            cmd.append("--save_per_sample_tensors")
        run_cmd(cmd, dry_run=args.dry_run)

    if not args.skip_exp5:
        cmd = [py, "xai_envelope_consistency.py"] + base + [
            "--output_dir",
            os.path.join(args.output_root, "envelope_consistency"),
            "--env_lowpass_hz",
            str(args.env_lowpass_hz),
        ]
        if args.use_mixture_minus_target_as_unattended:
            cmd.append("--use_mixture_minus_target_as_unattended")
        run_cmd(cmd, dry_run=args.dry_run)

    if not args.dry_run:
        table_path = aggregate_summary(args)
        print("\n[Done] Aggregated summary table:")
        print(table_path)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run all NeuroCodec XAI experiments and aggregate a summary table")

    p.add_argument("--root", type=str, default="/home/jaliya/eeg_speech/navindu/data/Apr-1/lmdb")
    p.add_argument("--checkpoint", type=str, default="/home/jaliya/eeg_speech/shaveen/NeuroCodec/checkpoints/neurocodec_KUL/best_model.pth")
    p.add_argument("--output_root", type=str, default="results/xai/all_runs_new")

    p.add_argument("--dataset", type=str, default="kul", choices=["kul", "cocktail"])
    p.add_argument("--subset", type=str, default="val")
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--backbone", type=str, default="mamba", choices=["mamba", "transformer"])
    p.add_argument("--activation", type=str, default="gelu", choices=["gelu", "snake", "relu"])

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=10)
    p.add_argument("--original_fs", type=int, default=16000)

    p.add_argument("--metrics", type=str, default="si_sdr,estoi,pesq,stoi,latent_mse")
    p.add_argument("--primary_metric", type=str, default="si_sdr")
    p.add_argument("--run_significance", action="store_true")

    p.add_argument("--shift_seconds", type=str, default="0.5,1.0,2.0")
    p.add_argument("--shift_pad_mode", type=str, default="zero", choices=["zero", "circular"])
    p.add_argument("--eeg_fs", type=float, default=128.0)

    p.add_argument("--occlusion_fill_mode", type=str, default="zero", choices=["zero", "mean"])
    p.add_argument("--electrode_names_path", type=str, default=None)
    p.add_argument("--top_k_channels", type=int, default=10)

    p.add_argument("--bands", type=str, default="delta,theta,alpha,beta,gamma")
    p.add_argument("--band_modes", type=str, default="bandstop,bandpass")
    p.add_argument("--filter_order", type=int, default=4)

    p.add_argument("--attn_layers", type=str, default="-1")
    p.add_argument("--attn_head_idx", type=int, default=-1)
    p.add_argument("--avg_audio_tokens", type=int, default=128)
    p.add_argument("--avg_eeg_tokens", type=int, default=128)
    p.add_argument("--save_attention_tensors", action="store_true")

    p.add_argument("--env_lowpass_hz", type=float, default=8.0)
    p.add_argument("--use_mixture_minus_target_as_unattended", action="store_true")

    p.add_argument("--skip_exp1", action="store_true")
    p.add_argument("--skip_exp2", action="store_true")
    p.add_argument("--skip_exp3", action="store_true")
    p.add_argument("--skip_exp4", action="store_true")
    p.add_argument("--skip_exp5", action="store_true")

    p.add_argument("--dry_run", action="store_true")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
