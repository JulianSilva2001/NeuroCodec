import argparse
import json
import math
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset_neurocodec import load_KUL_NeuroCodecDataset, load_NeuroCodecDataset
from models.neurocodec import NeuroCodec


def get_config_path(dataset_key: str) -> str:
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    mapping = {
        "CP": "train_cocktail.json",
        "KUL": "train_kul.json",
    }
    return os.path.join(config_dir, mapping[dataset_key])


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return config


def apply_eeg_cue_transform(eeg: torch.Tensor, cue_mode: str) -> torch.Tensor:
    if cue_mode == "real":
        return eeg
    if cue_mode == "noise":
        eeg_mean = eeg.mean()
        eeg_std = eeg.std()
        return torch.randn_like(eeg) * eeg_std + eeg_mean
    if cue_mode == "zero":
        return torch.zeros_like(eeg)
    if cue_mode == "shuffle":
        if eeg.shape[0] <= 1:
            return eeg
        perm = torch.randperm(eeg.shape[0], device=eeg.device)
        return eeg[perm]
    raise ValueError(f"Unknown cue_mode: {cue_mode}")


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def flatten_tensor(tensor: torch.Tensor) -> np.ndarray:
    return tensor_to_numpy(tensor).reshape(-1)


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a_np = flatten_tensor(a)
    b_np = flatten_tensor(b)
    denom = np.linalg.norm(a_np) + 1e-12
    return float(np.linalg.norm(a_np - b_np) / denom)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_np = flatten_tensor(a)
    b_np = flatten_tensor(b)
    denom = (np.linalg.norm(a_np) * np.linalg.norm(b_np)) + 1e-12
    return float(np.dot(a_np, b_np) / denom)


def tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    arr = tensor_to_numpy(tensor)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "abs_mean": float(np.abs(arr).mean()),
        "norm": float(np.linalg.norm(arr.reshape(-1))),
    }


def decode_prediction(model: NeuroCodec, z_pred: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        z_q = model.dac.quantizer(z_pred, n_quantizers=9)[0]
        pred_audio = model.dac.decode(z_q)
    return pred_audio


def detailed_forward(model: NeuroCodec, mixture: torch.Tensor, eeg: torch.Tensor) -> Dict[str, torch.Tensor]:
    outputs: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        z_mix, codes_mix, _, _, _ = model.dac.encode(mixture)

    outputs["z_mix"] = z_mix
    outputs["codes_mix"] = codes_mix.float() if torch.is_tensor(codes_mix) else torch.tensor(0.0, device=mixture.device)

    eeg_after_bn = model.eeg_encoder.BN1(eeg)
    outputs["eeg_after_bn"] = eeg_after_bn

    if model.eeg_encoder.A.device != eeg_after_bn.device:
        model.eeg_encoder.A = model.eeg_encoder.A.to(eeg_after_bn.device)
    from utility.utils import normalize_A

    L = normalize_A(model.eeg_encoder.A)
    eeg_after_gcn = model.eeg_encoder.layer1(eeg_after_bn, L)
    outputs["eeg_after_gcn"] = eeg_after_gcn

    eeg_after_projection = model.eeg_encoder.projection(eeg_after_gcn)
    outputs["eeg_after_projection"] = eeg_after_projection

    eeg_feat = model.eeg_encoder.eeg_encoder(eeg_after_projection)
    outputs["eeg_feat"] = eeg_feat

    x_audio_pre = model.audio_proj(z_mix.transpose(1, 2))
    outputs["x_audio_pre_pe"] = x_audio_pre

    x_eeg_pre = model.eeg_proj(eeg_feat.transpose(1, 2))
    outputs["x_eeg_pre_pe"] = x_eeg_pre

    scale = math.sqrt(model.audio_proj.out_features)
    x_audio = model.pos_encoder(x_audio_pre * scale)
    x_eeg = model.pos_encoder(x_eeg_pre * scale)
    outputs["x_audio_post_pe"] = x_audio
    outputs["x_eeg_post_pe"] = x_eeg

    x = x_audio
    for layer_idx, layer in enumerate(model.layers):
        x_norm = layer.ln1(x)
        attn_out, attn_weights = layer.attn(query=x_norm, key=x_eeg, value=x_eeg)
        x_after_attn = x + layer.dropout1(attn_out)

        outputs[f"layer{layer_idx}_attn_out"] = attn_out
        outputs[f"layer{layer_idx}_attn_weights"] = attn_weights
        outputs[f"layer{layer_idx}_x_after_attn"] = x_after_attn

        if layer.backbone_type == "mamba":
            x_norm_core = layer.ln2(x_after_attn)
            core_out = layer.core(x_norm_core)
            x = x_after_attn + layer.dropout2(core_out)
        else:
            batch_size, time_steps, _ = x_after_attn.shape
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(time_steps, device=x_after_attn.device)
            core_out = layer.core(x_after_attn, src_mask=causal_mask, is_causal=True)
            x = core_out

        outputs[f"layer{layer_idx}_core_out"] = core_out
        outputs[f"layer{layer_idx}_x_out"] = x

    outputs["x_hidden"] = x
    z_pred = model.output_proj(x).transpose(1, 2)
    outputs["z_pred"] = z_pred
    outputs["pred_audio"] = decode_prediction(model, z_pred)
    return outputs


def eeg_gradient_score(model: NeuroCodec, mixture: torch.Tensor, eeg: torch.Tensor) -> float:
    mixture = mixture.detach()
    eeg = eeg.detach().clone().requires_grad_(True)
    z_pred, _, _, _, _ = model(mixture, eeg)
    loss = z_pred.pow(2).mean()
    grad = torch.autograd.grad(loss, eeg, retain_graph=False, create_graph=False)[0]
    return float(grad.norm().detach().cpu().item())


def load_batch(args, dataset_name: str, target_fs: int):
    if dataset_name == "kul":
        loader = load_KUL_NeuroCodecDataset(
            lmdb_path=args.root,
            subset=args.subset,
            batch_size=args.batch_size,
            num_gpus=1,
            target_fs=target_fs,
            original_fs=16000,
            shuffle=args.shuffle,
        )
    else:
        loader = load_NeuroCodecDataset(
            root=args.root,
            subset=args.subset,
            batch_size=args.batch_size,
            num_gpus=1,
            shuffle=args.shuffle,
        )
    return iter(loader)


def compare_against_real(real_outputs: Dict[str, torch.Tensor], cue_outputs: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for key, real_tensor in real_outputs.items():
        cue_tensor = cue_outputs[key]
        metrics[key] = {
            "relative_l2": relative_l2(real_tensor, cue_tensor),
            "cosine_similarity": cosine_similarity(real_tensor, cue_tensor),
            "real_norm": tensor_stats(real_tensor)["norm"],
            "cue_norm": tensor_stats(cue_tensor)["norm"],
        }
    return metrics


def save_stage_barplot(comparisons: Dict[str, Dict[str, Dict[str, float]]], output_path: str):
    stage_names = [k for k in comparisons["noise"].keys() if k.endswith(("eeg_feat", "x_eeg_post_pe", "x_hidden", "z_pred", "pred_audio"))]
    if not stage_names:
        stage_names = list(comparisons["noise"].keys())[:10]

    cue_modes = ["noise", "zero", "shuffle"]
    x = np.arange(len(stage_names))
    width = 0.25

    plt.figure(figsize=(16, 6))
    for idx, cue_mode in enumerate(cue_modes):
        values = [comparisons[cue_mode][stage]["relative_l2"] for stage in stage_names]
        plt.bar(x + idx * width, values, width=width, label=cue_mode)

    plt.xticks(x + width, stage_names, rotation=45, ha="right")
    plt.ylabel("Relative L2 delta vs real EEG")
    plt.title("Where cue ablations change the network")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_attention_maps(real_outputs: Dict[str, torch.Tensor], ablated_outputs: Dict[str, Dict[str, torch.Tensor]], output_dir: str):
    attn_keys = [key for key in real_outputs.keys() if key.endswith("_attn_weights")]
    cue_modes = ["real", "noise", "zero", "shuffle"]
    for attn_key in attn_keys:
        fig, axes = plt.subplots(1, len(cue_modes), figsize=(5 * len(cue_modes), 4))
        for idx, cue_mode in enumerate(cue_modes):
            attn = real_outputs[attn_key] if cue_mode == "real" else ablated_outputs[cue_mode][attn_key]
            attn_np = tensor_to_numpy(attn[0])
            axes[idx].imshow(attn_np, aspect="auto", origin="lower", cmap="magma")
            axes[idx].set_title(f"{cue_mode}: {attn_key}")
            axes[idx].set_xlabel("EEG time")
            axes[idx].set_ylabel("Audio time")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{attn_key}.png"))
        plt.close(fig)


def save_waveform_plot(real_outputs: Dict[str, torch.Tensor], ablated_outputs: Dict[str, Dict[str, torch.Tensor]], clean: torch.Tensor, output_path: str):
    cue_modes = ["real", "noise", "zero", "shuffle"]
    clean_np = tensor_to_numpy(clean[0, 0])

    plt.figure(figsize=(16, 8))
    plt.plot(clean_np, label="clean", linewidth=1.0, alpha=0.8)
    for cue_mode in cue_modes:
        pred_audio = real_outputs["pred_audio"] if cue_mode == "real" else ablated_outputs[cue_mode]["pred_audio"]
        pred_np = tensor_to_numpy(pred_audio[0, 0])
        limit = min(len(clean_np), len(pred_np))
        plt.plot(pred_np[:limit], label=cue_mode, linewidth=0.8)
    plt.title("Decoded waveform comparison by cue mode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_eeg_feature_heatmaps(real_outputs: Dict[str, torch.Tensor], ablated_outputs: Dict[str, Dict[str, torch.Tensor]], output_path: str):
    cue_modes = ["real", "noise", "zero", "shuffle"]
    fig, axes = plt.subplots(len(cue_modes), 1, figsize=(14, 12), sharex=True)
    for idx, cue_mode in enumerate(cue_modes):
        eeg_feat = real_outputs["eeg_feat"] if cue_mode == "real" else ablated_outputs[cue_mode]["eeg_feat"]
        feat_np = tensor_to_numpy(eeg_feat[0])
        axes[idx].imshow(feat_np, aspect="auto", origin="lower", cmap="viridis")
        axes[idx].set_title(f"eeg_feat ({cue_mode})")
        axes[idx].set_ylabel("Feature channel")
    axes[-1].set_xlabel("Latent time")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def build_summary(
    real_outputs: Dict[str, torch.Tensor],
    ablated_outputs: Dict[str, Dict[str, torch.Tensor]],
    comparisons: Dict[str, Dict[str, Dict[str, float]]],
    gradient_scores: Dict[str, float],
) -> Dict:
    summary = {
        "gradient_scores": gradient_scores,
        "cue_deltas": comparisons,
        "real_stage_stats": {key: tensor_stats(value) for key, value in real_outputs.items()},
    }

    critical_stages = [
        "eeg_after_bn",
        "eeg_after_gcn",
        "eeg_after_projection",
        "eeg_feat",
        "x_eeg_pre_pe",
        "x_eeg_post_pe",
        "layer0_attn_out",
        "layer0_x_out",
        "x_hidden",
        "z_pred",
        "pred_audio",
    ]
    summary["critical_stage_table"] = {}
    for cue_mode in ["noise", "zero", "shuffle"]:
        summary["critical_stage_table"][cue_mode] = {}
        for stage in critical_stages:
            if stage in comparisons[cue_mode]:
                summary["critical_stage_table"][cue_mode][stage] = comparisons[cue_mode][stage]
    return summary


def main(args):
    config_path = args.config or get_config_path(args.dataset_key)
    config = load_config(config_path)
    dataset_name = "cocktail" if args.dataset_key == "CP" else "kul"
    checkpoint = args.checkpoint or config.get("checkpoint", os.path.join(config["checkpoint_dir"], "best_model.pth"))
    root = args.root or config["root"]
    gpu = args.gpu if args.gpu is not None else config.get("gpu", 0)
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else config["hidden_dim"]
    num_layers = args.num_layers if args.num_layers is not None else config["num_layers"]
    backbone = args.backbone or config["backbone"]
    activation = args.activation or config["activation"]
    dropout = args.dropout if args.dropout is not None else config["dropout"]
    eeg_channels = args.eeg_channels if args.eeg_channels is not None else config["eeg_channels"]
    batch_size = args.batch_size if args.batch_size is not None else max(config.get("batch_size", 2), 2)
    subset = args.subset
    target_fs = 16000 if dataset_name == "kul" else 44100
    dac_model_type = "16khz" if dataset_name == "kul" else "44khz"

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Loaded config: {config_path}")
    print(f"Inspecting on {device} with dataset={dataset_name}, batch_size={batch_size}")

    model = NeuroCodec(
        dac_model_type=dac_model_type,
        eeg_in_channels=eeg_channels,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        backbone=backbone,
        activation=activation,
        dropout=dropout,
    ).to(device)

    if os.path.exists(checkpoint):
        print(f"Loading checkpoint: {checkpoint}")
        try:
            ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        except pickle.UnpicklingError:
            ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Checkpoint not found. Continuing with random weights.")

    model.eval()

    loader_iter = load_batch(
        argparse.Namespace(root=root, subset=subset, batch_size=batch_size, shuffle=args.shuffle),
        dataset_name,
        target_fs,
    )

    output_root = args.output_dir or os.path.join("results", "NeuroCodec_DeepInspect", args.dataset_key)
    os.makedirs(output_root, exist_ok=True)

    for batch_idx in range(args.num_batches):
        try:
            noisy, eeg, clean = next(loader_iter)
        except StopIteration:
            print("Reached end of dataset.")
            break

        noisy = noisy.to(device)
        eeg = eeg.to(device)
        clean = clean.to(device)

        batch_dir = os.path.join(output_root, f"batch_{batch_idx:03d}")
        os.makedirs(batch_dir, exist_ok=True)

        cue_outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        gradient_scores: Dict[str, float] = {}

        for cue_mode in ["real", "noise", "zero", "shuffle"]:
            eeg_variant = apply_eeg_cue_transform(eeg, cue_mode)
            with torch.no_grad():
                cue_outputs[cue_mode] = detailed_forward(model, noisy, eeg_variant)
            gradient_scores[cue_mode] = eeg_gradient_score(model, noisy, eeg_variant)

        real_outputs = cue_outputs["real"]
        comparisons = {
            cue_mode: compare_against_real(real_outputs, cue_outputs[cue_mode])
            for cue_mode in ["noise", "zero", "shuffle"]
        }

        summary = build_summary(real_outputs, cue_outputs, comparisons, gradient_scores)
        summary["dataset"] = dataset_name
        summary["checkpoint"] = checkpoint
        summary["subset"] = subset
        summary["batch_index"] = batch_idx

        with open(os.path.join(batch_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        save_stage_barplot(comparisons, os.path.join(batch_dir, "stage_delta_barplot.png"))
        save_attention_maps(real_outputs, cue_outputs, batch_dir)
        save_waveform_plot(real_outputs, cue_outputs, clean, os.path.join(batch_dir, "waveform_comparison.png"))
        save_eeg_feature_heatmaps(real_outputs, cue_outputs, os.path.join(batch_dir, "eeg_feature_heatmaps.png"))

        print(f"Saved deep inspection outputs to {batch_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_key", choices=["CP", "KUL"], help="Inspection profile to use")
    parser.add_argument("--config", type=str, default=None, help="Optional explicit config file")
    parser.add_argument("--root", type=str, default=None, help="Override dataset root")
    parser.add_argument("--gpu", type=int, default=None, help="Override GPU id")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    parser.add_argument("--subset", type=str, default="val", help="Dataset subset")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for inspection; use >=2 for shuffle")
    parser.add_argument("--num_batches", type=int, default=1, help="How many batches to inspect")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before sampling")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to save plots and summaries")
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--backbone", type=str, default=None, choices=["mamba", "transformer"])
    parser.add_argument("--activation", type=str, default=None, choices=["gelu", "snake", "relu"])
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--eeg_channels", type=int, default=None)
    args = parser.parse_args()
    main(args)
