import argparse
import os
import torch
from torch.utils.data import DataLoader

from dataset_neurocodec import (
    NeuroCodecDataset,
    load_KUL_NeuroCodecDataset,
)


def _as_subject(x):
    if x is None:
        return "Unknown"
    if torch.is_tensor(x):
        if x.numel() == 1:
            return str(x.item())
        return str(x.detach().cpu().tolist())
    return str(x)


def _check_tensor(name, t, abs_max):
    issues = []
    if not torch.isfinite(t).all():
        issues.append("non-finite")
    max_abs = t.abs().max().item()
    mean = t.mean().item()
    std = t.std().item()
    if abs_max is not None and max_abs > abs_max:
        issues.append(f"abs_max {max_abs:.4g} > {abs_max}")
    return issues, max_abs, mean, std


def build_loader(args):
    if args.dataset == "cocktail":
        dataset = NeuroCodecDataset(args.root, mode=args.subset, return_subject=True)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )
        return loader

    # KUL: use existing split logic, then rebuild loader with desired num_workers
    base_loader = load_KUL_NeuroCodecDataset(
        lmdb_path=args.root,
        subset=args.subset,
        batch_size=1,
        num_gpus=1,
        target_fs=args.target_fs,
        original_fs=args.original_fs,
        shuffle=False,
        return_subject=True,
    )
    dataset = base_loader.dataset
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    return loader


def scan(args):
    if not os.path.exists(args.root):
        raise FileNotFoundError(f"Path not found: {args.root}")

    loader = build_loader(args)

    bad_count = 0
    total = 0
    for idx, batch in enumerate(loader):
        if args.max_samples > 0 and idx >= args.max_samples:
            break

        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            noisy, eeg, clean, subject = batch
        else:
            noisy, eeg, clean = batch
            subject = None

        # Batch size is 1 by design, but keep generic
        noisy = noisy.float()
        eeg = eeg.float()
        clean = clean.float()

        issues = []
        noisy_issues, noisy_max, noisy_mean, noisy_std = _check_tensor(
            "noisy", noisy, args.audio_abs_max
        )
        eeg_issues, eeg_max, eeg_mean, eeg_std = _check_tensor(
            "eeg", eeg, args.eeg_abs_max
        )
        clean_issues, clean_max, clean_mean, clean_std = _check_tensor(
            "clean", clean, args.audio_abs_max
        )

        if noisy_issues:
            issues.append(f"noisy: {', '.join(noisy_issues)}")
        if eeg_issues:
            issues.append(f"eeg: {', '.join(eeg_issues)}")
        if clean_issues:
            issues.append(f"clean: {', '.join(clean_issues)}")

        total += 1
        if issues:
            bad_count += 1
            subj_str = _as_subject(subject)
            print(f"[BAD] idx={idx} subject={subj_str}")
            print(f"  issues: {', '.join(issues)}")
            print(f"  noisy: max={noisy_max:.4g} mean={noisy_mean:.4g} std={noisy_std:.4g}")
            print(f"  clean: max={clean_max:.4g} mean={clean_mean:.4g} std={clean_std:.4g}")
            print(f"  eeg:   max={eeg_max:.4g} mean={eeg_mean:.4g} std={eeg_std:.4g}")

        if args.log_every > 0 and (idx + 1) % args.log_every == 0:
            print(f"Scanned {idx + 1} samples...")

    print("\nScan complete.")
    print(f"Scanned: {total}")
    print(f"Bad:     {bad_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cocktail", choices=["cocktail", "kul"])
    parser.add_argument("--root", type=str, required=True, help="Cocktail root dir or KUL LMDB path")
    parser.add_argument("--subset", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--log_every", type=int, default=200, help="Progress log interval")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")

    parser.add_argument("--audio_abs_max", type=float, default=5.0, help="Audio abs max threshold")
    parser.add_argument("--eeg_abs_max", type=float, default=1000.0, help="EEG abs max threshold")

    # KUL only
    parser.add_argument("--target_fs", type=int, default=16000)
    parser.add_argument("--original_fs", type=int, default=16000)

    args = parser.parse_args()
    scan(args)
