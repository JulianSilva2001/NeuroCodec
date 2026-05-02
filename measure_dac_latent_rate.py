import argparse

import torch


def measure_synthetic(dac_model, device, sample_rate, durations):
    print("Synthetic inputs")
    for seconds in durations:
        samples = int(round(sample_rate * seconds))
        x = torch.zeros(1, 1, samples, device=device)
        with torch.no_grad():
            z, codes, *_ = dac_model.encode(x)

        frames = z.shape[-1]
        duration = samples / sample_rate
        print(
            f"  {duration:.3f}s | samples={samples} | "
            f"latent_frames={frames} | frames_per_sec={frames / duration:.6f} | "
            f"samples_per_frame={samples / frames:.6f}"
        )


def measure_kul_batch(dac_model, device, args):
    from dataset_neurocodec import load_KUL_NeuroCodecDataset

    loader = load_KUL_NeuroCodecDataset(
        lmdb_path=args.kul_root,
        subset=args.subset,
        batch_size=1,
        num_gpus=1,
        target_fs=args.sample_rate,
        original_fs=args.original_fs,
        shuffle=False,
    )

    noisy, eeg, clean = next(iter(loader))[:3]
    noisy = noisy.to(device)

    with torch.no_grad():
        z, codes, *_ = dac_model.encode(noisy)

    audio_samples = noisy.shape[-1]
    duration = audio_samples / args.sample_rate

    print("\nKUL dataloader batch")
    print(f"  noisy shape: {tuple(noisy.shape)}")
    print(f"  eeg shape:   {tuple(eeg.shape)}")
    print(f"  clean shape: {tuple(clean.shape)}")
    print(
        f"  duration={duration:.6f}s | samples={audio_samples} | "
        f"latent_frames={z.shape[-1]} | frames_per_sec={z.shape[-1] / duration:.6f} | "
        f"samples_per_frame={audio_samples / z.shape[-1]:.6f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Measure DAC latent frame rate.")
    parser.add_argument("--model_type", default="16khz", choices=["16khz", "24khz", "44khz"])
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--durations", type=float, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--kul_root", default=None, help="Optional KUL LMDB path to measure a real batch.")
    parser.add_argument("--subset", default="val", choices=["train", "val", "test"])
    parser.add_argument("--original_fs", type=int, default=16000)
    args = parser.parse_args()

    import dac

    device = torch.device(args.device)
    model_path = dac.utils.download(model_type=args.model_type)
    dac_model = dac.DAC.load(model_path).to(device).eval()

    print(f"DAC model: {args.model_type}")
    print(f"Sample rate used for duration math: {args.sample_rate}")
    print(f"Device: {device}\n")

    measure_synthetic(dac_model, device, args.sample_rate, args.durations)

    if args.kul_root:
        measure_kul_batch(dac_model, device, args)


if __name__ == "__main__":
    main()
