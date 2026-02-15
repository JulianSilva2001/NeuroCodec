#!/usr/bin/env python3
"""
Build an LMDB directly from KUL mix CSV (mixture_data_list_2mix.csv format) to speed training.

CSV (no header) columns:
0 split (train/val/test)
1 subject (int)
2 trial (int)
3 target wav filename
4 target start time (sec)  -> also EEG start
5 unused
6 interferer wav filename
7 interferer start time (sec)
8 SNR (dB)
9 duration (sec)

Stored per LMDB entry:
  mixture (float16 1-D), target (float16 1-D), eeg (float16 2-D CxT),
  subject (str), trial_idx (int), split (str)

Variable-length segments are kept; training loader pads.
"""

import argparse
import os
import pickle
import math
import numpy as np
import soundfile as sf
import librosa
import lmdb
from tqdm import tqdm


def parse_csv(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 10:
                continue
            rows.append({
                "split": parts[0].lower(),
                "subject": int(parts[1]),
                "trial": int(parts[2]),
                "tgt_wav": parts[3],
                "tgt_start": float(parts[4]),
                "int_wav": parts[6],
                "int_start": float(parts[7]),
                "snr": float(parts[8]),
                "dur": float(parts[9]),
            })
    return rows


def load_audio_slice(path, start_sec, dur_sec, target_sr):
    audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    start = int(start_sec * file_sr)
    stop = start + int(dur_sec * file_sr)
    if start >= len(audio):
        audio = np.zeros(int(dur_sec * target_sr), dtype=np.float32)
    else:
        audio = audio[start:stop]
        need = int(dur_sec * file_sr) - len(audio)
        if need > 0:
            audio = np.pad(audio, (0, need))
    if file_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=target_sr)
    return audio.astype(np.float32, copy=False)


def fix_length_audio(a, target_len):
    if a.shape[0] < target_len:
        a = np.pad(a, (0, target_len - a.shape[0]))
    elif a.shape[0] > target_len:
        a = a[:target_len]
    return a


def fix_length_eeg(eeg, target_len):
    if eeg.shape[1] < target_len:
        eeg = np.pad(eeg, ((0, 0), (0, target_len - eeg.shape[1])))
    elif eeg.shape[1] > target_len:
        eeg = eeg[:, :target_len]
    return eeg


def snr_mix(a_tgt, a_int, snr_db):
    target_power = np.linalg.norm(a_tgt, 2) ** 2 / max(a_tgt.size, 1)
    inter_power = np.linalg.norm(a_int, 2) ** 2 / max(a_int.size, 1)
    if inter_power > 0:
        a_int = a_int * np.sqrt(target_power / (inter_power + 1e-8))
    snr_lin = 10 ** (snr_db / 20.0)
    max_snr = max(1.0, snr_lin)
    a_tgt = a_tgt / max_snr
    a_int = a_int / max_snr
    a_int = a_int * snr_lin
    mix = a_tgt + a_int
    max_val = np.max(np.abs(mix)) if mix.size else 1.0
    if max_val > 1.0:
        mix = mix / max_val
        a_tgt = a_tgt / max_val
    return mix, a_tgt


def load_eeg_slice(eeg_dir, subject, trial, start_sec, dur_sec, ref_sr):
    eeg_path = os.path.join(eeg_dir, f"S{subject}Tra{trial}.npy")
    eeg = np.load(eeg_path)
    if eeg.ndim != 2:
        raise ValueError(f"EEG not 2D: {eeg.shape}")
    if eeg.shape[0] == 64:
        eeg_c_t = eeg
    elif eeg.shape[1] == 64:
        eeg_c_t = eeg.T
    else:
        raise ValueError(f"EEG shape unexpected: {eeg.shape}")
    start = int(start_sec * ref_sr)
    stop = start + int(dur_sec * ref_sr)
    if start >= eeg_c_t.shape[1]:
        seg = np.zeros((64, int(dur_sec * ref_sr)), dtype=np.float32)
    else:
        seg = eeg_c_t[:, start:stop]
        need = int(dur_sec * ref_sr) - seg.shape[1]
        if need > 0:
            seg = np.pad(seg, ((0, 0), (0, need)))
    return seg.astype(np.float32, copy=False)


MIN_SEGMENT_SEC = 4.0  # drop segments shorter than this threshold


def write_lmdb(rows, audio_dir, eeg_dir, out_path, audio_sr, ref_sr, segment_len_sec=None):
    if os.path.exists(out_path):
        raise FileExistsError(f"{out_path} exists. Delete it first.")
    parent = os.path.dirname(out_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    map_size = 5 * 1024 * 1024 * 1024  # 5GB start
    max_map = 100 * 1024 * 1024 * 1024  # 100GB cap
    env = lmdb.open(out_path, subdir=False, map_size=int(map_size), readonly=False, meminit=False, map_async=False, writemap=False, lock=True)
    txn = env.begin(write=True)
    commit_every = 500

    total = len(rows)
    target_audio_len = None
    target_eeg_len = None
    if segment_len_sec is not None:
        target_audio_len = int(segment_len_sec * audio_sr)
        target_eeg_len = int(segment_len_sec * ref_sr)

    written = 0

    for i, r in enumerate(tqdm(rows, desc="Writing LMDB")):
        if r["dur"] < MIN_SEGMENT_SEC:
            continue  # skip short segments instead of padding
        try:
            tgt = load_audio_slice(os.path.join(audio_dir, r["tgt_wav"]), r["tgt_start"], r["dur"], audio_sr)
            intr = load_audio_slice(os.path.join(audio_dir, r["int_wav"]), r["int_start"], r["dur"], audio_sr)
            if tgt.size == 0 or intr.size == 0:
                continue
            mix, tgt_proc = snr_mix(tgt, intr, r["snr"])
            eeg = load_eeg_slice(eeg_dir, r["subject"], r["trial"], r["tgt_start"], r["dur"], ref_sr)

            if target_audio_len is not None:
                mix = fix_length_audio(mix, target_audio_len)
                tgt_proc = fix_length_audio(tgt_proc, target_audio_len)
            if target_eeg_len is not None:
                eeg = fix_length_eeg(eeg, target_eeg_len)

            obj = {
                "mixture": mix.astype(np.float16),
                "target": tgt_proc.astype(np.float16),
                "eeg": eeg.astype(np.float16),
                "subject": f"S{r['subject']}",
                "trial_idx": r["trial"],
                "split": r["split"],
            }
            txn.put(f"{written}".encode("ascii"), pickle.dumps(obj))
            written += 1
            if written % commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)
        except lmdb.MapFullError:
            txn.abort()
            map_size = min(map_size + 5 * 1024 * 1024 * 1024, max_map)
            env.set_mapsize(map_size)
            txn = env.begin(write=True)
        except Exception as e:
            print(f"Skip idx {i} due to error: {e}")
            continue

    txn.put(b"__len__", str(written).encode("ascii"))
    txn.commit()
    env.sync()
    env.close()
    print(f"Done. Wrote {written} entries (of {total} rows) to {out_path} (map {map_size // (1024**3)}GB).")


def main():
    ap = argparse.ArgumentParser(description="Build LMDB from KUL mix CSV.")
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--eeg_dir", required=True)
    ap.add_argument("--out_path", default="kul_mixcsv.lmdb")
    ap.add_argument("--audio_sr", type=int, default=16000)
    ap.add_argument("--ref_sr", type=int, default=128)
    ap.add_argument("--segment_len_sec", type=float, default=4.0, help="Fix segments to this length; set 0 to keep original durations")
    args = ap.parse_args()

    rows = parse_csv(args.csv_path)
    print(f"Loaded {len(rows)} rows from CSV.")
    seg_len = args.segment_len_sec if args.segment_len_sec > 0 else None
    write_lmdb(rows, args.audio_dir, args.eeg_dir, args.out_path, args.audio_sr, args.ref_sr, segment_len_sec=seg_len)


if __name__ == "__main__":
    main()
