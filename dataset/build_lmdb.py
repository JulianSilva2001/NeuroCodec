import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import lmdb
from scipy.io import wavfile
from tqdm import tqdm

try:
    import soundfile as sf
except ImportError:
    sf = None

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def _load_audio(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        return torchaudio.load(path)
    except (ImportError, OSError, RuntimeError):
        if sf is not None:
            data, sample_rate = sf.read(path, always_2d=True, dtype='float32')
            waveform = torch.from_numpy(data.T)
            return waveform, sample_rate
        sample_rate, data = wavfile.read(path)
        if data.dtype != np.float32:
            max_val = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0
            data = data.astype(np.float32) / max_val
        waveform = torch.from_numpy(data.T if data.ndim > 1 else data[None, :])
        return waveform, sample_rate


def rms(x):
    return torch.sqrt(torch.mean(x ** 2) + 1e-8)


# ==========================================
# 2. THE DATASET BUILDER CLASS
# ==========================================

class KULRawDataset(torch.utils.data.Dataset):
    """Builds ALL subjects into segments. No split logic here.

    The LMDB stores every segment with its subject name.
    A separate __subject_to_indices__ metadata key maps each subject
    to its list of LMDB indices, so the training dataloader can pick
    train/val subjects at runtime without rebuilding.
    """

    def __init__(self, csv_path, eeg_dir, audio_dir, segment_len_sec=6,
                 hop_len_sec=1, audio_fs=8192, target_eeg_fs=128):
        self.csv_path = csv_path
        self.eeg_dir = eeg_dir
        self.audio_dir = audio_dir
        self.segment_len_sec = segment_len_sec
        self.hop_len_sec = hop_len_sec
        self.audio_fs = audio_fs
        self.target_eeg_fs = target_eeg_fs
        self.trial_info = []
        self.segment_info = []
        self.trial_data_cache = {}
        # subject -> list of segment indices (filled during segmenting)
        self.subject_to_indices = {}

        self._parse_csv_and_cache()
        self._create_segments()

    def _parse_csv_and_cache(self):
        import pandas as pd
        print(f"Parsing CSV and caching all trial data: {self.csv_path}")

        with open(self.csv_path, "r", encoding="utf-8-sig") as f:
            header = f.readline()
        sep = "\t" if "\t" in header else ","
        df = pd.read_csv(self.csv_path, sep=sep, engine="python")
        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

        required_cols = ['Subject', 'TrialIndex', 'AttendedEar', 'TargetWav', 'InterfererWav', 'EEGFile']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: {col}")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading & Caching Trials"):
            try:
                subject_name = str(row['Subject'])
                trial_index = None
                trial_index_raw = row.get('TrialIndex', None)
                if trial_index_raw is not None and not pd.isna(trial_index_raw):
                    try:
                        trial_index = int(trial_index_raw)
                    except Exception:
                        pass
                attended_ear = row.get('AttendedEar', '')
                if pd.isna(attended_ear):
                    attended_ear = ''
                attended_ear = str(attended_ear).strip().upper()
                eeg_file = str(row['EEGFile'])
                target_file = str(row['TargetWav'])
                inter_file = str(row['InterfererWav'])
                original_mat = row.get('OriginalMat', '')
                if pd.isna(original_mat):
                    original_mat = ''
                original_mat = str(original_mat).strip()

                # Load EEG
                eeg_path = eeg_file if os.path.isabs(eeg_file) else os.path.join(self.eeg_dir, eeg_file)
                if not os.path.exists(eeg_path):
                    print(f"  EEG not found: {eeg_path} (skip)")
                    continue
                raw_eeg_np = np.load(eeg_path)
                if raw_eeg_np.ndim != 2:
                    print(f"  EEG not 2D: {raw_eeg_np.shape} (skip)")
                    continue
                if raw_eeg_np.shape[0] == 64:
                    pass
                elif raw_eeg_np.shape[1] == 64:
                    raw_eeg_np = raw_eeg_np.T
                else:
                    print(f"  EEG shape unexpected: {raw_eeg_np.shape} (skip)")
                    continue

                # Load audio
                target_path = target_file if os.path.isabs(target_file) else os.path.join(self.audio_dir, target_file)
                inter_path = inter_file if os.path.isabs(inter_file) else os.path.join(self.audio_dir, inter_file)
                if not os.path.exists(target_path):
                    print(f"  Target audio not found: {target_path} (skip)")
                    continue
                if not os.path.exists(inter_path):
                    print(f"  Interferer audio not found: {inter_path} (skip)")
                    continue

                int_wav, int_fs = _load_audio(inter_path)
                tgt_wav, tgt_fs = _load_audio(target_path)

                # Resample
                if int_fs != self.audio_fs:
                    int_wav = torchaudio.transforms.Resample(int_fs, self.audio_fs)(int_wav)
                if tgt_fs != self.audio_fs:
                    tgt_wav = torchaudio.transforms.Resample(tgt_fs, self.audio_fs)(tgt_wav)

                # Mono
                if int_wav.shape[0] > 1:
                    int_wav = int_wav.mean(dim=0, keepdim=True)
                if tgt_wav.shape[0] > 1:
                    tgt_wav = tgt_wav.mean(dim=0, keepdim=True)
                int_wav = int_wav.squeeze()
                tgt_wav = tgt_wav.squeeze()

                # Mix at 0 dB SNR and normalize
                tgt_rms = rms(tgt_wav)
                int_rms = rms(int_wav)
                int_wav_adj = int_wav * (tgt_rms / int_rms)
                mixture = int_wav_adj + tgt_wav
                mixture = mixture / (rms(mixture) + 1e-8)
                tgt_wav = tgt_wav / (tgt_rms + 1e-8)
                int_wav_normalized = int_wav_adj / (int_rms + 1e-8)

                # Align EEG duration to audio
                audio_duration = tgt_wav.shape[0] / self.audio_fs
                eeg_target_samples = int(audio_duration * self.target_eeg_fs)
                if raw_eeg_np.shape[1] > eeg_target_samples:
                    raw_eeg_np = raw_eeg_np[:, :eeg_target_samples]
                elif raw_eeg_np.shape[1] < eeg_target_samples:
                    raw_eeg_np = np.pad(raw_eeg_np, ((0, 0), (0, eeg_target_samples - raw_eeg_np.shape[1])))
                raw_eeg_np = np.clip(raw_eeg_np.astype(np.float32), -20.0, 20.0)

                self.trial_data_cache[idx] = {
                    'mix': mixture.half(),
                    'tgt': tgt_wav.half(),
                    'int': int_wav_normalized.half(),
                    'eeg': torch.from_numpy(raw_eeg_np).float(),
                    'subject': subject_name,
                    'trial_index': trial_index,
                    'attended_ear': attended_ear,
                    'original_mat': original_mat,
                    'target_file': target_file,
                    'inter_file': inter_file
                }
                self.trial_info.append({
                    'subject': subject_name,
                    'idx': idx,
                    'trial_index': trial_index,
                    'attended_ear': attended_ear,
                    'original_mat': original_mat
                })
            except Exception as e:
                print(f"  Error row {idx}: {e}")

        print(f"Cached {len(self.trial_info)} trials.")

    def _create_segments(self):
        """Segment ALL trials (no splitting). Build subject_to_indices map."""
        segment_len_samples = int(self.segment_len_sec * self.audio_fs)
        hop_len_samples = int(self.hop_len_sec * self.audio_fs)

        seg_idx = 0
        for trial_idx, info in enumerate(tqdm(self.trial_info, desc="Segmenting")):
            cache_idx = info['idx']
            subject_name = info['subject']
            cached_mix = self.trial_data_cache[cache_idx]['mix']
            total_samples = cached_mix.shape[0]

            per_subject_trial_num = info.get('trial_index', trial_idx)

            for start_sample in range(0, total_samples - segment_len_samples + 1, hop_len_samples):
                self.segment_info.append({
                    'trial_idx': trial_idx,
                    'cache_idx': cache_idx,
                    'start_sample': start_sample,
                    'subject_trial_num': per_subject_trial_num
                })
                self.subject_to_indices.setdefault(subject_name, []).append(seg_idx)
                seg_idx += 1

        all_subjects = sorted(self.subject_to_indices.keys())
        print(f"Created {len(self.segment_info)} total segments across {len(all_subjects)} subjects")
        for s in all_subjects:
            print(f"   {s}: {len(self.subject_to_indices[s])} segments")

    def __len__(self):
        return len(self.segment_info)

    def __getitem__(self, idx):
        seg_info = self.segment_info[idx]
        cache_idx = seg_info['cache_idx']
        cached_data = self.trial_data_cache[cache_idx]

        start = seg_info['start_sample']
        target_length = int(self.segment_len_sec * self.audio_fs)
        end = start + target_length

        mix_seg = cached_data['mix'][start:end].float()
        tgt_seg = cached_data['tgt'][start:end].float()
        int_seg = cached_data['int'][start:end].float()

        if mix_seg.shape[0] < target_length:
            pad_amt = target_length - mix_seg.shape[0]
            mix_seg = F.pad(mix_seg, (0, pad_amt))
            tgt_seg = F.pad(tgt_seg, (0, pad_amt))
            int_seg = F.pad(int_seg, (0, pad_amt))

        eeg_seg = self._process_eeg_segment(cached_data['eeg'], start, target_length)

        subject_name = cached_data['subject']
        subject_trial_num = seg_info.get('subject_trial_num', seg_info['trial_idx'])
        attended_ear = cached_data.get('attended_ear', '')
        original_mat = cached_data.get('original_mat', '')

        try:
            subject_label = int(''.join(filter(str.isdigit, subject_name))) - 1
        except Exception:
            subject_label = 0

        target_file = cached_data.get('target_file', '')
        speaker_label = self._extract_speaker_label(target_file, subject_trial_num)

        return {
            'mixture': mix_seg.numpy().astype(np.float16),
            'target': tgt_seg.numpy().astype(np.float16),
            'interferer': int_seg.numpy().astype(np.float16),
            'eeg': eeg_seg.numpy().astype(np.float16),
            'subject': subject_name,
            'trial_idx': subject_trial_num,
            'subject_label': subject_label,
            'speaker_label': speaker_label,
            'attended_ear': attended_ear,
            'original_mat': original_mat
        }

    def _process_eeg_segment(self, full_eeg, audio_start_sample, audio_target_length):
        audio_start_sec = audio_start_sample / self.audio_fs
        audio_duration_sec = audio_target_length / self.audio_fs
        eeg_start = int(audio_start_sec * self.target_eeg_fs)
        eeg_len = int(audio_duration_sec * self.target_eeg_fs)
        eeg_end = eeg_start + eeg_len
        if eeg_end > full_eeg.shape[1]:
            eeg_end = full_eeg.shape[1]
            eeg_start = max(0, eeg_end - eeg_len)
        eeg_seg = full_eeg[:, eeg_start:eeg_end]
        if eeg_seg.shape[1] < eeg_len:
            eeg_seg = F.pad(eeg_seg, (0, eeg_len - eeg_seg.shape[1]))
        return eeg_seg

    def _extract_speaker_label(self, target_file, subject_trial_num):
        speaker_label = 0
        try:
            basename = os.path.basename(target_file).lower()
            if 'track1' in basename:
                speaker_label = 0
            elif 'track2' in basename:
                speaker_label = 1
            else:
                speaker_label = 1 if subject_trial_num >= 4 else 0
        except Exception:
            speaker_label = 0
        return speaker_label


# ==========================================
# 3. LMDB WRITER
# ==========================================

def write_lmdb(dataset, out_path):
    if os.path.exists(out_path):
        print(f"Output {out_path} already exists. Delete it first:")
        print(f"  rm {out_path}")
        return

    parent_dir = os.path.dirname(out_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    initial_map_size = 5 * 1024 * 1024 * 1024
    max_map_size = 50 * 1024 * 1024 * 1024
    current_map_size = initial_map_size

    print(f"Creating LMDB at: {out_path}")
    print(f"Total segments: {len(dataset)}")

    db = lmdb.open(out_path, subdir=False, map_size=int(current_map_size),
                   readonly=False, meminit=False, map_async=False,
                   sync=True, writemap=False)
    txn = db.begin(write=True)
    commit_interval = 100

    i = 0
    while i < len(dataset):
        try:
            data = dataset[i]
            txn.put(f"{i}".encode('ascii'), pickle.dumps(data))
            i += 1
            if i % commit_interval == 0:
                txn.commit()
                txn = db.begin(write=True)
                print(f"\r  Writing: {i}/{len(dataset)} ({100*i/len(dataset):.1f}%)", end="", flush=True)
        except lmdb.MapFullError:
            try:
                txn.abort()
            except Exception:
                pass
            increment = 5 * 1024 * 1024 * 1024
            new_size = min(current_map_size + increment, max_map_size)
            if new_size == current_map_size:
                print(f"\nMap size limit reached ({max_map_size // (1024**3)}GB)")
                break
            print(f"\n  Resizing: {current_map_size // (1024**3)}GB -> {new_size // (1024**3)}GB")
            current_map_size = new_size
            db.set_mapsize(current_map_size)
            txn = db.begin(write=True)
        except Exception as e:
            print(f"\n  Error at {i}: {e}")
            try:
                txn.abort()
            except Exception:
                pass
            txn = db.begin(write=True)
            i += 1

    # Store metadata: total length and subject-to-indices mapping
    txn.put(b'__len__', str(len(dataset)).encode('ascii'))
    txn.put(b'__subject_to_indices__', pickle.dumps(dataset.subject_to_indices))
    txn.commit()
    db.sync()
    db.close()
    print(f"\nDone! {len(dataset)} segments written to {out_path}")
    print(f"Subjects stored: {sorted(dataset.subject_to_indices.keys())}")


# ==========================================
# 4. MAIN
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build single LMDB with all subjects (split at training time)")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--eeg_dir", type=str, required=True)
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--out_path", type=str, default="kul_all_subjects.lmdb")
    parser.add_argument("--audio_fs", type=int, default=16000)
    parser.add_argument("--segment_len_sec", type=float, default=4.0)
    parser.add_argument("--hop_len_sec", type=float, default=1.0)

    args = parser.parse_args()

    print("=" * 50)
    print("KUL LMDB Builder (All Subjects, Single File)")
    print("=" * 50)
    print(f"CSV:          {args.csv_path}")
    print(f"EEG Dir:      {args.eeg_dir}")
    print(f"Audio Dir:    {args.audio_dir}")
    print(f"Audio SR:     {args.audio_fs} Hz")
    print(f"Segment:      {args.segment_len_sec}s, Hop: {args.hop_len_sec}s")
    print(f"Output:       {args.out_path}")

    ds = KULRawDataset(
        csv_path=args.csv_path,
        eeg_dir=args.eeg_dir,
        audio_dir=args.audio_dir,
        segment_len_sec=args.segment_len_sec,
        hop_len_sec=args.hop_len_sec,
        audio_fs=args.audio_fs,
        target_eeg_fs=128
    )

    print(f"\nTotal: {len(ds.trial_info)} trials -> {len(ds)} segments")
    write_lmdb(ds, args.out_path)
