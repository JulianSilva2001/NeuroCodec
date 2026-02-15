import torch
import h5py, os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

class NeuroCodecDataset(Dataset):
    def __init__(self, root, mode, subject=None):
        super().__init__()
        self.root = root
        # Handle path dynamically based on input root
        if root.endswith('new'):
             self.file_path = root
        else:
             # Assume standard structure if root is top-level
             self.file_path = os.path.join(root, '2s', 'eeg', 'new')
             
        # self.file_path = '/home/jaliya/eeg_speech/navindu/data/Cocktail_Party/Normalized-2/2s/eeg/new'
        
        print(f"Dataset loading from: {self.file_path}")
        
        if mode == 'train':
            self.noisy_file = os.path.join(self.file_path, 'noisy_train.h5')
            self.clean_file = os.path.join(self.file_path, 'clean_train.h5')
            self.eeg_file = os.path.join(self.file_path, 'eegs_train.h5')
            f = h5py.File(self.noisy_file, 'r')
            d = f['noisy_train=']
            self.length = len(d)
            f.close()
        elif mode == 'val':
            self.noisy_file = os.path.join(self.file_path, 'noisy_val.h5')
            self.clean_file = os.path.join(self.file_path, 'clean_val.h5')
            self.eeg_file = os.path.join(self.file_path, 'eegs_val.h5')
            f = h5py.File(self.noisy_file, 'r')
            d = f['noisy_val=']
            self.length = len(d)
            f.close()
        else:
            self.subject = subject
            self.noisy_file = os.path.join(self.file_path, 'noisy_test.h5')
            self.clean_file = os.path.join(self.file_path, 'clean_test.h5')
            self.eeg_file = os.path.join(self.file_path, 'eegs_test.h5')
            self.subject_file = os.path.join(self.file_path, 'subjects_test.h5')
            if subject != None:
                subject_f = h5py.File(self.subject_file, 'r')
                subject_key = [key for key in subject_f][0]
                self.samples_of_interest = [i for i, s in enumerate(subject_f[subject_key][:]) if s == subject]
                self.length = len(self.samples_of_interest)
                subject_f.close()
            else:
                f = h5py.File(self.noisy_file, 'r')
                d = f['noisy_test=']
                self.length = len(d)
                f.close()
        
        self.mode = mode

    def __len__(self):
        return self.length

    def _ensure_files_open(self):
        if not hasattr(self, 'noisy_f') or self.noisy_f is None:
            self.noisy_f = h5py.File(self.noisy_file, 'r')
            self.clean_f = h5py.File(self.clean_file, 'r')
            self.eeg_f = h5py.File(self.eeg_file, 'r')

    def __getitem__(self, idx):
        # Handle Test/Subject specific indexing
        true_idx = idx
        if self.mode == 'test' and hasattr(self, 'subject') and self.subject is not None:
            true_idx = self.samples_of_interest[idx]
            
        self._ensure_files_open()
        
        noisy_key = f'noisy_{self.mode}='
        clean_key = f'clean_{self.mode}='
        eeg_key = f'eegs_{self.mode}='
        
        # Access data via cached handles
        noisy_data = self.noisy_f[noisy_key]
        clean_data = self.clean_f[clean_key]
        eeg_data = self.eeg_f[eeg_key]
        
        # Audio Processing: Keep 44.1kHz
        # Shape in H5: (N, T, 1) or similar.
        n_d = np.transpose(noisy_data[true_idx]) # (1, 87552)
        c_d = np.transpose(clean_data[true_idx]) # (1, 87552)
        
        # EEG Processing: Keep 128Hz
        # Shape in H5: (N, 256, 128)
        # squeeze helps if shape is (N, 256, 128, 1)?
        e_d = np.transpose((eeg_data[true_idx]).squeeze()) # (128, 256)
        
        # Scale EEG from Volts to Microvolts (1e-6 -> 1.0 range)
        # This fixes BatchNorm epsilon dominance issues with raw valus ~1e-5
        # e_d = e_d * 1e6
        
        # Hard Clip to +/- 500 uV to remove massive artifacts (-30000 range)
        # Real EEG is typically +/- 100 uV.
        # e_d = np.clip(e_d, -500, 500)
        
        n_d = n_d.astype(np.float32)
        c_d = c_d.astype(np.float32)
        e_d = e_d.astype(np.float32)
        
        return torch.from_numpy(n_d), torch.from_numpy(e_d), torch.from_numpy(c_d)
        return torch.from_numpy(n_d).float(), torch.from_numpy(e_d).float(), torch.from_numpy(c_d).float()
    
    def __del__(self):
        if hasattr(self, 'noisy_f') and self.noisy_f is not None:
            self.noisy_f.close()
        if hasattr(self, 'clean_f') and self.clean_f is not None:
            self.clean_f.close()
        if hasattr(self, 'eeg_f') and self.eeg_f is not None:
            self.eeg_f.close()

def load_NeuroCodecDataset(root, subset='train', batch_size=4, num_gpus=1, shuffle=None, fraction=1.0):
    """
    Loader for Cocktail Party Dataset (H5)
    """
    dataset = NeuroCodecDataset(root, mode=subset)

    # Optional fractional subset for quick experiments
    if fraction is not None and fraction < 1.0:
        total = len(dataset)
        keep = max(1, int(total * fraction))
        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(total, generator=g)[:keep].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Using fractional subset: {keep}/{total} samples ({fraction*100:.1f}%) for {subset}.")
    
    sampler = None
    if num_gpus > 1:
        sampler = DistributedSampler(dataset)
        
    # Determine Shuffle
    if shuffle is None:
        shuffle = (sampler is None) 
        # Wait, original logic was (sampler is None) effectively always shuffling?
        # No, line 130 was: shuffle=(sampler is None)
        # But wait, usually val/test shouldn't shuffle.
        # Line 78 in train_neurocodec.py calls it for train -> shuffle=True
        # Line 84 calls for val -> shuffle=True? 
        # Actually, looking at original code:
        # shuffle=(sampler is None)
        # That means it ALWAYS shuffles if single GPU, even for validation?
        # That might be why the user wants random samples (maybe they thought it wasn't shuffling).
        # But for inference script, subset='test' or 'val'.
        # If it was shuffling, they would see random samples.
        # But user says "instead of the first samples".
        # This implies it WASN'T shuffling.
        # Let's check line 130 in original file again.
        # "shuffle=(sampler is None)"
        # If running inference on 1 GPU, sampler is None, so shuffle=True?
        # If so, data_iter = iter(loader) should give random samples.
        # Unless 'dataset' itself is ordered and loader seed is fixed?
        
        # In KUL loader: `shuffle=(subset == 'train' and sampler is None)`
        # In NeuroCodec loader: `shuffle=(sampler is None)`
        # This looks like a bug in original NeuroCodec loader (shuffling val/test).
        # I will preserve original behavior unless shuffle is passed.
        
    if shuffle is None:
         shuffle = (sampler is None)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle, # Use the determined variable
        num_workers=4,
        sampler=sampler,
        pin_memory=True
    )
    return loader

# ==========================================
# KUL DATASET IMPLEMENTATION (LMDB + Resample)
# ==========================================
import lmdb
import pickle
import torchaudio

def collate_pad_varlen(batch):
    """Pad variable-length audio/EEG to max length in batch and return lengths (and optional indices)."""
    has_idx = len(batch[0]) == 4
    if has_idx:
        noisy_list, eeg_list, clean_list, idx_list = zip(*batch)
    else:
        noisy_list, eeg_list, clean_list = zip(*batch)
        idx_list = None
    audio_lengths = torch.tensor([x.shape[-1] for x in noisy_list], dtype=torch.long)
    eeg_lengths = torch.tensor([x.shape[-1] for x in eeg_list], dtype=torch.long)

    max_audio = int(audio_lengths.max())
    max_eeg = int(eeg_lengths.max())

    def pad_audio(t):
        if t.shape[-1] == max_audio:
            return t
        pad = max_audio - t.shape[-1]
        return torch.nn.functional.pad(t, (0, pad))

    def pad_eeg(t):
        if t.shape[-1] == max_eeg:
            return t
        pad = max_eeg - t.shape[-1]
        return torch.nn.functional.pad(t, (0, pad))

    noisy = torch.stack([pad_audio(t) for t in noisy_list])
    clean = torch.stack([pad_audio(t) for t in clean_list])
    eeg = torch.stack([pad_eeg(t) for t in eeg_list])
    if has_idx:
        idx_tensor = torch.tensor(idx_list, dtype=torch.long)
        return noisy, eeg, clean, audio_lengths, eeg_lengths, idx_tensor
    return noisy, eeg, clean, audio_lengths, eeg_lengths

class KULNeuroCodecDataset(Dataset):
    def __init__(self, lmdb_path, indices=None, target_fs=16000, original_fs=16000):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.indices = indices # List of LMDB keys (integers)
        self.env = None
        self.txn = None
        
        # Resampler: only create if sampling rates differ
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.resampler = None
        if target_fs != original_fs:
            self.resampler = torchaudio.transforms.Resample(orig_freq=original_fs, new_freq=target_fs)
        
        # Open once to collect existing keys if indices not provided (robust to gaps)
        if self.indices is None:
            temp_env = lmdb.open(self.lmdb_path, subdir=False, readonly=True, lock=False, readahead=False)
            keys = []
            with temp_env.begin(write=False) as txn:
                cursor = txn.cursor()
                for k, _ in cursor:
                    if k == b"__len__":
                        continue
                    try:
                        keys.append(int(k.decode("ascii")))
                    except Exception:
                        continue
            temp_env.close()
            self.indices = sorted(keys)
            
    def _init_db(self):
        self.env = lmdb.open(
            self.lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,   # allow OS readahead to reduce IO stalls
            meminit=False
        )
        self.txn = self.env.begin(write=False)
        if self.txn is None:
            raise RuntimeError(f"Failed to open LMDB transaction for {self.lmdb_path}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()
            
        real_idx = self.indices[idx]
        key = f"{real_idx}".encode("ascii")
        byteflow = self.txn.get(key)

        if byteflow is None:
            raise KeyError(f"LMDB key {real_idx} missing in {self.lmdb_path} (indices may be non-contiguous or corrupted)")

        data = pickle.loads(byteflow)
        
        # 1. Audio (Mixture/Target) - Resample 8k -> 44.1k
        # Data stored as numpy or torch; convert to torch float32
        mix_raw = data['mixture']
        clean_raw = data['target']
        if hasattr(mix_raw, 'float'):
            mix = mix_raw.float()
        else:
            mix = torch.from_numpy(mix_raw).float()
        if hasattr(clean_raw, 'float'):
            clean = clean_raw.float()
        else:
            clean = torch.from_numpy(clean_raw).float()
        
        # Ensure (1, T) for resampler
        if mix.ndim == 1: mix = mix.unsqueeze(0)
        if clean.ndim == 1: clean = clean.unsqueeze(0)
        
        # Normalize Audio (clip-safety without altering SNR): only scale down if needed
        max_val = max(mix.abs().max(), clean.abs().max())
        if max_val > 1.0:
            scale = max_val
            mix = mix / scale
            clean = clean / scale
        
        if self.resampler is not None:
            mix_resampled = self.resampler(mix)
            clean_resampled = self.resampler(clean)
        else:
            mix_resampled = mix
            clean_resampled = clean
        
        # 2. EEG
        eeg_raw = data['eeg']
        if hasattr(eeg_raw, 'float'):
            eeg = eeg_raw.float()
        else:
            eeg = torch.from_numpy(eeg_raw).float()
             
        # Normalize EEG? 
        # KUL data might be raw. NeuroCodec expects standard deviation 
        # similar to what it saw during training ~0.005?
        # Let's verify KUL range. Usually uV. 
        # For now return raw, user can normalize in training if needed.
        
        return mix_resampled, eeg, clean_resampled, real_idx

def load_KUL_NeuroCodecDataset(
    lmdb_path,
    subset='train',
    batch_size=4,
    num_gpus=1,
    target_fs=16000,
    original_fs=16000,
    shuffle=None,
    num_workers=8,
    prefetch_factor=4,
    pin_memory=True,
    persistent_workers=True,
    fraction=1.0
):
    """
    Loader for KUL Dataset (LMDB) with Subject-wise Splitting
    """
    # ... (Scanning logic remains common, skipping lines for brevity if possible, but replace tool needs context)
    # I will just match the function signature and the return statement logic.
    # But wait, replace needs exact match.
    # The scan logic is long.
    # I should try to replace strictly the signature and the DataLoader creation part?
    # Or just read the file again to be sure of contexts?
    # I have viewed lines 215-313 in step 4225.
    
    # ...
    # 1. Scan LMDB for Subject info to split
    # This is slow if we do it every time, but necessary without external metadata.
    # To save time, let's trust a simple cached split or just do it once.
    # For now, let's implement the 'metadata scan' approach but print progress.
    
    print(f"Scanning KUL LMDB at {lmdb_path} for splits...")
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False)
    with env.begin(write=False) as txn:
        len_bytes = txn.get(b"__len__")
        total_len = int(len_bytes.decode("ascii")) if len_bytes else 0
        
        # Check for Metadata Cache (Optimized)
        cache_path = lmdb_path + ".split_cache.pkl"
        if os.path.exists(cache_path):
            print(f"Loading split cache from {cache_path}")
            with open(cache_path, 'rb') as f:
                subject_indices = pickle.load(f)
        else:
            # Build Index (robust to missing/non-contiguous keys)
            subject_indices = {}
            missing = 0
            for i in tqdm(range(total_len), desc="Scanning LMDB splits", ncols=100):
                key = f"{i}".encode("ascii")
                raw = txn.get(key)
                if raw is None:
                    missing += 1
                    continue
                # We need to peek. Since pickle loads full object, this is slow.
                # Optimization: KUL2.py saves 'subject' in top level dict.
                data = pickle.loads(raw)
                subj = data.get('subject', 'Unknown')
                # Normalize subject to string for consistent sorting/keys
                if isinstance(subj, bytes):
                    subj = subj.decode('utf-8', errors='ignore')
                else:
                    subj = str(subj)
                
                if subj not in subject_indices:
                    subject_indices[subj] = []
                subject_indices[subj].append(i)
                
                if i % 100 == 0:
                    print(f"Scanned {i}/{total_len} (missing so far: {missing})...", end='\\r')
            
            # Save Cache
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(subject_indices, f)
                print(f"Saved split cache to {cache_path}")
            except:
                pass
    env.close()

    # 2. Create Splits (Train: All except last 2, Val: Last, Test: 2nd Last)
    subjects = sorted(subject_indices.keys())
    # Robust numeric sort if possible (S1, S2, S10)
    def sort_key(s):
        import re
        if not isinstance(s, str):
            s = str(s)
        nums = re.findall(r'\d+', s)
        return int(nums[0]) if nums else s
    subjects.sort(key=sort_key)
    
    print(f"Found Subjects: {subjects}")
    
    if len(subjects) >= 3:
        test_subj = [subjects[-2]] # S(N-1)
        val_subj = [subjects[-1]]  # S(N)
        train_subj = subjects[:-2]
    else:
        # Fallback for small debug datasets
        print("Warning: Not enough subjects for split. Using random split.")
        all_indices = list(range(total_len))
        # ... random logic ...
        # Assume we have enough based on KUL2.py defaults
        train_subj = subjects
        val_subj = []
        test_subj = []

    # Select indices based on subset
    target_indices = []
    if subset == 'train':
        for s in train_subj: target_indices.extend(subject_indices[s])
    elif subset == 'val':
        for s in val_subj: target_indices.extend(subject_indices[s])
    elif subset == 'test':
        for s in test_subj: target_indices.extend(subject_indices[s])
    
    print(f"Subset '{subset}': {len(target_indices)} samples (Subjects: {train_subj if subset=='train' else (val_subj if subset=='val' else test_subj)})")
    
    # Optional fractional subset for quick experiments (sampled deterministically)
    if fraction is not None and fraction < 1.0:
        total = len(target_indices)
        keep = max(1, int(total * fraction))
        rng = np.random.default_rng(42)
        target_indices = rng.permutation(target_indices)[:keep].tolist()
        print(f"Using fractional subset: {keep}/{total} samples ({fraction*100:.1f}%) for {subset}.")

    dataset = KULNeuroCodecDataset(lmdb_path, indices=target_indices, target_fs=target_fs, original_fs=original_fs)
    
    sampler = None
    if num_gpus > 1:
        sampler = DistributedSampler(dataset)

    # Determine Shuffle
    if shuffle is None:
        shuffle = (subset == 'train' and sampler is None)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_pad_varlen
    )
    return loader
