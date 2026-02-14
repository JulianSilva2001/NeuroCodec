import torch
import h5py, os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.distributed import DistributedSampler

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

def load_NeuroCodecDataset(root, subset='train', batch_size=4, num_gpus=1, shuffle=None):
    """
    Loader for Cocktail Party Dataset (H5)
    """
    dataset = NeuroCodecDataset(root, mode=subset)
    
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

class KULNeuroCodecDataset(Dataset):
    def __init__(self, lmdb_path, indices=None, target_fs=44100, original_fs=8000):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.indices = indices # List of LMDB keys (integers)
        self.env = None
        self.txn = None
        
        # Resampler: 8kHz -> 44.1kHz
        self.resampler = torchaudio.transforms.Resample(orig_freq=original_fs, new_freq=target_fs)
        
        # Open once to get length if indices not provided
        if self.indices is None:
            temp_env = lmdb.open(self.lmdb_path, subdir=False, readonly=True, lock=False)
            with temp_env.begin(write=False) as txn:
                len_bytes = txn.get(b"__len__")
                total_len = int(len_bytes.decode("ascii")) if len_bytes else 0
            temp_env.close()
            self.indices = list(range(total_len))
            
    def _init_db(self):
        self.env = lmdb.open(self.lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()
            
        real_idx = self.indices[idx]
        key = f"{real_idx}".encode("ascii")
        byteflow = self.txn.get(key)
        
        data = pickle.loads(byteflow)
        
        # 1. Audio (Mixture/Target) - Resample 8k -> 44.1k
        # Data is FP16 (Usually), convert to FP32
        mix = data['mixture'].float()   # (T,)
        clean = data['target'].float()  # (T,)
        
        # Ensure (1, T) for resampler
        if mix.ndim == 1: mix = mix.unsqueeze(0)
        if clean.ndim == 1: clean = clean.unsqueeze(0)
        
        # Normalize Audio (KUL is unnormalized, DAC expects -1..1)
        # Peak Normalize per sample
        max_val = max(mix.abs().max(), clean.abs().max())
        if max_val > 1.0:
            mix = mix / max_val
            clean = clean / max_val
        
        mix_resampled = self.resampler(mix)
        clean_resampled = self.resampler(clean)
        
        # 2. EEG
        eeg = data['eeg'].float() # (C, T_eeg)
        if hasattr(eeg, 'numpy'): # If it's a tensor
             pass
        else: 
             eeg = torch.from_numpy(eeg).float()
             
        # Normalize EEG? 
        # KUL data might be raw. NeuroCodec expects standard deviation 
        # similar to what it saw during training ~0.005?
        # Let's verify KUL range. Usually uV. 
        # For now return raw, user can normalize in training if needed.
        
        return mix_resampled, eeg, clean_resampled

def load_KUL_NeuroCodecDataset(lmdb_path, subset='train', batch_size=4, num_gpus=1, target_fs=44100, shuffle=None):
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
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)
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
            # Build Index
            subject_indices = {}
            for i in range(total_len):
                key = f"{i}".encode("ascii")
                # We need to peek. Since pickle loads full object, this is slow.
                # Optimization: KUL2.py saves 'subject' in top level dict.
                data = pickle.loads(txn.get(key))
                subj = data.get('subject', 'Unknown')
                
                if subj not in subject_indices:
                    subject_indices[subj] = []
                subject_indices[subj].append(i)
                
                if i % 100 == 0:
                    print(f"Scanned {i}/{total_len}...", end='\r')
            
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
    
    dataset = KULNeuroCodecDataset(lmdb_path, indices=target_indices, target_fs=target_fs)
    
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
        num_workers=4,
        sampler=sampler,
        pin_memory=True
    )
    return loader
