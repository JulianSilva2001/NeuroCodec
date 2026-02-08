import torch
import h5py, os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.distributed import DistributedSampler

class NeuroCodecDataset(Dataset):
    def __init__(self, root, mode, subject=None):
        super().__init__()
        self.root = root
        # Using the same hardcoded path as dataset.py for now, but user passed root
        # self.file_path = '/home/jaliya/eeg_speech/navindu/data/Cocktail_Party/Normalized/2s/eeg/new'
        # Actually better to use 'root' if passed, but dataset.py ignored it.
        # Let's inspect dataset.py again. It ignores root=root inside init except for super?
        # No, it sets self.file_path hardcoded.
        # We should respect the hardcoded path as it seems to be where data is.
        
        self.file_path = '/home/jaliya/eeg_speech/navindu/data/Cocktail_Party/Normalized-2/2s/eeg/new'
        
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
        e_d = e_d * 1e6
        
        n_d = n_d.astype(np.float32)
        c_d = c_d.astype(np.float32)
        e_d = e_d.astype(np.float32)
        
        return torch.from_numpy(n_d), torch.from_numpy(e_d), torch.from_numpy(c_d)
    
    def __del__(self):
        if hasattr(self, 'noisy_f') and self.noisy_f is not None:
            self.noisy_f.close()
        if hasattr(self, 'clean_f') and self.clean_f is not None:
            self.clean_f.close()
        if hasattr(self, 'eeg_f') and self.eeg_f is not None:
            self.eeg_f.close()

def load_NeuroCodecDataset(root, subset, batch_size, num_gpus=1):
    dataset = NeuroCodecDataset(root=root, mode=subset)
    kwargs = {"batch_size": batch_size, "num_workers": 8, "pin_memory": True, "drop_last": False}
    if num_gpus > 1:
        train_sampler = DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **kwargs)
    else:
        if subset == 'train':
            dataloader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=True, **kwargs)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=False, **kwargs)

    return dataloader
