import argparse
import logging
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
matplotlib_cache_dir = PROJECT_DIR / '.matplotlib-cache'
font_cache_dir = PROJECT_DIR / '.cache'

matplotlib_cache_dir.mkdir(exist_ok=True)
font_cache_dir.mkdir(exist_ok=True)

os.environ.setdefault('MPLCONFIGDIR', str(matplotlib_cache_dir))
os.environ.setdefault('XDG_CACHE_HOME', str(font_cache_dir))

logger.info('Using Matplotlib cache directory: %s', matplotlib_cache_dir)
logger.info('Using font cache directory: %s', font_cache_dir)
logger.info('Importing numpy, MNE, and matplotlib')

import mne
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

logger.info('Imports finished')

CHANNEL_NAMES_64 = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
    'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
    'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
    'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
    'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
    'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
    'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
    'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2',
]

importance_cmap = LinearSegmentedColormap.from_list(
    'importance_purple_to_red',
    ['#4b0082', '#2a6fdb', '#19a974', '#ffe45c', '#d90429'],
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot EEG channel-importance topomap from CSV data.',
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='results/xai/all_runs_new/channel_occlusion/channel_occlusion_importance.csv',
        help='Path to the channel importance CSV file. Default: results/xai/all_runs_new/channel_occlusion/channel_occlusion_importance.csv',
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='drop_si_sdr',
        choices=['drop_si_sdr', 'drop_estoi', 'drop_pesq', 'drop_stoi', 'drop_latent_mse', 'combined'],
        help='Which importance metric to use. Default: drop_si_sdr',
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize importance values to 0-1 range.',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save plot to this file path (e.g., topomap.png). If not provided, display interactively.',
    )
    return parser.parse_args()


def load_channel_importance(csv_path, metric='drop_si_sdr'):
    """Load channel importance from CSV file."""
    logger.info('Loading channel importance from %s', csv_path)
    df = pd.read_csv(csv_path)
    
    # Create mapping from channel_idx to channel name
    ch_idx_to_name = {i: CHANNEL_NAMES_64[i] for i in range(len(CHANNEL_NAMES_64))}
    
    # Map channel indices to importance values
    importance_dict = {}
    for _, row in df.iterrows():
        ch_idx = row['channel_idx']
        ch_name = ch_idx_to_name[ch_idx]
        
        if metric == 'combined':
            # Combine all metrics (normalize and average)
            metrics = ['drop_si_sdr', 'drop_estoi', 'drop_pesq', 'drop_stoi', 'drop_latent_mse']
            values = np.array([abs(row[m]) for m in metrics])
            importance_dict[ch_name] = np.mean(values)
        else:
            importance_dict[ch_name] = abs(row[metric])
    
    # Create importance array in the order of CHANNEL_NAMES_64
    importance = np.array([importance_dict[ch_name] for ch_name in CHANNEL_NAMES_64])
    
    logger.info('Loaded importance values for %d channels using metric: %s', len(importance), metric)
    logger.info('Importance range: [%.6f, %.6f]', importance.min(), importance.max())
    
    return importance


def main():
    args = parse_args()
    start_time = time.perf_counter()
    logger.info('Starting EEG channel importance topomap script')

    # Load importance data from CSV
    importance = load_channel_importance(args.csv, metric=args.metric)
    
    # Normalize if requested
    if args.normalize:
        logger.info('Normalizing importance values to 0-1 range')
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    
    ch_names = CHANNEL_NAMES_64
    
    logger.info('Creating EEG info for %d channels', len(ch_names))
    info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types='eeg')
    
    logger.info('Loading standard_1020 montage')
    montage = mne.channels.make_standard_montage('standard_1020')
    logger.info('Applying montage')
    info.set_montage(montage)

    logger.info('Creating matplotlib figure')
    fig, ax = plt.subplots(figsize=(8, 8))

    logger.info('Calculating and drawing topomap')
    im, cm = mne.viz.plot_topomap(
        importance,
        info,
        axes=ax,
        show=False,
        contours=6,
        sphere=0.1,
        cmap=importance_cmap,
        vlim=(importance.min(), importance.max()),
        sensors=True,
    )

    logger.info('Adding colorbar and title')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label=f'Channel Importance ({args.metric})')
    ax.set_title(f'EEG Channel Importance Topomap\nMetric: {args.metric}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    
    elapsed = time.perf_counter() - start_time
    logger.info('Plot is ready after %.2f seconds', elapsed)

    if args.output:
        logger.info('Saving plot to %s', args.output)
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        logger.info('Plot saved successfully')
    
    backend = plt.get_backend()
    if backend.lower().endswith('agg'):
        logger.info('Matplotlib backend is %s, so no interactive window will open', backend)
    else:
        logger.info(
            'Opening plot window with %s backend. This can block until you close the window.',
            backend,
        )
        plt.show()
        logger.info('Plot window closed')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logger.exception('Topomap script failed')
        raise
