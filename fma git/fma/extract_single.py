#!/usr/bin/env python3
"""Extract FMA-compatible features from a single audio file."""

import sys
import warnings
import numpy as np
from scipy import stats
import pandas as pd
import librosa


def compute_features(filepath):
    feature_sizes = dict(
        chroma_stft=12, chroma_cqt=12, chroma_cens=12,
        tonnetz=6, mfcc=20, rmse=1, zcr=1,
        spectral_centroid=1, spectral_bandwidth=1,
        spectral_contrast=7, spectral_rolloff=1,
    )
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    cols = []
    for name, size in feature_sizes.items():
        for moment in moments:
            cols.extend((name, moment, f'{i+1:02d}') for i in range(size))
    index = pd.MultiIndex.from_tuples(cols, names=('feature', 'statistics', 'number')).sort_values()

    features = pd.Series(index=index, dtype=np.float32, name=filepath)

    def feature_stats(name, values):
        features[name, 'mean']     = np.mean(values, axis=1)
        features[name, 'std']      = np.std(values, axis=1)
        features[name, 'skew']     = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median']   = np.median(values, axis=1)
        features[name, 'min']      = np.min(values, axis=1)
        features[name, 'max']      = np.max(values, axis=1)

    warnings.filterwarnings('ignore')

    x, sr = librosa.load(filepath, sr=None, mono=True)

    f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
    feature_stats('zcr', f)

    cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=84, tuning=None))
    f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cqt', f)
    f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cens', f)
    f = librosa.feature.tonnetz(chroma=f)
    feature_stats('tonnetz', f)
    del cqt

    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    del x

    f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
    feature_stats('chroma_stft', f)

    f = librosa.feature.rms(S=stft)
    feature_stats('rmse', f)

    f = librosa.feature.spectral_centroid(S=stft)
    feature_stats('spectral_centroid', f)
    f = librosa.feature.spectral_bandwidth(S=stft)
    feature_stats('spectral_bandwidth', f)
    f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
    feature_stats('spectral_contrast', f)
    f = librosa.feature.spectral_rolloff(S=stft)
    feature_stats('spectral_rolloff', f)

    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    del stft
    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    feature_stats('mfcc', f)

    return features


OUTPUT_CSV = 'customFeatures.csv'


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: py extract_single.py <audio_file>')
        sys.exit(1)

    filepath = sys.argv[1]
    print(f'Extracting features from: {filepath}')
    features = compute_features(filepath)

    row = features.to_frame().T
    row.index = [filepath]

    if pd.io.common.file_exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV, index_col=0, header=[0, 1, 2])
        combined = pd.concat([existing, row])
    else:
        combined = row

    combined.to_csv(OUTPUT_CSV)
    print(f'Saved to {OUTPUT_CSV} ({len(combined)} tracks total)')
