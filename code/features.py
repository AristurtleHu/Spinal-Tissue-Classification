# -*- coding: utf-8 -*-
"""Task 1/2 features: DWT (db4, level 5) coefficients + 6 statistics per component.

Components: cA5 (approximation) and cD5..cD1 (details), computed on the
preprocessed signal. Statistics: mean, variance, median, kurtosis (Fisher,
scipy default), RMS, skewness. 6 components x 6 stats = 36 features/signal.
"""
import numpy as np
import pywt
from scipy.stats import kurtosis, skew

WAVELET = "db4"
LEVEL = 5
STAT_NAMES = ("mean", "var", "median", "kurtosis", "rms", "skewness")


def _stats(c):
    return [float(c.mean()), float(c.var()), float(np.median(c)),
            float(kurtosis(c)), float(np.sqrt(np.mean(c ** 2))), float(skew(c))]


def dwt_features(signal):
    """One signal (896,) -> feature vector (36,)."""
    coeffs = pywt.wavedec(signal, WAVELET, level=LEVEL)  # [cA5, cD5, ..., cD1]
    return np.concatenate([_stats(c) for c in coeffs])


def extract_features(x):
    """(n, 896) -> (n, 36)."""
    return np.stack([dwt_features(s) for s in x])


def feature_names():
    comps = [f"cA{LEVEL}"] + [f"cD{l}" for l in range(LEVEL, 0, -1)]
    return [f"{c}_{s}" for c in comps for s in STAT_NAMES]


if __name__ == "__main__":
    from sklearn.feature_selection import f_classif
    from common import load_split, preprocess

    x, y = load_split("train")
    f = extract_features(preprocess(x))
    assert f.shape == (len(x), len(feature_names())) and np.isfinite(f).all(), f.shape

    names = np.array(feature_names())
    keep = f.var(axis=0) > 0  # flat baseline makes some detail stats constant (always 0)
    F = np.full(f.shape[1], np.nan)
    F[keep] = f_classif(f[:, keep], y)[0]  # signal-level F-score, sanity peek only
    print(f"constant features: {list(names[~keep])}")
    print("top discriminative features (train, signal level):")
    order = np.argsort(np.nan_to_num(F, nan=-np.inf))[::-1][:8]
    for i in order:
        print(f"  {names[i]:>20}  F={F[i]:8.1f}")
    print(f"features.py self-check passed: {f.shape}")
