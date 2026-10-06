# -*- coding: utf-8 -*-
"""Shared helpers: loading, preprocessing, sample-level voting, metrics, CV folds."""
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"
CLASSES = np.array([0, 1, 2, 5])
CLASS_NAMES = {0: "spinal_cord", 1: "nucleus_pulposus", 2: "adipose_tissue", 5: "nerve_root"}
WAVE = 14  # signals per biological sample


def load_split(split):
    """Raw signals (n, 896) and labels (n,) of one split ('train'/'val'/'test')."""
    x = np.load(DATA_DIR / f"{split}_data.npy").astype(np.float64)
    y = np.load(DATA_DIR / f"{split}_label.npy")
    return x, y


def preprocess(x):
    """Project spec: 127/128 -> 127.5, then (x - 127.5) / 127.5 -> [-1, 1]."""
    x = np.asarray(x, dtype=np.float64).copy()
    x[(x == 127) | (x == 128)] = 127.5
    return (x - 127.5) / 127.5


def sample_ids(n):
    """Group id (biological sample index) of every row."""
    return np.arange(n) // WAVE


def vote_to_samples(preds, probs=None):
    """Majority vote over each block of 14 signal predictions.

    preds: (n,) predicted labels. probs: (n, 4) optional, columns ordered as
    CLASSES, used only to break ties (largest mean probability).
    Returns (n // 14,) sample-level labels.
    """
    n_blocks = len(preds) // WAVE
    p = np.asarray(preds).reshape(n_blocks, WAVE)
    q = np.asarray(probs).reshape(n_blocks, WAVE, -1) if probs is not None else None
    out = np.empty(n_blocks, dtype=int)
    for b in range(n_blocks):
        counts = np.array([(p[b] == c).sum() for c in CLASSES])
        tied = np.flatnonzero(counts == counts.max())
        if len(tied) > 1 and q is not None:
            tied = [tied[np.argmax(q[b].mean(axis=0)[tied])]]
        out[b] = CLASSES[tied[0]]
    return out


def eval_metrics(y_true, y_pred):
    """One-vs-rest per-class metrics + micro aggregation, per project spec.

    micro.sensitivity == micro-precision == overall accuracy in single-label
    multi-class problems. The spec's literal formula (TP+TN)/(...) on OvR micro
    sums double-counts TN of correct samples; it is reported as
    micro.accuracy_ovr_spec for reference, (1 + accuracy) / 2.
    """
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    spec = np.divide(tn, tn + fp, out=np.zeros_like(tp), where=(tn + fp) > 0)
    per_class = {
        CLASS_NAMES[c]: {
            "precision": float(prec[i]), "recall": float(rec[i]),
            "specificity": float(spec[i]),
            "f1": float(2 * prec[i] * rec[i] / (prec[i] + rec[i])) if prec[i] + rec[i] else 0.0,
            "support": int(cm[i].sum()),
        } for i, c in enumerate(CLASSES)
    }
    TP, FP, FN, TN = tp.sum(), fp.sum(), fn.sum(), tn.sum()
    n = cm.sum()
    return {
        "micro": {
            "accuracy": float(np.trace(cm) / n),
            "sensitivity": float(TP / (TP + FN)),
            "specificity": float(TN / (TN + FP)),
            "accuracy_ovr_spec": float((TP + TN) / (TP + TN + FP + FN)),
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES.tolist(),
    }


def sample_metrics(y, pred, scores=None):
    """Sample-level metrics from signal-level predictions (majority vote)."""
    y_smp = y.reshape(-1, WAVE)[:, 0]
    m = eval_metrics(y_smp, vote_to_samples(pred, scores))
    m["signal_accuracy"] = float((np.asarray(pred) == y).mean())
    return m


def cv_folds(y, n_splits=5, seed=42):
    """Sample-level stratified group k-fold: whole 14-signal blocks stay together."""
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return skf.split(np.zeros_like(y), y, groups=sample_ids(len(y)))


def plot_cm(cm, path, title):
    """Confusion-matrix heatmap for the report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    names = [CLASS_NAMES[c] for c in CLASSES]
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(CLASSES)), yticks=range(len(CLASSES)),
           xticklabels=names, yticklabels=names,
           xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def demo():
    x = np.array([[127, 128, 255, 0]], dtype=float)
    assert preprocess(x).tolist() == [[0.0, 0.0, 1.0, -1.0]]

    _, y = load_split("train")
    assert (y.reshape(-1, WAVE) == y.reshape(-1, WAVE)[:, :1]).all()

    m = eval_metrics([0, 1, 2, 5], [0, 1, 2, 5])
    assert m["micro"]["accuracy"] == 1.0
    m = eval_metrics([0, 1, 0, 0], [0, 0, 0, 0])  # 3/4 correct
    assert m["micro"]["accuracy"] == 0.75, m
    assert abs(m["micro"]["accuracy_ovr_spec"] - 0.875) < 1e-12, m  # (1+acc)/2

    assert vote_to_samples([1] * 7 + [2] * 7) == 1            # tie -> lowest label
    probs = np.zeros((14, 4))
    probs[:7, 1], probs[7:, 2] = 0.3, 0.9                     # tie broken by probs
    assert vote_to_samples([1] * 7 + [2] * 7, probs) == 2

    folds = list(cv_folds(y))
    assert len(folds) == 5 and sum(len(te) for _, te in folds) == len(y)
    gid = sample_ids(len(y))
    for tr, te in folds:
        assert not set(gid[tr]) & set(gid[te]), "group leakage between folds"
    print("common.py self-check passed")


if __name__ == "__main__":
    demo()
