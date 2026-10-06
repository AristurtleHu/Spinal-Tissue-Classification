# -*- coding: utf-8 -*-
"""Sample-level aggregation study for the selected Task 2 model (SVM RBF, C=100).

Compares the two strategies offered by the project plus soft voting, using the
same sample-level stratified 5-fold CV on the training set.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import CLASSES, cv_folds, load_split, preprocess, sample_ids, sample_metrics
from features import extract_features

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    RESULTS.mkdir(exist_ok=True)
    x, y = load_split("train")
    f = extract_features(preprocess(x))
    n_feat = f.shape[1]
    g = sample_ids(len(y))
    y_smp = y.reshape(-1, 14)[:, 0]
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=100, random_state=42))

    res = {"majority_vote": [], "soft_vote": [], "averaged_features": []}
    for tr, te in cv_folds(y):
        p = clone(pipe).fit(f[tr], y[tr])
        pred, sc = p.predict(f[te]), p.decision_function(f[te])
        yte = y[te].reshape(-1, 14)[:, 0]
        res["majority_vote"].append(sample_metrics(y[te], pred, sc)["micro"]["accuracy"])
        res["soft_vote"].append(float((CLASSES[sc.reshape(-1, 14, 4).mean(axis=1).argmax(axis=1)] == yte).mean()))
        p_avg = clone(pipe).fit(f[tr].reshape(-1, 14, n_feat).mean(axis=1), y_smp[np.unique(g[tr])])
        res["averaged_features"].append(float((p_avg.predict(f[te].reshape(-1, 14, n_feat).mean(axis=1)) == yte).mean()))

    out = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "folds": [float(a) for a in v]}
           for k, v in res.items()}
    for k, v in out.items():
        print(f"{k:18s} {v['mean']:.4f} +/- {v['std']:.4f}")
    with open(RESULTS / "aggregation_study.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
