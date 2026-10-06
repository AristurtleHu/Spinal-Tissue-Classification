# -*- coding: utf-8 -*-
"""Task 2: ML classification on Task 1 (DWT) features.

Protocol: sample-level stratified 5-fold CV on the training set for model and
hyperparameter selection (scaler fitted inside each fold); the winner is
retrained on the full training set and evaluated on val, then test. Test is
touched only after everything is frozen.
"""
import json
from pathlib import Path

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import (WAVE, eval_metrics, cv_folds, load_split, plot_cm,
                    preprocess, vote_to_samples)
from features import extract_features

RESULTS = Path(__file__).resolve().parent / "results"
SEED = 42


def candidates():
    svc = lambda **kw: SVC(kernel="rbf", random_state=SEED, **kw)
    rf = lambda **kw: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, **kw)
    return {
        "svm_C1":      make_pipeline(StandardScaler(), svc(C=1)),
        "svm_C10":     make_pipeline(StandardScaler(), svc(C=10)),
        "svm_C100":    make_pipeline(StandardScaler(), svc(C=100)),
        "svm_C10_bal": make_pipeline(StandardScaler(), svc(C=10, class_weight="balanced")),
        "rf":          make_pipeline(StandardScaler(), rf()),
        "rf_bal":      make_pipeline(StandardScaler(), rf(class_weight="balanced")),
        "knn_5":       make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "knn_15":      make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
        "knn_45":      make_pipeline(StandardScaler(), KNeighborsClassifier(45)),
    }


def _scores(pipe, x):
    """Per-class scores for tie-breaking votes (probabilities or decision values)."""
    return pipe.predict_proba(x) if hasattr(pipe, "predict_proba") else pipe.decision_function(x)


def cv_eval(pipe, x, y):
    """Five-fold CV; returns per-fold signal- and sample-level accuracy."""
    sig_acc, smp_acc = [], []
    for tr, te in cv_folds(y):
        p = clone(pipe).fit(x[tr], y[tr])
        pred = p.predict(x[te])
        sig_acc.append(float((pred == y[te]).mean()))
        y_smp = y[te].reshape(-1, WAVE)[:, 0]
        smp_acc.append(eval_metrics(y_smp, vote_to_samples(pred, _scores(p, x[te])))["micro"]["accuracy"])
    return sig_acc, smp_acc


def summarize(acc):
    return {"mean": sum(acc) / len(acc), "std": (sum((a - sum(acc) / len(acc)) ** 2 for a in acc) / len(acc)) ** 0.5,
            "folds": acc}


def evaluate_split(pipe, split, tag, out):
    x, y = load_split(split)
    f = extract_features(preprocess(x))
    pred = pipe.predict(f)
    probs = _scores(pipe, f)
    y_smp = y.reshape(-1, WAVE)[:, 0]
    m = eval_metrics(y_smp, vote_to_samples(pred, probs))
    m["signal_accuracy"] = float((pred == y).mean())
    out[tag] = m
    plot_cm(m["confusion_matrix"], RESULTS / f"cm_task2_{tag}.png", f"Task 2 ({pipe.steps[-1][0]}) - {tag}")
    print(f"[{tag}] sample acc={m['micro']['accuracy']:.4f} | sens={m['micro']['sensitivity']:.4f} "
          f"| spec={m['micro']['specificity']:.4f} | signal acc={m['signal_accuracy']:.4f}")
    print("  cm:", m["confusion_matrix"])
    return m


def main():
    RESULTS.mkdir(exist_ok=True)
    x, y = load_split("train")
    f = extract_features(preprocess(x))
    print(f"train features: {f.shape}")

    cv = {}
    for name, pipe in candidates().items():
        sig_acc, smp_acc = cv_eval(pipe, f, y)
        cv[name] = {"signal": summarize(sig_acc), "sample": summarize(smp_acc)}
        print(f"{name:12s} signal {cv[name]['signal']['mean']:.4f} | "
              f"sample {cv[name]['sample']['mean']:.4f} +/- {cv[name]['sample']['std']:.4f}")

    best = max(cv, key=lambda k: cv[k]["sample"]["mean"])
    print(f"selected by CV: {best}")
    with open(RESULTS / "task2_cv.json", "w") as fh:
        json.dump({"cv": cv, "best": best}, fh, indent=2)

    final = candidates()[best].fit(f, y)  # retrain on the full training set
    out = {"config": best}
    evaluate_split(final, "val", "val", out)
    evaluate_split(final, "test", "test", out)  # frozen: evaluated once, never used for tuning
    with open(RESULTS / "task2_eval.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
