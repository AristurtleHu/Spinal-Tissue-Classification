# -*- coding: utf-8 -*-
"""Task 1: explicit interpretable statistical rule from DWT statistical features.

Candidate rules are shallow decision trees (depth x class_weight) over the
Task 1 features; the configuration is selected by sample-level stratified
5-fold CV on the training set, then the rule is refit on all training data and
exported as threshold ("if feature <= value") rules. Validation confirms the
choice; test is evaluated once, after the rule is frozen.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier, export_text

from common import (CLASSES, CLASS_NAMES, cv_folds, load_split, plot_cm,
                    preprocess, sample_metrics)
from features import extract_features, feature_names

RESULTS = Path(__file__).resolve().parent / "results"
SEED = 42


def candidates():
    return {f"depth{d}_{'bal' if w else 'plain'}":
            DecisionTreeClassifier(max_depth=d, class_weight=w, random_state=SEED)
            for d in (3, 4, 5) for w in (None, "balanced")}


def cv_eval(tree, f, y):
    accs = []
    for tr, te in cv_folds(y):
        t = clone(tree).fit(f[tr], y[tr])
        accs.append(sample_metrics(y[te], t.predict(f[te]), t.predict_proba(f[te]))["micro"]["accuracy"])
    return accs


def evaluate(tree, split):
    x, y = load_split(split)
    f = extract_features(preprocess(x))
    return sample_metrics(y, tree.predict(f), tree.predict_proba(f))


def boxplots(f, y, idx, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5))
    for ax, k in zip(axes.ravel(), idx):
        ax.boxplot([f[y == c, k] for c in CLASSES],
                   tick_labels=[CLASS_NAMES[c] for c in CLASSES], showfliers=False)
        ax.set_title(names[k])
        ax.tick_params(axis="x", rotation=20)
    for ax in axes.ravel()[len(idx):]:
        ax.axis("off")
    fig.suptitle("DWT statistical features by tissue type (training signals, outliers hidden)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tree_plot(tree, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    fig, ax = plt.subplots(figsize=(24, 11))
    plot_tree(tree, feature_names=list(names), class_names=[CLASS_NAMES[c] for c in CLASSES],
              filled=True, rounded=True, impurity=False, fontsize=6, ax=ax)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    x, y = load_split("train")
    f = extract_features(preprocess(x))
    names = np.array(feature_names())
    print(f"train features: {f.shape}")

    out = {"cv": {}}
    for name, tree in candidates().items():
        accs = cv_eval(tree, f, y)
        out["cv"][name] = {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "folds": accs}
        print(f"{name:14s} sample CV acc {out['cv'][name]['mean']:.4f} +/- {out['cv'][name]['std']:.4f}")

    best = max(out["cv"], key=lambda k: out["cv"][k]["mean"])
    out["config"] = best
    d, w = best.replace("depth", "").split("_")
    tree = DecisionTreeClassifier(max_depth=int(d), class_weight=None if w == "plain" else "balanced",
                                  random_state=SEED).fit(f, y)
    out["rule_text"] = export_text(tree, feature_names=list(names), decimals=4)
    print(f"selected by training CV: {best}\n{out['rule_text']}")

    # Per-class statistics of the features the rule uses (+ top F-score features), for the report.
    used = np.unique(tree.tree_.feature[tree.tree_.feature >= 0])
    keep = f.var(axis=0) > 0
    F = np.full(f.shape[1], -np.inf)
    F[keep] = f_classif(f[:, keep], y)[0]
    idx = np.unique(np.concatenate([used, np.argsort(F)[::-1][:6]]))
    plot_idx = used[np.argsort(F[used])[::-1][:4]]
    out["feature_stats"] = {
        names[k]: {CLASS_NAMES[c]: {"mean": float(f[y == c, k].mean()),
                                    "std": float(f[y == c, k].std()),
                                    "median": float(np.median(f[y == c, k]))}
                   for c in CLASSES} for k in idx
    }

    for split in ("train", "val", "test"):
        m = evaluate(tree, split)
        out[split] = m
        print(f"[{split}] sample acc={m['micro']['accuracy']:.4f} | sens={m['micro']['sensitivity']:.4f} "
              f"| spec={m['micro']['specificity']:.4f} | signal acc={m['signal_accuracy']:.4f}")
        print("  cm:", m["confusion_matrix"])
    plot_cm(out["test"]["confusion_matrix"], RESULTS / "cm_task1_test.png",
            f"Task 1 statistical rule ({best}) - test")
    boxplots(f, y, plot_idx, names, RESULTS / "box_task1_features.png")
    tree_plot(tree, names, RESULTS / "tree_task1.png")
    (RESULTS / "task1_rule.txt").write_text(out["rule_text"])

    with open(RESULTS / "task1_rule.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
