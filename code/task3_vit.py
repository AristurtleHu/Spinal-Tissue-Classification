# -*- coding: utf-8 -*-
"""Task 3 (mandatory): inference with the provided ViT checkpoint.

Reuses the provided model, dataset and checkpoint (raw amplitude inputs, no
Task 1 normalization, as documented in dataset/README.md) and reports val/test
with the project's sample-level metrics. The checkpoint's historical training
data overlaps 88/126 current val and 88/126 current test samples, so these
scores are diagnostic, not independent generalization estimates.
"""
import json
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
VIT_DIR = ROOT / "Research_BMEL_ViT" / "code"
sys.path.insert(0, str(VIT_DIR))  # provided code: model def, dataset, checkpoint

from common import CLASSES, load_split, plot_cm, sample_metrics  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
OVERLAP_CAVEAT = ("provided checkpoint's historical training data overlaps 88/126 "
                  "current val and 88/126 current test samples (dataset/README.md)")


def evaluate_split(model, vit, split, device):
    ds = vit.Signalset(vit.DATA_DIR / f"{split}_data.npy", vit.DATA_DIR / f"{split}_label.npy")
    loader = DataLoader(ds, batch_size=vit.BEAMNUM, shuffle=False, drop_last=vit.DROP_LAST)
    probs, preds = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["signal"].float().to(device)
            p = torch.softmax(model(x), dim=1).cpu().numpy()
            probs.append(p)
            preds.append(p.argmax(axis=1))
    probs, preds = np.concatenate(probs), np.concatenate(preds)

    _, y = load_split(split)  # original labels, same row order as the dataset file
    assert len(y) == len(preds)
    return sample_metrics(y, CLASSES[preds], probs), preds, probs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None, help="checkpoint .pt (default: provided model)")
    ap.add_argument("--tag", default="provided", help="name used in output files")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    import inference as vit  # the provided script (its main() is not run on import)

    device = vit.DEVICE
    path = Path(args.checkpoint) if args.checkpoint else vit.CHECKPOINT
    model = vit.build_model(4).to(device)
    state = torch.load(path, map_location=device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
        print("[warn] checkpoint loaded with strict=False")
    model.eval()

    caveat = OVERLAP_CAVEAT if args.tag == "provided" else "independently trained on the released training split only"
    try:
        stored = str(path.relative_to(ROOT))
    except ValueError:
        stored = str(path)
    out = {"checkpoint": stored, "caveat": caveat}
    for split in ("val", "test"):
        m, preds, _ = evaluate_split(model, vit, split, device)
        out[split] = {"metrics": m, "signal_predictions": preds.tolist()}
        print(f"[{split}] sample acc={m['micro']['accuracy']:.4f} | sens={m['micro']['sensitivity']:.4f} "
              f"| spec={m['micro']['specificity']:.4f} | signal acc={m['signal_accuracy']:.4f}")
        print("  cm:", m["confusion_matrix"])
    plot_cm(out["test"]["metrics"]["confusion_matrix"], RESULTS / f"cm_task3_{args.tag}_test.png",
            f"Task 3 {args.tag} ViT - test")

    with open(RESULTS / f"task3_{args.tag}.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
