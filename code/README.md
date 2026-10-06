# Project 1: Biomedical Signal Processing — Code

Spinal ultrasound tissue classification (4 classes: spinal cord, nucleus
pulposus, adipose tissue, nerve root) with DWT statistical features,
interpretable and ML classifiers, and ViT inference.

## Requirements

Python packages: `numpy`, `scipy`, `scikit-learn`, `pywavelets`, `matplotlib`,
and for Task 3 `torch`, `einops`. Tested with Python 3.14, numpy 2.5,
scipy 1.18, scikit-learn 1.9, PyWavelets 1.8, torch 2.14 (CUDA).

Data: the released `dataset/` folder must sit next to this `code/` folder
(`../dataset/train_data.npy`, ...). The provided ViT code/checkpoint must stay
in `../Research_BMEL_ViT/code/`.

## How to run

```bash
conda run -n ai bash run_all.sh        # everything, ~10 min
```

or step by step:

```bash
python common.py            # self-check: preprocessing, voting, metrics, folds
python features.py          # self-check: DWT features + discriminative ranking
python task1_rule.py        # Task 1: interpretable threshold rule
python task2_ml.py          # Task 2: SVM/RF/kNN, sample-level 5-fold CV, val/test
python aggregation_study.py # voting vs soft voting vs averaged features
python task3_vit.py         # Task 3: provided ViT checkpoint (test/val)
```

All results are written to `results/` as JSON + figures. `task3_vit.py`
accepts `--checkpoint <path.pt> --tag <name>` to evaluate a fine-tuned ViT
(e.g. the bonus model trained with `../Research_BMEL_ViT/code/train_minimal.py`).

## Protocol notes

- Preprocessing (Tasks 1–2): 127/128 → 127.5, then (x − 127.5)/127.5.
- Features: DWT (`db4`, level 5) on the preprocessed signal; mean, variance,
  median, kurtosis, RMS and skewness of cA5 and cD5..cD1 → 36 features/signal.
- All model selection uses **sample-level stratified 5-fold CV**
  (`StratifiedGroupKFold`, 14 signals of a biological sample always together);
  scalers are fitted inside each fold only.
- Predictions are aggregated per sample by majority voting (ties broken by the
  largest mean decision value).
- The test set is evaluated only after all decisions are frozen.
- `task3_vit.py` feeds raw amplitudes to the provided ViT (its original
  training convention, see `dataset/README.md`) and reports that the provided
  checkpoint overlaps 88/126 val and 88/126 test samples.

## Layout

`common.py` shared helpers · `features.py` DWT features · `task1_rule.py`
statistical rule · `task2_ml.py` ML classifiers · `aggregation_study.py`
sample-level fusion study · `task3_vit.py` ViT evaluation · `results/` outputs.
