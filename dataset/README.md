# Released ultrasound dataset

Use the fixed train/validation/test split. Do not reshuffle it. The requested train:test:val ratio is 4:1:1; integer rounding gives 506:126:126 while retaining all 758 samples.

| Subset | Biological samples | Signal array | Label array |
|---|---:|---|---|
| train | 506 | train_data.npy: (7084, 896) | train_label.npy: (7084,) |
| val | 126 | val_data.npy: (1764, 896) | val_label.npy: (1764,) |
| test | 126 | test_data.npy: (1764, 896) | test_label.npy: (1764,) |

All files have dtype int64. Each consecutive block of 14 rows belongs to one sample; preserve group membership and row order. Each signal has 896 points sampled at 75 MHz. Raw amplitudes lie in [0,255]. Labels are 0 (spinal cord), 1 (nucleus pulposus), 2 (adipose tissue), 5 (nerve root).

The 758 unique samples were stratified and shuffled at the sample level using seed 20260907. No identical 14-signal sample appears in two released subsets. Shapes, checksums, and class counts are in manifest.json.

```python
from pathlib import Path
import numpy as np
data_dir = Path("dataset")  # Run from the release folder.
x = np.load(data_dir / "train_data.npy", allow_pickle=False)
y = np.load(data_dir / "train_label.npy", allow_pickle=False)
sample_signals = x.reshape(-1, 14, 896)
sample_labels = y.reshape(-1, 14)[:, 0]
```

For Tasks 1 and 2, convert to floating point, replace 127 and 128 with 127.5, then apply (x - 127.5) / 127.5. Any learned preprocessing must be fitted only on the allowed training data, within each CV fold where relevant.

Use the training set for fitting and five-fold model selection. Validation can refine the statistical rule and support optional ViT training. Evaluate the test set only after decisions are frozen.

The provided ViT uses raw floating-point inputs, not the Task 1 normalization. Its historical training data overlap 88/126 current validation samples and 88/126 current test samples. Its scores are a diagnostic comparison only; explain this limitation.
