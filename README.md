# Project 1 - Folder Structure

```text
2026_Project1/
├── README.md                                  # Folder structure
├── Project_1_Biomedical_Signal_Processing.pdf   # Project task sheet
├── Project_1_Biomedical_Signal_Processing.tex   # LaTeX source of the task sheet
├── Paper3-BMEL2024.pdf                         # Background paper
├── dataset/                                  # Dataset and documentation
│   ├── README.md                             # Dataset description
│   ├── manifest.json                         # Dataset metadata and checksums
│   ├── train_data.npy                        # Training signals
│   ├── train_label.npy                       # Training labels
│   ├── val_data.npy                          # Validation signals
│   ├── val_label.npy                         # Validation labels
│   ├── test_data.npy                         # Test signals
│   └── test_label.npy                        # Test labels
├── Research_BMEL_ViT/
│   └── code/                                 # ViT code and pretrained weights
│       ├── inference.py                      # Inference script
│       ├── train_minimal.py                  # Training script
│       ├── run_train_minimal.bash            # Training launcher
│       ├── requirements.txt                  # Python dependencies
│       ├── checkpoints/
│       │   └── best_exp_vit.pt                # Provided model weights
│       └── models/
│           ├── __init__.py                   # Python package initializer
│           └── vit_1d_cls.py                 # ViT model definition
└── paper/                                    # Reference papers
    ├── Empirical_Wavelet_Transform.pdf
    ├── Nearest_neighbor_pattern_classification.pdf
    ├── Random_Forests.pdf
    └── Support-vector_networks.pdf
```
