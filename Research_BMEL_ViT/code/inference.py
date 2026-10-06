# -*- coding: utf-8 -*-
"""
inference.py
一键推理脚本：直接 `python inference.py` 即可运行。

功能：
- 加载 ViT-1D 模型与权重
- 使用原本的 Signalset 数据集读取 .npy 信号与标签
- 按 beamnum=14 将每 14 个 beam 聚合为 1 个样本（softmax 概率的平均池化）
- 打印宏平均 Precision/Recall/F1/Acc、混淆矩阵、平均推理时延(ms)

依赖：
numpy, scipy, torch
项目内需要：models/vit_1d_cls.py (包含 ViT 实现)
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import signal as spsig



# 今年发布包统一使用根目录 dataset，不依赖某台电脑的绝对路径。
CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parents[1] / "dataset"
CHECKPOINT = CODE_DIR / "checkpoints" / "best_exp_vit.pt"
IMAGES_TS  = DATA_DIR / "test_data.npy"
LABELS_TS  = DATA_DIR / "test_label.npy"

# ViT 超参（需与训练一致）
SEQ_LEN     = 896
PATCH_SIZE  = 7
NUM_CLASSES = 4       
DIM         = 512
DEPTH       = 8
HEADS       = 10
MLP_DIM     = 512
DROPOUT     = 0.15
EMB_DROPOUT = 0.15

# 推理设置
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
BEAMNUM  = 14          # 14 个 beam = 1 个样本
DROP_LAST = True       # 若最后一组不足 14 个 beam，则丢弃（避免分组错位）


# ================
# 数据集定义
# ================
class Signalset(Dataset):
    def __init__(self, data_path, label_path, train=False) -> None:
        self.data  = np.load(data_path, allow_pickle=True)
        self.label = np.load(label_path, allow_pickle=True).astype(np.int64)
        self.train = train

        # 将任意标签集合映射为从 0 开始的连续 ID（防越界）
        uniq = np.unique(self.label)
        self.remap = {old: i for i, old in enumerate(sorted(uniq))}
        self.label = np.vectorize(self.remap.get)(self.label)
        self.num_classes_in_file = len(uniq)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        sig = self.data[idx, :]
        sig = np.expand_dims(sig, axis=1)
        sig = self._resample(sig, SEQ_LEN)
        sig = sig.reshape(1, SEQ_LEN).astype(np.float32)
        sig = torch.from_numpy(sig)
        lab = torch.tensor(self.label[idx]).long()
        return {"signal": sig, "label": lab}

    @staticmethod
    def _resample(sig, target_point_num=896):
        return spsig.resample(sig, target_point_num) if target_point_num else sig


# ================
# 加载 ViT-1D
# ================
def build_model(num_classes: int):
    # 需保证项目有 models/vit_1d_cls.py，且包含 ViT 类
    from models.vit_1d_cls import ViT
    model = ViT(
        seq_len     = SEQ_LEN,
        patch_size  = PATCH_SIZE,
        num_classes = num_classes,
        dim         = DIM,
        depth       = DEPTH,
        heads       = HEADS,
        mlp_dim     = MLP_DIM,
        dropout     = DROPOUT,
        emb_dropout = EMB_DROPOUT,
    )
    return model


# ==========================
# evaluation
# ==========================
@torch.no_grad()
def evaluate(model, loader, num_classes: int, device: str, beamnum: int):
    model.eval()
    cm = torch.zeros(num_classes, num_classes, device=device, dtype=torch.int64)

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)
    latencies = []

    for _, batch in enumerate(loader):
        if use_cuda:
            start_event.record()
        else:
            t0 = time.perf_counter()

        x = batch["signal"].float().to(device)   # [beamnum, 1, L]
        y = batch["label"].to(device)            # [beamnum]

        # 同组标签需一致（否则说明数据分组对齐错误）
        if not torch.all(y == y[0]):
            raise AssertionError(f"Expect all elements in label are the same, but got: {y}")

        probs = F.softmax(model(x), dim=1)  # [beamnum, C]
        fused = probs.mean(dim=0)           # [C] 平均池化
        pred  = fused.argmax(dim=0)

        cm[y[0].item(), pred.item()] += 1

        if use_cuda:
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))   # ms
        else:
            latencies.append((time.perf_counter() - t0) * 1000.0)   # ms

    # 计算宏平均指标
    TP = torch.diag(cm).float()
    FP = cm.sum(0).float() - TP
    FN = cm.sum(1).float() - TP
    TN = cm.sum().float() - (TP + FP + FN)

    eps = 1e-8
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = (2 * precision * recall) / (precision + recall + eps)
    acc       = (TP + TN) / (TP + TN + FP + FN + eps)

    return {
        "macro_precision": precision.mean().item(),
        "macro_recall":    recall.mean().item(),
        "macro_f1":        f1.mean().item(),
        "macro_acc":       acc.mean().item(),
        "confusion_matrix": cm.detach().cpu().numpy(),
        "avg_latency_ms":  float(np.mean(latencies)) if latencies else 0.0,
    }


def main():
    # 加载数据
    if not (os.path.exists(IMAGES_TS) and os.path.exists(LABELS_TS)):
        print(f"[ERROR] 测试集路径不存在：\n  {IMAGES_TS}\n  {LABELS_TS}")
        sys.exit(1)

    test_set = Signalset(IMAGES_TS, LABELS_TS, train=False)

    # 自动对齐类别数
    num_classes = test_set.num_classes_in_file
    if NUM_CLASSES != num_classes:
        print(f"[WARN] NUM_CLASSES={NUM_CLASSES} 与数据中类别数 {num_classes} 不一致，自动使用 {num_classes}")
    num_classes = num_classes

    test_loader = DataLoader(
        test_set,
        batch_size=BEAMNUM,   # 14 个 beam = 1 组
        shuffle=False,
        drop_last=DROP_LAST
    )

    # 构建模型
    model = build_model(num_classes).to(DEVICE)

    # 加载权重
    if not os.path.exists(CHECKPOINT):
        print(f"[ERROR] 权重文件不存在：{CHECKPOINT}")
        sys.exit(1)

    state = torch.load(CHECKPOINT, map_location=DEVICE)
    # 兼容：state_dict / 整模型
    loaded = False
    if isinstance(state, dict):
        try:
            model.load_state_dict(state, strict=False)
            loaded = True
        except Exception:
            pass
        if not loaded and "state_dict" in state:
            try:
                model.load_state_dict(state["state_dict"], strict=False)
                loaded = True
            except Exception:
                pass
    if not loaded:
        try:
            # 可能是 torch.save(model) 存的整模型
            model = state.to(DEVICE).eval()
            loaded = True
        except Exception:
            pass

    if not loaded:
        print("[ERROR] 无法从 checkpoint 解析出可用权重，请确认保存方式（state_dict 或整模型）。")
        sys.exit(1)

    # 评测
    metrics = evaluate(model, test_loader, num_classes=num_classes, device=DEVICE, beamnum=BEAMNUM)

    # 打印结果
    print("\n[TEST] ===== Results =====")
    print(f"macro_acc      : {metrics['macro_acc']:.4f}")
    print(f"macro_precision: {metrics['macro_precision']:.4f}")
    print(f"macro_recall   : {metrics['macro_recall']:.4f}")
    print(f"macro_f1       : {metrics['macro_f1']:.4f}")
    print("confusion_matrix:\n", metrics["confusion_matrix"])
    print(f"avg_latency_ms : {metrics['avg_latency_ms']:.2f}")


if __name__ == "__main__":
    main()

