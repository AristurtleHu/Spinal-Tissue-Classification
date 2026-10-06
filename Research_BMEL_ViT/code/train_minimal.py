# -*- coding: utf-8 -*-
import os
import argparse
import time
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import signal as spsig  # 用别名，避免变量名冲突
import torch.nn.functional as F
# -------------------------
# 0. 可复现实验：固定随机种子
# -------------------------
def set_seed(seed: int = 2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================================================
# CFG
# =========================================================
DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"

class CFG:
    # 路径
    imagesTr = str(DATA_DIR / 'train_data.npy')
    labelsTr = str(DATA_DIR / 'train_label.npy')
    imagesVal = str(DATA_DIR / 'val_data.npy')
    labelsVal = str(DATA_DIR / 'val_label.npy')
    imagesTs = str(DATA_DIR / 'test_data.npy')
    labelsTs = str(DATA_DIR / 'test_label.npy')

    # 训练与 ViT 超参默认值（与你之前一致）
    seq_len     = 896
    patch_size  = 7
    num_classes = 4
    dim         = 512
    depth       = 8
    heads       = 10
    mlp_dim     = 512
    dropout     = 0.15
    emb_dropout = 0.15
    beam_num    = 14

# =========================================================
# 2. 数据集
# =========================================================
class Signalset(Dataset):
    def __init__(self, data_path, label_path, train=False) -> None:
        self.data  = np.load(data_path, allow_pickle=True)
        self.label = np.load(label_path, allow_pickle=True).astype(np.int64)
        self.train = train

        # ——可选：将任意标签集合映射到从0开始的连续ID，防越界——
        uniq = np.unique(self.label)
        self.remap = {old: i for i, old in enumerate(sorted(uniq))}
        self.label = np.vectorize(self.remap.get)(self.label)
        self.num_classes_in_file = len(uniq)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        sig = self.data[idx, :]
        sig = np.expand_dims(sig, axis=1)
        sig = self.resample(sig, 896)
        sig = sig.reshape(1, 896).astype(np.float32)
        sig = torch.from_numpy(sig)
        lab = torch.tensor(self.label[idx]).long()
        return {'signal': sig, 'label': lab}

    def resample(self, sig, target_point_num=896):
        return spsig.resample(sig, target_point_num) if target_point_num else sig

# =========================================================
# 3. ViT-1D 
# =========================================================
def fetch_model(args):
    from models.vit_1d_cls import ViT
    model = ViT(
        seq_len     = args.seq_len,
        patch_size  = args.patch_size,
        num_classes = args.num_classes,
        dim         = args.dim,
        depth       = args.depth,
        heads       = args.heads,
        mlp_dim     = args.mlp_dim,
        dropout     = args.dropout,
        emb_dropout = args.emb_dropout,
    )
    return model

# =========================================================
# 4. 指标（宏平均）
# =========================================================
@torch.no_grad()
def compute_metrics(preds: torch.Tensor, labels: torch.Tensor, num_classes: int):
    device = preds.device
    cm = torch.zeros(num_classes, num_classes, device=device, dtype=torch.int64)
    for p, y in zip(preds, labels):
        cm[y, p] += 1
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
        "confusion_matrix": cm.cpu().numpy(),
    }

# =========================================================
# 5. 训练器
# =========================================================
class Trainer:
    def __init__(self, args):
        self.args   = args
        self.device = args.device
        self.model  = fetch_model(args).to(self.device)
        self.crit   = nn.CrossEntropyLoss()
        self.optim  = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        os.makedirs("./student_checkpoints", exist_ok=True)
        self.best_macro_acc = -1.0

        # 数据
        self.train_set = Signalset(args.imagesTr, args.labelsTr, train=True)
        self.val_set   = Signalset(args.imagesVal, args.labelsVal, train=False)

        # 自动校正 num_classes（防止标签越界）
        auto_classes = max(self.train_set.num_classes_in_file,
                           self.val_set.num_classes_in_file)
        if args.num_classes != auto_classes:
            print(f"[WARN] args.num_classes={args.num_classes} 与数据中类别数 {auto_classes} 不一致，已自动改为 {auto_classes}")
            args.num_classes = auto_classes
            # 同时也更新模型分类头（最简做法：重建模型）
            self.model  = fetch_model(args).to(self.device)
            self.optim  = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        self.train_loader = DataLoader(self.train_set, batch_size=args.batch_size, shuffle=True)
        self.val_loader   = DataLoader(self.val_set,   batch_size=args.beamnum, shuffle=False)

    def train(self):
        print(f"[INFO] Start training for {self.args.epochs} epochs.")
        for epoch in range(1, self.args.epochs + 1):
            t0 = time.perf_counter()
            tr_loss = self._train_one_epoch()
            val_m = self.evaluate_beam(self.val_loader)

            improved = ""
            if val_m["macro_acc"] > self.best_macro_acc:
                self.best_macro_acc = val_m["macro_acc"]
                torch.save(self.model.state_dict(), f"./student_checkpoints/best_{self.args.tag}.pt")
                improved = " (improved)"

            print(
                f"[Epoch {epoch:03d}] "
                f"loss={tr_loss:.4f} | "
                f"val_acc={val_m['macro_acc']:.4f} | "
                f"val_f1={val_m['macro_f1']:.4f} | "
                f"time={time.perf_counter() - t0:.1f}s{improved}"
            )

        print(f"[INFO] Best val macro_acc = {self.best_macro_acc:.4f}")
        # 训练和验证集选模全部结束后，才读取测试集用于最终评估。
        self.test_set = Signalset(self.args.imagesTs, self.args.labelsTs, train=False)
        self.test_loader = DataLoader(self.test_set, batch_size=self.args.beamnum, shuffle=False)
        self.model.load_state_dict(torch.load(f"./student_checkpoints/best_{self.args.tag}.pt", map_location=self.device))
        test_m = self.evaluate_beam(self.test_loader)
        print("avg_latency_ms:", test_m["avg_latency_ms"])
        print("\n[TEST] ===== Results =====")
        print(f"macro_acc      : {test_m['macro_acc']:.4f}")
        print(f"macro_precision: {test_m['macro_precision']:.4f}")
        print(f"macro_recall   : {test_m['macro_recall']:.4f}")
        print(f"macro_f1       : {test_m['macro_f1']:.4f}")
        print("confusion_matrix:\n", test_m["confusion_matrix"])

    def _train_one_epoch(self):
        self.model.train()
        total = 0.0
        for batch in self.train_loader:
            x = batch["signal"].to(self.device)
            y = batch["label"].to(self.device)
            logits = self.model(x)
            # 防御：训练期再次断言范围
            if not ((y >= 0).all() and (y < self.args.num_classes).all()):
                raise ValueError(f"Found labels out of range [0, {self.args.num_classes-1}]: {int(y.min())}..{int(y.max())}")
            loss = self.crit(logits, y)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            total += loss.item()
        return total / max(1, len(self.train_loader))

    import torch.nn.functional as F

    @torch.no_grad()
    def evaluate_beam(self, loader):
        """
        每个 batch 视为一个样本（大小 = self.args.beamnum），
        对该组 beam 的概率做“平均池化”得到最终预测。
        """
        self.model.eval()
        device   = self.device
        beamnum  = self.args.beamnum
        cm = torch.zeros(self.args.num_classes, self.args.num_classes, device=device, dtype=torch.int64)

        # 计时
        use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)
        else:
            import time as _time
        latencies = []

        for _, batch in enumerate(loader):
            if use_cuda:
                start_event.record()
            else:
                t0 = _time.perf_counter()

            x = batch["signal"].float().to(device)   # [beamnum, 1, L]
            y = batch["label"].to(device)            # [beamnum]
            # 同组标签需一致
            if not torch.all(y == y[0]):
                raise AssertionError(f"Expect all elements in label are the same, but got: {y}")

            # 前向 -> 概率
            probs = F.softmax(self.model(x), dim=1)  # [beamnum, C]
            # 平均池化融合到 1 个样本
            fused = probs.mean(dim=0)                # [C]
            pred  = fused.argmax(dim=0)

            # 混淆矩阵累计
            cm[y[0].item(), pred.item()] += 1

            if use_cuda:
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))   # ms
            else:
                latencies.append((_time.perf_counter() - t0) * 1000.0)  # ms

        # 计算评价指标
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



# =========================================================
# 6. 参数 & 主函数
# =========================================================
def get_args():
    cfg = CFG()
    p = argparse.ArgumentParser("Minimal ViT-1D on your Signalset")
    p.add_argument("--tag", type=str, default="exp_vit")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--beamnum", type=int, default=14)

    # ViT 超参（默认取 CFG，可在命令行覆盖）
    p.add_argument("--seq_len", type=int, default=cfg.seq_len)
    p.add_argument("--patch_size", type=int, default=cfg.patch_size)
    p.add_argument("--num_classes", type=int, default=cfg.num_classes)
    p.add_argument("--dim", type=int, default=cfg.dim)
    p.add_argument("--depth", type=int, default=cfg.depth)
    p.add_argument("--heads", type=int, default=cfg.heads)
    p.add_argument("--mlp_dim", type=int, default=cfg.mlp_dim)
    p.add_argument("--dropout", type=float, default=cfg.dropout)
    p.add_argument("--emb_dropout", type=float, default=cfg.emb_dropout)

    # 数据路径（默认取 CFG，可在命令行覆盖）
    p.add_argument("--imagesTr", type=str, default=cfg.imagesTr)
    p.add_argument("--labelsTr", type=str, default=cfg.labelsTr)
    p.add_argument("--imagesVal", type=str, default=cfg.imagesVal)
    p.add_argument("--labelsVal", type=str, default=cfg.labelsVal)
    p.add_argument("--imagesTs", type=str, default=cfg.imagesTs)
    p.add_argument("--labelsTs", type=str, default=cfg.labelsTs)
    return p.parse_args()

def main():
    set_seed(2023)
    args = get_args()
    print(f"[INFO] args: {args}")
    Trainer(args).train()

if __name__ == "__main__":
    main()
