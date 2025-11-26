#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
CSR: 单模型训练脚本（PerturbAwareNet）
- 读取：mydata_read.load_adsb_aug5_strict（严格适配你的 .mat 字段/形状）
- 模型：mymodel1.create(cfg.model_name, num_classes, fs)
- 训练：联合训练（分类 + 扰动分类 z + 扰动参数回归 s）
- 验证/测试：可选是否使用 GT 的 z/s；否则用模型预测的 z^/s^
- 可视化：训练/验证曲线、混淆矩阵、分类报告、结果CSV/JSON
"""

import os, os.path as osp, time, json, csv, datetime
from types import SimpleNamespace
from collections import defaultdict
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mydata_read                     # ← 用你的严格读取函数
import mymodel1                        # ← 只保留 PerturbAwareNet
from perturb_utils import (
    DEFAULT_SPLIT_SEED,
    normalize_perturb_params,
    stratified_split_indices,
    summarize_perturb_metrics,
    set_all_seeds,
)
import torch.nn.functional as F





# ----- EarlyStopping helper -----
class EarlyStopper:
    def __init__(self, mode="max", min_delta=0.0, patience=10, warmup=0):
        self.mode = mode
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.warmup = int(warmup)
        self.best = None
        self.num_bad = 0

    def _is_better(self, current, best):
        if best is None:
            return True
        if self.mode == "max":
            return (current - best) > self.min_delta
        else:  # "min"
            return (best - current) > self.min_delta

    def step(self, current, epoch_idx):
        # 返回 True 表示应当 early stop
        if epoch_idx < self.warmup:
            return False
        if self._is_better(current, self.best):
            self.best = current
            self.num_bad = 0
            return False
        else:
            self.num_bad += 1
            return self.num_bad >= self.patience

# ============= 数据集封装（简单、确定） =============
class NumpySignalDataset(Dataset):
    """把 numpy 数组打包为 PyTorch Dataset。"""
    def __init__(self, X, Y, Z, S):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()
        self.Z = torch.from_numpy(Z).float()
        self.S = torch.from_numpy(S).float()

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.Z[i], self.S[i]


# ============= 常用工具 =============
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / max(1, self.count)

def _mixup(x, y, alpha=0.0):
    if not alpha or alpha <= 0: return x, y, None, None, 1.0
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[perm]
    return x_mix, y, y[perm], perm, lam


@torch.no_grad()
def _model_supports_perturb_heads(model):
    """检查模型是否实现了扰动辅助分支需要的属性。"""
    required_attrs = ["_x_as_32ch", "_perturb_reducer", "z_head", "s_head"]
    if getattr(model, "disable_perturb_loss", False):
        return False
    return all(hasattr(model, attr) for attr in required_attrs)


def ensure_zs_inputs(model, z_val, s_val, z_ref, s_ref, device):
    """
    确保传入模型的 z/s 始终是张量：
    - 若模型无辅助头或推理未返回预测，则退化为全零张量；
    - 对需要 z/s 的模型（带 first 的群模型）尤其重要。
    """
    requires_zs = hasattr(model, "first")

    def _ensure_tensor(candidate, ref):
        if candidate is not None:
            return candidate.to(device)
        if ref is not None:
            return torch.zeros_like(ref)
        if requires_zs:
            raise RuntimeError("Model requires z/s inputs but no fallback is available")
        return None

    z_out = _ensure_tensor(z_val, z_ref)
    s_out = _ensure_tensor(s_val, s_ref)
    return z_out, s_out


def infer_zs(model, x, z_thresh=0.5):
    """
    测试/验证不允许用标签时，先用模型的扰动感知支路得到 z^/s^。
    对于没有扰动辅助头的模型，返回 (None, None)。
    """
    if not _model_supports_perturb_heads(model):
        return None, None

    red_fn = getattr(model, "_x_as_32ch")
    reducer = getattr(model, "_perturb_reducer")
    z_head = getattr(model, "z_head")
    s_head = getattr(model, "s_head")

    red = reducer(red_fn(x))                # [B,64,1,1]
    red = red.view(red.size(0), -1)         # [B,64]
    z_logit = z_head(red)
    s_pred  = s_head(red)
    z_prob  = torch.sigmoid(z_logit)
    z_hat   = (z_prob > z_thresh).float()
    return z_hat, s_pred


def summarize_zs(loader, tag="train", max_batches=30):
    tot = 0; act = None; s_sum = None; s_sq = None
    for i, batch in enumerate(loader):
        x, y, z, s = batch
        m = (z > 0.5).float()
        if act is None:
            act = m.sum(0); s_sum = torch.nan_to_num(s).sum(0)
            s_sq = (torch.nan_to_num(s)**2).sum(0)
        else:
            act += m.sum(0); s_sum += torch.nan_to_num(s).sum(0)
            s_sq += (torch.nan_to_num(s)**2).sum(0)
        tot += z.size(0)
        if i+1 >= max_batches: break
    if tot == 0: return
    act_rate = (act / tot).tolist()
    s_mean   = (s_sum / tot).tolist()
    s_std    = ((s_sq / tot) - (s_sum / tot)**2).clamp(min=0).sqrt().tolist()
    print(f"[Z/S Summary-{tag}] act_rate={act_rate}")
    print(f"[Z/S Summary-{tag}] s_mean  ={s_mean}")
    print(f"[Z/S Summary-{tag}] s_std   ={s_std}")


@torch.no_grad()
def compute_s_norm_stats(loader, device):
    """
    仅在 z_i=1 的位置上统计 s 的均值/标准差（按维度），用于 s 回归的标准化。
    """
    s_sum = None; s_sq = None; cnt = None
    for x, y, z, s in loader:
        z = z.to(device); s = s.to(device)
        m = (z > 0.5)  # 布尔掩码 [B,D]
        if s_sum is None:
            D = s.size(1)
            s_sum = torch.zeros(D, device=device)
            s_sq  = torch.zeros(D, device=device)
            cnt   = torch.zeros(D, device=device)
        # 把被激活的位置（z==1）的 s 累加
        s_masked = torch.where(m, s, torch.zeros_like(s))
        s_sum += s_masked.sum(dim=0)
        s_sq  += (s_masked ** 2).sum(dim=0)
        cnt   += m.sum(dim=0)
    cnt = torch.clamp(cnt, min=1)  # 防止除零
    mu  = s_sum / cnt
    var = torch.clamp(s_sq / cnt - mu**2, min=0.0)
    std = torch.sqrt(var)
    std = torch.clamp(std, min=1e-6)  # 防止极小方差导致爆炸
    print("[S-Norm] μ:", mu.detach().cpu().tolist())
    print("[S-Norm] σ:", std.detach().cpu().tolist())
    return mu.detach(), std.detach()


# ============= 训练/评估（联合训练） =============
def _unpack_model_outputs(output):
    if isinstance(output, (tuple, list)):
        logits = output[0] if len(output) > 0 else None
        feat = output[1] if len(output) > 1 else None
        z_logit = output[2] if len(output) > 2 else None
        s_pred = output[3] if len(output) > 3 else None
        return logits, feat, z_logit, s_pred
    return output, None, None, None


def train_one_epoch(model, optimizer, loader, device, cfg, epoch_idx):
    model.train()
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    reg = nn.SmoothL1Loss()

    losses = AverageMeter()
    correct = 0.0; total = 0

    # s 回归 warmup：前 cfg.s_warmup_epochs 个 epoch 不参与 s 回归
    lambda_s_eff = 0.0 if (epoch_idx < cfg.s_warmup_epochs) else float(cfg.lambda_s)
    has_perturb_heads = _model_supports_perturb_heads(model)

    for it, (x, y, z, s) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device); z = z.to(device); s = s.to(device)

        # 1) 分类支路（mixup）
        x_mix, y_a, y_b, perm, lam = _mixup(x, y, cfg.mixup_alpha)
        if cfg.use_gt_zs_train:
            z_cls, s_cls = z, s
        else:
            z_pred, s_pred = infer_zs(model, x_mix, cfg.z_thresh)
            z_cls, s_cls = ensure_zs_inputs(model, z_pred, s_pred, z, s, device=x_mix.device)

        logits, _, _, _ = _unpack_model_outputs(model(x_mix, z_cls, s_cls))
        if y_b is not None:
            cls_loss = lam * ce(logits, y_a) + (1. - lam) * ce(logits, y_b)
        else:
            cls_loss = ce(logits, y)

        # 2) 扰动支路监督（不用mixup）
        _, _, z_logit, s_pred = _unpack_model_outputs(model(x, z, s))
        loss_z = torch.tensor(0.0, device=device)
        if z_logit is not None and has_perturb_heads:
            loss_z = bce(z_logit, z) * float(cfg.lambda_z)

        # —— 关键改动：对 s 做按维标准化再回归 —— #
        if s_pred is not None and has_perturb_heads and lambda_s_eff > 0:
            m = (z > 0.5).float()  # 仅在 z=1 的位置监督 s
            active = m.sum()
            if active > 0:
                s_pred_n, s_true_n = normalize_perturb_params(
                    s_pred, s, signal_length=x.size(-1)
                )
                per_elem = F.smooth_l1_loss(s_pred_n, s_true_n, reduction="none")
                loss_s = (per_elem * m).sum() / active
                loss_s = loss_s * lambda_s_eff
            else:
                loss_s = torch.tensor(0.0, device=device)
        else:
            loss_s = torch.tensor(0.0, device=device)

        loss = cls_loss + loss_z + loss_s

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(1)
            if y_b is not None:
                acc = lam * (pred == y_a).float() + (1. - lam) * (pred == y_b).float()
                correct += acc.sum().item()
            else:
                correct += (pred == y).float().sum().item()
            total += y.size(0)
        losses.update(loss.item(), y.size(0))

        if (it + 1) % cfg.print_freq == 0:
            print(f"  iter {it+1:04d} | loss {losses.avg:.4f} | cls {cls_loss:.3f} z {loss_z:.3f} s {loss_s:.3f}")

    acc = 100.0 * correct / max(1, total)
    return acc, losses.avg


@torch.no_grad()
def _forward_eval(model, batch, device, cfg):
    x, y, z, s = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device)
    z = z.to(device)
    s = s.to(device)
    if cfg.use_gt_zs_eval:
        z_use, s_use = z, s
    else:
        z_pred, s_pred = infer_zs(model, x, cfg.z_thresh)
        z_use, s_use = ensure_zs_inputs(model, z_pred, s_pred, z, s, device=device)
    logits, feat, _, _ = _unpack_model_outputs(model(x, z_use, s_use))
    return logits, feat, y


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    ce = nn.CrossEntropyLoss()
    losses = AverageMeter(); correct = 0; total = 0
    for batch in loader:
        logits, feat, y = _forward_eval(model, batch, device, cfg)
        loss = ce(logits, y)
        losses.update(loss.item(), y.size(0))
        pred = logits.argmax(1)
        correct += (pred == y).sum().item(); total += y.size(0)
    return 100.0 * correct / max(1, total), losses.avg


@torch.no_grad()
def evaluate_perturb_metrics(model, loader, device, cfg):
    model.eval()
    z_acc_meter = AverageMeter()
    mae_meter = AverageMeter()
    mse_meter = AverageMeter()
    for batch in loader:
        x, _, z_true, s_true = batch
        x = x.to(device)
        z_true = z_true.to(device)
        s_true = s_true.to(device)
        z_logit, s_pred = None, None
        if _model_supports_perturb_heads(model):
            red_fn = getattr(model, "_x_as_32ch")
            reducer = getattr(model, "_perturb_reducer")
            z_head = getattr(model, "z_head")
            s_head = getattr(model, "s_head")
            red = reducer(red_fn(x))
            red = red.view(red.size(0), -1)
            z_logit = z_head(red)
            s_pred = s_head(red)

        if z_logit is None or s_pred is None:
            continue
        z_acc, mae, mse = summarize_perturb_metrics(z_true, s_true, z_logit, s_pred, cfg.z_thresh)
        bs = z_true.size(0)
        z_acc_meter.update(z_acc, bs)
        if not np.isnan(mae):
            mae_meter.update(mae, int((z_true > 0.5).sum().item()))
            mse_meter.update(mse, int((z_true > 0.5).sum().item()))
    return {
        "z_acc": z_acc_meter.avg if z_acc_meter.count else float("nan"),
        "s_mae": mae_meter.avg if mae_meter.count else float("nan"),
        "s_mse": mse_meter.avg if mse_meter.count else float("nan"),
    }


@torch.no_grad()
def evaluate_perturb_metrics(model, loader, device, cfg):
    model.eval()
    z_acc_meter = AverageMeter()
    mae_meter = AverageMeter()
    mse_meter = AverageMeter()
    for batch in loader:
        x, _, z_true, s_true = batch
        x = x.to(device)
        z_true = z_true.to(device)
        s_true = s_true.to(device)
        z_logit, s_pred = None, None
        if _model_supports_perturb_heads(model):
            red_fn = getattr(model, "_x_as_32ch")
            reducer = getattr(model, "_perturb_reducer")
            z_head = getattr(model, "z_head")
            s_head = getattr(model, "s_head")
            red = reducer(red_fn(x))
            red = red.view(red.size(0), -1)
            z_logit = z_head(red)
            s_pred = s_head(red)

        if z_logit is None or s_pred is None:
            continue
        z_acc, mae, mse = summarize_perturb_metrics(z_true, s_true, z_logit, s_pred, cfg.z_thresh)
        bs = z_true.size(0)
        z_acc_meter.update(z_acc, bs)
        if not np.isnan(mae):
            mae_meter.update(mae, int((z_true > 0.5).sum().item()))
            mse_meter.update(mse, int((z_true > 0.5).sum().item()))
    return {
        "z_acc": z_acc_meter.avg if z_acc_meter.count else float("nan"),
        "s_mae": mae_meter.avg if mae_meter.count else float("nan"),
        "s_mse": mse_meter.avg if mse_meter.count else float("nan"),
    }


@torch.no_grad()
def test_and_visualize(model, loader, device, cfg, save_dir, class_names=None):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        logits, feat, y = _forward_eval(model, batch, device, cfg)
        y_true.append(y.cpu().numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    top1 = (y_true == y_pred).mean() * 100.0

    # 混淆矩阵 & 分类报告
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    except Exception:
        cm, report = None, None

    # 保存可视化
    if cm is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation='nearest')
        ax.figure.colorbar(im, ax=ax)
        ax.set_title('Confusion Matrix'); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        if class_names is not None:
            ax.set_xticks(np.arange(len(class_names))); ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha='right'); ax.set_yticklabels(class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        plt.savefig(osp.join(save_dir, 'confusion_matrix.png'), dpi=200)
        plt.close(fig)
        with open(osp.join(save_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report or '')

    return dict(top1=top1)


def plot_curves(histories, out_path):
    if not histories: return
    epochs = [h['epoch'] for h in histories]
    tr = [h['train_acc'] for h in histories]
    va = [h['val_acc'] for h in histories]
    plt.figure(figsize=(7,4))
    plt.plot(epochs, tr, label='Train')
    plt.plot(epochs, va, label='Val', linestyle='--')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.title('Training / Validation Curves'); plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============= 主流程 =============
def run(cfg, device):
    print("程序：（联合训练：分类 + 扰动分类 + 扰动参数）")
    print(f"数据路径：{cfg.data}")
    print(f"保存目录：{cfg.save_root}")
    print(f"模型：{cfg.model_name}")
    os.makedirs(cfg.save_root, exist_ok=True)
    cfg.split_seed = cfg.split_seed if cfg.split_seed is not None else DEFAULT_SPLIT_SEED

    # 1) 读取数据（严格匹配你生成的数据结构）
    X, Y, Z, S, fs, snr_db, noise_var, order = mydata_read.load_adsb_aug5_strict(cfg.data, shuffle=False, seed=cfg.seed)
    Y = Y.astype(np.int64)  # 明确整数标签

    # 类别检查
    uniq = np.unique(Y); print("[CHECK] classes in data =", uniq.tolist())
    assert len(uniq) == cfg.class_num, f"class_num({cfg.class_num}) 与数据({len(uniq)})不一致"

    # 2) 分层划分 80/10/10
    # 2) 分层随机划分（Stratified Random Split）
    val_ratio = float(getattr(cfg, "val_ratio", 0.10))
    test_ratio = float(getattr(cfg, "test_ratio", 0.10))
    assert 0 < val_ratio < 1 and 0 < test_ratio < 1 and (val_ratio + test_ratio) < 1, "val/test 比例不合法"

    tr, va, te, _used_seed = stratified_split_indices(
        Y, val_ratio=val_ratio, test_ratio=test_ratio, split_seed=cfg.split_seed
    )
    print(f"[SPLIT] seed={_used_seed}  sizes: train={len(tr)}  val={len(va)}  test={len(te)}")
    # 互斥检查
    st_tr, st_va, st_te = set(tr), set(va), set(te)
    print("[SPLIT] overlaps:", len(st_tr & st_va), len(st_tr & st_te), len(st_va & st_te))

    # 一定要在循环外创建 Dataset / DataLoader
    train_set = Subset(NumpySignalDataset(X, Y, Z, S), tr)
    val_set = Subset(NumpySignalDataset(X, Y, Z, S), va)
    test_set = Subset(NumpySignalDataset(X, Y, Z, S), te)

    # 可选：保存索引方便复现实验
    if getattr(cfg, "save_split", True):
        split_dir = osp.join(cfg.save_root, "split_indices")
        os.makedirs(split_dir, exist_ok=True)
        np.save(osp.join(split_dir, "train_idx.npy"), np.array(tr))
        np.save(osp.join(split_dir, "val_idx.npy"), np.array(va))
        np.save(osp.join(split_dir, "test_idx.npy"), np.array(te))
        with open(osp.join(split_dir, "meta.txt"), "w", encoding="utf-8") as f:
            f.write(f"split_seed={_used_seed}\nval_ratio={val_ratio}\ntest_ratio={test_ratio}\n")

    pin = device.type == 'cuda'
    trainloader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.workers, pin_memory=pin)
    valloader   = DataLoader(val_set,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=pin)
    testloader  = DataLoader(test_set,  batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=pin)

    summarize_zs(trainloader, tag="train")
    summarize_zs(valloader,  tag="val")

    # 3) 模型
    model = mymodel1.create(name=cfg.model_name, num_classes=cfg.class_num, fs=fs)
    model = model.to(device)

    # 类别权重（按训练集频次）
    from collections import Counter
    cnt = Counter([Y[i] for i in tr])
    freq = np.array([cnt.get(c, 1) for c in range(cfg.class_num)], dtype=np.float32)
    class_weight = 1.0 / np.maximum(freq, 1.0)
    class_weight *= (cfg.class_num / class_weight.sum())
    class_weight = torch.as_tensor(class_weight, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    # 4) 训练循环
    history = []
    best_val = -1e9
    best_state = None
    t0 = time.time()

    # ===== 早停器初始化（如果 cfg.early_stop=False 就不会启用） =====
    stopper = EarlyStopper(
        mode=cfg.es_mode,  # "max" 或 "min"
        min_delta=cfg.es_min_delta,  # 认为“有提升”的最小幅度
        patience=cfg.es_patience,  # 连续多少个 epoch 无提升就停
        warmup=cfg.es_warmup  # 前多少个 epoch 不触发早停
    ) if getattr(cfg, "early_stop", False) else None
    # ==================================================================

    for epoch in range(cfg.max_epoch):
        print(f"==> Epoch {epoch + 1}/{cfg.max_epoch}")

        a_tr, l_tr = train_one_epoch(
            model, optimizer, trainloader, device, cfg,
            epoch_idx=epoch
        )

        a_va, l_va = evaluate(model, valloader, device, cfg)
        print(f"Train_Acc: {a_tr:.2f}%  Val_Acc: {a_va:.2f}%  (loss {l_tr:.4f}/{l_va:.4f})")

        history.append(dict(epoch=epoch + 1, train_acc=float(a_tr), val_acc=float(a_va)))

        # 保存“最佳验证准确率”的权重（与早停指标解耦，保持主口径统一）
        if a_va > best_val:
            best_val = a_va
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        # ===== 早停判断（根据 cfg.es_metric 选择 val_acc 或 val_loss） =====
        if stopper is not None:
            current_metric = a_va if cfg.es_metric == "val_acc" else l_va
            should_stop = stopper.step(current_metric, epoch_idx=epoch)
            if should_stop:
                # 防御性打印：best 可能是 None（极端配置/代码修改时）
                best_str = (
                    f"{stopper.best:.4f}" if isinstance(stopper.best, (int, float)) else str(stopper.best)
                )
                print(
                    f"[EARLY STOP] No improvement in {cfg.es_patience} epochs "
                    f"(metric={cfg.es_metric}, best={best_str}). "
                    f"Stop at epoch {epoch + 1}."
                )
                break
    elapsed = time.time() - t0
    print("训练耗时：", str(datetime.timedelta(seconds=int(elapsed))))

    # 保存最好模型
    ckpt = osp.join(cfg.save_root, f"{cfg.model_name}_best.pt")
    torch.save(best_state, ckpt); print("[CKPT] 保存：", ckpt)

    # 5) 评估与可视化
    model.load_state_dict(torch.load(ckpt, map_location=device))
    acc_val, _ = evaluate(model, valloader, device, cfg)
    test_stats = test_and_visualize(model, testloader, device, cfg, save_dir=cfg.save_root)
    perturb_stats = evaluate_perturb_metrics(model, testloader, device, cfg)

    # 曲线
    plot_curves(history, osp.join(cfg.save_root, "train_val_curves.png"))

    # 写 CSV/JSON
    metrics = dict(
        val_acc=float(acc_val),
        test_top1=float(test_stats['top1']),
        test_z_acc=float(perturb_stats.get("z_acc", float("nan"))),
        test_s_mae=float(perturb_stats.get("s_mae", float("nan"))),
        test_s_mse=float(perturb_stats.get("s_mse", float("nan"))),
        train_time_sec=float(elapsed),
    )
    csv_path = osp.join(cfg.save_root, "summary.csv")
    write_header = (not osp.exists(csv_path))
    with open(csv_path, "a", newline='', encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "val_acc",
                "test_top1",
                "test_z_acc",
                "test_s_mae",
                "test_s_mse",
                "train_time_sec",
                "timestamp",
            ],
        )
        if write_header: w.writeheader()
        w.writerow({**metrics, "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    with open(osp.join(cfg.save_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "history": history}, f, ensure_ascii=False, indent=2)

    print(
        f"Val_Acc: {metrics['val_acc']:.2f}%  TestTop1: {metrics['test_top1']:.2f}%  "
        f"TestZ_Acc: {metrics['test_z_acc']:.4f}  TestS_MAE: {metrics['test_s_mae']:.4f}  "
        f"TestS_MSE: {metrics['test_s_mse']:.4f}"
    )
    print("[DONE] 结果已写入：", cfg.save_root)


def main():
    # ==== 多组实验配置 ====
    configs = [
        dict(
            data=r"E:\毕业\数据集\ADS-B_Train_100_20dB.mat",
            save_root="./runs_perturbawarenet_20dB",
            model_name="perturbawarenet",
        ),
        dict(
            data=r"E:\毕业\数据集\ADS-B_Train_100_20dB.mat",
            save_root="./runs_cnn_transformer_20dB",
            model_name="cnn_transformer",
        ),

    ]

    # ==== 公共训练参数 ====
    base_cfg = dict(
        class_num=100,
        batch_size=32,
        workers=0,
        lr=1e-4,
        wd=1e-4,
        max_epoch=60,
        gpu='0',
        seed=42,
        print_freq=10,
        val_ratio=0.10,
        test_ratio=0.10,
        split_seed=DEFAULT_SPLIT_SEED,
        save_split=True,
        early_stop=True,
        es_metric="val_acc",
        es_mode="max",
        es_min_delta=0.1,
        es_patience=8,
        es_warmup=5,
        lambda_z=1.0,
        lambda_s=1.0,
        s_warmup_epochs=5,
        use_gt_zs_train=True,
        use_gt_zs_eval=False,
        z_thresh=0.5,
        mixup_alpha=0.2,
        use_ema=True,
        ema_decay=0.999,
    )

    # ==== 依次执行每个实验 ====
    for i, exp in enumerate(configs):
        print(f"\n==================== 实验 {i+1}/{len(configs)} ====================")
        print(f"数据: {exp['data']}")
        cfg = SimpleNamespace(**{**base_cfg, **exp})

        # 设置GPU与随机种子
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_all_seeds(cfg.seed)

        # 执行训练
        run(cfg, device)

    print("\n 所有模型训练完成！")


if __name__ == '__main__':
    main()