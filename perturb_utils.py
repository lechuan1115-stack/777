#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""辅助工具：扰动范围、归一化与分层划分。"""
import random
from collections import defaultdict
from typing import Tuple

import numpy as np
import torch

# MATLAB augment_iq_with_perturbations_keepSNR.m 中的范围配置
PERTURB_RANGES = {
    "CFO": (-500.0, 500.0),
    "SCALE": (0.98, 1.02),
    "GAIN": (0.8, 1.2),
    "SHIFT_RATIO": 0.02,  # 相对序列长度的比例；实际 shift 采样数 = ratio * L
    "CHIRP": (-1e3, 1e3),
}

DEFAULT_SPLIT_SEED = 42


def set_all_seeds(seed: int) -> None:
    """同时固定 random / numpy / torch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _range_tensors(signal_length: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    shift_max = PERTURB_RANGES["SHIFT_RATIO"] * float(signal_length if signal_length is not None else 1)
    mins = torch.tensor([
        PERTURB_RANGES["CFO"][0],
        PERTURB_RANGES["SCALE"][0],
        PERTURB_RANGES["GAIN"][0],
        0.0,
        PERTURB_RANGES["CHIRP"][0],
    ], device=device)
    maxs = torch.tensor([
        PERTURB_RANGES["CFO"][1],
        PERTURB_RANGES["SCALE"][1],
        PERTURB_RANGES["GAIN"][1],
        shift_max,
        PERTURB_RANGES["CHIRP"][1],
    ], device=device)
    return mins, maxs


def normalize_perturb_params(s_pred: torch.Tensor, s_true: torch.Tensor, *, signal_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    依据已知参数范围对 s_pred/s_true 进行按维归一化，使不同扰动处于可比量级。
    返回的张量与输入 shape 相同。
    """
    device = s_pred.device if isinstance(s_pred, torch.Tensor) else s_true.device
    mins, maxs = _range_tensors(signal_length, device)
    center = (mins + maxs) / 2.0
    scale = torch.clamp((maxs - mins) / 2.0, min=1e-6)
    s_pred_n = (torch.nan_to_num(s_pred) - center) / scale
    s_true_n = (torch.nan_to_num(s_true) - center) / scale
    return s_pred_n, s_true_n


def stratified_split_indices(Y: np.ndarray, val_ratio: float, test_ratio: float, split_seed: int = DEFAULT_SPLIT_SEED):
    """按类别分层划分索引，返回 (train,val,test, used_seed)。"""
    by_cls = defaultdict(list)
    for i, y in enumerate(Y.tolist()):
        by_cls[y].append(i)

    rng = np.random.RandomState(split_seed)

    tr, va, te = [], [], []
    for _, ids in by_cls.items():
        ids = np.array(ids)
        rng.shuffle(ids)
        n = len(ids)

        n_va = int(round(val_ratio * n))
        n_te = int(round(test_ratio * n))
        n_tr = n - n_va - n_te

        if n >= 3:
            n_tr = max(n_tr, 1)
            n_va = max(n_va, 1)
            n_te = max(n - n_tr - n_va, 1)
        cut1 = n_tr
        cut2 = n_tr + n_va
        tr.extend(ids[:cut1])
        va.extend(ids[cut1:cut2])
        te.extend(ids[cut2:])

    return tr, va, te, split_seed


def summarize_perturb_metrics(z_true: torch.Tensor, s_true: torch.Tensor, z_logit: torch.Tensor, s_pred: torch.Tensor, z_thresh: float = 0.5):
    """计算扰动分类准确率与参数 MAE/MSE。"""
    z_prob = torch.sigmoid(z_logit)
    z_hat = (z_prob > z_thresh).float()
    z_acc = (z_hat == z_true).float().mean().item()

    mask = (z_true > 0.5)
    if mask.any():
        mae = torch.abs(torch.nan_to_num(s_pred) - torch.nan_to_num(s_true))[mask].mean().item()
        mse = torch.pow(torch.nan_to_num(s_pred) - torch.nan_to_num(s_true), 2)[mask].mean().item()
    else:
        mae = float("nan")
        mse = float("nan")
    return z_acc, mae, mse