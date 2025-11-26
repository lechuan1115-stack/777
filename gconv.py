#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# 扰动顺序固定，与数据生成一致
PERTURB_ORDER = ['CFO', 'SCALE', 'GAIN', 'SHIFT', 'CHIRP']

# ---------- 基础核操作（逐样本） ----------
def _mix_IQ_by_phase_per_sample(w, phi):
    """
    w:   [B, OC, IC, 1, kW]   (IC>=2，通道0=I, 通道1=Q)
    phi: [B, kW]
    返回对每个样本施加相位旋转后的核
    """
    B, OC, IC, _, kW = w.shape
    if IC < 2:
        return w
    I = w[:, :, 0, 0, :]                  # [B,OC,kW]
    Q = w[:, :, 1, 0, :]
    c = torch.cos(phi).unsqueeze(1)       # [B,1,kW]
    s = torch.sin(phi).unsqueeze(1)
    I2 = I * c - Q * s
    Q2 = I * s + Q * c
    w2 = w.clone()
    w2[:, :, 0, 0, :] = I2
    w2[:, :, 1, 0, :] = Q2
    return w2

def steer_scale_per_sample(w, alpha):
    """
    对核做时间尺度伸缩（线性插值到新长度再回到原长度）
    w:     [B,OC,IC,1,kW]
    alpha: [B] （1.0为不变）
    """
    B, OC, IC, _, kW = w.shape
    ker = w.view(B * OC * IC, 1, kW)
    out = []
    start = 0
    for b in range(B):
        a = torch.clamp(alpha[b], min=1e-3)
        new_W = int(torch.clamp((kW / a).round(), min=1).item())
        seg = ker[start:start + OC * IC]
        seg2 = F.interpolate(seg, size=new_W, mode='linear', align_corners=False)
        seg3 = F.interpolate(seg2, size=kW,   mode='linear', align_corners=False)
        out.append(seg3)
        start += OC * IC
    ker2 = torch.cat(out, dim=0)
    return ker2.view(B, OC, IC, 1, kW)

def steer_cfo_per_sample(w, fs, f0):
    """对核施加载频偏移相位：phi = 2π f0 n / fs"""
    kW = w.shape[-1]
    n  = torch.arange(kW, device=w.device, dtype=torch.float32)[None, :]
    phi = 2.0 * torch.pi * f0[:, None] * n / fs
    return _mix_IQ_by_phase_per_sample(w, phi)

def steer_chirp_per_sample(w, fs, a):
    """对核施加二次相位：phi = π a t^2"""
    kW = w.shape[-1]
    n  = torch.arange(kW, device=w.device, dtype=torch.float32)[None, :]
    t  = n / fs
    phi = torch.pi * a[:, None] * (t ** 2)
    return _mix_IQ_by_phase_per_sample(w, phi)

def steer_gain_per_sample(w, rho):
    """复增益幅度缩放（相位固定为0；相位扰动已在 CFO/CHIRP 里处理）"""
    return rho[:, None, None, None, None] * w

def steer_shift_per_sample(w, k):
    """循环移位（样本级）"""
    if torch.all(k == 0): return w
    w2 = w.clone()
    for b in range(w.shape[0]):
        s = int(k[b].item())
        if s != 0:
            w2[b] = torch.roll(w2[b], shifts=s, dims=-1)
    return w2

# ---------- 最简“有群”第一层 ----------
class SteerableFirstLayer(nn.Module):
    """
    纯群等变第一层：
      - 基核: Conv2d(in=2, out=out_ch, kernel=(1,k))
      - 逐样本核转向（SCALE→CFO→CHIRP→GAIN→SHIFT）
      - grouped conv 实现逐样本不同核
    约定输入:
      x: [B, 2, 1, L]
      z: [B, 5]  (0/1，是否激活对应扰动)
      s: [B, 5]  (扰动参数; 对未激活项可为任意值/NaN)
    """
    def __init__(self, in_ch=2, out_ch=32, k=5, fs=50e6):
        super().__init__()
        self.fs = float(fs)
        self.base = nn.Conv2d(in_ch, out_ch, kernel_size=(1, k),
                              padding=(0, k // 2), bias=False)
        nn.init.kaiming_uniform_(self.base.weight, a=0.2, nonlinearity='relu')

    def forward(self, x, z, s):
        assert x.dim() == 4 and x.size(1) == 2, "x应为[B,2,1,L]"
        assert z is not None and s is not None, "本层为纯群版本，需要同时提供 z/s"
        B, _, _, L = x.shape
        fs = torch.tensor(self.fs, dtype=torch.float32, device=x.device)

        # 基核复制到每个样本
        w0 = self.base.weight                         # [OC,IC,1,kW]
        wB = w0.unsqueeze(0).expand(B, *w0.shape).contiguous()  # [B,OC,IC,1,kW]

        # 依序对激活样本做核转向（未激活的样本保持单位/中性）
        def apply_if(tag, neutral, fn):
            idx = PERTURB_ORDER.index(tag)
            active = (z[:, idx] > 0.5)
            if not torch.any(active):  # 无激活则跳过
                return
            p = torch.nan_to_num(s[:, idx])
            # 构造“全体参数=neutral，激活样本=真实参数”的向量
            full = torch.full_like(p, neutral)
            full[active] = p[active]
            nonlocal wB
            if tag == 'SCALE':
                wB = steer_scale_per_sample(wB, full)
            elif tag == 'CFO':
                wB = steer_cfo_per_sample(wB, fs, full)
            elif tag == 'CHIRP':
                wB = steer_chirp_per_sample(wB, fs, full)
            elif tag == 'GAIN':
                wB = steer_gain_per_sample(wB, full)
            elif tag == 'SHIFT':
                wB = steer_shift_per_sample(wB, full.round())

        apply_if('SCALE', 1.0, steer_scale_per_sample)
        apply_if('CFO',   0.0, steer_cfo_per_sample)
        apply_if('CHIRP', 0.0, steer_chirp_per_sample)
        apply_if('GAIN',  1.0, steer_gain_per_sample)
        apply_if('SHIFT', 0.0, steer_shift_per_sample)

        # grouped conv 实现逐样本不同权重
        OC, IC, _, kW = w0.shape
        xG = x.reshape(1, B * IC, 1, L)                # [1,B*IC,1,L]
        wG = wB.reshape(B * OC, IC, 1, kW)             # [B*OC,IC,1,kW]
        yG = F.conv2d(xG, wG, bias=None, stride=1, padding=(0, kW // 2), groups=B)
        return yG.view(B, OC, 1, L)



