#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Single model: PerturbAwareNet  —— 只保留“有群”版本
- 第一层：SteerableFirstLayer（根据 z/s 对卷积核逐样本转向）
- 主干：纯 CNN 堆叠到 1024 维特征
- 头部：
    1) 分类头（n_classes）
    2) 扰动感知联合训练所需的两个头：
       - z_head：5 维（二分类 logits，顺序 ['CFO','SCALE','GAIN','SHIFT','CHIRP']）
       - s_head：5 维（对应扰动的实数参数回归）
注意：
- forward(x, z, s) 需要同时提供 z/s（训练/验证）。若你要在“测试时不提供标签”，
  请在 CSR 里先用 z_head/s_head 预测得到 ẑ/ŝ 再喂给本模型第一层。
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from gconv import SteerableFirstLayer  # 与你的 gconv 文件保持一致

PERTURB_ORDER = ['CFO', 'SCALE', 'GAIN', 'SHIFT', 'CHIRP']
N_PERT = len(PERTURB_ORDER)


# ------------------------------
# 基础模块：1×k 的时序卷积块
# ------------------------------
class Conv2dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int = 1, pool: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, k), stride=(1, stride),
                      padding=(0, k // 2), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _make_backbone() -> nn.Sequential:
    """与之前的 P4 主干等价的纯 CNN 堆叠，输出通道 1024。"""
    layers = []
    layers += [Conv2dBlock(32,  64, 11, pool=True)]
    layers += [Conv2dBlock(64, 128,  9, pool=True)]
    layers += [Conv2dBlock(128, 256, 7,  pool=True)]
    layers += [Conv2dBlock(256, 256, 7,  pool=True)]
    layers += [Conv2dBlock(256, 512, 5,  pool=True)]
    layers += [Conv2dBlock(512, 512, 5,  pool=True)]
    layers += [Conv2dBlock(512, 1024, 3, pool=True)]
    return nn.Sequential(*layers)


# ------------------------------
# 扰动分支复用工具
# ------------------------------
class PerturbBranchMixin:
    """提供共享的扰动感知分支实现。"""

    _lift32_layer: nn.Module = None

    @staticmethod
    def _x_as_32ch(x: torch.Tensor) -> torch.Tensor:
        """将输入升到 32 通道，供扰动分支使用（与原模型保持一致）。"""
        if PerturbBranchMixin._lift32_layer is None:
            PerturbBranchMixin._lift32_layer = nn.Sequential(
                nn.Conv2d(2, 32, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
        return PerturbBranchMixin._lift32_layer.to(x.device)(x)

    def _init_perturb_branch(self):
        self._perturb_reducer = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.z_head = nn.Linear(64, N_PERT)
        self.s_head = nn.Linear(64, N_PERT)

    def _forward_perturb_branch(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        red = self._perturb_reducer(self._x_as_32ch(x))
        red = red.view(red.size(0), -1)
        z_logit = self.z_head(red)
        s_pred = self.s_head(red)
        return z_logit, s_pred


# ------------------------------
# 单一模型：PerturbAwareNet
# ------------------------------
class PerturbAwareNet(PerturbBranchMixin, nn.Module):
    """
    输入:
        x: [B, 2, 1, L]
        z: [B, 5]   0/1 扰动激活标签（训练/验证必须提供）
        s: [B, 5]   扰动参数（训练/验证必须提供；未激活项可 NaN/任意）
    输出:
        logits: [B, n_classes]
        feat:   [B, 1024]   (分类前的全局特征)
        z_logit:[B, 5]      (联合训练用的扰动分类 logits)
        s_pred: [B, 5]      (联合训练用的扰动回归预测)
    """
    def __init__(self, n_classes: int, fs: float = 50e6):
        super().__init__()
        # 第一层：纯群等变（需要 z/s）
        self.first = SteerableFirstLayer(in_ch=2, out_ch=32, k=5, fs=fs)

        # 主干 & 池化
        self.features = _make_backbone()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, n_classes),
        )

        # 扰动感知头（联合训练）
        # 先压到一个较小的共享表征，再分出 z/s 两个分支
        self._init_perturb_branch()

    # —— 为了不改 CSR，保留这三个空接口（不做任何事）——
    def set_steer_disabled(self, flag: bool):  # 兼容旧调用；现在始终启用群
        return
    def set_first_debug(self, flag: bool):
        return
    def get_first_last_debug(self):
        return {}

    def forward(self, x: torch.Tensor, z: torch.Tensor, s: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1) 扰动感知辅助头：只看输入 x，提取第一层输入的全局表征
        #    注意：这是用来做联合训练监督的“感知器”，不参与卷积核转向。
        with torch.no_grad():
            pass  # 显式说明不需要这里对 x 额外处理；直接从 first 的输入侧取特征
        # 直接用一个浅层卷积对 x 提取共享表征（不依赖 z/s）
        z_logit, s_pred = self._forward_perturb_branch(x)

        # 2) 核转向 + 主干分类
        x = self.first(x, z, s)          # 需要 z/s；测试若无标签，请先用 z_logit/s_pred 生成 ẑ/ŝ
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)  # [B,1024]
        feat = x
        logits = self.classifier(x)
        return logits, feat, z_logit, s_pred

# ------------------------------
# CNN + Transformer 结合模型
# ------------------------------
class CNNTransformerNet(PerturbBranchMixin, nn.Module):
    def __init__(self, n_classes: int, fs: float = 50e6, *, d_model: int = 256,
                 nhead: int = 4, num_encoder_layers: int = 4,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        # CNN-Transformer 不参与扰动损失，保持纯分类训练
        self.disable_perturb_loss = True
        self.first = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        backbone = _make_backbone()
        self.cnn_stem = nn.Sequential(*list(backbone.children())[:3])
        self._stem_channels = 256

        # 限制序列长度（新增）
        self.n_tokens = 256
        self.seq_down = nn.AdaptiveAvgPool1d(self.n_tokens)

        self.channel_to_model = nn.Linear(self._stem_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.seq_pool = nn.AdaptiveAvgPool1d(1)
        hidden_dim = max(d_model // 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, n_classes),
        )
        self._init_perturb_branch()

    def forward(self, x, z=None, s=None):
        z_logit, s_pred = self._forward_perturb_branch(x)
        x = self.first(x)
        x = self.cnn_stem(x)
        b, c, _, l = x.shape
        x = x.view(b, c, l)
        x = self.seq_down(x)  # ★ 新增：固定序列长度
        x = x.permute(2, 0, 1).contiguous()
        x = self.channel_to_model(x)
        x = self.transformer(x)
        x = x.permute(1, 2, 0).contiguous()
        feat = self.seq_pool(x).squeeze(-1)
        logits = self.classifier(feat)
        return logits, feat, z_logit, s_pred





# ------------------------------
# 工厂：只保留“perturbawarenet”一个名字
# ------------------------------
def create(name: str, num_classes: int, fs: float = 50e6, **kwargs) -> nn.Module:
    key = (name or "").strip().lower()
    if key == "perturbawarenet":
        return PerturbAwareNet(n_classes=num_classes, fs=fs)
    if key == "cnn_transformer":
        return CNNTransformerNet(n_classes=num_classes, fs=fs, **kwargs)
    raise KeyError(f"Unsupported model name: {name}")
