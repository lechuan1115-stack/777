#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os, os.path as osp, json
import numpy as np
import h5py

IQ_KEY        = "signal"        # (2, L, N)
FS_KEY        = "fs"            # (1,1)
Z_KEY         = "Z"             # (5, N)
S_KEY         = "S"             # (5, N)
ORDER_KEY     = "perturb_order" # 1x5 cell -> ['CFO','SCALE','GAIN','SHIFT','CHIRP']
SNR_DB_KEY    = "snr_db"        # (1, N)
NOISE_VAR_KEY = "noise_var"     # (1, N)
LABEL_KEY     = "trainlabel"    # (1, N)
ORDER_EXPECT  = ['CFO','SCALE','GAIN','SHIFT','CHIRP']

def _check(p):
    p = osp.abspath(p); print(f"[DATA] {p}")
    if not osp.exists(p): raise FileNotFoundError(p)
    return p

def _h5_cellstr_1d(hf, obj):
    ds = hf[obj] if isinstance(obj, str) else obj
    if getattr(ds, "dtype", None) is not None and ds.dtype.kind == "O":
        refs = np.array(ds); out = []
        for r in refs.reshape(-1):
            arr = np.array(hf[r])
            if arr.dtype == np.uint16:
                out.append(bytes(arr.ravel()).decode("utf-16le", errors="ignore"))
            else:
                out.append(bytes(arr.ravel()).decode("utf-8", errors="ignore"))
        return out
    return [(x.decode("utf-8") if isinstance(x,(bytes,bytearray)) else str(x))
            for x in np.array(ds).reshape(-1)]

def _signal_2LN_to_P4(A_2LN):
    # A_2LN: (2, L, N) -> X [N,2,1,L]
    I = A_2LN[0, :, :].astype(np.float32)  # (L,N)
    Q = A_2LN[1, :, :].astype(np.float32)
    I = I.T  # (N,L)
    Q = Q.T
    return np.stack([I, Q], axis=1)[:, :, None, :]  # [N,2,1,L]

def load_adsb_aug5_strict(path, shuffle=False, seed=42):
    path = _check(path)
    with h5py.File(path, "r") as hf:
        keys = sorted(hf.keys())
        print(json.dumps({"keys": keys}, ensure_ascii=False))

        for k in [IQ_KEY, FS_KEY, Z_KEY, S_KEY, ORDER_KEY, SNR_DB_KEY, NOISE_VAR_KEY, LABEL_KEY]:
            if k not in hf.keys():
                raise KeyError(f"缺少字段: {k}")

        A         = np.array(hf[IQ_KEY])                 # (2,L,N)
        fs_arr    = np.array(hf[FS_KEY])                 # (1,1)
        Z_raw     = np.array(hf[Z_KEY]).astype(np.float32)       # (5,N)
        S_raw     = np.array(hf[S_KEY]).astype(np.float32)       # (5,N)
        snr_raw   = np.array(hf[SNR_DB_KEY]).astype(np.float32)  # (1,N)
        nvar_raw  = np.array(hf[NOISE_VAR_KEY]).astype(np.float32)# (1,N)
        y_raw     = np.array(hf[LABEL_KEY]).astype(np.float32)   # (1,N)
        order     = [s.strip() for s in _h5_cellstr_1d(hf, hf[ORDER_KEY])]

    # —— 严格形状检查（与你的文件完全一致）——
    if not (A.ndim == 3 and A.shape[0] == 2):
        raise ValueError(f"'signal' 必须是 (2,L,N)，实际 {A.shape}")
    L, N = A.shape[1], A.shape[2]
    if Z_raw.shape != (5, N):  raise ValueError(f"Z 必须 (5,N)，实际 {Z_raw.shape}")
    if S_raw.shape != (5, N):  raise ValueError(f"S 必须 (5,N)，实际 {S_raw.shape}")
    if y_raw.shape != (1, N):  raise ValueError(f"trainlabel 必须 (1,N)，实际 {y_raw.shape}")
    if snr_raw.shape != (1, N):raise ValueError(f"snr_db 必须 (1,N)，实际 {snr_raw.shape}")
    if nvar_raw.shape != (1, N):raise ValueError(f"noise_var 必须 (1,N)，实际 {nvar_raw.shape}")
    if np.array(fs_arr).shape != (1,1): raise ValueError(f"fs 必须 (1,1)，实际 {fs_arr.shape}")
    if order != ORDER_EXPECT:
        raise ValueError(f"perturb_order 必须等于 {ORDER_EXPECT}，实际 {order}")

    # —— 确定性转换 ——
    X  = _signal_2LN_to_P4(A)       # [N,2,1,L]
    Z  = Z_raw.T                    # (N,5)
    S  = S_raw.T                    # (N,5)
    Y  = y_raw.reshape(-1)          # (N,)
    snr_db   = snr_raw.reshape(-1)  # (N,)
    noise_var= nvar_raw.reshape(-1) # (N,)
    fs = float(fs_arr.squeeze())

    if shuffle:
        idx = np.random.RandomState(seed).permutation(N)
        X, Y, Z, S, snr_db, noise_var = X[idx], Y[idx], Z[idx], S[idx], snr_db[idx], noise_var[idx]

    print(json.dumps({
        "summary": f"N={N}, L={L}, X{X.shape}, Y{Y.shape}, Z{Z.shape}, S{S.shape}, "
                   f"fs={fs}, order={order}, SNR[min={snr_db.min():.2f}, max={snr_db.max():.2f}] dB"
    }, ensure_ascii=False))
    return X, Y, Z, S, fs, snr_db, noise_var, order

# 直接在这里填路径运行
if __name__ == "__main__":
    p = r"E:\数据集\ADS-B_Train_-5dB.mat"
    X, Y, Z, S, fs, snr, nvar, order = load_adsb_aug5_strict(p, shuffle=False)
    print("加载完成")
