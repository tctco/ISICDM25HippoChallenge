#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate hippocampus segmentation (LH=1, RH=2) using MONAI metrics (Dice, HD95).
- Folder interface similar to seg-metrics: --gt, --pred, --out, --metrics, --verbose
- Computes per-case Dice and/or HD95 (mm) for label 1 and 2 only (no combined >0)

Usage:
  pip install monai nibabel torch pandas numpy
  python hippo_eval_monai.py \
      --gt ./gt --pred ./pred --out hippo_metrics.csv --metrics dice hd95
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from tqdm import tqdm


def one_hot_bg_lh_rh(arr: np.ndarray) -> torch.Tensor:
    """arr:(Z,Y,X) int -> (C=3,Z,Y,X) one-hot, 通道顺序: [bg, LH(1), RH(2)]"""
    bg = (arr == 0)
    lh = (arr == 1)
    rh = (arr == 2)
    oh = np.stack([bg, lh, rh], axis=0).astype(np.float32)
    return torch.from_numpy(oh)


def load_nii_int(path: Path):
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(), dtype=np.int16)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data, spacing


def eval_case_monai(gt_path: Path, pred_path: Path, need_dice: bool, need_hd95: bool):
    gt_np, spacing = load_nii_int(gt_path)
    pr_np, _ = load_nii_int(pred_path)

    # (B=1,C=3,Z,Y,X), C=[bg, LH, RH]
    y = one_hot_bg_lh_rh(gt_np).unsqueeze(0)
    p = one_hot_bg_lh_rh(pr_np).unsqueeze(0)

    # 各类是否存在（用于空类时的边界处理）
    gt_any = [(gt_np == 1).any(), (gt_np == 2).any()]
    pr_any = [(pr_np == 1).any(), (pr_np == 2).any()]

    out = {}

    # include_background=False => 只计算 C=1,2 两个通道（LH/RH）
    if need_dice:
        dice = DiceMetric(include_background=False, reduction="none")
        d = dice(y_pred=p, y=y).detach().cpu().numpy()[0]  # shape (2,) -> [LH, RH]
        # 处理空类：两边皆空 => Dice=1；一边空 => Dice=0
        for i, side in enumerate(["lh", "rh"]):
            if not gt_any[i] and not pr_any[i]:
                out[f"dice_{side}"] = 1.0
            elif not gt_any[i] or not pr_any[i]:
                out[f"dice_{side}"] = 0.0
            else:
                out[f"dice_{side}"] = float(np.asarray(d[i]).item())

    if need_hd95:
        hd95m = HausdorffDistanceMetric(include_background=False, percentile=95)
        h = hd95m(y_pred=p, y=y, spacing=spacing).detach().cpu().numpy()[0]  # (2,)
        # 处理空类：两边皆空 => HD95=0；一边空 => HD95=inf
        for i, side in enumerate(["lh", "rh"]):
            if not gt_any[i] and not pr_any[i]:
                out[f"hd95_{side}_mm"] = 0.0
            elif not gt_any[i] or not pr_any[i]:
                out[f"hd95_{side}_mm"] = float("inf")
            else:
                out[f"hd95_{side}_mm"] = float(np.asarray(h[i]).item())

    return out


def main():
    ap = argparse.ArgumentParser(
        description="Compute metrics (e.g., Dice/HD95) for LH(1) and RH(2) using MONAI (folder interface)."
    )
    ap.add_argument("--gt", required=True, type=Path, help="Ground-truth folder (contains NIfTI files)")
    ap.add_argument("--pred", required=True, type=Path, help="Prediction folder (filenames match GT)")
    ap.add_argument("--out", default="hippo_metrics.csv", type=Path, help="Output CSV path")
    ap.add_argument(
        "--metrics", nargs="+", default=["dice", "hd95"],
        choices=["dice", "hd95"],
        help="Metrics to compute (one or more). Default: dice hd95"
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    # Only dice & hd95 are implemented in this MONAI version
    requested = set(args.metrics)
    supported = {"dice", "hd95"}
    unsupported = requested - supported
    if unsupported:
        print(f"[WARN] Ignoring unsupported metrics for MONAI evaluator: {sorted(unsupported)}; supported: {sorted(supported)}")
    need_dice = "dice" in requested
    need_hd95 = "hd95" in requested

    gt_files = sorted(p for p in args.gt.glob("*.nii.gz") if p.is_file())
    pred_map = {p.name: p for p in args.pred.glob("*.nii.gz")}

    rows, missing = [], []

    for g in tqdm(gt_files):
        name = g.name
        if name not in pred_map:
            missing.append(name)
            continue
        # try:
        metrics = eval_case_monai(g, pred_map[name], need_dice, need_hd95)
        metrics["case"] = name[:-7] if name.endswith(".nii.gz") else Path(name).stem
        rows.append(metrics)
        # except Exception as e:
        #     rows.append({"case": name, "error": str(e)})

    # Build DataFrame with consistent columns
    columns = ["case"]
    if need_dice:
        columns += ["dice_lh", "dice_rh"]
    if need_hd95:
        columns += ["hd95_lh_mm", "hd95_rh_mm"]
    columns += ["error"]

    df = pd.DataFrame(rows)

    # 补全缺失列并按既定顺序排列
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    df = df[columns]

    # 计算均值行（仅对数值列，忽略 NaN/Inf）
    metric_cols = [c for c in df.columns if c not in ("case", "error")]
    mean_vals = {}
    for c in metric_cols:
        col = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        mean_vals[c] = float(np.nanmean(col)) if col.notna().any() else np.nan

    mean_row = {"case": "MEAN", "error": np.nan}
    mean_row.update(mean_vals)

    # 将均值行置顶，其余按 case 排序
    df = pd.concat([pd.DataFrame([mean_row]), df.sort_values("case")], ignore_index=True)

    # 写出
    df.to_csv(args.out, index=False)
    print(f"Saved metrics to: {args.out}")


if __name__ == "__main__":
    main()
