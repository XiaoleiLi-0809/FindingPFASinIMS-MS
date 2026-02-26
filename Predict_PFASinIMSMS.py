#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict_PFASinIMSMS.py

Load a trained unified bundle (.joblib) and predict PFAS probabilities for POS/NEG.
Includes PFAS_level (0-3) + topN ranking using low/high FDR thresholds.

Requires:
  - numpy, pandas, joblib
  - PrePFASinIMSMS.py providing mc_predict_proba_pipeline

Input must contain (depending on mode):
  POS: mz_M+H, CCS_M+H, riMp1
  NEG: mz_M-H, CCS_M-H, riMp1

Outputs:
  - full output CSV (df_out)
  - optional topN CSV (df_top)
  - prints a JSON-ish info summary

Example:
  python predict_PrePFAS.py \
    --model model_unified.joblib \
    --input exp_features.xlsx \
    --mode pos \
    --out pred_full.csv \
    --top-out pred_top.csv \
    --low-fdr 0.10 --high-fdr 0.20 --topN 200
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import joblib

from PrePFASinIMSMS import mc_predict_proba_pipeline


def _load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _prepare_mode(df: pd.DataFrame, mode: str) -> Tuple[pd.DataFrame, str, str]:
    """
    Returns:
      df2: cleaned df with mz_use + CCS_use + riMp1
      MZ_COL, CCS_COL: original column names used
    """
    mode = mode.lower().strip()
    if mode not in ("pos", "neg"):
        raise ValueError("mode must be 'pos' or 'neg'")

    MZ_COL = "mz_M-H" if mode == "neg" else "mz_M+H"
    CCS_COL = "CCS_M-H" if mode == "neg" else "CCS_M+H"

    need_cols = [MZ_COL, CCS_COL, "riMp1"]
    for c in need_cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}' in input. Required: {need_cols}")

    df2 = df.copy()
    for c in need_cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

    df2 = df2.dropna(subset=need_cols).copy()
    df2["mz_use"] = df2[MZ_COL]
    df2["CCS_use"] = df2[CCS_COL]
    return df2, MZ_COL, CCS_COL


def _get_threshold(thresholds_by_fdr: Dict[Any, Any], fdr: float) -> Dict[str, Any]:
    """
    Robustly fetch threshold object from a thresholds_by_fdr dict where keys might be float or str.
    Returns the stored dict like {"t":..., "precision":..., ...}
    """
    target = float(fdr)

    # exact float key
    if target in thresholds_by_fdr and thresholds_by_fdr[target] is not None:
        return thresholds_by_fdr[target]

    # try string key
    s = str(target)
    if s in thresholds_by_fdr and thresholds_by_fdr[s] is not None:
        return thresholds_by_fdr[s]

    # try "near match"
    for k, v in thresholds_by_fdr.items():
        if v is None:
            continue
        try:
            kk = float(k)
        except Exception:
            continue
        if abs(kk - target) < 1e-12:
            return v

    raise RuntimeError(f"No stored threshold for fdr={target}. Available keys: {list(thresholds_by_fdr.keys())}")


# ----------------------------
# Main notebook-style function
# ----------------------------
def predict_unified_notebook(
    model_path: str,
    mode: str,
    input_path: str,
    low_fdr: float = 0.10,
    high_fdr: float = 0.20,
    topN: int = 200,
    K: Optional[int] = None,
    ri_noise: Optional[float] = None,
    ccs_noise: Optional[float] = None,
    quantiles=(0.05, 0.50, 0.95),
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Returns:
      df_out: full results with probabilities + PFAS_level (0-3)
      df_top: topN candidates (level>=1), ranked by PFAS_level then p_mean
      info: dict of settings used (thresholds, mode, etc.)
    """
    mode = mode.lower().strip()
    if mode not in ("pos", "neg"):
        raise ValueError("mode must be 'pos' or 'neg'")

    # ---- Load unified bundle ----
    bundle = joblib.load(model_path)
    if bundle.get("bundle_type") != "mode_aware_unified":
        raise RuntimeError("Expected bundle_type='mode_aware_unified'.")

    if "models" not in bundle or mode not in bundle["models"]:
        raise RuntimeError(f"Bundle does not contain model for mode='{mode}'")

    pipe = bundle["models"][mode]
    meta = bundle.get("meta", {}).get(mode, {})
    thresholds_by_fdr = meta.get("thresholds_by_fdr", {})
    settings = bundle.get("settings", {})

    if not isinstance(thresholds_by_fdr, dict) or len(thresholds_by_fdr) == 0:
        raise RuntimeError(f"No thresholds_by_fdr found in bundle meta for mode='{mode}'")

    # Defaults from bundle if not provided
    if K is None:
        K = int(settings.get("K_infer", 200))
    if ri_noise is None:
        ri_noise = float(settings.get("riMp1_rel_noise", 0.20))
    if ccs_noise is None:
        ccs_noise = float(settings.get("ccs_rel_noise", 0.03))
    if seed is None:
        seed = int(settings.get("random_state", 42))

    # ---- Load input ----
    df_raw = _load_table(input_path)
    df, MZ_COL, CCS_COL = _prepare_mode(df_raw, mode)
    X = df[["mz_use", "CCS_use", "riMp1"]].copy()

    # ---- MC prediction ----
    p_mean, p_std, q = mc_predict_proba_pipeline(
        pipe, X,
        K=int(K),
        rel_noise_ri=float(ri_noise),
        rel_noise_ccs=float(ccs_noise),
        seed=int(seed),
        quantiles=quantiles,
        return_samples=False
    )

    # q is dict with float keys (0.05, 0.50, 0.95, ...)
    q05_key = float(quantiles[0]) if len(quantiles) > 0 else 0.05
    p05 = q.get(float(0.05), None)
    if p05 is None:
        p05 = q.get(q05_key, p_mean)
    # final fallback
    if p05 is None:
        p05 = p_mean

    # ---- Thresholds ----
    low_obj = _get_threshold(thresholds_by_fdr, float(low_fdr))
    high_obj = _get_threshold(thresholds_by_fdr, float(high_fdr))

    t_low = float(low_obj["t"])
    t_high = float(high_obj["t"])

    # Ensure: t_low is the more conservative (higher threshold) one
    t_low, t_high = max(t_low, t_high), min(t_low, t_high)

    # ---- PFAS level 0-3 ----
    level = np.zeros(len(df), dtype=int)
    level[p_mean >= t_high] = 1
    level[p_mean >= t_low] = 2
    level[p05   >= t_low] = 3

    label = np.full(len(df), "not_PFAS", dtype=object)
    label[level == 1] = "candidate_PFAS"
    label[level == 2] = "medium_conf_PFAS"
    label[level == 3] = "high_conf_PFAS"

    # ---- Output tables ----
    df_out = df.copy()
    df_out["p_PFAS_mean"] = p_mean
    df_out["p_PFAS_std"] = p_std
    df_out["p_PFAS_p05"] = p05
    df_out["PFAS_level"] = level
    df_out["PFAS_label"] = label

    df_out["mode_used"] = mode
    df_out["MZ_COL_used"] = MZ_COL
    df_out["CCS_COL_used"] = CCS_COL
    df_out["thr_low_fdr"] = float(low_fdr)
    df_out["thr_high_fdr"] = float(high_fdr)
    df_out["t_low"] = float(t_low)
    df_out["t_high"] = float(t_high)

    # Add quantiles as columns (optional)
    for qq, arr in q.items():
        if qq == "samples":
            continue
        try:
            qqf = float(qq)
        except Exception:
            continue
        df_out[f"p_PFAS_q{int(round(qqf*100)):02d}"] = arr

    # TopN: keep only level>=1, sort by level then probability
    df_cand = df_out[df_out["PFAS_level"] >= 1].copy()
    df_cand = df_cand.sort_values(["PFAS_level", "p_PFAS_mean"], ascending=[False, False])
    df_top = df_cand.head(int(topN)).copy()

    info = dict(
        mode=mode,
        MZ_COL=MZ_COL,
        CCS_COL=CCS_COL,
        K=int(K),
        ri_noise=float(ri_noise),
        ccs_noise=float(ccs_noise),
        seed=int(seed),
        low_fdr=float(low_fdr),
        high_fdr=float(high_fdr),
        t_low=float(t_low),
        t_high=float(t_high),
        topN=int(topN),
        n_input=int(len(df_out)),
        n_level_counts=df_out["PFAS_level"].value_counts().sort_index().to_dict(),
    )

    return df_out, df_top, info


# ----------------------------
# CLI wrapper
# ----------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict PFAS with unified model; output PFAS_level and topN.")
    p.add_argument("--model", required=True, help="Unified .joblib bundle path")
    p.add_argument("--input", required=True, help="Input .xlsx/.xls or .csv")
    p.add_argument("--mode", required=True, choices=["pos", "neg"], help="Ionization mode to use")
    p.add_argument("--out", required=True, help="Output CSV for full results")
    p.add_argument("--top-out", default=None, help="Output CSV for topN candidates (optional)")

    p.add_argument("--low-fdr", type=float, default=0.10)
    p.add_argument("--high-fdr", type=float, default=0.20)
    p.add_argument("--topN", type=int, default=200)

    p.add_argument("--k-infer", type=int, default=None)
    p.add_argument("--ri-noise", type=float, default=None)
    p.add_argument("--ccs-noise", type=float, default=None)
    p.add_argument("--mc-quantiles", type=float, nargs=3, default=(0.05, 0.50, 0.95))
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--print-info", action="store_true", help="Print info dict to stdout")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    df_out, df_top, info = predict_unified_notebook(
        model_path=args.model,
        mode=args.mode,
        input_path=args.input,
        low_fdr=float(args.low_fdr),
        high_fdr=float(args.high_fdr),
        topN=int(args.topN),
        K=args.k_infer,
        ri_noise=args.ri_noise,
        ccs_noise=args.ccs_noise,
        quantiles=tuple(args.mc_quantiles),
        seed=args.seed,
    )

    df_out.to_csv(args.out, index=False)
    print(f"Saved full results to: {args.out}")

    if args.top_out:
        df_top.to_csv(args.top_out, index=False)
        print(f"Saved topN results to: {args.top_out}")

    if args.print_info:
        print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()