# =========================
# train_PFASinIMSMS.py
# =========================
"""
Portable training script for a unified ionization-mode PFAS classifier.

Goal:
- One single model for POS/NEG ("unified mode model")
- Uses mode-aware mz/CCS (choose mz_M+H/CCS_M+H for POS; mz_M-H/CCS_M-H for NEG)
- Uses features: mz_mode, CCS_mode, riMp1, mode_is_pos
- Labels: PFAS_2 (0/1)
- Sample weights: F_No * balanced class weights (computed on TRAIN only)
- Balanced class_weight in models where supported

Outputs:
- joblib bundle with preprocessing + fitted model + metadata
- metrics table on test split
- optional ROC+PR curves (best model or all models) saved to disk

Examples:
python train_PFASinIMSMS.py --input data.csv --out_bundle unified_model.joblib --mode_col Mode
python train_PFASinIMSMS.py --input data.xlsx --sheet Sheet1 --compare_all --metric_select ap
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# Mode parsing helpers
# -----------------------------
POS_TOKENS = {"pos", "positive", "+", "m+H".lower(), "m+h", "m+H".lower(), "m+ h", "m+H".lower(), "m+h"}
NEG_TOKENS = {"neg", "negative", "-", "m-H".lower(), "m-h", "m- h", "m-h"}

def _norm_mode(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    return s

def mode_to_is_pos(mode_val: Any) -> int:
    s = _norm_mode(mode_val)
    if s in POS_TOKENS:
        return 1
    if s in NEG_TOKENS:
        return 0
    # common variants
    if "pos" in s or "+" in s:
        return 1
    if "neg" in s or "-" == s or "m-h" in s:
        return 0
    raise ValueError(f"Unrecognized mode value: {mode_val!r}. Please map your mode values to POS/NEG.")


# -----------------------------
# Feature engineering
# -----------------------------
def build_unified_features(
    df: pd.DataFrame,
    mode_col: str,
    mz_pos_col: str,
    ccs_pos_col: str,
    mz_neg_col: str,
    ccs_neg_col: str,
    rimp1_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create unified features:
      mz_mode  = mz_M+H if POS else mz_M-H
      CCS_mode = CCS_M+H if POS else CCS_M-H
      riMp1
      mode_is_pos (0/1)

    Returns: (X_df, mode_is_pos_series)
    """
    if mode_col not in df.columns:
        raise ValueError(f"mode_col '{mode_col}' not found in input columns.")

    # mode -> is_pos
    mode_is_pos = df[mode_col].apply(mode_to_is_pos).astype(int)

    # Choose mz/ccs by mode
    required_cols = [mz_pos_col, ccs_pos_col, mz_neg_col, ccs_neg_col, rimp1_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for unified features: {missing}")

    # ensure numeric
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    mz_mode = np.where(mode_is_pos.values == 1, df[mz_pos_col].values, df[mz_neg_col].values)
    ccs_mode = np.where(mode_is_pos.values == 1, df[ccs_pos_col].values, df[ccs_neg_col].values)

    X = pd.DataFrame({
        "mz_mode": mz_mode,
        "CCS_mode": ccs_mode,
        "riMp1": df[rimp1_col].values,
        "mode_is_pos": mode_is_pos.values.astype(float),  # keep numeric
    }, index=df.index)

    return X, mode_is_pos


# -----------------------------
# Metrics
# -----------------------------
def compute_pos_neg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # POS (class=1)
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_pos    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos        = (2 * precision_pos * recall_pos / (precision_pos + recall_pos)
                     if (precision_pos + recall_pos) > 0 else 0.0)

    # NEG (class=0)
    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_neg    = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # specificity
    f1_neg        = (2 * precision_neg * recall_neg / (precision_neg + recall_neg)
                     if (precision_neg + recall_neg) > 0 else 0.0)

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "TN": float(tn), "FP": float(fp), "FN": float(fn), "TP": float(tp),
        "Selected": float(tp + fp),
        "accuracy": float(acc),
        "precision_POS": float(precision_pos),
        "recall_POS": float(recall_pos),
        "f1_POS": float(f1_pos),
        "precision_NEG": float(precision_neg),
        "recall_NEG": float(recall_neg),
        "f1_NEG": float(f1_neg),
        "FDR": float(1.0 - precision_pos),
    }


def get_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise RuntimeError("Model has neither predict_proba nor decision_function.")


def fit_with_weights(name: str, model, X_tr, y_tr, w_tr):
    """
    Fit model using sample weights where possible.
    - Pipeline: pass <last_step>__sample_weight
    - CalibratedClassifierCV: pass sample_weight directly
    Fallback: fit without weights.
    """
    try:
        if isinstance(model, Pipeline):
            last_step = model.steps[-1][0]
            model.fit(X_tr, y_tr, **{f"{last_step}__sample_weight": w_tr})
        else:
            model.fit(X_tr, y_tr, sample_weight=w_tr)
    except TypeError:
        print(f"[WARN] {name}: sample_weight not supported; fitting without weights.")
        model.fit(X_tr, y_tr)
    return model


def plot_roc_prc(
    curves: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    out_png: str,
    title_suffix: str = ""
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC
    ax = axes[0]
    for name, d in curves.items():
        ax.plot(d["fpr"], d["tpr"], label=f'{name} (AUC={d["roc_auc"]:.3f})')
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(f"ROC curves{title_suffix}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)

    # PR
    ax = axes[1]
    for name, d in curves.items():
        ax.plot(d["recall_curve"], d["precision_curve"], label=f'{name} (AP={d["ap"]:.3f})')
    prevalence = (y_test == 1).mean()
    ax.hlines(prevalence, 0, 1, linestyles="--")
    ax.set_title(f"Precision–Recall curves{title_suffix}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -----------------------------
# Model zoo
# -----------------------------
def build_model_zoo(random_state: int) -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    models["LogisticRegression"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=8000,
            solver="lbfgs",
            class_weight="balanced"
        ))
    ])

    models["RandomForest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=600,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    models["SVM_RBF"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state
        ))
    ])

    ridge_base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", RidgeClassifier(class_weight="balanced", random_state=random_state))
    ])
    models["Ridge(calibrated)"] = CalibratedClassifierCV(
        estimator=ridge_base,
        method="sigmoid",
        cv=5
    )

    models["KNN"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=15))
    ])

    # Optional XGBoost
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=900,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=random_state,
                n_jobs=-1,
                eval_metric="logloss",
            ))
        ])
    except Exception as e:
        print("XGBoost not available (skipping). Reason:", str(e))

    # Optional CatBoost
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", CatBoostClassifier(
                iterations=1600,
                learning_rate=0.03,
                depth=6,
                loss_function="Logloss",
                random_seed=random_state,
                verbose=False
            ))
        ])
    except Exception as e:
        print("CatBoost not available (skipping). Reason:", str(e))

    return models


# -----------------------------
# Main
# -----------------------------
def read_table(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV/XLSX")
    ap.add_argument("--sheet", default=None, help="Excel sheet name (if xlsx)")
    ap.add_argument("--out_bundle", required=True, help="Output joblib bundle path")
    ap.add_argument("--out_metrics", default=None, help="Optional: save metrics CSV")
    ap.add_argument("--out_curve_png", default=None, help="Optional: save ROC+PR plot PNG")

    # column config
    ap.add_argument("--label_col", default="PFAS_2")
    ap.add_argument("--weight_col", default="F_No")
    ap.add_argument("--mode_col", default="Mode", help="Mode column (POS/NEG etc.)")
    ap.add_argument("--mz_pos_col", default="mz_M+H")
    ap.add_argument("--ccs_pos_col", default="CCS_M+H")
    ap.add_argument("--mz_neg_col", default="mz_M-H")
    ap.add_argument("--ccs_neg_col", default="CCS_M-H")
    ap.add_argument("--rimp1_col", default="riMp1")

    # training config
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)

    # model selection
    ap.add_argument("--compare_all", action="store_true",
                    help="Train all models and pick the best by --metric_select")
    ap.add_argument("--model", default="LogisticRegression",
                    help="Which model to train if not --compare_all")
    ap.add_argument("--metric_select", default="ap",
                    choices=["ap", "roc_auc", "f1_pos", "precision_pos", "recall_pos"],
                    help="Selection metric when --compare_all is set")

    # threshold for predicted label in predict.py
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Classification threshold saved into bundle (default=0.5)")

    args = ap.parse_args()

    df = read_table(args.input, args.sheet)

    # Basic checks
    for c in [args.label_col, args.weight_col, args.mode_col,
              args.mz_pos_col, args.ccs_pos_col, args.mz_neg_col, args.ccs_neg_col, args.rimp1_col]:
        if c not in df.columns:
            raise ValueError(f"Required column missing: {c}")

    # label + weights
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
    df[args.weight_col] = pd.to_numeric(df[args.weight_col], errors="coerce")

    df = df.dropna(subset=[args.label_col, args.mode_col])
    y = df[args.label_col].astype(int).values

    if set(np.unique(y)) - {0, 1}:
        raise ValueError(f"{args.label_col} must be 0/1 only. Found: {np.unique(y)}")

    w_base = df[args.weight_col].fillna(1.0).astype(float).values
    w_base = np.where(np.isfinite(w_base) & (w_base > 0), w_base, 1.0)

    # unified features
    X_df, mode_is_pos = build_unified_features(
        df=df,
        mode_col=args.mode_col,
        mz_pos_col=args.mz_pos_col,
        ccs_pos_col=args.ccs_pos_col,
        mz_neg_col=args.mz_neg_col,
        ccs_neg_col=args.ccs_neg_col,
        rimp1_col=args.rimp1_col,
    )

    # Drop rows where mz/CCS/riMp1 missing after selection
    keep = np.isfinite(X_df["mz_mode"].values) & np.isfinite(X_df["CCS_mode"].values) & np.isfinite(X_df["riMp1"].values)
    X_df = X_df.loc[keep].copy()
    y = y[keep]
    w_base = w_base[keep]

    # split
    X_train, X_test, y_train, y_test, w_train_base, w_test_base = train_test_split(
        X_df.values, y, w_base,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    # balanced class sample weights (train only)
    w_bal_train = compute_sample_weight(class_weight="balanced", y=y_train)
    w_train = w_train_base * w_bal_train

    # build models
    zoo = build_model_zoo(args.random_state)

    if not args.compare_all:
        if args.model not in zoo:
            raise ValueError(f"--model {args.model!r} not found. Available: {list(zoo.keys())}")
        zoo = {args.model: zoo[args.model]}

    all_results: List[Dict[str, Any]] = []
    curves: Dict[str, Dict[str, Any]] = {}

    # train/eval
    for name, model in zoo.items():
        model = fit_with_weights(name, model, X_train, y_train, w_train)

        y_pred = model.predict(X_test)
        y_score = get_scores(model, X_test)

        # metrics
        m = compute_pos_neg_metrics(y_test, y_pred)
        try:
            roc_auc = roc_auc_score(y_test, y_score)
        except Exception:
            roc_auc = float("nan")
        try:
            ap_auc = average_precision_score(y_test, y_score)
        except Exception:
            ap_auc = float("nan")

        m.update({"model": name, "ROC_AUC": roc_auc, "AP(PR_AUC)": ap_auc})

        all_results.append(m)

        # curves
        fpr, tpr, _ = roc_curve(y_test, y_score)
        prec, rec, _ = precision_recall_curve(y_test, y_score)
        curves[name] = {
            "fpr": fpr, "tpr": tpr,
            "precision_curve": prec,
            "recall_curve": rec,
            "roc_auc": roc_auc,
            "ap": ap_auc
        }

    res_df = pd.DataFrame(all_results)

    # choose best
    if args.compare_all:
        if args.metric_select == "ap":
            best_idx = res_df["AP(PR_AUC)"].astype(float).idxmax()
        elif args.metric_select == "roc_auc":
            best_idx = res_df["ROC_AUC"].astype(float).idxmax()
        elif args.metric_select == "f1_pos":
            best_idx = res_df["f1_POS"].astype(float).idxmax()
        elif args.metric_select == "precision_pos":
            best_idx = res_df["precision_POS"].astype(float).idxmax()
        elif args.metric_select == "recall_pos":
            best_idx = res_df["recall_POS"].astype(float).idxmax()
        else:
            best_idx = res_df["AP(PR_AUC)"].astype(float).idxmax()

        best_name = res_df.loc[best_idx, "model"]
        best_model = build_model_zoo(args.random_state)[best_name]
        # refit best model on full training data
        best_model = fit_with_weights(best_name, best_model, X_train, y_train, w_train)
        selected_models_for_plot = curves  # plot all models if compare_all
    else:
        best_name = res_df.loc[0, "model"]
        best_model = list(zoo.values())[0]
        selected_models_for_plot = curves  # plot that single model

    # print table
    cols_order = [
        "model", "accuracy",
        "precision_POS", "recall_POS", "f1_POS", "FDR",
        "precision_NEG", "recall_NEG", "f1_NEG",
        "ROC_AUC", "AP(PR_AUC)",
        "TP", "FP", "TN", "FN", "Selected"
    ]
    for c in cols_order:
        if c not in res_df.columns:
            res_df[c] = np.nan
    res_df = res_df[cols_order].sort_values(by="AP(PR_AUC)", ascending=False)

    print("\n=== Unified-mode model comparison ===")
    with pd.option_context("display.max_columns", 200):
        print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # save metrics
    if args.out_metrics:
        res_df.to_csv(args.out_metrics, index=False)

    # plot curves
    if args.out_curve_png:
        title_suffix = " (all models)" if args.compare_all else f" ({best_name})"
        plot_roc_prc(selected_models_for_plot, y_test, args.out_curve_png, title_suffix=title_suffix)

    # save bundle
    bundle = {
        "bundle_version": "unified_mode_v1",
        "best_model_name": best_name,
        "model": best_model,
        "feature_names": ["mz_mode", "CCS_mode", "riMp1", "mode_is_pos"],
        "threshold": float(args.threshold),
        "mode_config": {
            "mode_col": args.mode_col,
            "mz_pos_col": args.mz_pos_col,
            "ccs_pos_col": args.ccs_pos_col,
            "mz_neg_col": args.mz_neg_col,
            "ccs_neg_col": args.ccs_neg_col,
            "rimp1_col": args.rimp1_col,
            "label_col": args.label_col,
            "weight_col": args.weight_col,
            "pos_tokens": sorted(list(POS_TOKENS)),
            "neg_tokens": sorted(list(NEG_TOKENS)),
        },
        "train_meta": {
            "input": os.path.basename(args.input),
            "n_total": int(len(X_df)),
            "test_size": float(args.test_size),
            "random_state": int(args.random_state),
        }
    }

    joblib.dump(bundle, args.out_bundle)
    print(f"\nSaved bundle -> {args.out_bundle}")
    print(f"Best model: {best_name} | threshold={bundle['threshold']}")


if __name__ == "__main__":
    main()