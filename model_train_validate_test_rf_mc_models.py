from __future__ import annotations

"""Official sklearn-based reanalysis pipeline for the PFAS manuscript.

This script follows the current manuscript/SI decisions:
- Training labels use PFAS = 1 when F_No > 3.
- CCS values in the cleaned training file are already calibrated.
- MC inference perturbs both riMp1 (+/-20%) and CCS (+/-3%).
- t_low and t_high are selected on the validation set from model results.
- Model outputs are tiered as Level 0-3 rather than binary only.

It requires: scikit-learn, matplotlib, scipy, joblib.
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_shared_rf_mc_data_loading_and_metrics import (
    ROOT,
    TRAIN_FILE,
    UNUSED_FILE,
    SRM_FILE,
    TORBAY_FILE,
    BAKER_FILE,
    LABEL_POLICY,
    PFAS_LABEL_THRESHOLD,
    RANDOM_STATE,
    RIMP1_REL_NOISE,
    CCS_REL_NOISE,
K_MC,
    make_output_dir,
    ensure_dirs,
    standardize_training,
    stratified_train_valid_test,
    metrics_from_pred,
    roc_auc,
    average_precision,
    threshold_for_fdr,
    assign_tier,
    tier_metrics,
    two_step_band_fit,
    two_step_band_predict,
    apply_external,
)


FEATURE_COLS = [
    "mz_percentile",
    "mz",
    "CCS",
    "riMp1",
    "CCS_over_m",
    "CCS_over_Mp1",
    "M_over_Mp1",
    "CCS_over_sqrtm",
    "CCS_over_m23",
]
K_TRAIN_AUG = 20
PFAS_WEIGHT_ALPHA = 1.0
HIGHCONF_MIN_PRECISION = 0.98
HIGHRECALL_TARGET_FDR = 0.18


def make_features(
    raw: pd.DataFrame,
    medians: pd.Series | None = None,
    train_mz_sorted: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build manuscript-compatible engineered features without CCS residual leakage."""
    d = raw[["mz", "CCS", "riMp1"]].copy()
    eps = 1e-12
    for c in ["mz", "CCS", "riMp1"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if train_mz_sorted is None:
        train_mz_sorted = np.sort(d["mz"].dropna().to_numpy(float))
    denom = max(len(train_mz_sorted) - 1, 1)
    d["mz_percentile"] = np.searchsorted(train_mz_sorted, d["mz"].to_numpy(float), side="right") / denom
    d["CCS_over_m"] = d["CCS"] / (d["mz"] + eps)
    d["CCS_over_Mp1"] = d["CCS"] / (d["riMp1"] + eps)
    d["M_over_Mp1"] = d["mz"] / (d["riMp1"] + eps)
    d["CCS_over_sqrtm"] = d["CCS"] / (np.sqrt(d["mz"]) + eps)
    d["CCS_over_m23"] = d["CCS"] / ((d["mz"] ** (2 / 3)) + eps)
    X = d[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = X.median(numeric_only=True)
    return X.fillna(medians), medians


def augment_training(raw: pd.DataFrame, y: np.ndarray, k_aug: int = K_TRAIN_AUG) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create manuscript-style MC-augmented training rows and sample weights."""
    rng = np.random.default_rng(RANDOM_STATE)
    raw_list = [raw.copy()]
    y_list = [y.copy()]
    f_no = pd.to_numeric(raw.get("F_No", pd.Series(np.zeros(len(raw)))), errors="coerce").fillna(0).to_numpy(float)
    base_weight = np.where(y == 1, 1.0 + PFAS_WEIGHT_ALPHA * np.log1p(np.maximum(f_no, 0)), 1.0)
    w_list = [base_weight.copy()]
    for _ in range(k_aug):
        d = raw.copy()
        d["riMp1"] = np.clip(
            d["riMp1"].to_numpy(float) * (1 + rng.uniform(-RIMP1_REL_NOISE, RIMP1_REL_NOISE, len(d))),
            1e-12,
            None,
        )
        d["CCS"] = np.clip(
            d["CCS"].to_numpy(float) * (1 + rng.uniform(-CCS_REL_NOISE, CCS_REL_NOISE, len(d))),
            1e-12,
            None,
        )
        raw_list.append(d)
        y_list.append(y.copy())
        w_list.append(base_weight.copy())
    return pd.concat(raw_list, ignore_index=True), np.concatenate(y_list), np.concatenate(w_list)


def fit_rf(train_raw: pd.DataFrame, y_train: np.ndarray) -> tuple[Pipeline, pd.Series, np.ndarray]:
    """Fit the official Random Forest model used for diagnostic and test outputs."""
    train_mz_sorted = np.sort(train_raw["mz"].dropna().to_numpy(float))
    aug_raw, aug_y, aug_w = augment_training(train_raw, y_train)
    X_train, medians = make_features(aug_raw, train_mz_sorted=train_mz_sorted)
    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=800,
                    max_depth=16,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf.fit(X_train, aug_y, rf__sample_weight=aug_w)
    return rf, medians, train_mz_sorted


def predict_rf(model: Pipeline, raw: pd.DataFrame, medians: pd.Series, train_mz_sorted: np.ndarray) -> np.ndarray:
    """Predict PFAS probability for raw mz/CCS/riMp1 rows."""
    X, _ = make_features(raw, medians, train_mz_sorted=train_mz_sorted)
    return model.predict_proba(X)[:, 1]


def mc_predict_sklearn(
    model: Pipeline,
    raw: pd.DataFrame,
    medians: pd.Series,
    train_mz_sorted: np.ndarray,
    k: int = K_MC,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perturb riMp1 and CCS, then return pmean, pstd, and p05."""
    rng = np.random.default_rng(seed)
    base = raw[["mz", "CCS", "riMp1"]].copy()
    probs = np.zeros((k, len(base)), dtype=np.float32)
    for i in range(k):
        d = base.copy()
        d["riMp1"] = np.clip(
            d["riMp1"].to_numpy(float) * (1 + rng.uniform(-RIMP1_REL_NOISE, RIMP1_REL_NOISE, len(d))),
            1e-12,
            None,
        )
        d["CCS"] = np.clip(
            d["CCS"].to_numpy(float) * (1 + rng.uniform(-CCS_REL_NOISE, CCS_REL_NOISE, len(d))),
            1e-12,
            None,
        )
        probs[i] = predict_rf(model, d, medians, train_mz_sorted)
    return probs.mean(axis=0), probs.std(axis=0), np.quantile(probs, 0.05, axis=0)


def threshold_for_min_precision(y_valid: np.ndarray, score: np.ndarray, min_precision: float) -> float:
    """Choose the lowest threshold that maximizes recall while meeting a precision target."""
    best = None
    for thr in np.linspace(0, 1, 1001):
        pred = (score >= thr).astype(int)
        m = metrics_from_pred(y_valid, pred)
        feasible = m["precision"] >= min_precision and (m["TP"] + m["FP"]) > 0
        candidate = (1 if feasible else 0, m["recall"], m["precision"], -thr)
        if best is None or candidate > best[0]:
            best = (candidate, float(thr))
    return best[1]


def plot_mc_space(df: pd.DataFrame, mode: str, bundle: dict, out_png: Path) -> None:
    """Plot the manuscript-style pmean vs pmean-p05 space."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    d0 = df[df["PFAS_label"] == 0]
    d1 = df[df["PFAS_label"] == 1]
    ax.scatter(d0["PFAS_prob_mean"], d0["PFAS_MC_uncertainty"], s=2, c="0.55", alpha=0.55, label="non-PFAS")
    ax.scatter(d1["PFAS_prob_mean"], d1["PFAS_MC_uncertainty"], s=2, c="#d62728", alpha=0.75, label="PFAS")
    t_low = bundle["t_low_fdr10"]
    t_high = bundle.get("t_high_fdr18", bundle["t_high_fdr20"])
    ax.axvline(t_high, color="#1f77b4", lw=1.2, ls="--", label=f"t_high={t_high:.3f}")
    ax.axvline(t_low, color="#008060", lw=1.2, ls="--", label=f"t_low={t_low:.3f}")
    x = np.linspace(t_low, 1.0, 100)
    ax.plot(x, x - t_low, color="#008060", lw=1.5, label="p05 >= t_low")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.35, float(df["PFAS_MC_uncertainty"].quantile(0.995))))
    ax.set_xlabel("pmean")
    ax.set_ylabel("pmean - p05")
    ax.set_title(f"{mode}: Monte Carlo probability-uncertainty space")
    ax.legend(markerscale=4, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_curves(test_out: pd.DataFrame, mode: str, figdir: Path) -> None:
    """Write ROC and PR curves for the sklearn RF model."""
    y = test_out["PFAS_label"].to_numpy(int)
    p = test_out["PFAS_prob_mean"].to_numpy(float)
    order = np.argsort(-p)
    yy = y[order]
    tpr = np.r_[0, np.cumsum(yy == 1) / max((yy == 1).sum(), 1), 1]
    fpr = np.r_[0, np.cumsum(yy == 0) / max((yy == 0).sum(), 1), 1]
    precision = np.cumsum(yy == 1) / (np.arange(len(yy)) + 1)
    recall = np.cumsum(yy == 1) / max((yy == 1).sum(), 1)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    ax.plot(fpr, tpr, label=f"AUC={roc_auc(y, p):.3f}")
    ax.plot([0, 1], [0, 1], color="0.7", ls="--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{mode}: ROC")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / f"ROC_RF_sklearn_{mode}.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    ax.plot(np.r_[0, recall], np.r_[1, precision], label=f"AP={average_precision(y, p):.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{mode}: Precision-Recall")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / f"PR_RF_sklearn_{mode}.png")
    plt.close(fig)


def train_mode(mode: str, train_df_all: pd.DataFrame, dirs: dict[str, Path]) -> dict:
    """Train one ion-mode model and write internal validation/test diagnostics."""
    data = standardize_training(train_df_all, mode)
    train, valid, test = stratified_train_valid_test(data)
    y_train = train["PFAS_label"].to_numpy(int)
    y_valid = valid["PFAS_label"].to_numpy(int)
    y_test = test["PFAS_label"].to_numpy(int)

    model, medians, train_mz_sorted = fit_rf(train[["mz", "CCS", "riMp1", "F_No"]], y_train)
    p_valid, pstd_valid, p05_valid = mc_predict_sklearn(model, valid[["mz", "CCS", "riMp1"]], medians, train_mz_sorted)
    # Level 3 uses the MC-stability score p05. It is tuned more strictly than
    # the broad candidate threshold so "high-confidence" remains meaningful.
    t_low = threshold_for_min_precision(y_valid, p05_valid, min_precision=HIGHCONF_MIN_PRECISION)
    t_high = threshold_for_fdr(y_valid, p_valid, target_fdr=HIGHRECALL_TARGET_FDR)
    valid_level, valid_level_name = assign_tier(p_valid, p05_valid, t_low, t_high)

    p_test, pstd_test, p05_test = mc_predict_sklearn(model, test[["mz", "CCS", "riMp1"]], medians, train_mz_sorted, seed=RANDOM_STATE + 9)
    test_level, test_level_name = assign_tier(p_test, p05_test, t_low, t_high)

    two_params = two_step_band_fit(train, valid)
    two_pred_test = two_step_band_predict(test, two_params)
    heuristic_mz = (test["CCS"] < 0.2 * test["mz"] + 100).astype(int).to_numpy()
    heuristic_mp1 = (test["CCS"] > (35 / 9) * test["riMp1"] + 125).astype(int).to_numpy()
    heuristic_3d = (heuristic_mz & heuristic_mp1).astype(int)

    rows = []
    for name, pred in [
        ("heuristic_mz_CCS", heuristic_mz),
        ("heuristic_riMp1_CCS", heuristic_mp1),
        ("heuristic_3D_AND", heuristic_3d),
        ("two_step_band", two_pred_test),
    ]:
        rows.append({"mode": mode, "model": name, **metrics_from_pred(y_test, pred)})
    metrics = pd.concat([pd.DataFrame(rows), tier_metrics(y_test, test_level, p_test, mode)], ignore_index=True)
    metrics.to_csv(dirs["tables"] / f"model_performance_sklearn_{mode}.csv", index=False)

    bundle = {
        "mode": mode,
        "label_policy": LABEL_POLICY,
        "model_type": "sklearn_RandomForestClassifier",
        "model": model,
        "medians": medians,
        "train_mz_sorted": train_mz_sorted,
        "t_low_fdr10": t_low,
        "t_high_fdr18": t_high,
        "t_high_fdr20": t_high,
        "t_low_selection": f"p05 validation threshold maximizing recall with precision>={HIGHCONF_MIN_PRECISION}",
        "t_high_selection": f"pmean validation threshold maximizing recall with FDR<={HIGHRECALL_TARGET_FDR}",
        "k_mc": K_MC,
        "riMp1_rel_noise": RIMP1_REL_NOISE,
        "ccs_rel_noise": CCS_REL_NOISE,
        "feature_cols": FEATURE_COLS,
        "k_train_aug": K_TRAIN_AUG,
        "pfas_weight_alpha": PFAS_WEIGHT_ALPHA,
        "metrics": metrics.to_dict(orient="records"),
    }
    with open(dirs["models"] / f"rf_sklearn_bundle_{mode}.pkl", "wb") as f:
        pickle.dump(bundle, f)

    valid_out = valid.copy()
    valid_out["PFAS_prob_mean"] = p_valid
    valid_out["PFAS_prob_std"] = pstd_valid
    valid_out["PFAS_prob_p05"] = p05_valid
    valid_out["PFAS_MC_uncertainty"] = p_valid - p05_valid
    valid_out["PFAS_level"] = valid_level
    valid_out["PFAS_level_name"] = valid_level_name
    valid_out.to_excel(dirs["data"] / f"validation_predictions_sklearn_{mode}.xlsx", index=False)

    test_out = test.copy()
    test_out["PFAS_prob_mean"] = p_test
    test_out["PFAS_prob_std"] = pstd_test
    test_out["PFAS_prob_p05"] = p05_test
    test_out["PFAS_MC_uncertainty"] = p_test - p05_test
    test_out["PFAS_level"] = test_level
    test_out["PFAS_level_name"] = test_level_name
    test_out.to_excel(dirs["data"] / f"internal_test_predictions_sklearn_{mode}.xlsx", index=False)

    plot_mc_space(test_out, mode, bundle, dirs["figures"] / f"MC_probability_uncertainty_sklearn_{mode}.png")
    plot_curves(test_out, mode, dirs["figures"])
    return bundle


def main() -> None:
    """Run sklearn-based internal diagnostics; external tests run after figure approval."""
    start = time.time()
    out = make_output_dir()
    dirs = ensure_dirs(out)
    train_df_all = pd.read_excel(TRAIN_FILE, sheet_name="Sheet1")
    bundles = {}
    for mode in ["POS", "NEG"]:
        print(f"Training sklearn {mode} model...")
        bundles[mode] = train_mode(mode, train_df_all, dirs)

    metadata = {
        "script": "run_reanalysis_sklearn.py",
        "elapsed_seconds": time.time() - start,
        "label_policy": LABEL_POLICY,
        "PFAS_label_threshold": PFAS_LABEL_THRESHOLD,
        "training_file": str(TRAIN_FILE),
        "external_files_pending": [str(UNUSED_FILE), str(SRM_FILE), str(TORBAY_FILE), str(BAKER_FILE)],
    }
    (out / "run_metadata_sklearn.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Done.")
    print(out)


if __name__ == "__main__":
    main()
