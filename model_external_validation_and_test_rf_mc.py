from __future__ import annotations

"""Apply approved sklearn PFAS tier models to external experimental/test files.

The script loads the trained model bundles from outputs_20260604_reanalysis_8,
adds row-level PFAS probabilities and tier labels, and writes new Excel files.
Original input files are never overwritten.
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model_shared_rf_mc_data_loading_and_metrics import (
    BAKER_FILE,
    SRM_FILE,
    TORBAY_FILE,
    UNUSED_FILE,
    annotate_known_labels,
    prepare_external,
)
from model_train_validate_test_rf_mc_models import (
    K_MC,
    RIMP1_REL_NOISE,
    CCS_REL_NOISE,
    assign_tier,
    make_features,
)


MODEL_DIR = ROOT / "outputs_20260604_reanalysis_8"
OUT_DATA = MODEL_DIR / "data"
OUT_TABLES = MODEL_DIR / "tables"
CHUNK_SIZE = 8000


def load_bundle(mode: str) -> dict:
    """Load one trained sklearn RF bundle."""
    with open(MODEL_DIR / "models" / f"rf_sklearn_bundle_{mode}.pkl", "rb") as f:
        return pickle.load(f)


def predict_proba_bundle(bundle: dict, raw: pd.DataFrame) -> np.ndarray:
    """Predict PFAS probability from raw mz/CCS/riMp1 rows."""
    X, _ = make_features(raw, bundle["medians"], train_mz_sorted=bundle["train_mz_sorted"])
    return bundle["model"].predict_proba(X)[:, 1]


def mc_predict_chunked(bundle: dict, raw: pd.DataFrame, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run MC prediction in chunks to avoid high memory use on large files."""
    rng = np.random.default_rng(seed)
    n = len(raw)
    pmean = np.full(n, np.nan, dtype=float)
    pstd = np.full(n, np.nan, dtype=float)
    p05 = np.full(n, np.nan, dtype=float)
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        base = raw.iloc[start:end][["mz", "CCS", "riMp1"]].reset_index(drop=True)
        probs = np.zeros((K_MC, len(base)), dtype=np.float32)
        for k in range(K_MC):
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
            probs[k] = predict_proba_bundle(bundle, d)
        pmean[start:end] = probs.mean(axis=0)
        pstd[start:end] = probs.std(axis=0)
        p05[start:end] = np.quantile(probs, 0.05, axis=0)
        print(f"    chunk {start}-{end} / {n}")
    return pmean, pstd, p05


def add_top200_columns(out: pd.DataFrame, mode: str, source: str) -> None:
    """Add optional Top-200 selection columns for real-sample screening outputs."""
    if source not in {"SRM2585", "Torbay"}:
        out[f"PFAS_top200_selected_{mode}"] = np.nan
        return
    level_col = f"PFAS_level_{mode}"
    prob_col = f"PFAS_prob_mean_{mode}"
    selected = np.zeros(len(out), dtype=int)
    candidates = out.index[(pd.to_numeric(out[level_col], errors="coerce") >= 1) & out[prob_col].notna()]
    ranked = out.loc[candidates].sort_values(prob_col, ascending=False).head(200).index
    selected[ranked] = 1
    out[f"PFAS_top200_selected_{mode}"] = selected


def apply_one_source(source: str, path: Path, bundles: dict[str, dict]) -> pd.DataFrame:
    """Apply POS/NEG models to one external source and write row-level output."""
    print(f"Applying external source: {source}")
    df = pd.read_excel(path, sheet_name="Sheet1")
    out = annotate_known_labels(df, source)
    out["known_label_note"] = (
        "Known positives only; blank experimental annotations are unknown, not true negatives."
        if source in {"SRM2585", "Torbay"}
        else "Known positive external test set."
    )
    summary_rows = []
    for mode, bundle in bundles.items():
        raw_all, mask = prepare_external(out, mode, source)
        if raw_all.empty or int(mask.sum()) == 0:
            continue
        print(f"  {mode}: predictable rows = {int(mask.sum())}")
        valid_idx = np.where(mask.to_numpy())[0]
        valid_raw = raw_all.loc[mask, ["mz", "CCS", "riMp1"]].reset_index(drop=True)
        pm, ps, pp05 = mc_predict_chunked(bundle, valid_raw, seed=101 if mode == "POS" else 202)

        pmean = np.full(len(out), np.nan)
        pstd = np.full(len(out), np.nan)
        p05 = np.full(len(out), np.nan)
        pmean[valid_idx] = pm
        pstd[valid_idx] = ps
        p05[valid_idx] = pp05
        level, level_name = assign_tier(
            np.nan_to_num(pmean, nan=-1.0),
            np.nan_to_num(p05, nan=-1.0),
            bundle["t_low_fdr10"],
            bundle.get("t_high_fdr18", bundle["t_high_fdr20"]),
        )
        level[np.isnan(pmean)] = -1
        level_name = level_name.astype(object)
        level_name[np.isnan(pmean)] = "Not predictable"

        out[f"PFAS_prob_mean_{mode}"] = pmean
        out[f"PFAS_prob_std_{mode}"] = pstd
        out[f"PFAS_prob_p05_{mode}"] = p05
        out[f"PFAS_MC_uncertainty_{mode}"] = pmean - p05
        out[f"PFAS_level_{mode}"] = level
        out[f"PFAS_level_name_{mode}"] = level_name
        add_top200_columns(out, mode, source)

        known = pd.to_numeric(out.get("PFAS_known_label"), errors="coerce")
        known_pos = known == 1
        summary_rows.append(
            {
                "source": source,
                "mode": mode,
                "n_rows": len(out),
                "n_predictable": int(mask.sum()),
                "n_known_positive": int(known_pos.sum()),
                "level_ge_1": int((level >= 1).sum()),
                "level_ge_2": int((level >= 2).sum()),
                "level_3": int((level == 3).sum()),
                "known_positive_level_ge_1_recall": float(((level >= 1) & known_pos.to_numpy()).sum() / known_pos.sum()) if known_pos.sum() else np.nan,
                "known_positive_level_3_recall": float(((level == 3) & known_pos.to_numpy()).sum() / known_pos.sum()) if known_pos.sum() else np.nan,
            }
        )

    out_path = OUT_DATA / f"{source}_with_sklearn_model_tiers.xlsx"
    out.to_excel(out_path, index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_TABLES / f"{source}_external_summary_sklearn.csv", index=False)
    return summary


def main() -> None:
    """Run all external tests using the approved sklearn model outputs."""
    start = time.time()
    OUT_DATA.mkdir(exist_ok=True)
    OUT_TABLES.mkdir(exist_ok=True)
    bundles = {"POS": load_bundle("POS"), "NEG": load_bundle("NEG")}
    sources = {
        "Unused": UNUSED_FILE,
        "Baker": BAKER_FILE,
        "SRM2585": SRM_FILE,
        "Torbay": TORBAY_FILE,
    }
    summaries = []
    for source, path in sources.items():
        if path.exists():
            summaries.append(apply_one_source(source, path, bundles))
    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(OUT_TABLES / "external_summary_sklearn_all.csv", index=False)
    print("External tests complete.")
    print(f"Elapsed seconds: {time.time() - start:.1f}")


if __name__ == "__main__":
    main()
