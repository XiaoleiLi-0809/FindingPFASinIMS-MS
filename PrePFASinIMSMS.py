import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def augment_train_mc(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    K_train: int = 20,
    rel_noise_ri: float = 0.20,
    rel_noise_ccs: float = 0.03,
    seed: int = 42,
):
    """
    Monte-Carlo augmentation for training:
      - repeats original data + K_train noisy copies
      - perturbs BOTH 'riMp1' and 'CCS_use' multiplicatively:
          x <- x * (1 + eps), eps ~ Uniform(-rel_noise, +rel_noise)

    Returns:
      X_aug (DataFrame), y_aug (Series) with length (K_train+1) * n
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("augment_train_mc expects X_train as a pandas DataFrame.")
    if "riMp1" not in X_train.columns:
        raise KeyError("augment_train_mc requires column 'riMp1' in X_train.")
    if "CCS_use" not in X_train.columns:
        raise KeyError("augment_train_mc requires column 'CCS_use' in X_train.")

    rng = np.random.default_rng(seed)
    n = len(X_train)

    X_list = [X_train.copy()]
    y_list = [y_train.copy()]

    base = X_train.copy()

    ri_base = pd.to_numeric(base["riMp1"], errors="coerce").to_numpy(dtype=float)
    ccs_base = pd.to_numeric(base["CCS_use"], errors="coerce").to_numpy(dtype=float)

    # Keep NaNs as NaNs (imputer downstream can handle). Only clip finite positives.
    for _ in range(int(K_train)):
        dfk = base.copy()

        eps_ri = rng.uniform(-rel_noise_ri, rel_noise_ri, size=n)
        eps_ccs = rng.uniform(-rel_noise_ccs, rel_noise_ccs, size=n)

        ri_k = ri_base * (1.0 + eps_ri)
        ccs_k = ccs_base * (1.0 + eps_ccs)

        # Avoid non-physical <=0 values while preserving NaNs
        ri_k = np.where(np.isfinite(ri_k), np.clip(ri_k, 1e-12, None), ri_k)
        ccs_k = np.where(np.isfinite(ccs_k), np.clip(ccs_k, 1e-12, None), ccs_k)

        dfk["riMp1"] = ri_k
        dfk["CCS_use"] = ccs_k

        X_list.append(dfk)
        y_list.append(y_train)

    X_aug = pd.concat(X_list, axis=0, ignore_index=True)
    y_aug = pd.concat(y_list, axis=0, ignore_index=True)
    return X_aug, y_aug


def mc_predict_proba_pipeline(
    pipe,
    X_raw_df: pd.DataFrame,
    K: int = 200,
    rel_noise_ri: float = 0.20,
    rel_noise_ccs: float = 0.03,
    seed: int = 42,
    quantiles=(0.05, 0.50, 0.95),
    return_samples: bool = False,
):
    """
    Monte-Carlo inference:
      - creates K noisy copies of X_raw_df
      - perturbs BOTH 'riMp1' and 'CCS_use'
      - predicts proba on each copy and aggregates

    Returns:
      p_mean (n,), p_std (n,), q_dict (quantile -> array(n,))
      If return_samples=True, q_dict also contains {"samples": P} where P is (K, n).
    """
    if not isinstance(X_raw_df, pd.DataFrame):
        raise TypeError("mc_predict_proba_pipeline expects X_raw_df as a pandas DataFrame.")
    if "riMp1" not in X_raw_df.columns:
        raise KeyError("mc_predict_proba_pipeline requires column 'riMp1' in X_raw_df.")
    if "CCS_use" not in X_raw_df.columns:
        raise KeyError("mc_predict_proba_pipeline requires column 'CCS_use' in X_raw_df.")

    rng = np.random.default_rng(seed)
    n = len(X_raw_df)
    K = int(K)
    if K <= 0:
        raise ValueError("K must be a positive integer.")

    base = X_raw_df.copy()
    ri_base = pd.to_numeric(base["riMp1"], errors="coerce").to_numpy(dtype=float)
    ccs_base = pd.to_numeric(base["CCS_use"], errors="coerce").to_numpy(dtype=float)

    P = np.zeros((K, n), dtype=float)

    for k in range(K):
        dfk = base.copy()

        eps_ri = rng.uniform(-rel_noise_ri, rel_noise_ri, size=n)
        eps_ccs = rng.uniform(-rel_noise_ccs, rel_noise_ccs, size=n)

        ri_k = ri_base * (1.0 + eps_ri)
        ccs_k = ccs_base * (1.0 + eps_ccs)

        ri_k = np.where(np.isfinite(ri_k), np.clip(ri_k, 1e-12, None), ri_k)
        ccs_k = np.where(np.isfinite(ccs_k), np.clip(ccs_k, 1e-12, None), ccs_k)

        dfk["riMp1"] = ri_k
        dfk["CCS_use"] = ccs_k

        P[k, :] = pipe.predict_proba(dfk)[:, 1]

    p_mean = P.mean(axis=0)
    p_std = P.std(axis=0)

    q_dict = {}
    if quantiles is not None and len(quantiles) > 0:
        qs = np.asarray(quantiles, dtype=float)
        Q = np.quantile(P, qs, axis=0)
        for i, q in enumerate(qs):
            q_dict[float(q)] = Q[i, :]

    if return_samples:
        q_dict["samples"] = P

    return p_mean, p_std, q_dict


def pick_threshold_by_fdr_max_recall(
    y_true,
    p,
    max_fdr: float = 0.10,
    min_recall: float = 0.0,
    grid_size: int = 2001,
):
    """
    Pick threshold t that satisfies:
      FDR <= max_fdr and Recall >= min_recall
    among those, maximize Recall; tie-breaker: maximize Precision; then larger t.

    Returns dict with confusion metrics, or None if no threshold meets constraints.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)

    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")

    thr_grid = np.linspace(0.0, 1.0, int(grid_size))
    best = None

    for t in thr_grid:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        fdr = fp / (tp + fp + 1e-12)
        selected = tp + fp

        if (fdr <= max_fdr) and (recall >= min_recall):
            cand = dict(
                t=float(t),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                precision=float(precision),
                recall=float(recall),
                fdr=float(fdr),
                selected=int(selected),
            )

            if best is None:
                best = cand
            else:
                # maximize recall, then precision, then threshold (more conservative)
                key_best = (best["recall"], best["precision"], best["t"])
                key_cand = (cand["recall"], cand["precision"], cand["t"])
                if key_cand > key_best:
                    best = cand

    return best