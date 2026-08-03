from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("reports/group3_v28_leaf_shrinkage")
OUT_DIR.mkdir(parents=True, exist_ok=True)

V28_SCRIPT = Path("baseline_ontology_v28_gfs_grid5.py")
TARGET = "kpx_group_3"
CAPACITY = 21000.0
VALID_START = pd.Timestamp("2024-01-01")
ALPHA = 1.10


CF_BINS = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, np.inf]
CF_LABELS = ["0-10", "10-20", "20-40", "40-60", "60-80", "80-90", "90-100", ">100"]


def load_v28_objects():
    source = V28_SCRIPT.read_text(encoding="utf-8")
    marker = "predictions = pd.DataFrame(index=sample_submission.index)"
    if marker not in source:
        raise RuntimeError("Cannot find v28 cutoff marker.")
    namespace = {"__name__": "__v28_leaf_loader__"}
    exec(source.split(marker)[0], namespace)
    return namespace


def bin_counts(values):
    bins = pd.cut(pd.Series(values), bins=CF_BINS, labels=CF_LABELS, right=False, include_lowest=True)
    counts = bins.value_counts().reindex(CF_LABELS).fillna(0).astype(int)
    return counts


def summarize_values(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {}
    counts = bin_counts(values)
    out = {
        "n_leaf_train_members_weighted": int(len(values)),
        "leaf_target_cf_mean": float(np.mean(values)),
        "leaf_target_cf_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "leaf_target_cf_q10": float(np.quantile(values, 0.10)),
        "leaf_target_cf_q25": float(np.quantile(values, 0.25)),
        "leaf_target_cf_median": float(np.quantile(values, 0.50)),
        "leaf_target_cf_q75": float(np.quantile(values, 0.75)),
        "leaf_target_cf_q90": float(np.quantile(values, 0.90)),
        "leaf_target_cf_lt60_ratio": float(np.mean(values < 0.60)),
        "leaf_target_cf_60_80_ratio": float(np.mean((values >= 0.60) & (values < 0.80))),
        "leaf_target_cf_ge80_ratio": float(np.mean(values >= 0.80)),
        "leaf_target_cf_ge90_ratio": float(np.mean(values >= 0.90)),
    }
    for label, count in counts.items():
        out[f"leaf_bin_{label}_count"] = int(count)
        out[f"leaf_bin_{label}_ratio"] = float(count / len(values))
    return out


def main():
    ns = load_v28_objects()
    train_df = ns["train_df"].copy()
    X = ns["X_train_imp"].copy()
    make_model = ns["make_model"]

    train_df["forecast_kst_dtm"] = pd.to_datetime(train_df["forecast_kst_dtm"])
    label_mask = train_df[TARGET].notna()
    fit_mask = label_mask & (train_df["forecast_kst_dtm"] < VALID_START)
    valid_mask = label_mask & (train_df["forecast_kst_dtm"] >= VALID_START)

    X_fit = X.loc[fit_mask].reset_index(drop=True)
    X_valid = X.loc[valid_mask].reset_index(drop=True)
    y_fit_cf = (train_df.loc[fit_mask, TARGET].to_numpy(dtype=float) / CAPACITY)
    y_valid_kwh = train_df.loc[valid_mask, TARGET].to_numpy(dtype=float)
    valid_time = train_df.loc[valid_mask, "forecast_kst_dtm"].reset_index(drop=True)

    model = make_model()
    model.fit(X_fit, y_fit_cf)
    raw_pred_cf = np.clip(model.predict(X_valid), 0, 1)
    pred_cf = np.clip(raw_pred_cf * ALPHA, 0, 1)
    actual_cf = y_valid_kwh / CAPACITY

    selected_mask = (actual_cf >= 0.90) & (actual_cf <= 1.00) & (pred_cf >= 0.60) & (pred_cf < 0.80)
    selected_idx = np.where(selected_mask)[0]
    if len(selected_idx) == 0:
        raise RuntimeError("No actual 90-100 / pred 60-80 samples found.")

    selected_rows = pd.DataFrame(
        {
            "valid_row_index": selected_idx,
            "forecast_kst_dtm": valid_time.iloc[selected_idx].to_numpy(),
            "actual_kwh": y_valid_kwh[selected_idx],
            "pred_kwh_alpha110": pred_cf[selected_idx] * CAPACITY,
            "actual_cf": actual_cf[selected_idx],
            "raw_pred_cf": raw_pred_cf[selected_idx],
            "pred_cf_alpha110": pred_cf[selected_idx],
            "error_cf_pred_minus_actual": pred_cf[selected_idx] - actual_cf[selected_idx],
        }
    )
    selected_rows.to_csv(
        OUT_DIR / "selected_actual90_100_pred60_80_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train_leaf = model.apply(X_fit)
    selected_leaf = model.apply(X_valid.iloc[selected_idx])

    # Cache each tree's leaf -> training target array. Then inspect the train target
    # distribution that each failed high-generation sample shares across the forest.
    tree_leaf_values = []
    for tree_i in range(train_leaf.shape[1]):
        mapping = {}
        leaves = train_leaf[:, tree_i]
        for leaf_id in np.unique(leaves):
            mapping[leaf_id] = y_fit_cf[leaves == leaf_id]
        tree_leaf_values.append(mapping)

    per_sample_rows = []
    pooled_values = []
    for row_pos, valid_idx in enumerate(selected_idx):
        values = []
        leaf_sizes = []
        leaf_means = []
        for tree_i in range(selected_leaf.shape[1]):
            leaf_id = selected_leaf[row_pos, tree_i]
            arr = tree_leaf_values[tree_i][leaf_id]
            values.append(arr)
            leaf_sizes.append(len(arr))
            leaf_means.append(float(np.mean(arr)))
        values = np.concatenate(values)
        pooled_values.append(values)
        summary = summarize_values(values)
        per_sample_rows.append(
            {
                "valid_row_index": int(valid_idx),
                "forecast_kst_dtm": valid_time.iloc[valid_idx],
                "actual_cf": float(actual_cf[valid_idx]),
                "raw_pred_cf": float(raw_pred_cf[valid_idx]),
                "pred_cf_alpha110": float(pred_cf[valid_idx]),
                "mean_leaf_size": float(np.mean(leaf_sizes)),
                "median_leaf_size": float(np.median(leaf_sizes)),
                "mean_of_tree_leaf_means": float(np.mean(leaf_means)),
                "std_of_tree_leaf_means": float(np.std(leaf_means, ddof=1)),
                **summary,
            }
        )

    per_sample = pd.DataFrame(per_sample_rows)
    per_sample.to_csv(
        OUT_DIR / "per_sample_leaf_target_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled_values = np.concatenate(pooled_values)
    pooled_summary = pd.DataFrame([summarize_values(pooled_values)])
    pooled_summary.insert(0, "selected_samples", len(selected_idx))
    pooled_summary.insert(1, "trees", len(model.estimators_))
    pooled_summary.insert(2, "alpha", ALPHA)
    pooled_summary.to_csv(
        OUT_DIR / "pooled_leaf_target_distribution_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled_bins = bin_counts(pooled_values).reset_index()
    pooled_bins.columns = ["train_target_cf_bin_in_shared_leaves", "weighted_count"]
    pooled_bins["weighted_ratio"] = pooled_bins["weighted_count"] / pooled_bins["weighted_count"].sum()
    pooled_bins.to_csv(
        OUT_DIR / "pooled_leaf_target_cf_bin_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Compare selected failed high samples with all actual 90-100 samples.
    all_high = pd.DataFrame(
        {
            "actual_cf": actual_cf,
            "pred_cf_alpha110": pred_cf,
        }
    )
    all_high = all_high[(all_high["actual_cf"] >= 0.90) & (all_high["actual_cf"] <= 1.00)]
    compare = pd.DataFrame(
        [
            {
                "subset": "all_actual_90_100",
                "rows": len(all_high),
                "pred_cf_mean": all_high["pred_cf_alpha110"].mean(),
                "pred_cf_median": all_high["pred_cf_alpha110"].median(),
                "pred_cf_60_80_count": int(((all_high["pred_cf_alpha110"] >= 0.60) & (all_high["pred_cf_alpha110"] < 0.80)).sum()),
                "pred_cf_80_90_count": int(((all_high["pred_cf_alpha110"] >= 0.80) & (all_high["pred_cf_alpha110"] < 0.90)).sum()),
                "pred_cf_90_100_count": int(((all_high["pred_cf_alpha110"] >= 0.90) & (all_high["pred_cf_alpha110"] <= 1.00)).sum()),
            },
            {
                "subset": "selected_actual_90_100_pred_60_80",
                "rows": len(selected_idx),
                "pred_cf_mean": np.mean(pred_cf[selected_idx]),
                "pred_cf_median": np.median(pred_cf[selected_idx]),
                "pred_cf_60_80_count": len(selected_idx),
                "pred_cf_80_90_count": 0,
                "pred_cf_90_100_count": 0,
            },
        ]
    )
    compare.to_csv(OUT_DIR / "selected_sample_size_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== Selected high-generation underprediction samples ===")
    print(compare.to_string(index=False))
    print("\n=== Pooled train target distribution in shared ExtraTrees leaves ===")
    print(pooled_summary.to_string(index=False))
    print("\n=== Pooled train target CF bin distribution in shared leaves ===")
    print(pooled_bins.to_string(index=False))
    print("\n=== Per-sample leaf distribution head ===")
    show_cols = [
        "forecast_kst_dtm",
        "actual_cf",
        "pred_cf_alpha110",
        "mean_leaf_size",
        "leaf_target_cf_mean",
        "leaf_target_cf_median",
        "leaf_target_cf_lt60_ratio",
        "leaf_target_cf_60_80_ratio",
        "leaf_target_cf_ge80_ratio",
        "leaf_target_cf_ge90_ratio",
    ]
    print(per_sample[show_cols].head(20).to_string(index=False))
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    main()
