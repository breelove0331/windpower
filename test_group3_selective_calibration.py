from pathlib import Path

import numpy as np
import pandas as pd


PRED_PATH = Path("reports/v28_group3_holdout_error/v28_group3_holdout_predictions_with_features.csv")
CLF_PATH = Path("reports/group3_cf80_classifier/group3_cf80_classifier_valid_predictions.csv")
OUT_DIR = Path("reports/group3_selective_calibration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAPACITY = 21000.0
MODEL_NAME = "extra_trees_balanced"
ALPHAS = [1.02, 1.03, 1.05, 1.08, 1.10, 1.12]
PROB_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
TOP_FRACS = [0.05, 0.10, 0.15, 0.20]
PRED_CF_FLOORS = [0.50, 0.55, 0.60]


def score(actual, pred):
    valid = actual >= CAPACITY * 0.10
    actual_v = actual[valid]
    pred_v = pred[valid]
    err = np.abs(pred_v - actual_v) / CAPACITY
    nmae = float(np.mean(err))
    unit_price = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
    ficr = float(np.sum(actual_v * unit_price) / np.sum(actual_v * 4.0))
    return {
        "official_rows": int(valid.sum()),
        "nmae": nmae,
        "one_minus_nmae": 1 - nmae,
        "ficr": ficr,
        "local_total": 0.5 * (1 - nmae) + 0.5 * ficr,
        "under_6_rate": float(np.mean(err <= 0.06)),
        "under_8_rate": float(np.mean(err <= 0.08)),
    }


def bin_summary(actual, pred):
    df = pd.DataFrame({"actual": actual, "pred": pred})
    df["actual_cf"] = df["actual"] / CAPACITY
    df["pred_cf"] = df["pred"] / CAPACITY
    bins = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, np.inf]
    labels = ["0-10", "10-20", "20-40", "40-60", "60-80", "80-90", "90-100", ">100"]
    df["actual_cf_bin"] = pd.cut(df["actual_cf"], bins=bins, labels=labels, right=False, include_lowest=True)
    rows = []
    for label, part in df.groupby("actual_cf_bin", observed=False):
        if len(part) == 0:
            continue
        rows.append(
            {
                "actual_cf_bin": str(label),
                "rows": len(part),
                "actual_cf_mean": part["actual_cf"].mean(),
                "pred_cf_mean": part["pred_cf"].mean(),
                "bias_cf": (part["pred_cf"] - part["actual_cf"]).mean(),
                "mae_kwh": np.abs(part["pred"] - part["actual"]).mean(),
            }
        )
    return pd.DataFrame(rows)


def load_joined():
    pred = pd.read_csv(PRED_PATH, encoding="utf-8-sig")
    pred["forecast_kst_dtm"] = pd.to_datetime(pred["forecast_kst_dtm"])
    clf = pd.read_csv(CLF_PATH, encoding="utf-8-sig")
    clf["forecast_kst_dtm"] = pd.to_datetime(clf["forecast_kst_dtm"])
    clf = clf[clf["model"] == MODEL_NAME][["forecast_kst_dtm", "prob_cf_ge80", "actual_cf_ge80"]].copy()
    df = pred.merge(clf, on="forecast_kst_dtm", how="left", validate="one_to_one")
    if df["prob_cf_ge80"].isna().any():
        raise ValueError("Missing classifier probabilities after merge.")
    df["actual"] = df["actual"].astype(float)
    df["base_pred"] = df["pred"].astype(float)
    df["actual_cf"] = df["actual"] / CAPACITY
    df["base_pred_cf"] = df["base_pred"] / CAPACITY
    return df


def evaluate_case(df, case_name, selected_mask, alpha):
    pred = df["base_pred"].to_numpy().copy()
    pred[selected_mask] = np.clip(pred[selected_mask] * alpha, 0, CAPACITY)
    actual = df["actual"].to_numpy()
    selected = df[selected_mask]
    out = {
        "case": case_name,
        "alpha": alpha,
        "selected_rows": int(selected_mask.sum()),
        "selected_actual_cf_ge80_rate": float((selected["actual_cf"] >= 0.8).mean()) if len(selected) else np.nan,
        "selected_actual_cf_ge90_rate": float((selected["actual_cf"] >= 0.9).mean()) if len(selected) else np.nan,
        "selected_base_pred_cf_mean": float(selected["base_pred_cf"].mean()) if len(selected) else np.nan,
        "selected_prob_mean": float(selected["prob_cf_ge80"].mean()) if len(selected) else np.nan,
        **score(actual, pred),
    }
    bins = bin_summary(actual, pred)
    bins.insert(0, "alpha", alpha)
    bins.insert(0, "case", case_name)
    return out, bins


def main():
    df = load_joined()
    actual = df["actual"].to_numpy()
    base_pred = df["base_pred"].to_numpy()

    rows = []
    bin_frames = []

    base_score = score(actual, base_pred)
    rows.append(
        {
            "case": "v28_base_no_selective_calibration",
            "alpha": 1.0,
            "selected_rows": 0,
            "selected_actual_cf_ge80_rate": np.nan,
            "selected_actual_cf_ge90_rate": np.nan,
            "selected_base_pred_cf_mean": np.nan,
            "selected_prob_mean": np.nan,
            **base_score,
        }
    )
    base_bins = bin_summary(actual, base_pred)
    base_bins.insert(0, "alpha", 1.0)
    base_bins.insert(0, "case", "v28_base_no_selective_calibration")
    bin_frames.append(base_bins)

    for pred_floor in PRED_CF_FLOORS:
        floor_mask = df["base_pred_cf"].to_numpy() >= pred_floor

        for threshold in PROB_THRESHOLDS:
            mask = (df["prob_cf_ge80"].to_numpy() >= threshold) & floor_mask
            for alpha in ALPHAS:
                out, bins = evaluate_case(df, f"prob_ge_{threshold:.2f}_predcf_ge_{pred_floor:.2f}", mask, alpha)
                rows.append(out)
                bin_frames.append(bins)

        order = np.argsort(-df["prob_cf_ge80"].to_numpy())
        n = len(df)
        for frac in TOP_FRACS:
            top_mask = np.zeros(n, dtype=bool)
            top_mask[order[: max(1, int(round(n * frac)))]] = True
            mask = top_mask & floor_mask
            for alpha in ALPHAS:
                out, bins = evaluate_case(df, f"top_{int(frac*100):02d}pct_predcf_ge_{pred_floor:.2f}", mask, alpha)
                rows.append(out)
                bin_frames.append(bins)

    summary = pd.DataFrame(rows)
    summary["delta_local_total_vs_base"] = summary["local_total"] - base_score["local_total"]
    summary["delta_ficr_vs_base"] = summary["ficr"] - base_score["ficr"]
    summary["delta_nmae_vs_base"] = summary["nmae"] - base_score["nmae"]
    summary = summary.sort_values(["local_total", "ficr"], ascending=False)
    bins_all = pd.concat(bin_frames, ignore_index=True)

    summary.to_csv(OUT_DIR / "selective_calibration_summary.csv", index=False, encoding="utf-8-sig")
    bins_all.to_csv(OUT_DIR / "selective_calibration_bin_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== Selective calibration summary top 30 ===")
    show_cols = [
        "case",
        "alpha",
        "selected_rows",
        "selected_actual_cf_ge80_rate",
        "selected_actual_cf_ge90_rate",
        "nmae",
        "one_minus_nmae",
        "ficr",
        "local_total",
        "delta_local_total_vs_base",
        "under_6_rate",
        "under_8_rate",
    ]
    print(summary[show_cols].head(30).to_string(index=False))

    best = summary.iloc[0]
    best_bins = bins_all[(bins_all["case"] == best["case"]) & (bins_all["alpha"] == best["alpha"])]
    print("\n=== Best case bin summary ===")
    print(best_bins.to_string(index=False))
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    main()
