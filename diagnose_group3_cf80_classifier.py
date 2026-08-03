from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT_DIR = Path("reports/group3_cf80_classifier")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "kpx_group_3"
CAPACITY = 21000.0
VALID_START = pd.Timestamp("2024-01-01")
TOP_FRACS = [0.05, 0.10, 0.20, 0.30]
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


def load_v28_feature_matrix():
    source = Path("baseline_ontology_v28_gfs_grid5.py").read_text(encoding="utf-8")
    marker = "predictions = pd.DataFrame(index=sample_submission.index)"
    if marker not in source:
        raise RuntimeError("Cannot find v28 cutoff marker.")
    namespace = {"__name__": "__v28_cf80_classifier_loader__"}
    exec(source.split(marker)[0], namespace)
    return namespace["train_df"].copy(), namespace["X_train_imp"].copy()


def make_models():
    return {
        "logistic_l2_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5,
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=600,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "catboost_logloss_balanced": CatBoostClassifier(
            iterations=1200,
            depth=6,
            learning_rate=0.035,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def top_k_metrics(y_true, prob, fracs):
    rows = []
    order = np.argsort(-prob)
    n = len(y_true)
    positives = int(np.sum(y_true))
    for frac in fracs:
        k = max(1, int(round(n * frac)))
        idx = order[:k]
        tp = int(np.sum(y_true[idx]))
        rows.append(
            {
                "top_fraction": frac,
                "top_k_rows": k,
                "true_positive_in_top_k": tp,
                "precision_at_top_k": tp / k,
                "recall_at_top_k": tp / positives if positives else np.nan,
                "lift_vs_base_rate": (tp / k) / (positives / n) if positives else np.nan,
            }
        )
    return pd.DataFrame(rows)


def threshold_metrics(y_true, prob, thresholds):
    rows = []
    for threshold in thresholds:
        pred = (prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": threshold,
                "pred_positive_rows": int(pred.sum()),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }
        )
    return pd.DataFrame(rows)


def probability_decile_table(y_true, prob):
    df = pd.DataFrame({"y": y_true, "prob": prob})
    # qcut can fail with tied probabilities; rank first to force stable deciles.
    df["prob_decile"] = pd.qcut(df["prob"].rank(method="first"), 10, labels=False) + 1
    rows = []
    for decile, part in df.groupby("prob_decile"):
        rows.append(
            {
                "prob_decile_low_to_high": int(decile),
                "rows": len(part),
                "prob_min": part["prob"].min(),
                "prob_mean": part["prob"].mean(),
                "prob_max": part["prob"].max(),
                "actual_cf_ge80_rate": part["y"].mean(),
                "actual_cf_ge80_count": int(part["y"].sum()),
            }
        )
    return pd.DataFrame(rows)


def main():
    train_df, X = load_v28_feature_matrix()
    train_df["forecast_kst_dtm"] = pd.to_datetime(train_df["forecast_kst_dtm"])
    valid_label = train_df[TARGET].notna()
    actual_cf = train_df[TARGET] / CAPACITY
    y = (actual_cf >= 0.80).astype(int)

    train_mask = valid_label & (train_df["forecast_kst_dtm"] < VALID_START)
    valid_mask = valid_label & (train_df["forecast_kst_dtm"] >= VALID_START)

    X_train = X.loc[train_mask]
    X_valid = X.loc[valid_mask]
    y_train = y.loc[train_mask].to_numpy()
    y_valid = y.loc[valid_mask].to_numpy()
    valid_meta = train_df.loc[valid_mask, ["forecast_kst_dtm", TARGET]].copy()
    valid_meta["actual_cf"] = valid_meta[TARGET] / CAPACITY

    summary_rows = []
    top_frames = []
    threshold_frames = []
    decile_frames = []
    pred_frames = []

    base_rate_train = float(np.mean(y_train))
    base_rate_valid = float(np.mean(y_valid))

    for name, model in make_models().items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_valid)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_valid, prob)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )
        best_idx = int(np.argmax(f1))
        best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 1.0

        summary_rows.append(
            {
                "model": name,
                "train_rows": int(len(y_train)),
                "valid_rows": int(len(y_valid)),
                "train_cf_ge80_rate": base_rate_train,
                "valid_cf_ge80_rate": base_rate_valid,
                "roc_auc": roc_auc_score(y_valid, prob),
                "average_precision": average_precision_score(y_valid, prob),
                "best_f1_threshold": best_threshold,
                "best_f1": float(f1[best_idx]),
                "best_f1_precision": float(precision[best_idx]),
                "best_f1_recall": float(recall[best_idx]),
            }
        )

        top = top_k_metrics(y_valid, prob, TOP_FRACS)
        top.insert(0, "model", name)
        top_frames.append(top)

        thr = threshold_metrics(y_valid, prob, THRESHOLDS)
        thr.insert(0, "model", name)
        threshold_frames.append(thr)

        dec = probability_decile_table(y_valid, prob)
        dec.insert(0, "model", name)
        decile_frames.append(dec)

        pred = valid_meta.copy()
        pred["model"] = name
        pred["prob_cf_ge80"] = prob
        pred["actual_cf_ge80"] = y_valid
        pred_frames.append(pred)

    summary = pd.DataFrame(summary_rows).sort_values(["roc_auc", "average_precision"], ascending=False)
    top_summary = pd.concat(top_frames, ignore_index=True)
    threshold_summary = pd.concat(threshold_frames, ignore_index=True)
    decile_summary = pd.concat(decile_frames, ignore_index=True)
    predictions = pd.concat(pred_frames, ignore_index=True)

    summary.to_csv(OUT_DIR / "group3_cf80_classifier_summary.csv", index=False, encoding="utf-8-sig")
    top_summary.to_csv(OUT_DIR / "group3_cf80_classifier_topk.csv", index=False, encoding="utf-8-sig")
    threshold_summary.to_csv(OUT_DIR / "group3_cf80_classifier_thresholds.csv", index=False, encoding="utf-8-sig")
    decile_summary.to_csv(OUT_DIR / "group3_cf80_classifier_deciles.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT_DIR / "group3_cf80_classifier_valid_predictions.csv", index=False, encoding="utf-8-sig")

    print("\n=== Group3 CF>=80 classifier summary, 2024 holdout ===")
    print(summary.to_string(index=False))
    print("\n=== Top-k precision/recall ===")
    print(top_summary.to_string(index=False))
    print("\n=== Probability deciles: actual CF>=80 rate ===")
    best_model = summary.iloc[0]["model"]
    print(decile_summary[decile_summary["model"] == best_model].to_string(index=False))
    print("\n=== Threshold metrics ===")
    print(threshold_summary[threshold_summary["model"] == best_model].to_string(index=False))
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    main()
