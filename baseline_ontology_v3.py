from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

DATA_DIR = Path(".")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY_KWH = {
    "kpx_group_1": 21600,
    "kpx_group_2": 21600,
    "kpx_group_3": 21000,
}
VALID_START = pd.Timestamp("2024-01-01 00:00:00")
ALPHA_GRID = np.round(np.arange(0.80, 1.251, 0.005), 3)

train_labels = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8-sig")

ldaps_train = pd.read_csv(TRAIN_DIR / "ldaps_train.csv", encoding="utf-8-sig")
gfs_train = pd.read_csv(TRAIN_DIR / "gfs_train.csv", encoding="utf-8-sig")
ldaps_test = pd.read_csv(TEST_DIR / "ldaps_test.csv", encoding="utf-8-sig")
gfs_test = pd.read_csv(TEST_DIR / "gfs_test.csv", encoding="utf-8-sig")

train_labels["kst_dtm"] = pd.to_datetime(train_labels["kst_dtm"])
sample_submission["forecast_kst_dtm"] = pd.to_datetime(sample_submission["forecast_kst_dtm"])

print("train_labels:", train_labels.shape)
print("sample_submission:", sample_submission.shape)
print("ldaps_train:", ldaps_train.shape, "gfs_train:", gfs_train.shape)
print("ldaps_test:", ldaps_test.shape, "gfs_test:", gfs_test.shape)

WIND_VECTOR_PAIRS = {
    "ldaps": [
        ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "wind_10m"),
        ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax", "wind_50m_max"),
        ("heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin", "wind_50m_min"),
        ("heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS", "wind_5m_blws"),
    ],
    "gfs": [
        ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "wind_10m"),
        ("heightAboveGround_80_u", "heightAboveGround_80_v", "wind_80m"),
        ("heightAboveGround_100_100u", "heightAboveGround_100_100v", "wind_100m"),
        ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v", "wind_pbl"),
        ("isobaricInhPa_850_u", "isobaricInhPa_850_v", "wind_850hpa"),
        ("isobaricInhPa_700_u", "isobaricInhPa_700_v", "wind_700hpa"),
        ("isobaricInhPa_500_u", "isobaricInhPa_500_v", "wind_500hpa"),
    ],
}


def add_wind_ontology_features(df, prefix):
    df = df.copy()
    for u_col, v_col, name in WIND_VECTOR_PAIRS[prefix]:
        if u_col not in df.columns or v_col not in df.columns:
            continue

        speed = np.sqrt(df[u_col] ** 2 + df[v_col] ** 2)
        df[f"{name}_speed"] = speed
        df[f"{name}_speed_sq"] = speed ** 2
        df[f"{name}_speed_cube"] = speed ** 3
        df[f"{name}_dir_sin"] = df[v_col] / speed.replace(0, np.nan)
        df[f"{name}_dir_cos"] = df[u_col] / speed.replace(0, np.nan)

    if prefix == "gfs" and "surface_0_gust" in df.columns:
        gust = df["surface_0_gust"].clip(lower=0)
        df["gust_speed_sq"] = gust ** 2
        df["gust_speed_cube"] = gust ** 3

    return df


def aggregate_weather(df, prefix):
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    df = add_wind_ontology_features(df, prefix)
    drop_cols = {"data_available_kst_dtm", "grid_id", "latitude", "longitude"}
    value_cols = [c for c in df.columns if c not in {"forecast_kst_dtm", *drop_cols}]
    agg = df.groupby("forecast_kst_dtm")[value_cols].mean()
    agg.columns = [f"{prefix}_{c}_mean" for c in agg.columns]
    return agg.reset_index()


def calendar_features(dt_series):
    dt = pd.to_datetime(dt_series)
    out = pd.DataFrame(index=dt.index)
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def make_model():
    return RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def calc_group_scores(y_true_kwh, y_pred_kwh, capacity_kwh):
    nmae_by_time = np.abs(y_pred_kwh - y_true_kwh) / capacity_kwh
    nmae = float(np.mean(nmae_by_time))

    settlement_rate = np.where(
        nmae_by_time <= 0.06,
        4,
        np.where(nmae_by_time <= 0.08, 3, 0),
    )
    ficr = float(np.sum(settlement_rate) / (4 * len(settlement_rate)))
    under_6 = float(np.mean(nmae_by_time <= 0.06))
    under_8 = float(np.mean(nmae_by_time <= 0.08))
    return nmae, ficr, under_6, under_8


def tune_alpha(y_true_kwh, y_pred_kwh, capacity_kwh):
    best = None
    base_nmae, base_ficr, base_under_6, base_under_8 = calc_group_scores(
        y_true_kwh,
        np.clip(y_pred_kwh, 0, capacity_kwh),
        capacity_kwh,
    )
    for alpha in ALPHA_GRID:
        pred = np.clip(y_pred_kwh * alpha, 0, capacity_kwh)
        nmae, ficr, under_6, under_8 = calc_group_scores(y_true_kwh, pred, capacity_kwh)
        # FICR is the settlement target; NMAE breaks ties in favor of accuracy.
        key = (ficr, -nmae)
        if best is None or key > best["key"]:
            best = {
                "alpha": float(alpha),
                "nmae": nmae,
                "ficr": ficr,
                "under_6": under_6,
                "under_8": under_8,
                "base_nmae": base_nmae,
                "base_ficr": base_ficr,
                "base_under_6": base_under_6,
                "base_under_8": base_under_8,
                "key": key,
            }
    return best


train_weather = aggregate_weather(ldaps_train, "ldaps").merge(
    aggregate_weather(gfs_train, "gfs"), on="forecast_kst_dtm", how="inner"
)
test_weather = aggregate_weather(ldaps_test, "ldaps").merge(
    aggregate_weather(gfs_test, "gfs"), on="forecast_kst_dtm", how="inner"
)

train_base = train_labels.rename(columns={"kst_dtm": "forecast_kst_dtm"})
train_df = train_base.merge(train_weather, on="forecast_kst_dtm", how="left")
test_df = sample_submission[["forecast_id", "forecast_kst_dtm"]].merge(
    test_weather, on="forecast_kst_dtm", how="left"
)

X_train = pd.concat(
    [calendar_features(train_df["forecast_kst_dtm"]), train_df.drop(columns=["forecast_kst_dtm", *TARGET_COLS])],
    axis=1,
)
X_test = pd.concat(
    [calendar_features(test_df["forecast_kst_dtm"]), test_df.drop(columns=["forecast_id", "forecast_kst_dtm"])],
    axis=1,
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

predictions = pd.DataFrame(index=sample_submission.index)
best_alphas = {}
valid_results = []

for target in TARGET_COLS:
    train_mask = train_df[target].notna()
    valid_mask = train_mask & (train_df["forecast_kst_dtm"] >= VALID_START)
    fit_mask = train_mask & (train_df["forecast_kst_dtm"] < VALID_START)

    valid_model = make_model()
    valid_y = train_df.loc[fit_mask, target] / CAPACITY_KWH[target]
    valid_model.fit(X_train_imp.loc[fit_mask], valid_y)

    valid_pred = valid_model.predict(X_train_imp.loc[valid_mask])
    valid_pred = np.clip(valid_pred, 0, 1) * CAPACITY_KWH[target]
    valid_true = train_df.loc[valid_mask, target].to_numpy()
    best_alpha = tune_alpha(valid_true, valid_pred, CAPACITY_KWH[target])
    best_alphas[target] = best_alpha["alpha"]
    valid_results.append(best_alpha)

    print(
        target,
        "fit rows:", int(fit_mask.sum()),
        "valid rows:", int(valid_mask.sum()),
        "alpha:", best_alpha["alpha"],
        "base nmae:", round(best_alpha["base_nmae"], 5),
        "base ficr:", round(best_alpha["base_ficr"], 5),
        "valid nmae:", round(best_alpha["nmae"], 5),
        "valid ficr:", round(best_alpha["ficr"], 5),
        "valid <=6%:", round(best_alpha["under_6"], 5),
        "valid <=8%:", round(best_alpha["under_8"], 5),
    )

    y_train = train_df.loc[train_mask, target] / CAPACITY_KWH[target]
    model = make_model()
    model.fit(X_train_imp.loc[train_mask], y_train)

    pred = model.predict(X_test_imp)
    pred = np.clip(pred * best_alphas[target], 0, 1) * CAPACITY_KWH[target]
    predictions[target] = pred

    print(target, "final train rows:", int(train_mask.sum()))

mean_valid_nmae = float(np.mean([r["nmae"] for r in valid_results]))
mean_valid_ficr = float(np.mean([r["ficr"] for r in valid_results]))
print("validation 1-NMAE:", round(1 - mean_valid_nmae, 5))
print("validation FICR:", round(mean_valid_ficr, 5))
print("best alphas:", best_alphas)

submission = sample_submission[["forecast_id", "forecast_kst_dtm"]].copy()
for target in TARGET_COLS:
    submission[target] = predictions[target]

submission["forecast_kst_dtm"] = pd.to_datetime(submission["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")

print(submission.head())
print(submission.shape)

output_path = DATA_DIR / "ontology_submit_v3.csv"
submission.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"Saved: {output_path}")
