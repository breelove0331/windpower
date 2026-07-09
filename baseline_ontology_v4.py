from pathlib import Path
import re

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
GROUP_IDS = {
    "kpx_group_1": 1,
    "kpx_group_2": 2,
    "kpx_group_3": 3,
}
VALID_START = pd.Timestamp("2024-01-01 00:00:00")
ALPHA_GRID = np.round(np.arange(0.90, 1.101, 0.005), 3)

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

TURBINE_META = [
    (1, "37°16'55.61\"N 128°57'02.10\"E"),
    (1, "37°17'04.05\"N 128°56'58.35\"E"),
    (1, "37°17'11.49\"N 128°56'58.99\"E"),
    (1, "37°17'23.11\"N 128°57'03.68\"E"),
    (1, "37°17'28.20\"N 128°57'15.58\"E"),
    (1, "37°17'19.48\"N 128°57'24.96\"E"),
    (2, "37°17'16.20\"N 128°57'34.67\"E"),
    (2, "37°17'11.29\"N 128°57'47.24\"E"),
    (2, "37°17'00.97\"N 128°57'57.44\"E"),
    (2, "37°16'52.77\"N 128°58'04.18\"E"),
    (2, "37°16'44.89\"N 128°58'01.12\"E"),
    (2, "37°16'30.58\"N 128°58'02.54\"E"),
    (3, "37°16'59.73\"N 128°57'44.97\"E"),
    (3, "37°16'40.41\"N 128°58'13.80\"E"),
    (3, "37°16'28.03\"N 128°58'22.54\"E"),
    (3, "37°16'18.58\"N 128°58'29.01\"E"),
    (3, "37°16'06.83\"N 128°58'35.68\"E"),
]


def dms_to_decimal(coord_text):
    pattern = r"(\d+)°(\d+)'([\d.]+)\"([NSEW])"
    matches = re.findall(pattern, coord_text)
    values = []
    for deg, minute, second, hemi in matches:
        value = float(deg) + float(minute) / 60 + float(second) / 3600
        if hemi in {"S", "W"}:
            value *= -1
        values.append(value)
    if len(values) != 2:
        raise ValueError(f"Cannot parse coordinate: {coord_text}")
    return values[0], values[1]


TURBINES = pd.DataFrame(
    [
        {"group_id": group_id, "latitude": dms_to_decimal(coord)[0], "longitude": dms_to_decimal(coord)[1]}
        for group_id, coord in TURBINE_META
    ]
)


def haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


def make_group_grid_weights(df):
    grid_meta = df[["grid_id", "latitude", "longitude"]].drop_duplicates().copy()
    weight_frames = []

    for group_id in sorted(TURBINES["group_id"].unique()):
        group_turbines = TURBINES[TURBINES["group_id"] == group_id]
        weights = np.zeros(len(grid_meta))

        for _, turbine in group_turbines.iterrows():
            dist_km = haversine_km(
                grid_meta["latitude"].to_numpy(),
                grid_meta["longitude"].to_numpy(),
                turbine["latitude"],
                turbine["longitude"],
            )
            weights += 1 / np.maximum(dist_km, 0.05) ** 2

        weights = weights / weights.sum()
        weight_frame = grid_meta[["grid_id"]].copy()
        weight_frame["group_id"] = group_id
        weight_frame["grid_weight"] = weights
        weight_frames.append(weight_frame)

    return pd.concat(weight_frames, ignore_index=True)


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
    grid_weights = make_group_grid_weights(df)
    drop_cols = {"data_available_kst_dtm", "grid_id", "latitude", "longitude"}
    value_cols = [c for c in df.columns if c not in {"forecast_kst_dtm", *drop_cols}]

    agg = df.groupby("forecast_kst_dtm")[value_cols].mean()
    agg.columns = [f"{prefix}_{c}_mean" for c in agg.columns]
    agg = agg.reset_index()

    weighted = df[["forecast_kst_dtm", "grid_id", *value_cols]].merge(grid_weights, on="grid_id", how="inner")
    weighted[value_cols] = weighted[value_cols].multiply(weighted["grid_weight"], axis=0)
    weighted_agg = weighted.groupby(["forecast_kst_dtm", "group_id"])[value_cols].sum().reset_index()

    for group_id in sorted(TURBINES["group_id"].unique()):
        group_agg = weighted_agg[weighted_agg["group_id"] == group_id].drop(columns=["group_id"])
        group_agg = group_agg.rename(
            columns={c: f"{prefix}_g{group_id}_{c}_distw" for c in value_cols}
        )
        agg = agg.merge(group_agg, on="forecast_kst_dtm", how="left")

    return agg


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

output_path = DATA_DIR / "ontology_submit_v4.csv"
submission.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"Saved: {output_path}")
