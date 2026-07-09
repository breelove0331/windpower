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

def aggregate_weather(df, prefix):
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
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

for target in TARGET_COLS:
    train_mask = train_df[target].notna()
    y_train = train_df.loc[train_mask, target]

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_imp.loc[train_mask], y_train)

    pred = model.predict(X_test_imp)
    pred = np.clip(pred, 0, CAPACITY_KWH[target])
    predictions[target] = pred

    print(target, "train rows:", int(train_mask.sum()))

submission = sample_submission[["forecast_id", "forecast_kst_dtm"]].copy()
for target in TARGET_COLS:
    submission[target] = predictions[target]

submission["forecast_kst_dtm"] = pd.to_datetime(submission["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")

print(submission.head())
print(submission.shape)

output_path = DATA_DIR / "baseline_submit.csv"
submission.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"Saved: {output_path}")
