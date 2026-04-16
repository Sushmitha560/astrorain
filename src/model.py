"""
model.py — Two-Stage ML Pipeline

Stage 1: Rain Classifier — "Will it rain today?" (XGBoost Binary Classifier)
Stage 2: Rain Regressor  — "How many mm?"       (XGBoost on log-transformed rainfall)

All features come from ancient astronomical calculations.
Target (Y) is REAL observed daily rainfall from ERA5 reanalysis.
"""

import numpy as np
import pandas as pd
import pickle, os, math
from datetime import datetime
from src.astronomy import compute_panchanga

# ─── Feature Column Definitions ───

FEATURE_COLUMNS = [
    # Location & calendar (cyclical encoding)
    "lat", "lon", "day_of_year",
    "day_sin", "day_cos", "month_sin", "month_cos",
    # Lunar Mansion features
    "mansion_index", "mansion_rain_tendency", "mansion_is_water", "mansion_is_fire", "quarter",
    # Zodiac features
    "moon_sign_index", "sun_sign_index", "sign_rain_modifier", "sign_is_water",
    # Lunar Day features
    "lunar_day", "lunar_day_rain", "day_in_phase",
    # Yoga & Karana
    "yoga_index", "karana_index",
    # Raw longitudes
    "moon_longitude", "sun_longitude", "sun_moon_angle",
    # Derived
    "moon_speed_proxy", "ruler_is_benefic", "nature_divine", "nature_demonic",
]  # 28 features total


def build_features(date, lat, lon):
    """Build a feature dict for a single date + location."""
    p = compute_panchanga(date)
    doy = date.timetuple().tm_yday
    month = date.month
    return {
        "lat": lat, "lon": lon, "day_of_year": doy,
        "day_sin": math.sin(2 * math.pi * doy / 365),
        "day_cos": math.cos(2 * math.pi * doy / 365),
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
        "mansion_index": p["mansion_index"],
        "mansion_rain_tendency": p["mansion_rain_tendency"],
        "mansion_is_water": p["mansion_is_water"],
        "mansion_is_fire": p["mansion_is_fire"],
        "quarter": p["quarter"],
        "moon_sign_index": p["moon_sign_index"],
        "sun_sign_index": p["sun_sign_index"],
        "sign_rain_modifier": p["sign_rain_modifier"],
        "sign_is_water": p["sign_is_water"],
        "lunar_day": p["lunar_day"],
        "lunar_day_rain": p["lunar_day_rain"],
        "day_in_phase": p["day_in_phase"],
        "yoga_index": p["yoga_index"],
        "karana_index": p["karana_index"],
        "moon_longitude": p["moon_longitude"],
        "sun_longitude": p["sun_longitude"],
        "sun_moon_angle": p["sun_moon_angle"],
        "moon_speed_proxy": p["moon_speed_proxy"],
        "ruler_is_benefic": p["ruler_is_benefic"],
        "nature_divine": p["nature_divine"],
        "nature_demonic": p["nature_demonic"],
        # Panchanga metadata (not used as features, but passed through)
        "_panchanga": p,
    }


def engineer_dataset(df_raw, progress_callback=None):
    """Add astronomical features to the raw rainfall dataframe."""
    records = []
    total = len(df_raw)
    for i, (_, row) in enumerate(df_raw.iterrows()):
        date = pd.Timestamp(row["date"])
        feat = build_features(date, row["lat"], row["lon"])
        feat.pop("_panchanga")  # remove metadata
        feat["rainfall_mm"] = row["rainfall_mm"]
        feat["region"] = row["region"]
        feat["date"] = date
        records.append(feat)
        if progress_callback and i % 5000 == 0:
            progress_callback(i / total)
    return pd.DataFrame(records)


# ─── Model Training ───

MODEL_PATH = "model.pkl"


def _create_xgb_classifier(pos_weight):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.85, min_child_weight=5,
        scale_pos_weight=pos_weight,
        tree_method="hist", device="cuda", max_bin=256,
        n_jobs=-1, random_state=42, eval_metric="logloss",
    )


def _create_xgb_regressor():
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=800, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.85, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        tree_method="hist", device="cuda", max_bin=256,
        n_jobs=-1, random_state=42, eval_metric="mae",
    )


def _fit_with_fallback(model, X_train, y_train, X_val, y_val):
    """Try GPU first, fall back to CPU."""
    try:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    except Exception:
        model.set_params(device="cpu", tree_method="hist")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train(df_raw, progress_fn=None):
    """
    Train the two-stage model on real rainfall data.

    Returns a dict with classifier, regressor, metrics, and feature importances.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score, accuracy_score, f1_score, roc_auc_score)

    # Engineer features
    if progress_fn: progress_fn(5, "Computing astronomical positions...")
    df = engineer_dataset(df_raw)
    df = df.dropna(subset=["rainfall_mm"])

    X = df[FEATURE_COLUMNS].values
    y_mm = df["rainfall_mm"].values
    y_binary = (y_mm > 0.1).astype(int)
    y_log = np.log1p(y_mm)

    # Split
    if progress_fn: progress_fn(35, f"{len(df):,} samples ready. Splitting...")
    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=0.15, random_state=42)

    # Stage 1: Classifier
    if progress_fn: progress_fn(40, "Training rain detection classifier...")
    neg = (y_binary[idx_tr] == 0).sum()
    pos = max((y_binary[idx_tr] == 1).sum(), 1)
    clf = _create_xgb_classifier(neg / pos)
    clf = _fit_with_fallback(clf, X[idx_tr], y_binary[idx_tr], X[idx_te], y_binary[idx_te])

    clf_pred = clf.predict(X[idx_te])
    clf_prob = clf.predict_proba(X[idx_te])[:, 1]

    # Stage 2: Regressor (rainy days only)
    if progress_fn: progress_fn(65, "Training rainfall amount regressor...")
    rain_tr = y_binary[idx_tr] == 1
    rain_te = y_binary[idx_te] == 1

    reg = _create_xgb_regressor()
    reg = _fit_with_fallback(reg, X[idx_tr][rain_tr], y_log[idx_tr][rain_tr],
                             X[idx_te][rain_te], y_log[idx_te][rain_te])

    # Combined evaluation
    if progress_fn: progress_fn(85, "Evaluating...")
    combined = np.zeros(len(idx_te))
    pred_rain = clf_pred == 1
    if pred_rain.any():
        combined[pred_rain] = np.expm1(reg.predict(X[idx_te][pred_rain]))
    combined = np.clip(combined, 0, None)

    # Rainy-day-only metrics
    if rain_te.any():
        rain_pred = np.expm1(reg.predict(X[idx_te][rain_te]))
        rain_mae = mean_absolute_error(y_mm[idx_te][rain_te], rain_pred)
        rain_r2 = r2_score(y_mm[idx_te][rain_te], rain_pred)
    else:
        rain_mae, rain_r2 = 0.0, 0.0

    # Feature importances
    clf_fi = sorted(zip(FEATURE_COLUMNS, clf.feature_importances_.tolist()), key=lambda x: x[1], reverse=True)
    reg_fi = sorted(zip(FEATURE_COLUMNS, reg.feature_importances_.tolist()), key=lambda x: x[1], reverse=True)

    result = {
        "classifier": clf,
        "regressor": reg,
        "metrics": {
            "clf_accuracy": accuracy_score(y_binary[idx_te], clf_pred),
            "clf_f1": f1_score(y_binary[idx_te], clf_pred),
            "clf_auc": roc_auc_score(y_binary[idx_te], clf_prob),
            "overall_mae": mean_absolute_error(y_mm[idx_te], combined),
            "overall_rmse": np.sqrt(mean_squared_error(y_mm[idx_te], combined)),
            "overall_r2": r2_score(y_mm[idx_te], combined),
            "rain_only_mae": rain_mae,
            "rain_only_r2": rain_r2,
        },
        "clf_importance": clf_fi,
        "reg_importance": reg_fi,
        "data_stats": {
            "total": len(df), "train": len(idx_tr), "test": len(idx_te),
            "rain_days": int(y_binary.sum()),
            "dry_days": int((y_binary == 0).sum()),
            "avg_rainfall": float(y_mm.mean()),
            "max_rainfall": float(y_mm.max()),
            "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
            "regions": sorted(df["region"].unique().tolist()),
        },
        "trained_at": datetime.now().isoformat(),
    }

    if progress_fn: progress_fn(95, "Saving model...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(result, f)

    if progress_fn: progress_fn(100, "Done!")
    return result


def load():
    """Load a previously trained model."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


def predict(saved, date, lat, lon):
    """
    Two-stage prediction.
    Returns: (rain_probability, predicted_mm, panchanga_dict)
    """
    feat = build_features(date, lat, lon)
    panchanga = feat.pop("_panchanga")

    X = np.array([[feat[f] for f in FEATURE_COLUMNS]])
    rain_prob = float(saved["classifier"].predict_proba(X)[0, 1])

    if rain_prob > 0.5:
        amount = max(0.0, float(np.expm1(saved["regressor"].predict(X)[0])))
    else:
        amount = 0.0

    return rain_prob, amount, panchanga
