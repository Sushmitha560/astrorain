# ☽ AstroRain — Rainfall Prediction Using Ancient Astronomy

A machine learning system that uses **ancient Indian astronomical calculations** as the sole input features to predict **real observed daily rainfall** across 12 Indian cities.

The project answers a simple question: **can the positions of the Sun and Moon — as tracked by ancient astronomers — predict whether it will rain?**

---

## Overview

| Component | Description |
|-----------|-------------|
| **Features (X)** | 28 variables derived from ancient astronomical calculations — lunar mansions, zodiac signs, lunar days, planetary yogas, and raw ecliptic longitudes |
| **Target (Y)** | Real daily rainfall in mm from ERA5 reanalysis (Open-Meteo Historical API), 2000–2025 |
| **Model** | Two-stage XGBoost: Stage 1 classifies rain/no-rain, Stage 2 regresses amount on log-scale |
| **Regions** | 12 Indian cities spanning different climate zones |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download real rainfall data (runs ~30 seconds)
python download_data.py

# 3. Launch the app
streamlit run app.py

# 4. Click "Train Model" in the sidebar
```

---

### The 28 Features

All features come from astronomical calculations — no modern weather data:

| Category | Features | Count |
|----------|----------|-------|
| Location + Calendar | lat, lon, day_of_year (sin/cos encoded), month (sin/cos encoded) | 7 |
| Lunar Mansion | index, rain tendency, is-water-element, is-fire-element, quarter | 5 |
| Zodiac Sign | moon sign, sun sign, rain modifier, is-water-sign | 4 |
| Lunar Day | day number, rain association, position in phase | 3 |
| Yoga & Karana | yoga index, karana index | 2 |
| Raw Positions | moon longitude, sun longitude, sun-moon angle | 3 |
| Derived | moon speed proxy, ruler benefic/malefic, divine/demonic nature | 4 |

### Two-Stage ML Model (`src/model.py`)

**Why two stages?** Rainfall data is heavily skewed — most days are dry. A single regressor averages everything toward low values.

**Stage 1 — Rain Detection (XGBoost Classifier)**
- Binary: will it rain today (> 0.1 mm)?
- Uses `scale_pos_weight` to handle class imbalance (more dry days than wet)
- Outputs a rain probability (0–100%)

**Stage 2 — Rainfall Amount (XGBoost Regressor)**
- Trained only on days that actually had rain
- Predicts `log(rainfall + 1)` to handle the skewed distribution
- Back-transforms with `exp(prediction) - 1` to get mm

**Combined prediction:** If Stage 1 says "rain" (prob > 50%), Stage 2 predicts the amount. Otherwise, prediction is 0 mm.

## Results

- **Rain Detection (Classifier)**
  - Accuracy: **82.2%**
  - AUC: **0.901**

- **Rainfall Amount (Regressor)**
  - MAE: 3.85 mm 

**Insight:**  
Even without modern weather data, the model learns seasonal patterns (e.g., monsoon cycles).

---

## Disclaimer

- Astronomical calculations are **scientifically accurate**
- Panchanga-based features are **traditional and heuristic**
- No atmospheric data is used (humidity, pressure, etc.)

This is an **experimental project**, not a production weather forecasting system.

---

## Future Work

- Add weather features (temperature, humidity)
- Try other models (LightGBM, CatBoost)
- Use time-series models (LSTM, Transformer)
- Perform feature importance & ablation studies
- Add SHAP for model interpretability
- Expand dataset (more regions, longer time range)

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9+ | 3.11+ |
| RAM | 4 GB | 8 GB |

---

## License

MIT
