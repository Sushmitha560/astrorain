"""
download_data.py — Fetch real daily rainfall from Open-Meteo (ERA5 reanalysis).
Run once before launching the app.  Takes ~30 seconds.

Usage:
    python download_data.py
"""

import requests, time, os
import pandas as pd

REGIONS = {
    "Mumbai":     {"lat": 19.076, "lon": 72.877},
    "Chennai":    {"lat": 13.083, "lon": 80.270},
    "Delhi":      {"lat": 28.613, "lon": 77.209},
    "Bengaluru":  {"lat": 12.972, "lon": 77.594},
    "Kolkata":    {"lat": 22.572, "lon": 88.364},
    "Jaipur":     {"lat": 26.912, "lon": 75.787},
    "Cherrapunji":{"lat": 25.300, "lon": 91.700},
    "Trivandrum": {"lat":  8.524, "lon": 76.936},
    "Lucknow":    {"lat": 26.847, "lon": 80.947},
    "Hyderabad":  {"lat": 17.385, "lon": 78.487},
    "Guwahati":   {"lat": 26.144, "lon": 91.736},
    "Pune":       {"lat": 18.520, "lon": 73.856},
}

API = "https://archive-api.open-meteo.com/v1/archive"
OUTPUT = "data/rainfall.csv"


def fetch(name, lat, lon, start="2000-01-01", end="2025-12-31"):
    r = requests.get(API, params={
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": "precipitation_sum",
        "timezone": "Asia/Kolkata",
    }, timeout=60)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(d["time"]), "rainfall_mm": d["precipitation_sum"]})
    df["region"], df["lat"], df["lon"] = name, lat, lon
    return df


def main():
    print("━" * 55)
    print("  AstroRain — Downloading Real Rainfall Data")
    print("  Source : Open-Meteo Historical API (ERA5)")
    print("  Period : 2000–2025 (24 years)")
    print("  Regions: 12 Indian cities")
    print("━" * 55)

    os.makedirs("data", exist_ok=True)
    frames = []
    for i, (name, info) in enumerate(REGIONS.items(), 1):
        print(f"  [{i:2d}/12]  {name:<14s}", end="  ")
        try:
            df = fetch(name, info["lat"], info["lon"])
            total = df["rainfall_mm"].sum()
            print(f"✓  {len(df):,} days  |  {total:,.0f} mm total")
            frames.append(df)
        except Exception as e:
            print(f"✗  {e}")
        time.sleep(1)

    if frames:
        out = pd.concat(frames, ignore_index=True).dropna(subset=["rainfall_mm"])
        out.to_csv(OUTPUT, index=False)
        print(f"\n{'━' * 55}")
        print(f"  ✓ Saved to {OUTPUT}")
        print(f"    {len(out):,} rows  |  {out['region'].nunique()} regions")
        print(f"    {out['date'].min().date()}  →  {out['date'].max().date()}")
        print(f"{'━' * 55}")
        print(f"\n  Next step:  streamlit run app.py")


if __name__ == "__main__":
    main()
