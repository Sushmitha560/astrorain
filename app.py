"""
AstroRain — Rainfall Prediction Using Ancient Astronomy
Streamlit application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from src.astronomy import compute_panchanga, LUNAR_MANSIONS
from src.model import train, load, predict, FEATURE_COLUMNS

# ─── Page Setup ───
st.set_page_config(page_title="AstroRain", page_icon="☽", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg:       #f6f4f0;
    --surface:  #ffffff;
    --surface2: #f0ede8;
    --ink:      #1a1a1a;
    --ink2:     #5a5a5a;
    --ink3:     #8a8a8a;
    --accent:   #2563eb;
    --gold:     #b8860b;
    --rain:     #1d4ed8;
    --dry:      #65a30d;
    --heavy:    #dc2626;
    --border:   #e5e2dc;
    --radius:   12px;
}

.stApp { background: var(--bg); font-family: 'IBM Plex Sans', sans-serif; color: var(--ink); }

/* Header */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 2.2rem 2.5rem; border-radius: var(--radius); margin-bottom: 2rem;
    display: flex; justify-content: space-between; align-items: center;
}
.app-header .left h1 {
    font-family: 'DM Serif Display', serif; color: #f1f5f9;
    font-size: 1.9rem; margin: 0; letter-spacing: -0.3px;
}
.app-header .left p { color: #94a3b8; font-size: 0.9rem; margin: 0.2rem 0 0; font-weight: 300; }
.app-header .badge {
    background: #166534; color: #bbf7d0; padding: 0.3rem 0.9rem;
    border-radius: 20px; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.5px; text-transform: uppercase;
}

/* Cards */
.card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.4rem; margin-bottom: 0.8rem;
}
.card-dark {
    background: #0f172a; border: 1px solid #1e293b;
    border-radius: var(--radius); padding: 1.6rem; text-align: center;
}
.card-dark h2 {
    font-family: 'IBM Plex Mono', monospace; font-size: 2.8rem;
    font-weight: 500; margin: 0;
}
.card-dark .label {
    color: #94a3b8; font-size: 0.78rem; text-transform: uppercase;
    letter-spacing: 1.5px; margin-top: 0.3rem;
}
.card-dark .tag {
    display: inline-block; padding: 0.2rem 0.6rem; border-radius: 6px;
    font-size: 0.75rem; font-weight: 600; margin-top: 0.6rem;
}

/* Panchanga Grid */
.pan-item {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
}
.pan-item .title {
    font-size: 0.7rem; color: var(--ink3); text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 0.3rem;
}
.pan-item .value {
    font-family: 'DM Serif Display', serif; font-size: 1.15rem;
    color: var(--ink); margin-bottom: 0.15rem;
}
.pan-item .detail { font-size: 0.78rem; color: var(--ink2); }

/* Stat Box */
.stat {
    background: var(--surface2); border-radius: 10px;
    padding: 1rem; text-align: center;
}
.stat .num {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem;
    font-weight: 500; color: var(--ink);
}
.stat .lbl {
    font-size: 0.7rem; color: var(--ink3); text-transform: uppercase;
    letter-spacing: 1px; margin-top: 0.15rem;
}

/* Winner */
.winner {
    background: var(--surface); border: 2px solid var(--accent);
    border-radius: var(--radius); padding: 1rem; text-align: center;
}
.winner .num {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem;
    font-weight: 500; color: var(--accent);
}
</style>
""", unsafe_allow_html=True)


# ─── Constants ───
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

DATA_PATH = "data/rainfall.csv"


def rain_color(mm):
    if mm < 0.5: return "#65a30d"
    if mm < 10: return "#2563eb"
    if mm < 30: return "#d97706"
    if mm < 60: return "#ea580c"
    return "#dc2626"

def rain_tag(mm):
    if mm < 0.5: return "Dry", "#1a3a0a", "#bbf7d0"
    if mm < 10:  return "Light", "#0c2461", "#bfdbfe"
    if mm < 30:  return "Moderate", "#5a3e00", "#fde68a"
    if mm < 60:  return "Heavy", "#5a2000", "#fed7aa"
    return "Very Heavy", "#5a0a0a", "#fecaca"


# ─── App ───
def main():

    # Header
    st.markdown("""
    <div class="app-header">
        <div class="left">
            <h1>☽ AstroRain</h1>
            <p>Rainfall prediction using ancient astronomical features</p>
        </div>
        <div class="badge">Trained on Real ERA5 Data</div>
    </div>
    """, unsafe_allow_html=True)

    has_data = os.path.exists(DATA_PATH)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("#### Region")
        region = st.selectbox("City", list(REGIONS.keys()), index=3, label_visibility="collapsed")
        ri = REGIONS[region]
        st.caption(f"{ri['lat']:.2f}°N, {ri['lon']:.2f}°E")

        st.markdown("#### Date")
        pdate = st.date_input("Date", value=datetime.now().date(),
                              min_value=datetime(1950,1,1).date(),
                              max_value=datetime(2100,12,31).date(),
                              label_visibility="collapsed")
        pdt = datetime.combine(pdate, datetime.min.time())

        st.divider()
        st.markdown("#### Model")
        if has_data:
            df_raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
            st.caption(f"Data: {len(df_raw):,} days across {df_raw['region'].nunique()} cities")
        else:
            df_raw = None
            st.error("No data. Run `python download_data.py`")

        saved = load()
        if saved:
            m = saved["metrics"]
            st.success("Model ready")
            st.caption(f"Rain detection: {m['clf_accuracy']:.1%} accuracy, {m['clf_auc']:.3f} AUC")
            st.caption(f"Amount: {m['overall_mae']:.2f} mm MAE (overall)")
            st.caption(f"Rainy days: {m['rain_only_mae']:.2f} mm MAE")
        else:
            st.warning("Not trained yet")

        if has_data and st.button("Train Model", use_container_width=True, type="primary"):
            prog = st.progress(0)
            saved = train(df_raw, progress_fn=lambda p, t: prog.progress(int(p), text=t))
            st.rerun()

    if not has_data:
        st.info("**Getting started:** run `python download_data.py` to fetch 24 years of real rainfall data, then click Train Model in the sidebar.")
        return

    # ── Panchanga Display ──
    pan = compute_panchanga(pdt)

    st.markdown(f"### Astronomical Reading — {pdate.strftime('%B %d, %Y')}")

    c1, c2, c3, c4, c5 = st.columns(5)
    items = [
        (c1, "Lunar Mansion", pan["mansion_name"],
         f"Ruler: {pan['mansion_ruler']}  ·  {pan['mansion_element'].title()}  ·  Quarter {pan['quarter']}"),
        (c2, "Lunar Day", f"Day {pan['lunar_day']} ({pan['lunar_day_name']})",
         f"{pan['lunar_phase']} phase  ·  Rain index: {pan['lunar_day_rain']:.0%}"),
        (c3, "Moon Sign", pan["moon_sign_name"],
         f"Element: {pan['sign_element'].title()}  ·  Modifier: {pan['sign_rain_modifier']:+.2f}"),
        (c4, "Yoga", pan["yoga_name"],
         f"Index: {pan['yoga_index']}  ·  Karana: {pan['karana_index']}"),
        (c5, "Season", pan["season"],
         f"Sun in {pan['sun_sign_name']}  ·  Moon: {pan['moon_longitude']:.1f}°"),
    ]
    for col, title, value, detail in items:
        with col:
            st.markdown(f"""
            <div class="pan-item">
                <div class="title">{title}</div>
                <div class="value">{value}</div>
                <div class="detail">{detail}</div>
            </div>""", unsafe_allow_html=True)

    # Rain tendency from text
    rain_pct = pan["mansion_rain_tendency"]
    tendency = "very high" if rain_pct > 0.8 else "high" if rain_pct > 0.6 else "moderate" if rain_pct > 0.4 else "low"
    st.caption(f"Lunar mansion **{pan['mansion_name']}** has a **{tendency}** traditional rain association ({rain_pct:.0%}) according to ancient texts. Moon is in **{pan['moon_sign_name']}** ({pan['sign_element']}) during the **{pan['season']}** season.")

    st.markdown("---")

    # ── Prediction ──
    if saved:
        prob, mm, _ = predict(saved, pdt, ri["lat"], ri["lon"])

        st.markdown(f"### Prediction for {region}")

        pc1, pc2 = st.columns(2)
        with pc1:
            pc = "#2563eb" if prob > 0.5 else "#65a30d"
            st.markdown(f"""
            <div class="card-dark">
                <h2 style="color: {pc};">{prob:.0%}</h2>
                <div class="label">Rain Probability</div>
            </div>""", unsafe_allow_html=True)
            st.progress(prob)

        with pc2:
            rc = rain_color(mm)
            tag_text, tag_bg, tag_fg = rain_tag(mm)
            st.markdown(f"""
            <div class="card-dark">
                <h2 style="color: {rc};">{mm:.1f} mm</h2>
                <div class="label">Predicted Rainfall</div>
                <div class="tag" style="background:{tag_bg};color:{tag_fg};">{tag_text}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Tabs ──
        t1, t2, t3, t4 = st.tabs(["7-Day Forecast", "Model Evaluation", "Feature Analysis", "Explore Data"])

        with t1:
            rows = []
            for i in range(7):
                d = pdt + timedelta(days=i)
                p, mm_, pan_ = predict(saved, d, ri["lat"], ri["lon"])
                rows.append({
                    "Date": d.strftime("%b %d, %a"),
                    "Lunar Mansion": pan_["mansion_name"],
                    "Lunar Day": pan_["lunar_day"],
                    "Moon Sign": pan_["moon_sign_name"],
                    "Season": pan_["season"],
                    "Rain Prob": f"{p:.0%}",
                    "Rainfall (mm)": round(mm_, 1),
                })
            fdf = pd.DataFrame(rows)
            st.dataframe(fdf, use_container_width=True, hide_index=True)

            st.markdown("#### Rainfall Chart")
            st.bar_chart(
                pd.DataFrame({"Date": [r["Date"] for r in rows],
                               "mm": [r["Rainfall (mm)"] for r in rows]}).set_index("Date"),
                color="#2563eb"
            )
            st.metric("7-Day Total", f"{sum(r['Rainfall (mm)'] for r in rows):.1f} mm")

        with t2:
            m = saved["metrics"]
            ds = saved["data_stats"]

            st.markdown("### Model Performance")
            st.markdown(f"Trained on **{ds['total']:,}** real daily rainfall observations ({ds['date_range']}) across **{len(ds['regions'])}** Indian cities. Target variable is actual measured precipitation from ERA5 reanalysis.")

            st.markdown("#### Stage 1 — Rain Detection (Binary Classifier)")
            st.caption("Predicts: will it rain today (> 0.1 mm)?")
            s1, s2, s3 = st.columns(3)
            with s1: st.markdown(f'<div class="stat"><div class="num">{m["clf_accuracy"]:.1%}</div><div class="lbl">Accuracy</div></div>', unsafe_allow_html=True)
            with s2: st.markdown(f'<div class="stat"><div class="num">{m["clf_f1"]:.3f}</div><div class="lbl">F1 Score</div></div>', unsafe_allow_html=True)
            with s3: st.markdown(f'<div class="stat"><div class="num">{m["clf_auc"]:.3f}</div><div class="lbl">AUC-ROC</div></div>', unsafe_allow_html=True)

            st.markdown("#### Stage 2 — Rainfall Amount (Regressor)")
            st.caption("Predicts: how many mm (log-transformed, trained on rainy days only)")
            s1, s2, s3 = st.columns(3)
            with s1: st.markdown(f'<div class="stat"><div class="num">{m["overall_mae"]:.2f} mm</div><div class="lbl">Overall MAE</div></div>', unsafe_allow_html=True)
            with s2: st.markdown(f'<div class="stat"><div class="num">{m["rain_only_mae"]:.2f} mm</div><div class="lbl">Rainy Day MAE</div></div>', unsafe_allow_html=True)
            with s3: st.markdown(f'<div class="stat"><div class="num">{m["overall_r2"]:.4f}</div><div class="lbl">R² Score</div></div>', unsafe_allow_html=True)

            st.markdown("#### Training Data")
            d1, d2, d3, d4 = st.columns(4)
            with d1: st.metric("Total Days", f"{ds['total']:,}")
            with d2: st.metric("Rain Days", f"{ds['rain_days']:,}")
            with d3: st.metric("Dry Days", f"{ds['dry_days']:,}")
            with d4: st.metric("Max Single Day", f"{ds['max_rainfall']:.0f} mm")

        with t3:
            st.markdown("### Which Astronomical Features Matter?")
            st.markdown("Feature importance scores from the trained XGBoost models, showing which Panchanga elements have the strongest statistical association with real rainfall.")

            st.markdown("#### Rain Detection — Top 15 Features")
            st.caption("What predicts whether it rains at all")
            st.bar_chart(
                pd.DataFrame(saved["clf_importance"][:15], columns=["Feature", "Importance"]).set_index("Feature"),
                horizontal=True, color="#2563eb"
            )

            st.markdown("#### Rain Amount — Top 15 Features")
            st.caption("What predicts how much it rains (on rainy days)")
            st.bar_chart(
                pd.DataFrame(saved["reg_importance"][:15], columns=["Feature", "Importance"]).set_index("Feature"),
                horizontal=True, color="#d97706"
            )

            # Breakdown
            astro_feats = {"mansion_index","mansion_rain_tendency","mansion_is_water","mansion_is_fire","quarter",
                           "moon_sign_index","sun_sign_index","sign_rain_modifier","sign_is_water",
                           "lunar_day","lunar_day_rain","day_in_phase","yoga_index","karana_index",
                           "moon_longitude","sun_longitude","sun_moon_angle",
                           "moon_speed_proxy","ruler_is_benefic","nature_divine","nature_demonic"}
            loc_feats = {"lat","lon","day_of_year","day_sin","day_cos","month_sin","month_cos"}

            st.markdown("#### Astronomy vs Location/Calendar Contribution")
            for label, fi_key in [("Classifier", "clf_importance"), ("Regressor", "reg_importance")]:
                fi = saved[fi_key]
                a = sum(v for f,v in fi if f in astro_feats)
                l = sum(v for f,v in fi if f in loc_feats)
                t = a + l
                if t > 0:
                    st.markdown(f"**{label}:** Astronomical features = **{a/t*100:.1f}%** · Location/Calendar = **{l/t*100:.1f}%**")

        with t4:
            if df_raw is not None:
                st.markdown("### Real Rainfall Data")
                rf = st.selectbox("Filter", ["All Regions"] + sorted(df_raw["region"].unique().tolist()))
                dff = df_raw if rf == "All Regions" else df_raw[df_raw["region"] == rf]

                d1,d2,d3,d4 = st.columns(4)
                with d1: st.metric("Records", f"{len(dff):,}")
                with d2: st.metric("Rain Days", f"{(dff['rainfall_mm']>0).sum():,}")
                with d3: st.metric("Avg Daily", f"{dff['rainfall_mm'].mean():.2f} mm")
                with d4: st.metric("Max Daily", f"{dff['rainfall_mm'].max():.1f} mm")

                st.markdown("#### Monthly Average Rainfall")
                tmp = dff.copy()
                tmp["month"] = pd.to_datetime(tmp["date"]).dt.month
                st.bar_chart(tmp.groupby("month")["rainfall_mm"].mean(), color="#2563eb")

                st.markdown("#### Sample Records")
                st.dataframe(dff.head(15), use_container_width=True, hide_index=True)
    else:
        st.info("Click **Train Model** in the sidebar to build the prediction model.")


if __name__ == "__main__":
    main()
