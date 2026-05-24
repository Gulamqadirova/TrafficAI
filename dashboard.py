"""
dashboard.py  -  TPBI Platform uchun veb-interfeys (UI / Dashboard).

Bu butun loyihani bitta chiroyli brauzer oynasiga jamlaydi:
  - Pipeline'ni bir tugma bilan ishga tushirish
  - KPI ko'rsatkichlar (kartalar)
  - Grafiklar (flow, heatmap, anomaly, forecast, site comparison)
  - Anomaliyalar jadvali
  - Annotatsiyalangan video
  - Chatbot bilan savol-javob

ISHLATISH:
    pip install streamlit
    streamlit run dashboard.py

So'ng brauzer avtomatik ochiladi (http://localhost:8501).
"""
from __future__ import annotations
import json
import os

import pandas as pd
import streamlit as st

import config

# Loyiha modullari (mavjud fayllaringiz)
import synthetic_data
import video_analytics
import anomaly_detection
import predictive_analytics
import bi_dashboard
from chatbot import TrafficChatbot

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

# YouTube video sozlamalari (youtube_detect.py dagi bilan bir xil)
YOUTUBE_URL = "https://www.youtube.com/watch?v=8JCk5M_xrBs"
YT_VIDEO_FILE = os.path.join(config.BASE_DIR, "video.mp4")
YT_DURATION_SEC = 60   # jonli efirdan shuncha sekund olamiz

# -------------------------------------------------------------------------- #
st.set_page_config(
    page_title="TPBI Platform — Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Uslub (CSS) ---------------------------------------------------------- #
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.5px; }
    .main-title {
        font-size: 2.4rem; font-weight: 700; color: #1B3A5B;
        border-left: 6px solid #ED7D31; padding-left: 16px; margin-bottom: 0;
    }
    .subtitle { color: #6b7785; font-size: 1rem; margin-top: 4px; }
    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, #f4f8fc 100%);
        border: 1px solid #e3e9f0; border-radius: 14px; padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(27,58,91,0.05);
    }
    .kpi-label { color: #6b7785; font-size: 0.8rem; text-transform: uppercase;
                 letter-spacing: 0.5px; }
    .kpi-value { color: #1B3A5B; font-size: 1.9rem; font-weight: 700;
                 font-family: 'Space Grotesk', sans-serif; }
    .band-Congested { color: #C00000; } .band-Heavy { color: #ED7D31; }
    .band-Busy { color: #C9A227; } .band-Free-flow { color: #548235; }
</style>
""", unsafe_allow_html=True)

SUMMARY_PATH = os.path.join(config.OUTPUT_DIR, "run_summary.json")


# ---- Pipeline'ni keshlab ishga tushirish --------------------------------- #
def _ensure_youtube_video() -> str:
    if os.path.exists(YT_VIDEO_FILE) and os.path.getsize(YT_VIDEO_FILE) > 100_000:
        return YT_VIDEO_FILE
    return ""   # video yo'q - synthetic'ga qaytamiz (qotib qolmaslik uchun)


@st.cache_resource(show_spinner=False)
def run_full_pipeline(write_video: bool = True):
    """Butun analitika quvurini ishga tushiradi va natijalarni qaytaradi."""
    df = synthetic_data.generate_traffic_timeseries()

    # Video manbasi: avval YouTube'ni urinamiz, bo'lmasa synthetic.
    yt_video = _ensure_youtube_video()
    if yt_video:
        cv = video_analytics.analyse_video(video_path=yt_video,
                                           write_annotated=write_video)
    else:
        synthetic_data.generate_traffic_clip()
        cv = video_analytics.analyse_video(write_annotated=write_video)

    scored = anomaly_detection.detect_anomalies(df)
    anomalies = anomaly_detection.anomaly_summary(scored)
    forecasts = predictive_analytics.forecast_all_zones(df)
    kpis = bi_dashboard.compute_kpis(df, cv, anomalies)
    charts = bi_dashboard.build_all_charts(df, scored, kpis, forecasts)
    bi_dashboard.write_excel_report(df, scored, kpis, anomalies, forecasts)
    return {"df": df, "scored": scored, "cv": cv, "anomalies": anomalies,
            "forecasts": forecasts, "kpis": kpis, "charts": charts}


def kpi_card(label: str, value, band: str = ""):
    cls = f"band-{band.replace(' ', '-')}" if band else ""
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {cls}">{value}</div>
        </div>""", unsafe_allow_html=True)



# SARLAVHA
st.markdown('<div class="main-title">🚦 TPBI Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Traffic &amp; Pedestrian Business Intelligence — '
            'Real-time detection, anomaly analysis &amp; forecasting</div>',
            unsafe_allow_html=True)
st.write("")

# ---- Yon panel ------------------------------------------------------------ #
with st.sidebar:
    st.header("⚙️ Boshqaruv")
    write_video = st.checkbox("Annotatsiyalangan video yaratish", value=True)
    run_clicked = st.button("▶  Pipeline'ni ishga tushirish", type="primary",
                            use_container_width=True)
    st.divider()
    st.caption(f"Model: `{config.YOLO_WEIGHTS}`")
    st.caption(f"Klasslar: {', '.join(c['name'] for c in config.CLASSES.values())}")
    st.caption(f"Zonalar: {len(config.ZONES)} ta")

# ---- Natijalarni olish ---------------------------------------------------- #
if run_clicked:
    with st.spinner("Pipeline ishlamoqda — detection, anomaly, forecast..."):
        st.session_state["results"] = run_full_pipeline(write_video)
    st.success("Pipeline muvaffaqiyatli tugadi!")

results = st.session_state.get("results")

if results is None:
    st.info("👈 Boshlash uchun chap paneldagi **Pipeline'ni ishga tushirish** "
            "tugmasini bosing. (Birinchi marta bir oz vaqt olishi mumkin.)")
    st.stop()

kpis = results["kpis"]
df = results["df"]
charts = results["charts"]
anomalies = results["anomalies"]
forecasts = results["forecasts"]


# KPI KARTALAR
st.subheader("📊 Asosiy ko'rsatkichlar (KPI)")
c1, c2, c3, c4 = st.columns(4)
with c1: kpi_card("Jami mashinalar", f"{kpis['total_cars']:,}")
with c2: kpi_card("Jami odamlar", f"{kpis['total_people']:,}")
with c3: kpi_card("Peak yo'l bandligi", f"{kpis['peak_road_occupancy_pct']}%",
                  kpis["peak_band"])
with c4: kpi_card("Anomaliyalar", kpis["anomalies_detected"])

c5, c6, c7, c8 = st.columns(4)
with c5: kpi_card("Kuzatuvlar", f"{kpis['total_observations']:,}")
with c6: kpi_card("Eng band (mashina)", kpis["busiest_car_site"])
with c7: kpi_card("Eng band (odam)", kpis["busiest_ped_site"])
with c8: kpi_card("CV backend", kpis["cv_backend"])

st.divider()


# TABLAR
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Grafiklar", "⚠️ Anomaliyalar", "🔮 Bashorat", "🎬 Video", "💬 Chatbot"])

# --- Grafiklar ---
with tab1:
    g1, g2 = st.columns(2)
    chart_titles = {
        "flow_trends": "Mashina vs Piyoda oqimi",
        "site_comparison": "Saytlar bo'yicha taqqoslash",
        "road_heatmap": "Yo'l bandligi (heatmap)",
        "anomaly_timeline": "Anomaliya vaqt chizig'i",
    }
    cols = [g1, g2, g1, g2]
    for (key, title), col in zip(chart_titles.items(), cols):
        path = charts.get(key)
        if path and os.path.exists(path):
            with col:
                st.markdown(f"**{title}**")
                st.image(path, use_container_width=True)

# --- Anomaliyalar ---
with tab2:
    st.markdown(f"**{len(anomalies)} ta anomaliya aniqlandi**")
    if anomalies.empty:
        st.success("Anomaliya topilmadi — trafik normal.")
    else:
        st.dataframe(anomalies, use_container_width=True, height=420)

# --- Bashorat ---
with tab3:
    fpath = charts.get("forecast")
    if fpath and os.path.exists(fpath):
        st.image(fpath, use_container_width=True)
    st.markdown("**Zonalar bo'yicha bashorat**")
    frows = []
    for z, f in forecasts.items():
        frows.append({
            "Zona": z,
            "Mashina (peak)": f.get("car_predicted_peak"),
            "Odam (peak)": f.get("person_predicted_peak"),
            "Yo'l sig'imi": f.get("road_capacity"),
            "Tirbandlik?": "Ha" if f.get("congestion_predicted") else "Yo'q",
            "MAE (mashina)": f.get("car_mae"),
        })
    st.dataframe(pd.DataFrame(frows), use_container_width=True)

# --- Video ---
with tab4:
    vpath = config.ANNOTATED_VIDEO
    if os.path.exists(vpath):
        st.markdown("**Annotatsiyalangan video** (aniqlangan obyektlar bilan)")
        with open(vpath, "rb") as vf:
            st.video(vf.read())
    else:
        st.info("Video hali yaratilmagan. Chap panelda 'video yaratish' ni "
                "belgilab, pipeline'ni qayta ishga tushiring.")

# --- Chatbot ---
with tab5:
    st.markdown("**Trafik bo'yicha savol bering** (ingliz tilida)")
    st.caption("Masalan: *Was there any congestion today?*, "
               "*What is the busiest zone?*, *Is the main road safe?*")

    if "bot" not in st.session_state:
        st.session_state["bot"] = TrafficChatbot(
            df, kpis, anomalies, forecasts)
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for role, msg in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.write(msg)

    user_q = st.chat_input("Savolingizni yozing...")
    if user_q:
        st.session_state["chat_history"].append(("user", user_q))
        with st.chat_message("user"):
            st.write(user_q)
        answer = st.session_state["bot"].ask(user_q)
        st.session_state["chat_history"].append(("assistant", answer))
        with st.chat_message("assistant"):
            st.write(answer)

# ---- Pastki qism ---------------------------------------------------------- #
st.divider()
xlsx = os.path.join(config.OUTPUT_DIR, "Traffic_BI_Report.xlsx")
if os.path.exists(xlsx):
    with open(xlsx, "rb") as f:
        st.download_button("📥 Excel hisobotni yuklab olish", f.read(),
                           file_name="Traffic_BI_Report.xlsx",
                           use_container_width=True)