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
    import metrics, governance
    timer = metrics.PerformanceTimer()

    df = synthetic_data.generate_traffic_timeseries()

    # --- Ma'lumot sifati tekshiruvi va tozalash (Task 4) ---
    quality = governance.validate_dataframe(df)
    df, clean_report = governance.clean_dataframe(df)

    # Video manbasi: avval YouTube'ni urinamiz, bo'lmasa synthetic.
    yt_video = _ensure_youtube_video()
    with timer.stage("detection"):
        if yt_video:
            cv = video_analytics.analyse_video(video_path=yt_video,
                                               write_annotated=write_video)
        else:
            synthetic_data.generate_traffic_clip()
            cv = video_analytics.analyse_video(write_annotated=write_video)
    timer.record_frames(cv.get("frames_processed", 0))

    with timer.stage("analytics"):
        scored = anomaly_detection.detect_anomalies(df)
        anomalies = anomaly_detection.anomaly_summary(scored)
        forecasts = predictive_analytics.forecast_all_zones(df)
        kpis = bi_dashboard.compute_kpis(df, cv, anomalies)
        charts = bi_dashboard.build_all_charts(df, scored, kpis, forecasts)
        bi_dashboard.write_excel_report(df, scored, kpis, anomalies, forecasts)

    # --- Hammasini DATABASE'ga saqlash (chatbot shu yerdan o'qiydi) ---
    try:
        import database
        database.init_db()
        database.save_dataframe(df)
        database.save_anomalies(anomalies)
        database.save_forecasts(forecasts)
        database.save_kpis(kpis)
    except Exception as e:
        print(f"Database'ga yozishda ogohlantirish: {e}")

    perf = timer.report()
    return {"df": df, "scored": scored, "cv": cv, "anomalies": anomalies,
            "forecasts": forecasts, "kpis": kpis, "charts": charts,
            "performance": perf, "data_quality": quality,
            "clean_report": clean_report}


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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["📈 Grafiklar", "⚠️ Anomaliyalar", "🔮 Bashorat", "🎬 Video", "💬 Chatbot",
     "📐 Baholash", "🔒 Boshqaruv & Maxfiylik"])

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
    st.markdown("**Zonalar bo'yicha bashorat** (95% ishonch oralig'i bilan)")
    frows = []
    for z, f in forecasts.items():
        car_lo = f.get("car_forecast_lower", [])
        car_hi = f.get("car_forecast_upper", [])
        peak = f.get("car_predicted_peak")
        ci = (f"{min(car_lo)}–{max(car_hi)}" if car_lo and car_hi else "—")
        frows.append({
            "Zona": z,
            "Mashina (peak)": peak,
            "Ishonch oralig'i (95%)": ci,
            "Odam (peak)": f.get("person_predicted_peak"),
            "Yo'l sig'imi": f.get("road_capacity"),
            "Tirbandlik?": "Ha" if f.get("congestion_predicted") else "Yo'q",
            "MAE (mashina)": f.get("car_mae"),
        })
    st.dataframe(pd.DataFrame(frows), use_container_width=True)
    st.caption("Ishonch oralig'i = bashorat ± 1.96σ (qoldiqlar standart "
               "og'ishiga asoslangan). Kengroq oraliq — kattaroq noaniqlik.")

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
    if "bot" not in st.session_state:
        st.session_state["bot"] = TrafficChatbot(df, kpis, anomalies, forecasts)
    bot = st.session_state["bot"]

    # Rejim ko'rsatkichi (LLM yoki SQL)
    if bot.use_llm:
        st.success("🤖 **Claude AI (LLM) rejimi** — tabiiy til, kontekstli "
                   "suhbat, database'dan o'qiydi.")
    else:
        st.warning(
            "⚠️ **SQL rejimi** (zaxira) — Claude API topilmadi. "
            "LLM yoqish uchun: Streamlit **Settings → Secrets** ga "
            "`ANTHROPIC_API_KEY = \"sk-ant-...\"` qo'shing. "
            "Yoki terminalda: `export ANTHROPIC_API_KEY=sk-ant-...` "
            "va dashboardni qayta ishga tushiring.")

    st.markdown("**Trafik bo'yicha savol bering**")
    st.caption(
        "Masalan: *Was there any congestion today?*, "
        "*Which zone has the most cars?*, *Is the main road safe?*, "
        "*Compare all zones*, *Forecast for JCT_MainRoad?*")

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
        answer = bot.ask(user_q)
        st.session_state["chat_history"].append(("assistant", answer))
        with st.chat_message("assistant"):
            st.write(answer)

    with st.expander("ℹ️ Chatbot arxitekturasi haqida (examiner uchun)"):
        st.markdown("""
**Chatbot dizayni (BTEC Task 1 — AI Chatbot):**

| Qatlam | Texnologiya | Maqsad |
|--------|-------------|--------|
| **Birlamchi** | Claude API (`claude-sonnet-4`) | Tabiiy til, kontekstli suhbat |
| **Zaxira** | SQL + Intent matching | API'siz ishlash kafolati |
| **Ma'lumot** | SQLite (`traffic.db`) | Haqiqiy DB raqamlari |

**RAG-lite pattern**: Har so'rovda tizim `traffic.db`'dan joriy holat 
ma'lumotini olib, LLM'ga kontekst sifatida beradi. Shu tarzda LLM 
"hallucinate" qilmaydi — faqat haqiqiy bazaviy raqamlarni tushuntiradi.

**Suhbat xotirasi**: LLM rejimida oxirgi 10 xabar saqlanadi, 
shuning uchun "avvalgi savolim haqida ko'proq" kabi davomli suhbat 
mumkin.
""")

# --- Baholash (metrics / performance / data quality) ---
with tab6:
    st.markdown("### Tizim baholash (System Evaluation)")
    st.caption("BTEC mezonlari: C.M3 (AI aniqligi), Task 4 (unumdorlik, sifat)")

    perf = results.get("performance", {})
    quality = results.get("data_quality", {})
    clean_report = results.get("clean_report", {})
    cv = results.get("cv", {})

    e1, e2, e3 = st.columns(3)
    with e1: kpi_card("FPS (kadr/sek)", perf.get("fps", "—"))
    with e2: kpi_card("Qayta ishlangan kadr", cv.get("frames_processed", "—"))
    with e3: kpi_card("Jami vaqt (s)", perf.get("total_runtime_sec", "—"))

    st.markdown("**Bosqichlar bo'yicha vaqt (pipeline timing)**")
    timings = perf.get("stage_timings_sec", {})
    if timings:
        st.dataframe(pd.DataFrame(
            [{"Bosqich": k, "Vaqt (s)": v} for k, v in timings.items()]),
            use_container_width=True)

    st.markdown("**Ma'lumot sifati (Data Quality)**")
    dq_col1, dq_col2 = st.columns(2)
    with dq_col1:
        st.metric("Qatorlar", quality.get("rows", "—"))
        st.metric("Bo'sh (NaN) qiymatlar", quality.get("missing_values", "—"))
    with dq_col2:
        st.metric("Takroriy qatorlar", quality.get("duplicate_rows", "—"))
        st.metric("Holat", "✅ Toza" if quality.get("valid") else "⚠️ Tuzatildi")
    if clean_report:
        st.caption(f"Tozalash: {clean_report}")

    st.markdown("**AI aniqligi (Detection accuracy)**")
    # evaluate_accuracy.py natijasini o'qiymiz
    acc_path = os.path.join(config.OUTPUT_DIR, "accuracy_report.json")
    if os.path.exists(acc_path):
        import json as _json
        with open(acc_path) as _f:
            acc_data = _json.load(_f)
        mode = acc_data.get("evaluation_mode", "unknown")
        a1, a2, a3 = st.columns(3)
        if mode == "full_iou":
            with a1: st.metric("Precision", acc_data.get("precision", "—"))
            with a2: st.metric("Recall",    acc_data.get("recall", "—"))
            with a3: st.metric("F1-score",  acc_data.get("f1_score", "—"))
            st.caption(
                f"IoU≥0.5, {acc_data.get('images_evaluated')} rasm, "
                f"{acc_data.get('ground_truth_count')} GT annotatsiya. "
                "Manba: `evaluate_accuracy.py`")
        else:
            with a1: st.metric("Baholangan rasmlar",
                               acc_data.get("images_evaluated", "—"))
            with a2: st.metric("Aniqlangan obyektlar",
                               acc_data.get("count", "—"))
            with a3: st.metric("O'rtacha ishonch",
                               acc_data.get("mean_confidence", "—"))
            st.warning(
                "**Baholash rejimi: confidence statistikasi** — "
                "ground-truth labellar (`datasets/train/labels/*.txt`) bo'sh. "
                "To'liq precision/recall/F1 uchun YOLO format labellar kerak. "
                "Baholash tizimi (`metrics.detection_accuracy`, IoU≥0.5) "
                "to'liq joriy etilgan — `evaluate_accuracy.py` ni ishga tushiring.")
    else:
        st.info("Baholash natijasi yo'q. "
                "`python evaluate_accuracy.py` ni ishga tushiring.")

# --- Boshqaruv & Maxfiylik + Power BI + Scalability ---
with tab7:
    import governance
    gov = governance.governance_summary()
    st.markdown("### Boshqaruv, axloq, maxfiylik va kengayish")
    st.caption("BTEC mezonlari: C.P5, Task 2 (Power BI), A.P1, B.M2 (scalability)")

    inner_tab1, inner_tab2, inner_tab3 = st.tabs(
        ["🔒 Maxfiylik & Axloq", "📊 Power BI integratsiya", "⚙️ Masshtablilik"])

    with inner_tab1:
        st.markdown(gov["privacy"])
        st.divider()
        st.markdown(gov["ethics"])
        st.divider()
        st.markdown(gov["compliance"])

    with inner_tab2:
        st.markdown("#### Power BI integratsiya (Task 2)")
        st.info(
            "Ushbu tizim **Streamlit + matplotlib** ishlatadi (bepul, Python "
            "bilan to'liq integratsiya, portativ). Ma'lumotlar Power BI "
            "iste'mol qiladigan formatlarda eksport qilinadi.")

        st.markdown("**Power BI'ga ulanish — 2 yo'l:**")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**1. SQLite orqali (tavsiya)**")
            st.code("""
# Power BI Desktop:
# Get Data → More → Database
# → SQLite Database
# → Fayl tanlang: traffic.db
# → Jadvallar: detections,
#   anomalies, forecasts
            """, language="text")

        with col_b:
            st.markdown("**2. Excel orqali**")
            st.code("""
# Power BI Desktop:
# Get Data → Excel workbook
# → Traffic_BI_Report.xlsx
# → Varaqlar: Sheet1, ...
# → Load → Vizualizatsiya
            """, language="text")

        st.markdown("**Tayyor eksport fayllar:**")
        db_path = os.path.join(config.BASE_DIR, "traffic.db")
        xlsx_path = os.path.join(config.OUTPUT_DIR, "Traffic_BI_Report.xlsx")
        c1, c2 = st.columns(2)
        with c1:
            exists = "✅ Tayyor" if os.path.exists(db_path) else "⏳ Kerak emas"
            st.metric("traffic.db (SQLite)", exists)
        with c2:
            exists = "✅ Tayyor" if os.path.exists(xlsx_path) else "⏳ Pipeline kerak"
            st.metric("Traffic_BI_Report.xlsx", exists)

        st.markdown("""
**Arxitektura qaroriga asoslash:**
Namoyish uchun Streamlit tanlandi, chunki:
- Python loyihasi bilan to'g'ridan-to'g'ri integratsiya (alohida litsenziya kerak emas)
- Ma'lumot modeli Power BI bilan mos (SQLite, Excel eksport)
- Akademik muhitda tezkor prototiplash uchun optimallar
""")

    with inner_tab3:
        st.markdown("#### Masshtablilik arxitekturasi (A.P1, B.M2)")

        st.markdown("""
**Joriy dizayn — ko'p kamera uchun tayyor:**

Har bir detection qatorida `zone` va `source` ustunlari mavjud. Yangi kamera 
qo'shish = yangi `zone` ta'riflash, kod o'zgarmaydi.
""")

        st.code("""
# Ko'p kamera — paralel ishlov (threading dizayni)
import threading, queue

detection_queue = queue.Queue(maxsize=500)

def camera_worker(zone_id, video_source):
    \"\"\"Har bir kamera — alohida thread\"\"\"
    detector = MultiClassDetector()
    cap = cv2.VideoCapture(video_source)
    while True:
        ok, frame = cap.read()
        if not ok: break
        detections = detector.detect(frame)
        detection_queue.put({
            "zone": zone_id,
            "detections": detections,
            "timestamp": datetime.now()
        })

def db_writer_worker():
    \"\"\"Bitta thread — bazaga yozish (thread-safe)\"\"\"
    while True:
        item = detection_queue.get()
        database.log_realtime_detection(
            source=item["zone"], ...)

# 4 ta kamera — 4 ta thread + 1 yozuvchi
cameras = {"JCT_MainRoad": 0, "GATE_Campus": 1,
           "HUB_StationFwd": 2, "MALL_CarParkIn": 3}
threads = [threading.Thread(target=camera_worker,
           args=(z, s)) for z, s in cameras.items()]
writer = threading.Thread(target=db_writer_worker)
""", language="python")

        st.markdown("""
**Kengayish strategiyasi:**

| Daraja | Yechim | Sabab |
|--------|--------|-------|
| **4 kamera** | Threading + Queue | Bitta mashinada, GIL muammo yo'q (I/O bound) |
| **10+ kamera** | Multiprocessing | CPU og'ir detection uchun alohida jarayonlar |
| **100+ kamera** | Distributed (Kafka + microservices) | Tarmoqli ishlov, yuqori mavjudlik |

**SQLite → PostgreSQL ko'chirish**: `database.py` da faqat ulanish qatorini 
o'zgartirish kifoya — barcha SQL so'rovlar bir xil qoladi. 
Bu arxitekturaviy moslashuvchanlikning asosiy afzalligi.

**Joriy cheklov**: namoyish maqsadida bitta jarayonli (single-threaded) 
ishlov qo'llanildi. Ko'p kamerada threading/multiprocessing qo'shimcha 
o'zgarishlar bilan amalga oshiriladi (yuqoridagi dizayn ko'rsatilganidek).
""")
        st.divider()
        st.markdown(gov["compliance"])

# ---- Pastki qism ---------------------------------------------------------- #
st.divider()
xlsx = os.path.join(config.OUTPUT_DIR, "Traffic_BI_Report.xlsx")
if os.path.exists(xlsx):
    with open(xlsx, "rb") as f:
        st.download_button("📥 Excel hisobotni yuklab olish", f.read(),
                           file_name="Traffic_BI_Report.xlsx",
                           use_container_width=True)