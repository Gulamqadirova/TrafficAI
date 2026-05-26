from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime

import pandas as pd

import config
import database

# Eksport papkasi
EXPORT_DIR = Path(config.OUTPUT_DIR) / "powerbi"


def export_all() -> dict[str, int]:
    """Barcha jadvallarni CSV sifatida eksport qiladi."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    database.init_db()

    exported = {}

    # 1. Detections — asosiy ma'lumot
    df = database.query_df("SELECT * FROM detections ORDER BY timestamp, zone")
    path = EXPORT_DIR / "detections.csv"
    df.to_csv(path, index=False)
    exported["detections.csv"] = len(df)
    print(f"  detections.csv    : {len(df)} qator")

    # 2. Anomalies
    an = database.query_df("SELECT * FROM anomalies ORDER BY timestamp")
    path = EXPORT_DIR / "anomalies.csv"
    an.to_csv(path, index=False)
    exported["anomalies.csv"] = len(an)
    print(f"  anomalies.csv     : {len(an)} qator")

    # 3. Forecasts — bashorat (ishonch oralig'i bilan)
    fc = database.query_df(
        "SELECT * FROM forecasts ORDER BY zone, horizon_step")
    path = EXPORT_DIR / "forecasts.csv"
    fc.to_csv(path, index=False)
    exported["forecasts.csv"] = len(fc)
    print(f"  forecasts.csv     : {len(fc)} qator")

    # 4. KPI snapshots
    kpi = database.query_df(
        "SELECT * FROM kpi_snapshots ORDER BY created_at")
    path = EXPORT_DIR / "kpi_snapshots.csv"
    kpi.to_csv(path, index=False)
    exported["kpi_snapshots.csv"] = len(kpi)
    print(f"  kpi_snapshots.csv : {len(kpi)} qator")

    # 5. Real-time detection log
    rt = database.query_df(
        "SELECT * FROM detection_log ORDER BY timestamp")
    path = EXPORT_DIR / "detection_log.csv"
    rt.to_csv(path, index=False)
    exported["detection_log.csv"] = len(rt)
    print(f"  detection_log.csv : {len(rt)} qator")

    # 6. Pivot view — Power BI dashboard uchun qulay
    if not df.empty:
        pivot = df.groupby(["zone", "timestamp"]).agg(
            car_count=("car_count", "sum"),
            person_count=("person_count", "sum"),
            road_occupancy=("road_occupancy", "max"),
        ).reset_index()
        path = EXPORT_DIR / "zone_timeseries.csv"
        pivot.to_csv(path, index=False)
        exported["zone_timeseries.csv"] = len(pivot)
        print(f"  zone_timeseries.csv: {len(pivot)} qator")

    _write_readme(exported)
    print(f"\nEksport tugadi → {EXPORT_DIR}/")
    return exported


def _write_readme(exported: dict[str, int]) -> None:
    """Power BI ulanish yo'riqnomasini yozadi."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Power BI — TPBI Ma'lumotlari Ulanish Yo'riqnomasi",
        f"\n*Yaratilgan: {now}*\n",
        "## Eksport qilingan fayllar\n",
    ]
    for fname, n in exported.items():
        lines.append(f"| `{fname}` | {n} qator |")

    lines += [
        "\n## Power BI Desktop — CSV orqali ulanish (tavsiya)\n",
        "```",
        "1. Power BI Desktop oching",
        "2. Home → Get Data → Text/CSV",
        "3. Quyidagi fayllarni navbatma-navbat yuklang:",
        "     outputs/powerbi/detections.csv       ← asosiy ma'lumot",
        "     outputs/powerbi/anomalies.csv         ← anomaliyalar",
        "     outputs/powerbi/forecasts.csv         ← bashorat",
        "     outputs/powerbi/zone_timeseries.csv   ← pivot jadval (dashboard uchun)",
        "4. Transform Data → jadvallarni 'zone' ustuni orqali bog'lang",
        "5. Vizualizatsiya: Bar chart (zone vs car_count), Line chart (timeseries),",
        "   Map (agar koordinatalar qo'shilsa), Card (KPI)",
        "```\n",
        "## Power BI Desktop — SQLite orqali to'g'ridan-to'g'ri ulanish\n",
        "```",
        "1. Home → Get Data → More → Database → SQLite Database",
        "2. Fayl tanlang: <loyiha>/code/traffic.db",
        "3. Jadvallar: detections, anomalies, forecasts, kpi_snapshots",
        "4. Load → Power Query → kerakli ustunlarni tanlang",
        "```\n",
        "## Jadvallar tuzilishi\n",
        "### detections",
        "| Ustun | Tur | Tavsif |",
        "|-------|-----|--------|",
        "| timestamp | TEXT | Vaqt tamg'asi (5 daqiqalik interval) |",
        "| zone | TEXT | Monitoring zonasi (JCT_MainRoad, GATE_Campus, ...) |",
        "| car_count | INTEGER | Mashina soni |",
        "| person_count | INTEGER | Odam soni |",
        "| road_occupancy | REAL | Yo'l bandligi (0.0–1.0) |",
        "| ped_occupancy | REAL | Piyoda bandligi (0.0–1.0) |\n",
        "### forecasts",
        "| Ustun | Tur | Tavsif |",
        "|-------|-----|--------|",
        "| zone | TEXT | Zona |",
        "| horizon_step | INTEGER | Bashorat qadami (0=keyingi 5 daqiqa) |",
        "| predicted | REAL | Bashorat qilingan qiymat |",
        "| lower_bound | REAL | 95% ishonch oralig'i — quyi chegara |",
        "| upper_bound | REAL | 95% ishonch oralig'i — yuqori chegara |\n",
        "## Arxitektura asoslash\n",
        "Ushbu loyiha Streamlit + matplotlib ishlatadi, chunki:",
        "- Python loyiha bilan to'liq integratsiya (alohida litsenziya kerak emas)",
        "- Akademik muhitda tezkor prototiplash uchun optimal",
        "- Ma'lumot modeli Power BI bilan to'liq mos (SQLite, CSV, Excel)",
        "- Bir xil ma'lumot ustida Power BI hisobotlari qurish mumkin\n",
        "Power BI uchun tayyor eksport har `python powerbi_export.py` da yangilanadi.",
    ]
    (EXPORT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  README.md         : ulanish yo'riqnomasi yozildi")


if __name__ == "__main__":
    print("Power BI eksport boshlanyapti...")
    result = export_all()
    print(f"\nJami {len(result)} fayl eksport qilindi: {EXPORT_DIR}/")