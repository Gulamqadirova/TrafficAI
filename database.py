from __future__ import annotations
import os
import sqlite3
from datetime import datetime
from typing import Any

import pandas as pd

import config

DB_PATH = os.path.join(config.BASE_DIR, "traffic.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Barcha jadvallarni yaratadi (agar mavjud bo'lmasa)."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            zone          TEXT    NOT NULL,
            site          TEXT,
            car_count     INTEGER DEFAULT 0,
            person_count  INTEGER DEFAULT 0,
            road_occupancy  REAL  DEFAULT 0,
            ped_occupancy   REAL  DEFAULT 0
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS anomalies (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp      TEXT,
            zone           TEXT,
            car_count      INTEGER,
            person_count   INTEGER,
            road_occupancy REAL,
            anomaly_reason TEXT
        )
    """)

    # Bashorat natijalari - ishonch oralig'i (confidence interval) bilan
    cur.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at    TEXT,
            zone          TEXT,
            metric        TEXT,
            horizon_step  INTEGER,
            predicted     REAL,
            lower_bound   REAL,
            upper_bound   REAL,
            mae           REAL
        )
    """)

    # KPI snapshot - har pipeline ishlaganda saqlanadi
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kpi_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT,
            kpi_name    TEXT,
            kpi_value   TEXT
        )
    """)

    # Real-time detection log - youtube_detect.py shu yerga yozadi
    cur.execute("""
        CREATE TABLE IF NOT EXISTS detection_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            source      TEXT,
            frame_index INTEGER,
            person_count INTEGER,
            car_count    INTEGER
        )
    """)

    conn.commit()
    conn.close()


#  Yozish funksiyalari
def save_dataframe(df: pd.DataFrame, table: str = "detections") -> int:
    init_db()
    conn = get_connection()
    if table == "detections":
        cols = ["timestamp", "zone", "site", "car_count", "person_count",
                "road_occupancy", "ped_occupancy"]
    else:
        cols = list(df.columns)
    keep = [c for c in cols if c in df.columns]
    sub = df[keep].copy()
    sub["timestamp"] = sub["timestamp"].astype(str)
    conn.execute(f"DELETE FROM {table}")
    sub.to_sql(table, conn, if_exists="append", index=False)
    conn.commit()
    n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.close()
    return n


def save_anomalies(anomalies: pd.DataFrame) -> int:
    init_db()
    conn = get_connection()
    conn.execute("DELETE FROM anomalies")
    if not anomalies.empty:
        cols = ["timestamp", "zone", "car_count", "person_count",
                "road_occupancy", "anomaly_reason"]
        keep = [c for c in cols if c in anomalies.columns]
        sub = anomalies[keep].copy()
        sub["timestamp"] = sub["timestamp"].astype(str)
        sub.to_sql("anomalies", conn, if_exists="append", index=False)
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM anomalies").fetchone()[0]
    conn.close()
    return n


def save_forecasts(forecasts: dict) -> int:
    """forecast_all_zones natijasini bazaga yozadi (ishonch oralig'i bilan)."""
    init_db()
    conn = get_connection()
    conn.execute("DELETE FROM forecasts")
    now = datetime.now().isoformat(timespec="seconds")
    rows = []
    for zone, f in (forecasts or {}).items():
        car = f.get("car_forecast", [])
        car_lo = f.get("car_forecast_lower", [None] * len(car))
        car_hi = f.get("car_forecast_upper", [None] * len(car))
        mae = f.get("car_mae")
        for i, val in enumerate(car):
            lo = car_lo[i] if i < len(car_lo) else None
            hi = car_hi[i] if i < len(car_hi) else None
            rows.append((now, zone, "car_count", i, float(val), lo, hi, mae))
    if rows:
        conn.executemany(
            "INSERT INTO forecasts (created_at, zone, metric, horizon_step, "
            "predicted, lower_bound, upper_bound, mae) VALUES (?,?,?,?,?,?,?,?)",
            rows)
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
    conn.close()
    return n


def save_kpis(kpis: dict) -> int:
    """KPI lug'atini snapshot sifatida saqlaydi."""
    init_db()
    conn = get_connection()
    now = datetime.now().isoformat(timespec="seconds")
    rows = [(now, str(k), str(v)) for k, v in (kpis or {}).items()]
    if rows:
        conn.executemany(
            "INSERT INTO kpi_snapshots (created_at, kpi_name, kpi_value) "
            "VALUES (?,?,?)", rows)
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM kpi_snapshots").fetchone()[0]
    conn.close()
    return n


def log_realtime_detection(source: str, frame_index: int,
                           person_count: int, car_count: int) -> None:
    """Real-time detection hodisasini yozadi (youtube_detect.py uchun)."""
    init_db()
    conn = get_connection()
    conn.execute(
        "INSERT INTO detection_log (timestamp, source, frame_index, "
        "person_count, car_count) VALUES (?,?,?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), source,
         frame_index, person_count, car_count))
    conn.commit()
    conn.close()



#  O'qish funksiyalari
def query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def table_summary() -> dict:
    init_db()
    conn = get_connection()
    out = {}
    for t in ("detections", "anomalies", "forecasts", "kpi_snapshots",
              "detection_log"):
        try:
            out[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception:
            out[t] = 0
    conn.close()
    return out


if __name__ == "__main__":
    import synthetic_data
    init_db()
    df = synthetic_data.generate_traffic_timeseries()
    n = save_dataframe(df)
    print(f"Database'ga {n} qator yozildi -> {DB_PATH}")
    print("Jadvallar:", table_summary())
    rows = query("SELECT zone, SUM(car_count) AS total_cars "
                 "FROM detections GROUP BY zone ORDER BY total_cars DESC")
    print("\nZona bo'yicha jami mashinalar:")
    for r in rows:
        print(f"  {r['zone']}: {r['total_cars']}")