from __future__ import annotations
import json, os, time, argparse
import config, synthetic_data, video_analytics
import anomaly_detection, predictive_analytics, bi_dashboard
import database, governance, metrics
from chatbot import TrafficChatbot


def run(write_annotated: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    log = print if verbose else (lambda *a: None)
    timer = metrics.PerformanceTimer()

    # ------------------------------------------------------------------ #
    log("\n[1/6] INGEST  - ma'lumot yuklanmoqda ...")
    with timer.stage("ingest"):
        synthetic_data.generate_traffic_clip()
        df = synthetic_data.generate_traffic_timeseries()

    # Ma'lumot sifati tekshiruvi va tozalash (Task 4, governance.py)
    quality = governance.validate_dataframe(df)
    df, clean_rep = governance.clean_dataframe(df)
    log(f"        {len(df)} qator, {df['zone'].nunique()} zona | "
        f"sifat: {'✓ toza' if quality['valid'] else '⚠ tuzatildi'} "
        f"({quality.get('missing_values', 0)} NaN, "
        f"{quality.get('duplicate_rows', 0)} takror)")

    # ------------------------------------------------------------------ #
    log("[2/6] PERCEIVE- aniqlash + kuzatuv ...")
    with timer.stage("detection"):
        cv = video_analytics.analyse_video(write_annotated=write_annotated)
    timer.record_frames(cv.get("frames_processed", 0))
    for name, st in cv.get("classes", {}).items():
        log(f"        {name:>7}: peak={st['peak_count']} "
            f"tracks={st['unique_tracks']} conf={st['mean_confidence']}")

    # ------------------------------------------------------------------ #
    log("[3/6] ANALYSE - anomaliyalar + bashorat ...")
    with timer.stage("analytics"):
        scored   = anomaly_detection.detect_anomalies(df)
        anomalies = anomaly_detection.anomaly_summary(scored)
        forecasts = predictive_analytics.forecast_all_zones(df)
    log(f"        anomaliyalar: {len(anomalies)} | "
        f"bashorat zonalar: {len(forecasts)}")

    # ------------------------------------------------------------------ #
    log("[4/6] PRESENT - KPI, grafiklar, Excel ...")
    with timer.stage("reporting"):
        kpis   = bi_dashboard.compute_kpis(df, cv, anomalies)
        charts = bi_dashboard.build_all_charts(df, scored, kpis, forecasts)
        xlsx   = bi_dashboard.write_excel_report(
            df, scored, kpis, anomalies, forecasts)
    log(f"        {len(charts)} grafik | hisobot: {os.path.basename(xlsx)}")

    # ------------------------------------------------------------------ #
    log("[5/6] STORE   - database'ga saqlash ...")
    with timer.stage("db_write"):
        database.init_db()
        n_det  = database.save_dataframe(df)
        n_an   = database.save_anomalies(anomalies)
        n_fc   = database.save_forecasts(forecasts)
        n_kpi  = database.save_kpis(kpis)
    log(f"        detections={n_det} anomalies={n_an} "
        f"forecasts={n_fc} kpi_snapshots={n_kpi}")
    log(f"        DB jadvallari: {database.table_summary()}")

    # ------------------------------------------------------------------ #
    log("[6/6] ASSIST  - chatbot demo ...")
    bot = TrafficChatbot(df, kpis, anomalies, forecasts)
    chatbot_demo = []
    for q in ["Was there any congestion today?",
              "Which zone is busiest?",
              "Compare zones",
              "Is the main road safe?"]:
        answer = bot.ask(q)
        chatbot_demo.append({"q": q, "a": answer})
        log(f"        Q: {q}\n           A: {answer}")

    # ------------------------------------------------------------------ #
    # Unumdorlik hisoboti (Task 4, metrics.py)
    perf = timer.report()
    log(f"\n{metrics.evaluation_report(cv_summary=cv, perf=perf)}")

    # ------------------------------------------------------------------ #
    # run_summary.json — ma'lumot sifati, unumdorlik, DB holati
    summary = {
        "kpis": kpis,
        "anomaly_count": int(len(anomalies)),
        "cv": cv,
        "charts": charts,
        "excel_report": xlsx,
        "annotated_video": cv.get("annotated_video"),
        "runtime_seconds": round(time.time() - t0, 2),
        # --- Yangi bo'limlar ---
        "performance": perf,
        "data_quality": quality,
        "data_clean_report": clean_rep,
        "database_summary": database.table_summary(),
        "chatbot_demo": chatbot_demo,
    }
    sp = os.path.join(config.OUTPUT_DIR, "run_summary.json")
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log(f"\nTugadi: {summary['runtime_seconds']}s | "
        f"FPS: {perf['fps']} | Xulosa: {os.path.basename(sp)}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TPBI pipeline")
    ap.add_argument("--no-video", action="store_true",
                    help="Annotatsiyalangan video yaratmaslik (tezroq)")
    ap.add_argument("--quiet", action="store_true",
                    help="Minimal chiqish")
    args = ap.parse_args()
    run(write_annotated=not args.no_video, verbose=not args.quiet)