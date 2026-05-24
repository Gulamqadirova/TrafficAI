from __future__ import annotations
import json, os, time
import config, synthetic_data, video_analytics
import anomaly_detection, predictive_analytics, bi_dashboard
from chatbot import TrafficChatbot


def run(write_annotated: bool = True, verbose: bool = True) -> dict:
    t0 = time.time(); log = print if verbose else (lambda *a: None)

    log("\n[1/5] INGEST  - generating traffic clip + history ...")
    synthetic_data.generate_traffic_clip()
    df = synthetic_data.generate_traffic_timeseries()
    log(f"        video + {len(df)} rows across {df['zone'].nunique()} zones")

    log("[2/5] PERCEIVE- multi-class detection + tracking ...")
    cv = video_analytics.analyse_video(write_annotated=write_annotated)
    for name, st in cv["classes"].items():
        log(f"        {name:>7}: peak={st['peak_count']} tracks={st['unique_tracks']} "
            f"crossings={st['total_crossings']} conf={st['mean_confidence']}")

    log("[3/5] ANALYSE - anomalies + forecasting ...")
    scored = anomaly_detection.detect_anomalies(df)
    anomalies = anomaly_detection.anomaly_summary(scored)
    forecasts = predictive_analytics.forecast_all_zones(df)
    log(f"        anomalies: {len(anomalies)} | zones forecast: {len(forecasts)}")

    log("[4/5] PRESENT - KPIs, charts, Excel report ...")
    kpis = bi_dashboard.compute_kpis(df, cv, anomalies)
    charts = bi_dashboard.build_all_charts(df, scored, kpis, forecasts)
    xlsx = bi_dashboard.write_excel_report(df, scored, kpis, anomalies, forecasts)
    log(f"        {len(charts)} charts | report: {os.path.basename(xlsx)}")

    log("[5/5] ASSIST  - chatbot demo ...")
    bot = TrafficChatbot(df, kpis, anomalies, forecasts)
    for q in ["Was there any congestion today?", "What is the busiest zone?",
              "Forecast for JCT_MainRoad?", "Is the main road safe?"]:
        log(f"        Q: {q}\n           A: {bot.ask(q)}")

    summary = {"kpis": kpis, "anomaly_count": int(len(anomalies)),
               "cv": cv, "charts": charts, "excel_report": xlsx,
               "annotated_video": cv.get("annotated_video"),
               "runtime_seconds": round(time.time() - t0, 2)}
    sp = os.path.join(config.OUTPUT_DIR, "run_summary.json")
    with open(sp, "w") as fh: json.dump(summary, fh, indent=2, default=str)
    log(f"\nDone in {summary['runtime_seconds']}s. Summary -> {os.path.basename(sp)}")
    return summary


if __name__ == "__main__":
    run()
