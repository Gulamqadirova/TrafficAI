from __future__ import annotations
from typing import Dict, List
import pandas as pd


class TrafficChatbot:
    def __init__(self, df, kpis, anomalies, forecasts):
        self.df = df.copy(); self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.kpis = kpis; self.anomalies = anomalies; self.forecasts = forecasts

    def ask(self, q: str) -> str:
        ql = q.lower().strip()
        if self._has(ql, ["anomal", "incident", "unusual", "congest", "jam"]):
            return self._anomalies()
        if self._has(ql, ["forecast", "predict", "next hour", "expect"]):
            return self._forecast(ql)
        if self._has(ql, ["peak", "busiest", "most"]):
            return self._peak()
        if self._has(ql, ["how many car", "total car", "cars"]):
            return self._cars(ql)
        if self._has(ql, ["how many people", "pedestrian", "people", "footfall"]):
            return self._people(ql)
        if self._has(ql, ["safe", "capacity", "risk"]):
            return self._safety()
        if self._has(ql, ["help", "what can you"]):
            return self._help()
        return ("I can report on traffic/pedestrian anomalies, car and people counts, "
                "peak times, congestion risk and next-hour forecasts. "
                "Try: 'Was there any congestion today?' or 'Forecast for JCT_MainRoad'.")

    @staticmethod
    def _has(q, ks): return any(k in q for k in ks)

    def _zone(self, q):
        for z in self.df["zone"].unique():
            if z.lower() in q or z.split("_")[-1].lower() in q:
                return z
        return None

    def _anomalies(self):
        n = len(self.anomalies)
        if n == 0:
            return "No traffic or pedestrian anomalies were detected today."
        t = self.anomalies.iloc[0]
        return (f"{n} anomalous reading(s) detected. Most severe at {t['zone']} at "
                f"{pd.to_datetime(t['timestamp']).strftime('%H:%M')} with {int(t['car_count'])} "
                f"cars ({t['road_occupancy']*100:.0f}% of road capacity) and "
                f"{int(t['person_count'])} people. Reason: {t['anomaly_reason']}.")

    def _forecast(self, q):
        z = self._zone(q) or self.kpis["peak_zone"]
        f = self.forecasts.get(z)
        if not f: return f"No forecast available for {z}."
        risk = ("and congestion is expected (>85% road capacity)"
                if f["congestion_predicted"] else "with no congestion expected")
        return (f"Forecast for {z}: next hour expected to peak at ~{f['car_predicted_peak']} "
                f"cars (capacity {f['road_capacity']}) and ~{f['person_predicted_peak']} "
                f"people {risk}. Car forecast MAE {f['car_mae']}.")

    def _peak(self):
        return (f"Peak road occupancy was at {self.kpis['peak_zone']} at "
                f"{self.kpis['peak_time']} ({self.kpis['peak_road_occupancy_pct']}% - "
                f"{self.kpis['peak_band']}). Busiest car site: {self.kpis['busiest_car_site']}; "
                f"busiest pedestrian site: {self.kpis['busiest_ped_site']}.")

    def _cars(self, q):
        z = self._zone(q)
        if z:
            v = int(self.df[self.df['zone'] == z]['car_count'].sum())
            return f"Estimated {v:,} car detections at {z} today."
        return f"Estimated {self.kpis['total_cars']:,} car detections across all zones today."

    def _people(self, q):
        z = self._zone(q)
        if z:
            v = int(self.df[self.df['zone'] == z]['person_count'].sum())
            return f"Estimated {v:,} pedestrian detections at {z} today."
        return f"Estimated {self.kpis['total_people']:,} pedestrian detections across all zones today."

    def _safety(self):
        crit = [z for z, p in self.kpis["zone_peak_road"].items() if p >= 85]
        if crit:
            return ("Congestion risk detected. Zones reaching >=85% road occupancy: "
                    f"{', '.join(crit)}. Recommend signal-timing review and operator alert.")
        return "All monitored roads stayed below the 85% congestion threshold today."

    def _help(self):
        return ("Ask me about: anomalies/congestion, car counts, pedestrian counts, "
                "peak/busiest times, capacity & safety risk, and next-hour forecasts. "
                "You can name a zone, e.g. 'cars at JCT_MainRoad'.")


if __name__ == "__main__":
    import synthetic_data, video_analytics, anomaly_detection, predictive_analytics, bi_dashboard
    df = synthetic_data.generate_traffic_timeseries(); synthetic_data.generate_traffic_clip()
    cv = video_analytics.analyse_video(write_annotated=False)
    scored = anomaly_detection.detect_anomalies(df); an = anomaly_detection.anomaly_summary(scored)
    fc = predictive_analytics.forecast_all_zones(df)
    k = bi_dashboard.compute_kpis(df, cv, an)
    bot = TrafficChatbot(df, k, an, fc)
    for q in ["Was there any congestion today?", "What is the busiest zone?",
              "How many cars in total?", "Is the main road safe?",
              "Forecast for JCT_MainRoad?"]:
        print(f"USER: {q}\n BOT: {bot.ask(q)}\n")
