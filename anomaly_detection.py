from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import config


def detect_anomalies(df: pd.DataFrame,
                     contamination: float = config.ANOMALY_CONTAMINATION,
                     z_threshold: float = config.ANOMALY_Z_THRESHOLD) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0

    feats = df[["car_count", "person_count", "road_occupancy",
                "ped_occupancy", "hour"]].to_numpy(float)
    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=config.RANDOM_SEED)
    df["iforest_flag"] = (iso.fit_predict(feats) == -1)

    for metric in ("car_count", "person_count"):
        g = df.groupby("zone")[metric]
        mean = g.transform("mean")
        std = g.transform("std").replace(0, np.nan)
        df[f"z_{metric}"] = ((df[metric] - mean) / std).fillna(0).round(2)
    df["zscore_flag"] = ((df["z_car_count"].abs() > z_threshold) |
                         (df["z_person_count"].abs() > z_threshold))

    df["is_anomaly"] = df["iforest_flag"] | df["zscore_flag"]

    def reason(r):
        out = []
        if r["iforest_flag"]: out.append("isolation_forest")
        if abs(r["z_car_count"]) > z_threshold: out.append(f"car z={r['z_car_count']}")
        if abs(r["z_person_count"]) > z_threshold: out.append(f"person z={r['z_person_count']}")
        return "; ".join(out)
    df["anomaly_reason"] = df.apply(reason, axis=1)
    return df


def anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    a = df[df["is_anomaly"]].copy()
    if a.empty:
        return a
    cols = ["timestamp", "zone", "site", "car_count", "person_count",
            "road_occupancy", "ped_occupancy", "anomaly_reason"]
    return a[cols].sort_values("road_occupancy", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import synthetic_data
    d = synthetic_data.generate_traffic_timeseries()
    s = detect_anomalies(d)
    summ = anomaly_summary(s)
    print(f"Anomalies: {len(summ)} / {len(s)}")
    print(summ.head(6).to_string(index=False))
