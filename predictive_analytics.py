from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import config


def _time_features(mins: np.ndarray) -> np.ndarray:
    f = (mins % 1440) / 1440.0
    return np.column_stack([np.sin(2*np.pi*f), np.cos(2*np.pi*f),
                            np.sin(4*np.pi*f), np.cos(4*np.pi*f)])


def _forecast_metric(zdf: pd.DataFrame, metric: str, horizon: int):
    zdf = zdf.sort_values("timestamp")
    mod = (zdf["timestamp"].dt.hour * 60 + zdf["timestamp"].dt.minute).to_numpy(float)
    X = _time_features(mod); y = zdf[metric].to_numpy(float)
    if len(y) > horizon * 3:
        Xtr, ytr, Xv, yv = X[:-horizon], y[:-horizon], X[-horizon:], y[-horizon:]
    else:
        Xtr, ytr, Xv, yv = X, y, X, y
    m = Ridge(alpha=1.0).fit(Xtr, ytr)
    resid = ytr - m.predict(Xtr)               # qoldiqlar (residuals)
    sigma = float(np.std(resid)) if len(resid) else 0.0
    mae = float(mean_absolute_error(yv, m.predict(Xv)))
    last = zdf["timestamp"].iloc[-1]
    step = pd.Timedelta(minutes=config.SAMPLE_INTERVAL_MIN)
    fut = [last + step*(i+1) for i in range(horizon)]
    fmod = np.array([(t.hour*60+t.minute) for t in fut], float)
    point = m.predict(_time_features(fmod))
    yhat = np.clip(np.round(point), 0, None).astype(int)
    # 95% ishonch oralig'i: nuqta ± 1.96*sigma (normal taqsimot taxmini)
    margin = 1.96 * sigma
    lower = np.clip(np.round(point - margin), 0, None).astype(int)
    upper = np.clip(np.round(point + margin), 0, None).astype(int)
    return (fut, yhat.tolist(), round(mae, 2),
            lower.tolist(), upper.tolist(), round(sigma, 2))


def forecast_zone(zone_df: pd.DataFrame, horizon: int = config.FORECAST_HORIZON) -> Dict:
    z = zone_df.copy(); z["timestamp"] = pd.to_datetime(z["timestamp"])
    fut, car_hat, car_mae, car_lo, car_hi, car_sigma = _forecast_metric(z, "car_count", horizon)
    _,   ped_hat, ped_mae, ped_lo, ped_hi, ped_sigma = _forecast_metric(z, "person_count", horizon)
    rc = int(z["road_capacity"].iloc[0]); pc = int(z["ped_capacity"].iloc[0])
    return {
        "zone": z["zone"].iloc[0], "future_timestamps": fut,
        "car_forecast": car_hat, "person_forecast": ped_hat,
        "car_forecast_lower": car_lo, "car_forecast_upper": car_hi,
        "person_forecast_lower": ped_lo, "person_forecast_upper": ped_hi,
        "car_sigma": car_sigma, "person_sigma": ped_sigma,
        "car_mae": car_mae, "person_mae": ped_mae,
        "road_capacity": rc, "ped_capacity": pc,
        "car_predicted_peak": int(max(car_hat)) if car_hat else 0,
        "person_predicted_peak": int(max(ped_hat)) if ped_hat else 0,
        "congestion_predicted": bool(any(c > 0.85*rc for c in car_hat)),
    }


def forecast_all_zones(df: pd.DataFrame, horizon: int = config.FORECAST_HORIZON,
                       now_cutoff: str = config.FORECAST_NOW) -> Dict[str, Dict]:
    work = df.copy()
    if now_cutoff is not None:
        work["timestamp"] = pd.to_datetime(work["timestamp"])
        work = work[work["timestamp"] <= pd.Timestamp(now_cutoff)]
    return {z: forecast_zone(zdf, horizon) for z, zdf in work.groupby("zone")}


def busiest_hours(df: pd.DataFrame, metric: str = "car_count", top_n: int = 3) -> pd.DataFrame:
    d = df.copy(); d["timestamp"] = pd.to_datetime(d["timestamp"])
    d["clock_hour"] = d["timestamp"].dt.hour
    by = d.groupby("clock_hour")[metric].mean().round(1).sort_values(ascending=False)
    return by.head(top_n).reset_index()


if __name__ == "__main__":
    import synthetic_data
    d = synthetic_data.generate_traffic_timeseries()
    fc = forecast_all_zones(d)
    for z, info in fc.items():
        print(f"{z:>16}: car_peak={info['car_predicted_peak']:>3} "
              f"ped_peak={info['person_predicted_peak']:>3} "
              f"congestion={info['congestion_predicted']} "
              f"car_MAE={info['car_mae']}")
    print("\nBusiest car hours:\n", busiest_hours(d).to_string(index=False))