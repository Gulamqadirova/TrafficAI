from __future__ import annotations
import numpy as np
import pandas as pd
import cv2
import config


#  1. Synthetic traffic video (cars + people)
def generate_traffic_clip(path: str = config.SYNTHETIC_VIDEO,
                          frames: int = 150,
                          fps: int = 20,
                          size=(720, 405),
                          n_cars: int = 5,
                          n_people: int = 4,
                          seed: int = config.RANDOM_SEED) -> str:
    """Render a road scene with cars on the road and people on the pavement."""
    rng = np.random.default_rng(seed)
    w, h = size
    road_top = int(h * 0.45)            # carriageway occupies lower half
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():           # pragma: no cover
        raise RuntimeError(f"Could not open VideoWriter for {path}")

    # Cars: move mostly horizontally along the road.
    car_pos = np.column_stack([
        rng.uniform(0, w - 90, n_cars),
        rng.uniform(road_top + 10, h - 50, n_cars),
    ])
    car_vel = np.column_stack([
        rng.choice([-4.0, -3.0, 3.0, 4.0], n_cars),
        rng.uniform(-0.4, 0.4, n_cars),
    ])
    car_wh = np.tile([78, 38], (n_cars, 1))

    # People: move slowly on the pavement (upper area).
    ped_pos = np.column_stack([
        rng.uniform(0, w - 30, n_people),
        rng.uniform(40, road_top - 80, n_people),
    ])
    ped_vel = rng.uniform(-1.8, 1.8, size=(n_people, 2))
    ped_wh = np.tile([26, 64], (n_people, 1))

    for _ in range(frames):
        frame = np.full((h, w, 3), 150, np.uint8)              # tarmac grey
        frame[:road_top] = (190, 185, 175)                     # pavement
        frame += rng.integers(-5, 5, frame.shape, dtype=np.int16).astype(np.uint8)
        # Lane markings + counting line.
        for lx in range(0, w, 60):
            cv2.line(frame, (lx, (road_top + h) // 2), (lx + 30, (road_top + h) // 2),
                     (255, 255, 255), 2)
        cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 0, 255), 2)

        # Update + draw cars.
        car_pos += car_vel
        for i in range(n_cars):
            if car_pos[i, 0] < -90: car_pos[i, 0] = w
            if car_pos[i, 0] > w:   car_pos[i, 0] = -car_wh[i, 0]
            car_pos[i, 1] = np.clip(car_pos[i, 1], road_top + 5, h - car_wh[i, 1] - 5)
            x, y = car_pos[i].astype(int); bw, bh = car_wh[i]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (40, 40, 130), -1)
            cv2.rectangle(frame, (x + 12, y + 6), (x + bw - 12, y + 18), (180, 200, 220), -1)
            cv2.circle(frame, (x + 16, y + bh), 7, (20, 20, 20), -1)
            cv2.circle(frame, (x + bw - 16, y + bh), 7, (20, 20, 20), -1)

        # Update + draw people.
        ped_pos += ped_vel
        for i in range(n_people):
            for ax, lim in enumerate((w - ped_wh[i, 0], road_top - 80 - ped_wh[i, 1])):
                if ped_pos[i, ax] < (0 if ax == 0 else 30) or ped_pos[i, ax] > lim:
                    ped_vel[i, ax] *= -1
                    ped_pos[i, ax] = np.clip(ped_pos[i, ax], 0 if ax == 0 else 30, lim)
            x, y = ped_pos[i].astype(int); bw, bh = ped_wh[i]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (70, 70, 70), -1)
            cv2.circle(frame, (x + bw // 2, y - 6), 9, (100, 100, 100), -1)

        writer.write(frame)

    writer.release()
    return path



#  2. Aggregated multi-camera, multi-class time-series
def generate_traffic_timeseries(path: str = config.TIMESERIES_CSV,
                                day: str = "2026-05-20",
                                seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Full-day person & car counts at 5-min resolution per zone."""
    rng = np.random.default_rng(seed)
    step = f"{config.SAMPLE_INTERVAL_MIN}min"
    times = pd.date_range(f"{day} 00:00", f"{day} 23:55", freq=step)
    hours = (times.hour + times.minute / 60.0).to_numpy(dtype=float)

    rows = []
    for zone, meta in config.ZONES.items():
        rc, pc = meta["road_capacity"], meta["ped_capacity"]
        # Two sharp commuter peaks for cars, broader daytime curve for people.
        am = np.exp(-((hours - 8.0) ** 2) / 1.5)
        pm = np.exp(-((hours - 17.5) ** 2) / 1.8)
        car_curve = 1.1 * am + 1.2 * pm
        ped_curve = (np.exp(-((hours - 9) ** 2) / 6) +
                     0.6 * np.exp(-((hours - 13) ** 2) / 9) +
                     0.9 * np.exp(-((hours - 17) ** 2) / 5))
        floor = 0.04 + 0.04 * (hours > 6) * (hours < 23)

        cars = rc * (floor + car_curve) + rng.normal(0, rc * 0.07, len(times))
        peds = pc * 0.5 * (floor + ped_curve) + rng.normal(0, pc * 0.06, len(times))

        # Inject an evening traffic-congestion incident on the main road.
        if zone == "JCT_MainRoad":
            inc = (hours >= 17.5) & (hours <= 18.0)
            cars[inc] += rc * 0.8

        cars = np.clip(np.round(cars), 0, None).astype(int)
        peds = np.clip(np.round(peds), 0, None).astype(int)

        for t, c, p in zip(times, cars, peds):
            rows.append({
                "timestamp": t, "zone": zone, "site": meta["site"],
                "car_count": int(c), "person_count": int(p),
                "road_capacity": rc, "ped_capacity": pc,
                "road_occupancy": round(min(c / rc, 2.0), 3),
                "ped_occupancy": round(min(p / pc, 2.0), 3),
            })

    df = pd.DataFrame(rows).sort_values(["timestamp", "zone"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    v = generate_traffic_clip()
    print("Traffic clip:", v)
    d = generate_traffic_timeseries()
    print("Rows:", len(d), "zones:", d["zone"].nunique())
    print(d.head())
