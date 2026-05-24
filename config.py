from __future__ import annotations
import os

#  File-system layout

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
for _d in (DATA_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

SYNTHETIC_VIDEO = os.path.join(DATA_DIR, "traffic_clip.mp4")
ANNOTATED_VIDEO = os.path.join(OUTPUT_DIR, "annotated_traffic.mp4")
TIMESERIES_CSV = os.path.join(DATA_DIR, "traffic_timeseries.csv")

#  Object classes (COCO ids used by YOLO)
# We monitor people and cars - exactly the two classes in the reference video.
CLASSES = {
    0: {"name": "person", "color": (0, 255, 0)},
    2: {"name": "car",    "color": (0, 165, 255)},
}
TARGET_CLASS_IDS = list(CLASSES.keys())


#  Monitored locations (cameras)

ZONES = {
    "JCT_MainRoad":   {"site": "Public Street",   "road_capacity": 40, "ped_capacity": 60},
    "GATE_Campus":    {"site": "Smart Campus",    "road_capacity": 25, "ped_capacity": 120},
    "HUB_StationFwd": {"site": "Transport Hub",   "road_capacity": 30, "ped_capacity": 200},
    "MALL_CarParkIn": {"site": "Commercial Area", "road_capacity": 50, "ped_capacity": 90},
}


#  Computer-vision settings
DETECTION_CONFIDENCE = 0.40
NMS_THRESHOLD = 0.45
YOLO_WEIGHTS = "yolov8n.pt"


# Centroid tracker
MAX_TRACK_DISTANCE = 70
MAX_DISAPPEARED = 20


#  Congestion thresholds (fraction of capacity)
DENSITY_BANDS = [
    (0.00, 0.35, "Free-flow"),
    (0.35, 0.65, "Busy"),
    (0.65, 0.85, "Heavy"),
    (0.85, 10.0, "Congested"),
]


#  Analytics settings
ANOMALY_CONTAMINATION = 0.02
ANOMALY_Z_THRESHOLD = 3.0
FORECAST_HORIZON = 12          # 12 x 5 min = 1 hour look-ahead
SAMPLE_INTERVAL_MIN = 5
FORECAST_NOW = "2026-05-20 08:00"

RANDOM_SEED = 42
