from __future__ import annotations
from typing import Dict, List
import numpy as np
import cv2
import config
from detection import MultiClassDetector, detection_centroid
from tracking import MultiClassTracker



# Xavfsiz klass ma'lumoti olish - topilmasa standart qiymat (xato bermaydi)
_DEFAULT_CLASS = {"name": "obj", "color": (200, 200, 200)}
def _cls(cid):
    return config.CLASSES.get(cid, _DEFAULT_CLASS)


def analyse_video(video_path: str = config.SYNTHETIC_VIDEO,
                  annotated_out: str = config.ANNOTATED_VIDEO,
                  write_annotated: bool = True) -> Dict[str, object]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    detector = MultiClassDetector()
    mtrack = MultiClassTracker(line_x=width // 2)
    writer = None
    if write_annotated:
        writer = cv2.VideoWriter(annotated_out,
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    per_frame = {cid: [] for cid in config.TARGET_CLASS_IDS}
    conf_acc = {cid: [] for cid in config.TARGET_CLASS_IDS}
    frame_idx = 0

    MAX_FRAMES = 600
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx > MAX_FRAMES:
            break
        detections = detector.detect(frame)
        objs = mtrack.update(detections, detection_centroid)
        for cid in config.TARGET_CLASS_IDS:
            cdets = [d for d in detections if d["class_id"] == cid]
            per_frame[cid].append(len(cdets))
            if cdets:
                conf_acc[cid].append(float(np.mean([d["confidence"] for d in cdets])))
        if writer is not None:
            _annotate(frame, detections, objs, mtrack, width)
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    summary = {"backend": detector.backend, "frames_processed": frame_idx,
               "annotated_video": annotated_out if write_annotated else None,
               "classes": {}}
    for cid in config.TARGET_CLASS_IDS:
        name = _cls(cid)["name"]
        counts = np.array(per_frame[cid]) if per_frame[cid] else np.array([0])
        summary["classes"][name] = {
            "peak_count": int(counts.max()),
            "mean_count": round(float(counts.mean()), 2),
            "unique_tracks": mtrack.trackers[cid].next_id,
            "crossings_l2r": mtrack.counters[cid].left_to_right,
            "crossings_r2l": mtrack.counters[cid].right_to_left,
            "total_crossings": mtrack.counters[cid].total,
            "mean_confidence": round(float(np.mean(conf_acc[cid])), 3) if conf_acc[cid] else 0.0,
        }
    return summary


def _annotate(frame, detections, objs, mtrack, width) -> None:
    cv2.line(frame, (width // 2, 0), (width // 2, frame.shape[0]), (0, 0, 255), 2)
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = _cls(d["class_id"])["color"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{d['label']} {d['confidence']:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    for cid, omap in objs.items():
        color = _cls(cid)["color"]
        for oid, (cx, cy) in omap.items():
            cv2.circle(frame, (cx, cy), 3, color, -1)
            cv2.putText(frame, str(oid), (cx - 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    # HUD
    y = 22
    for cid in config.TARGET_CLASS_IDS:
        name = _cls(cid)["name"]; c = mtrack.counters[cid]
        cv2.putText(frame, f"{name}:  ->{c.left_to_right}  <-{c.right_to_left}  tot {c.total}",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    _cls(cid)["color"], 2)
        y += 26


if __name__ == "__main__":
    import synthetic_data
    synthetic_data.generate_traffic_clip()
    s = analyse_video()
    print("backend:", s["backend"], "frames:", s["frames_processed"])
    for name, st in s["classes"].items():
        print(f"  {name:>7}: peak={st['peak_count']} tracks={st['unique_tracks']} "
              f"crossings={st['total_crossings']} conf={st['mean_confidence']}")