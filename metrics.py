from __future__ import annotations
import time
from typing import Dict, List, Optional

import numpy as np



#  1. DETECTION ANIQLIGI (precision / recall / F1)
def _iou(box_a, box_b) -> float:
    """Ikki quti orasidagi IoU (Intersection over Union)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detection_accuracy(predictions: List[dict], ground_truth: List[dict],
                       iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Bashorat qilingan qutilarni ground-truth bilan taqqoslab,
    precision, recall, F1 hisoblaydi.

    predictions / ground_truth: har biri {"bbox": (x1,y1,x2,y2), "class_id": int}
    """
    matched_gt = set()
    tp = 0
    for pred in predictions:
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(ground_truth):
            if j in matched_gt or gt.get("class_id") != pred.get("class_id"):
                continue
            i = _iou(pred["bbox"], gt["bbox"])
            if i > best_iou:
                best_iou, best_j = i, j
        if best_iou >= iou_threshold and best_j >= 0:
            tp += 1
            matched_gt.add(best_j)

    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
    }


def confidence_stats(detections: List[dict]) -> Dict[str, float]:
    """Ground-truth bo'lmaganda: model ishonch (confidence) statistikasi."""
    confs = [float(d.get("confidence", 0)) for d in detections]
    if not confs:
        return {"count": 0, "mean_confidence": 0.0, "min_confidence": 0.0,
                "max_confidence": 0.0}
    return {
        "count": len(confs),
        "mean_confidence": round(float(np.mean(confs)), 3),
        "min_confidence": round(float(np.min(confs)), 3),
        "max_confidence": round(float(np.max(confs)), 3),
    }



#  2. TIZIM UNUMDORLIGI (FPS, timing)
class PerformanceTimer:
    """Pipeline bosqichlari vaqtini va FPS ni o'lchaydi.

    Ishlatish:
        timer = PerformanceTimer()
        with timer.stage("detection"):
            ...  # detection kodi
        timer.record_frames(300)
        print(timer.report())
    """
    def __init__(self):
        self.stages: Dict[str, float] = {}
        self.frames_processed = 0
        self._t0 = time.perf_counter()

    def stage(self, name: str):
        return _StageContext(self, name)

    def record_frames(self, n: int) -> None:
        self.frames_processed += n

    def total_seconds(self) -> float:
        return time.perf_counter() - self._t0

    def fps(self) -> float:
        # FPS - detection bosqichi vaqtiga nisbatan (agar bor bo'lsa)
        det_time = self.stages.get("detection", self.total_seconds())
        return round(self.frames_processed / det_time, 2) if det_time > 0 else 0.0

    def report(self) -> Dict:
        return {
            "frames_processed": self.frames_processed,
            "fps": self.fps(),
            "total_runtime_sec": round(self.total_seconds(), 3),
            "stage_timings_sec": {k: round(v, 3) for k, v in self.stages.items()},
        }


class _StageContext:
    def __init__(self, timer: PerformanceTimer, name: str):
        self.timer = timer; self.name = name
    def __enter__(self):
        self._s = time.perf_counter(); return self
    def __exit__(self, *exc):
        dt = time.perf_counter() - self._s
        self.timer.stages[self.name] = self.timer.stages.get(self.name, 0.0) + dt



#  3. BAHOLASH HISOBOTI
def evaluation_report(cv_summary: Optional[dict] = None,
                      perf: Optional[dict] = None,
                      accuracy: Optional[dict] = None) -> str:
    """Inson o'qishi uchun matnli baholash hisoboti tuzadi."""
    lines = ["=" * 52, "TIZIM BAHOLASH HISOBOTI (System Evaluation)", "=" * 52]

    if cv_summary:
        lines.append("\n[ Computer Vision ]")
        lines.append(f"  Backend (model): {cv_summary.get('backend', 'n/a')}")
        lines.append(f"  Qayta ishlangan kadrlar: {cv_summary.get('frames_processed', 'n/a')}")

    if accuracy:
        lines.append("\n[ Detection aniqligi ]")
        if "precision" in accuracy:
            lines.append(f"  Precision: {accuracy['precision']}")
            lines.append(f"  Recall:    {accuracy['recall']}")
            lines.append(f"  F1-score:  {accuracy['f1_score']}")
        if "mean_confidence" in accuracy:
            lines.append(f"  O'rtacha ishonch: {accuracy['mean_confidence']}")

    if perf:
        lines.append("\n[ Tizim unumdorligi ]")
        lines.append(f"  FPS: {perf.get('fps', 'n/a')}")
        lines.append(f"  Jami ishlash vaqti: {perf.get('total_runtime_sec', 'n/a')} s")
        for stage, t in (perf.get("stage_timings_sec") or {}).items():
            lines.append(f"    - {stage}: {t} s")

    lines.append("=" * 52)
    return "\n".join(lines)


if __name__ == "__main__":
    # Demo: soxta bashorat va ground-truth bilan precision/recall
    preds = [
        {"bbox": (10, 10, 50, 50), "class_id": 2, "confidence": 0.9},
        {"bbox": (60, 60, 90, 90), "class_id": 0, "confidence": 0.7},
        {"bbox": (200, 200, 240, 240), "class_id": 2, "confidence": 0.4},  # FP
    ]
    gts = [
        {"bbox": (12, 12, 52, 52), "class_id": 2},   # preds[0] bilan mos
        {"bbox": (62, 62, 92, 92), "class_id": 0},   # preds[1] bilan mos
        {"bbox": (300, 300, 340, 340), "class_id": 0},  # topilmagan (FN)
    ]
    acc = detection_accuracy(preds, gts)
    print("Detection aniqligi:", acc)
    print("Ishonch statistikasi:", confidence_stats(preds))

    # Demo: performance timer
    timer = PerformanceTimer()
    with timer.stage("detection"):
        time.sleep(0.05)
    timer.record_frames(100)
    print("\n" + evaluation_report(perf=timer.report(), accuracy=acc))