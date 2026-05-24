from __future__ import annotations
from typing import List, Dict, Tuple
import os
import numpy as np
import cv2
import config

Detection = Dict[str, object]


class MultiClassDetector:

    def __init__(self, weights: str = config.YOLO_WEIGHTS,
                 conf: float = config.DETECTION_CONFIDENCE,
                 prefer_yolo: bool = True) -> None:
        self.conf = conf
        self.backend = "none"
        self._yolo = None
        self._hog = None
        self._car_cascade = None

        if prefer_yolo:
            self._try_init_yolo(weights)
        if self._yolo is None:
            self._init_opencv_fallback()

    # ------------------------------------------------------------------ #
    def _try_init_yolo(self, weights: str) -> None:
        try:
            from ultralytics import YOLO            # lazy import on purpose
            self._yolo = YOLO(weights)
            self.backend = "yolov8"
        except Exception:
            self._yolo = None                       # fall back below

    def _init_opencv_fallback(self) -> None:
        # Person detector (HOG + SVM) - always available in OpenCV.
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # Car detector (Haar cascade) - bundled with OpenCV data.
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_car.xml")
        if os.path.exists(cascade_path):
            self._car_cascade = cv2.CascadeClassifier(cascade_path)
        self.backend = "opencv_fallback"

    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self._yolo is not None:
            return self._detect_yolo(frame)
        return self._detect_opencv(frame)

    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:  # pragma: no cover
        results = self._yolo.predict(frame, conf=self.conf, verbose=False,
                                     classes=config.TARGET_CLASS_IDS)
        out: List[Detection] = []
        for res in results:
            for box in res.boxes:
                cid = int(box.cls[0])
                if cid not in config.CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                out.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": float(box.conf[0]),
                    "class_id": cid,
                    "label": config.CLASSES[cid]["name"],
                })
        return out

    def _detect_opencv(self, frame: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out: List[Detection] = []

        # --- People (HOG) ---
        rects, weights = self._hog.detectMultiScale(
            gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
        boxes = [[x, y, x + w, y + h] for (x, y, w, h) in rects]
        scores = [float(s) for s in weights]
        for i in self._nms(boxes, scores, config.NMS_THRESHOLD):
            confidence = float(1.0 / (1.0 + np.exp(-scores[i])))
            if confidence < self.conf:
                continue
            x1, y1, x2, y2 = boxes[i]
            out.append({"bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": round(confidence, 3),
                        "class_id": 0, "label": "person"})

        # --- Cars (Haar cascade if available, else colour+shape heuristic) ---
        car_boxes = []
        if self._car_cascade is not None and not self._car_cascade.empty():
            cars = self._car_cascade.detectMultiScale(gray, 1.1, 2)
            car_boxes = [[x, y, x + w, y + h] for (x, y, w, h) in cars]
        else:
            car_boxes = self._detect_cars_heuristic(frame)
        for (x1, y1, x2, y2) in car_boxes:
            out.append({"bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": 0.6, "class_id": 2, "label": "car"})
        return out

    @staticmethod
    def _detect_cars_heuristic(frame: np.ndarray) -> List[List[int]]:
        h, w = frame.shape[:2]
        road = frame[int(h * 0.45):, :]
        hsv = cv2.cvtColor(road, cv2.COLOR_BGR2HSV)
        # Dark / saturated regions = vehicle bodies against light tarmac.
        mask = cv2.inRange(hsv, (0, 0, 0), (179, 255, 110))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        offset = int(h * 0.45)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw >= 45 and 18 <= bh <= 90 and bw > bh:   # car-shaped
                boxes.append([x, y + offset, x + bw, y + offset + bh])
        return boxes

    @staticmethod
    def _nms(boxes, scores, iou_thr: float) -> List[int]:
        if not boxes:
            return []
        b = np.array(boxes, float); s = np.array(scores, float)
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = s.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]; keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
            wd = np.maximum(0, xx2 - xx1 + 1); ht = np.maximum(0, yy2 - yy1 + 1)
            inter = wd * ht
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= iou_thr]
        return keep


def detection_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2


if __name__ == "__main__":
    det = MultiClassDetector()
    print("Detector backend:", det.backend)
