from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import config
import database
import metrics

# Dataset yo'llari
DATASET_DIR = Path(config.BASE_DIR) / "datasets" / "train"
IMAGES_DIR  = DATASET_DIR / "images"
LABELS_DIR  = DATASET_DIR / "labels"
MODEL_PATH  = os.path.join(config.BASE_DIR, "yolov8n.pt")

# COCO klass ID → dataset klass ID moslik
# Dataset: 0=car, 1=person (Roboflow tartibi)
# COCO:    0=person, 2=car
DATASET_CLASSES = {0: "car", 1: "person"}


def parse_yolo_label(label_path: Path,
                     img_w: int, img_h: int) -> List[Dict]:
    """YOLO format .txt labelni o'qib, piksel koordinatalariga o'giradi.
    Qaytaradi: [{"class_id": int, "bbox": (x1,y1,x2,y2)}, ...]
    """
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    results = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        x1 = int((cx - w/2) * img_w)
        y1 = int((cy - h/2) * img_h)
        x2 = int((cx + w/2) * img_w)
        y2 = int((cy + h/2) * img_h)
        results.append({"class_id": cls, "bbox": (x1, y1, x2, y2)})
    return results


def run_yolo_on_image(model, img_path: Path,
                      conf: float = 0.25) -> List[Dict]:
    """Bitta rasmda YOLO detection ishlatadi.
    Qaytaradi: [{"class_id": int, "bbox": (x1,y1,x2,y2), "confidence": float}]
    """
    import cv2
    frame = cv2.imread(str(img_path))
    if frame is None:
        return []
    # COCO klasslarini dataset klasslariga moslashtirish
    # COCO 0=person→dataset 1, COCO 2=car→dataset 0
    coco_to_dataset = {0: 1, 2: 0}
    results = model.predict(frame, conf=conf, verbose=False,
                            classes=[0, 2])  # person, car
    detections = []
    for box in results[0].boxes:
        cid_coco = int(box.cls[0])
        cid_ds   = coco_to_dataset.get(cid_coco, cid_coco)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "class_id":  cid_ds,
            "bbox":      (x1, y1, x2, y2),
            "confidence": float(box.conf[0]),
        })
    return detections


def evaluate(max_images: int = 50) -> Dict:
    """
    Dataset ustida model baholaydi.
    Agar ground-truth labellar bo'sh bo'lsa, confidence statistikasi qaytaradi.
    """
    images = sorted(IMAGES_DIR.glob("*.jpg"))[:max_images]
    if not images:
        images = sorted(IMAGES_DIR.glob("*.png"))[:max_images]

    if not images:
        return {"error": "Rasm topilmadi", "images_dir": str(IMAGES_DIR)}

    # YOLO modelini yuklaymiz
    model = None
    try:
        from ultralytics import YOLO
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
        else:
            model = YOLO("yolov8n.pt")
        print(f"Model yuklandi: {MODEL_PATH if os.path.exists(MODEL_PATH) else 'yolov8n.pt'}")
    except Exception as e:
        print(f"Model yuklanmadi ({e}) — confidence stats rejimi")

    all_preds: List[Dict]   = []
    all_gts:   List[Dict]   = []
    has_gt = False

    import cv2
    for img_path in images:
        label_path = LABELS_DIR / (img_path.stem + ".txt")
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        # Ground-truth
        gt = parse_yolo_label(label_path, w, h)
        if gt:
            has_gt = True
            all_gts.extend(gt)

        # Model bashorati
        if model is not None:
            preds = run_yolo_on_image(model, img_path)
            all_preds.extend(preds)

    print(f"\nBaholash: {len(images)} rasm | "
          f"GT annotatsiyalar: {len(all_gts)} | "
          f"Bashoratlar: {len(all_preds)}")

    if has_gt and all_preds:
        # To'liq precision/recall/F1 (ground-truth mavjud)
        acc = metrics.detection_accuracy(all_preds, all_gts, iou_threshold=0.5)
        mode = "full_iou"
        print(f"Precision: {acc['precision']} | "
              f"Recall: {acc['recall']} | "
              f"F1: {acc['f1_score']}")
    else:
        # Confidence statistikasi (ground-truth bo'sh)
        acc = metrics.confidence_stats(all_preds)
        mode = "confidence_stats"
        print("Ground-truth labellar bo'sh — confidence statistikasi:")
        print(f"  Aniqlangan obyektlar: {acc.get('count', 0)}")
        print(f"  O'rtacha ishonch:     {acc.get('mean_confidence', 0)}")
        print(f"  Min/Max ishonch:      "
              f"{acc.get('min_confidence', 0)} / {acc.get('max_confidence', 0)}")
        print("\nEslatma: Roboflow'da annotatsiya qilingan labellar bu "
              "muhitda bo'sh. To'liq precision/recall uchun datasets/train/labels/ "
              "da YOLO format labellar kerak.")

    result = {
        "evaluation_mode": mode,
        "images_evaluated": len(images),
        "ground_truth_count": len(all_gts),
        "predictions_count": len(all_preds),
        **acc,
    }

    # Natijani database'ga saqlash (kpi_snapshots orqali)
    try:
        database.init_db()
        database.save_kpis({
            f"accuracy_{k}": v for k, v in result.items()
        })
        print(f"Natija database'ga yozildi.")
    except Exception as e:
        print(f"DB yozishda xato: {e}")

    # JSON fayliga ham saqlash
    out_path = os.path.join(config.OUTPUT_DIR, "accuracy_report.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"Natija: {out_path}")
    return result


if __name__ == "__main__":
    result = evaluate()
    print("\n=== YAKUNIY NATIJA ===")
    for k, v in result.items():
        print(f"  {k}: {v}")