from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import config
import database
import metrics

# Dataset yo'llari
DATASET_DIR  = Path(config.BASE_DIR) / "datasets" / "train"
IMAGES_DIR   = DATASET_DIR / "images"
LABELS_DIR   = DATASET_DIR / "labels"

# Modellar
STUDENT_MODEL  = "yolov8n.pt"   # loyiha modeli (baholanuvchi)
TEACHER_MODEL  = "yolov8s.pt"   # reference/proxy GT modeli (kuchliroq)
CONF_THRESHOLD = 0.30

# COCO → dataset klass moslik
# Dataset: 0=car, 1=person (Roboflow) | COCO: 0=person, 2=car
COCO_TO_DS = {0: 1, 2: 0}



#  Yordamchi funksiyalar
def _load_model(path: str):
    """YOLO modelini yuklab oladi. Yuklanmasa None qaytaradi."""
    try:
        from ultralytics import YOLO
        m = YOLO(path)
        print(f"  Model yuklandi: {path}")
        return m
    except Exception as e:
        print(f"  Model yuklanmadi ({path}): {e}")
        return None


def parse_yolo_label(label_path: Path, img_w: int, img_h: int) -> List[Dict]:
    """YOLO .txt labelni piksel koordinatalariga o'giradi."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    result = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        result.append({"class_id": cls, "bbox": (x1, y1, x2, y2)})
    return result


def detect_on_image(model, img_path: Path, conf: float = CONF_THRESHOLD) -> List[Dict]:
    """Bitta rasmda YOLO detection ishlatadi."""
    try:
        import cv2
        frame = cv2.imread(str(img_path))
        if frame is None:
            return []
        res = model.predict(frame, conf=conf, verbose=False, classes=[0, 2])
        out = []
        for box in res[0].boxes:
            cid_coco = int(box.cls[0])
            cid_ds   = COCO_TO_DS.get(cid_coco, cid_coco)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            out.append({
                "class_id":   cid_ds,
                "bbox":       (x1, y1, x2, y2),
                "confidence": float(box.conf[0]),
            })
        return out
    except Exception:
        return []


#  Asosiy baholash funksiyasi
def evaluate(max_images: int = 50) -> Dict:
    images = sorted(IMAGES_DIR.glob("*.jpg"))[:max_images]
    if not images:
        images = sorted(IMAGES_DIR.glob("*.png"))[:max_images]
    if not images:
        return {"error": "Rasm topilmadi", "images_dir": str(IMAGES_DIR)}

    print(f"\n{len(images)} ta rasm topildi: {IMAGES_DIR}")


    #  Rejim 1: Haqiqiy human labellar bormi?
    non_empty = sum(
        1 for img in images
        if (LABELS_DIR / (img.stem + ".txt")).stat().st_size > 0
        if (LABELS_DIR / (img.stem + ".txt")).exists()
    )
    print(f"  Bo'sh bo'lmagan label fayllari: {non_empty}/{len(images)}")

    if non_empty > 0:
        return _evaluate_full_iou(images)


    #  Rejim 2: Reference model (Teacher-Student)
    print("\n[Rejim 2] Reference model baholash (Teacher-Student)")
    print("  Sabab: datasets/train/labels/ da human annotation yo'q")
    print(f"  O'qituvchi (pseudo-GT): {TEACHER_MODEL}")
    print(f"  O'quvchi (baholanuvchi): {STUDENT_MODEL}")

    teacher = _load_model(TEACHER_MODEL)
    student = _load_model(STUDENT_MODEL)

    if teacher is None or student is None:
        print("  Modellar yuklanmadi — confidence_stats rejimiga o'tildi")
        return _evaluate_confidence(images, student)

    return _evaluate_reference_model(images, teacher, student)


def _evaluate_full_iou(images: List[Path]) -> Dict:
    """Rejim 1: Haqiqiy human labellar bilan IoU baholash."""
    print("\n[Rejim 1] To'liq IoU baholash (human ground-truth)")
    student = _load_model(STUDENT_MODEL)
    all_preds, all_gts = [], []
    import cv2
    for img_path in images:
        label_path = LABELS_DIR / (img_path.stem + ".txt")
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gt = parse_yolo_label(label_path, w, h)
        all_gts.extend(gt)
        if student:
            all_preds.extend(detect_on_image(student, img_path))

    acc = metrics.detection_accuracy(all_preds, all_gts, iou_threshold=0.5)
    mode = "full_iou"
    _print_accuracy(acc, mode, len(images))
    return _save_result(acc, mode, len(images), len(all_gts), len(all_preds))


def _evaluate_reference_model(images: List[Path],
                               teacher, student) -> Dict:
    """
    Rejim 2: Teacher-Student baholash.
    Teacher (yolov8s) → pseudo ground-truth.
    Student (yolov8n) → bashorat.
    metrics.detection_accuracy(student_preds, teacher_preds, IoU≥0.5).
    """
    teacher_preds_all: List[Dict] = []
    student_preds_all: List[Dict] = []

    for img_path in images:
        t_dets = detect_on_image(teacher, img_path, conf=0.40)
        s_dets = detect_on_image(student, img_path, conf=CONF_THRESHOLD)
        teacher_preds_all.extend(t_dets)
        student_preds_all.extend(s_dets)

    print(f"\n  Teacher deteksiyalar (pseudo-GT): {len(teacher_preds_all)}")
    print(f"  Student deteksiyalar:              {len(student_preds_all)}")

    if not teacher_preds_all:
        print("  Teacher hech narsa aniqlamadi — confidence_stats rejimiga o'tildi")
        return _evaluate_confidence(images, student)

    acc = metrics.detection_accuracy(
        student_preds_all, teacher_preds_all, iou_threshold=0.5)
    mode = "reference_model_eval"

    _print_accuracy(acc, mode, len(images))
    print("\n  *** MUHIM: Bu natijalar 'proxy ground truth' asosida ***")
    print(f"  *** {TEACHER_MODEL} to'g'ri deb qabul qilingan (Teacher-Student) ***")
    print("  *** Cheklov: teacher xatolari ham 'to'g'ri' hisoblangan ***")

    return _save_result(acc, mode, len(images),
                        len(teacher_preds_all), len(student_preds_all),
                        extra={
                            "teacher_model": TEACHER_MODEL,
                            "student_model": STUDENT_MODEL,
                            "evaluation_note": (
                                "Teacher-Student proxy evaluation. "
                                f"{TEACHER_MODEL} used as reference (pseudo ground-truth). "
                                "Not equivalent to human-annotated ground truth. "
                                "Teacher errors are counted as correct.")
                        })


def _evaluate_confidence(images: List[Path], model) -> Dict:
    """Rejim 3: Faqat confidence statistikasi (fallback)."""
    print("\n[Rejim 3] Confidence statistikasi (fallback)")
    all_preds = []
    if model:
        for img_path in images:
            all_preds.extend(detect_on_image(model, img_path))
    acc = metrics.confidence_stats(all_preds)
    mode = "confidence_stats"
    _print_accuracy(acc, mode, len(images))
    return _save_result(acc, mode, len(images), 0, len(all_preds))


def _print_accuracy(acc: Dict, mode: str, n_images: int) -> None:
    print(f"\n  === NATIJA [{mode}] ({n_images} rasm) ===")
    if "precision" in acc:
        print(f"  Precision : {acc['precision']}")
        print(f"  Recall    : {acc['recall']}")
        print(f"  F1-score  : {acc['f1_score']}")
        print(f"  TP={acc.get('true_positives')} "
              f"FP={acc.get('false_positives')} "
              f"FN={acc.get('false_negatives')}")
    else:
        print(f"  Aniqlangan: {acc.get('count', 0)}")
        print(f"  O'rtacha ishonch: {acc.get('mean_confidence', 0)}")


def _save_result(acc: Dict, mode: str,
                 n_images: int, n_gt: int, n_pred: int,
                 extra: Optional[Dict] = None) -> Dict:
    result = {
        "evaluation_mode": mode,
        "images_evaluated": n_images,
        "ground_truth_count": n_gt,
        "predictions_count": n_pred,
        **acc,
        **(extra or {}),
    }
    # JSON ga saqlash
    out_path = os.path.join(config.OUTPUT_DIR, "accuracy_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"\n  Natija yozildi: {out_path}")
    # DB ga saqlash
    try:
        database.init_db()
        database.save_kpis({f"accuracy_{k}": v for k, v in result.items()
                            if isinstance(v, (int, float, str))})
        print("  Natija DB ga yozildi (kpi_snapshots).")
    except Exception as e:
        print(f"  DB xato: {e}")
    return result


if __name__ == "__main__":
    result = evaluate()
    print("\n=== YAKUNIY NATIJA ===")
    for k, v in result.items():
        if k != "evaluation_note":
            print(f"  {k}: {v}")
    if "evaluation_note" in result:
        print(f"\n  Eslatma: {result['evaluation_note']}")