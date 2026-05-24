from __future__ import annotations
import argparse
import os
import sys

import config

# Standart yo'llar (config.py uslubiga mos)
DATASET_DIR = os.path.join(config.BASE_DIR, "datasets")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

# Fine-tuning sozlamalari - kichik dataset uchun KUCHAYTIRILGAN
DEFAULT_BASE_MODEL = "yolov8n.pt"   # eng yengil model - laptopda tez o'rganadi
DEFAULT_EPOCHS = 100                # ko'p epoch - kichik datasetda yaxshiroq o'rganadi
DEFAULT_IMG_SIZE = 640              # rasm o'lchami
DEFAULT_BATCH = 8                   # bir vaqtda nechta rasm (GPU yo'q bo'lsa kichik)
RUN_NAME = "traffic_finetune_v2"


def _pick_device() -> str:
    """Qaysi qurilmada o'rgatishni avtomatik aniqlaydi (GPU / Mac / CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "0"            # NVIDIA GPU
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"          # Apple Silicon (M1/M2/M3)
    except Exception:
        pass
    return "cpu"                  # GPU topilmadi - protsessorda (sekinroq)


def check_dataset() -> bool:
    """Dataset to'g'ri joydami va to'g'ri formatdami - tekshiradi."""
    if not os.path.isfile(DATA_YAML):
        print("=" * 64)
        print("DATASET TOPILMADI.")
        print(f"Kutilgan joy: {DATA_YAML}")
        print("-" * 64)
        print("Nima qilish kerak:")
        print("  1. https://universe.roboflow.com  saytiga kiring (bepul)")
        print("     Qidiruv: 'traffic vehicle detection'")
        print("  2. Dataset -> Download Dataset -> Format: YOLOv8")
        print("     -> 'Download zip to computer'")
        print(f"  3. ZIP ichidagilarni shu papkaga oching:")
        print(f"       {DATASET_DIR}")
        print("     (ichida data.yaml, train/, valid/ bo'lsin)")
        print("  4. Qaytadan:  python train.py")
        print("=" * 64)
        return False
    return True


def fine_tune(base_model: str, epochs: int, img_size: int,
              batch: int, device: str) -> str:
    """Asosiy fine-tuning jarayoni."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics o'rnatilmagan. Terminalda quyidagini bajaring:")
        print("    pip install ultralytics")
        sys.exit(1)

    print("=" * 64)
    print("YOLO FINE-TUNING boshlanmoqda")
    print(f"  Asos model : {base_model}")
    print(f"  Dataset    : {DATA_YAML}")
    print(f"  Epochs     : {epochs}")
    print(f"  Rasm o'lcham: {img_size}")
    print(f"  Batch      : {batch}")
    print(f"  Qurilma    : {device}")
    print("=" * 64)

    # Pre-trained modeldan boshlaymiz (transfer learning)
    model = YOLO(base_model)

    # Fine-tuning - kichik dataset uchun KUCHAYTIRILGAN sozlamalar
    model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        device=device,
        name=RUN_NAME,
        project=os.path.join(config.BASE_DIR, "runs", "detect"),
        patience=30,        # ko'proq sabr - erta to'xtamasin
        seed=config.RANDOM_SEED,
        verbose=True,
        # --- Data augmentation: kichik datasetni "sun'iy" ko'paytiradi ---
        hsv_h=0.015,        # rang tovlanishi
        hsv_s=0.7,          # to'yinganlik
        hsv_v=0.4,          # yorqinlik
        degrees=5.0,        # biroz aylantirish
        translate=0.1,      # siljitish
        scale=0.5,          # masshtab o'zgartirish
        fliplr=0.5,         # gorizontal aks ettirish
        mosaic=1.0,         # mosaic augmentation (juda foydali)
        # --- Kichik obyektlar uchun ---
        lr0=0.01,           # boshlang'ich o'rganish tezligi
    )

    best = os.path.join(config.BASE_DIR, "runs", "detect",
                        RUN_NAME, "weights", "best.pt")
    print("\n" + "=" * 64)
    print("FINE-TUNING TUGADI!")
    print(f"  Yangi model: {best}")
    print("-" * 64)
    print("Endi config.py da quyidagini o'zgartiring:")
    print(f'    YOLO_WEIGHTS = r"{best}"')
    print("Shunda butun loyiha fine-tuned modelni ishlatadi.")
    print("=" * 64)
    return best


def main():
    parser = argparse.ArgumentParser(
        description="YOLO modelni trafik datasetiga fine-tune qilish")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL,
                        help="Asos model (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--device", default=None,
                        help="cpu / mps / 0 (default: avtomatik)")
    args = parser.parse_args()

    if not check_dataset():
        sys.exit(1)

    device = args.device or _pick_device()
    fine_tune(args.model, args.epochs, args.imgsz, args.batch, device)


if __name__ == "__main__":
    main()