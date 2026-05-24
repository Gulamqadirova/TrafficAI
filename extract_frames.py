from __future__ import annotations
import os

import cv2

import config

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


# youtube_detect.py dagi bilan bir xil video
YOUTUBE_URL = "https://www.youtube.com/watch?v=8JCk5M_xrBs"

# Sozlamalar
NUM_FRAMES = 50                       # nechta rasm ajratish
VIDEO_FILE = os.path.join(config.BASE_DIR, "video.mp4")
FRAMES_DIR = os.path.join(config.BASE_DIR, "datasets", "raw_frames")


# Jonli efir (live stream) bo'lsa, faqat shu davomiylikni olamiz (sekund)
LIVE_DURATION_SEC = 60   # ~2 daqiqalik bo'lak - 50 kadr uchun yetarli


def download_video(url: str, out: str) -> str:
    """YouTube videosini (yoki jonli efirning bir bo'lagini) yuklab oladi."""
    if os.path.exists(out):
        print(f"'{os.path.basename(out)}' allaqachon mavjud, qaytadan yuklanmaydi.")
        return out
    if yt_dlp is None:
        raise RuntimeError("yt-dlp o'rnatilmagan. 'pip install yt-dlp' qiling.")

    ydl_opts = {
        "format": "best[height<=720]/best",
        "outtmpl": out,
        "quiet": False,
        "nocheckcertificate": True,
        "cookiesfrombrowser": ("chrome",),   # YouTube bloklamasligi uchun
        # Jonli efir cheksiz oqadi - ffmpeg ni LIVE_DURATION_SEC sekunddan
        # keyin to'xtatamiz, shunda faqat qisqa bo'lak yuklanadi.
        "external_downloader": "ffmpeg",
        "external_downloader_args": {
            "ffmpeg": ["-t", str(LIVE_DURATION_SEC)],
        },
    }
    print(f"Video yuklanmoqda... (jonli efir bo'lsa ~{LIVE_DURATION_SEC} sekund olinadi)")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return out


def extract_frames(video_path: str, out_dir: str, num_frames: int) -> int:
    """Videodan teng oraliqlarda num_frames ta kadr ajratadi."""
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video ochilmadi: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Ba'zi videolarda kadrlar soni noma'lum bo'ladi - ketma-ket o'qiymiz
        print("Kadrlar soni noma'lum, ketma-ket ajratiladi...")
        total = num_frames * 30

    # Teng oraliq: masalan 1500 kadrli videodan har 30-kadr
    step = max(1, total // num_frames)
    print(f"Jami ~{total} kadr, har {step}-kadr olinadi.")

    saved = 0
    idx = 0
    while saved < num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            fname = os.path.join(out_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
        idx += 1

    cap.release()
    return saved


def main():
    print("=" * 64)
    print("VIDEODAN KADR AJRATISH")
    print("=" * 64)

    # 1) Video yuklab olish
    video = download_video(YOUTUBE_URL, VIDEO_FILE)

    # 2) Kadr ajratish
    count = extract_frames(video, FRAMES_DIR, NUM_FRAMES)

    print("\n" + "=" * 64)
    print(f"TAYYOR! {count} ta rasm ajratildi.")
    print(f"  Joylashuvi: {FRAMES_DIR}")
    print("-" * 64)
    print("KEYINGI QADAM - rasmlarni belgilash (annotation):")
    print("  1. https://roboflow.com  -> bepul ro'yxatdan o'ting")
    print("  2. 'Create New Project' -> Object Detection")
    print("  3. Yuqoridagi papkadagi rasmlarni yuklang (drag & drop)")
    print("  4. Har bir rasmda mashina/odam atrofiga quticha chizing:")
    print("       - mashina -> class nomi: 'car'")
    print("       - odam    -> class nomi: 'person'")
    print("  5. 'Generate' -> 'Export' -> format: YOLOv8 -> ZIP")
    print("  6. ZIP ichini  datasets/traffic/  ga oching")
    print("  7. Keyin:  python train.py")
    print("=" * 64)


if __name__ == "__main__":
    main()