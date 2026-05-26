import os
import cv2
from ultralytics import YOLO

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

# Real-time detection natijalarini database'ga yozish uchun (A.P2)
try:
    import database
    database.init_db()
    _DB_OK = True
except Exception:
    _DB_OK = False



#  SOZLAMALAR
# Taipei trafik — jonli kamera
YOUTUBE_URL = "https://www.youtube.com/watch?v=1EiC9bvVGnk"
MODEL_NAME = "yolov8n.pt"
TARGET_CLASSES = [0, 2]            # COCO: 0 = person, 2 = car
USE_STREAM = True
DOWNLOADED_FILE = "video.mp4"
CONF = 0.30                        # ishonch chegarasi

# COCO klass nomlari
CLASS_NAMES = {0: "person", 2: "car"}
CLASS_COLORS = {0: (0, 255, 0), 2: (0, 165, 255)}  # yashil=odam, sariq=mashina (BGR)


#  1. Video manbasini tayyorlash
def get_stream_url(url: str) -> str:
    """YouTube'dan to'g'ridan-to'g'ri stream URL oladi."""
    if yt_dlp is None:
        raise RuntimeError("yt-dlp o'rnatilmagan. 'pip install yt-dlp' qiling.")
    ydl_opts = {
        "format": "best[ext=mp4][height<=720]/best[ext=mp4]/best",
        "quiet": True,
        "nocheckcertificate": True,
        "cookiesfrombrowser": ("chrome",),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info.get("url") or info["formats"][-1]["url"]


def download_video(url: str, out: str = DOWNLOADED_FILE) -> str:
    """YouTube videosini mahalliy faylga yuklab oladi (ishonchli usul)."""
    if yt_dlp is None:
        raise RuntimeError("yt-dlp o'rnatilmagan. 'pip install yt-dlp' qiling.")
    if os.path.exists(out):
        print(f"'{out}' allaqachon mavjud, qaytadan yuklanmaydi.")
        return out
    ydl_opts = {
        "format": "best[ext=mp4][height<=720]/best[ext=mp4]/best",
        "outtmpl": out,
        "quiet": False,
        "nocheckcertificate": True,
    }
    print("Video yuklanmoqda... (bir oz vaqt olishi mumkin)")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return out


def open_capture():
    """USE_STREAM sozlamasiga qarab video manbasini ochadi."""
    if USE_STREAM:
        print("YouTube stream URL olinmoqda...")
        src = get_stream_url(YOUTUBE_URL)
    else:
        src = download_video(YOUTUBE_URL)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("Video ochilmadi. USE_STREAM=False qilib ko'ring, "
                           "yoki internetni tekshiring.")
    return cap



#  2. Asosiy aniqlash sikli
def main():
    # Modelni yuklash (topilmasa avtomatik yuklab oladi).
    try:
        model = YOLO(MODEL_NAME)
        print(f"Model yuklandi: {MODEL_NAME}")
    except Exception as e:
        print(f"'{MODEL_NAME}' yuklanmadi ({e}). 'yolov8n.pt' bilan urinilmoqda...")
        model = YOLO("yolov8n.pt")

    cap = open_capture()
    print("Aniqlash boshlandi. Chiqish uchun video oynasida 'q' tugmasini bosing.")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video tugadi yoki frame olinmadi.")
            break
        frame_count += 1

        # YOLO predict -- faqat person va car klasslari.
        results = model.predict(frame, classes=TARGET_CLASSES,
                                conf=CONF, verbose=False)

        # Qutilarni o'zimiz chizamiz (klass nomi + rang + ishonch bilan).
        person_n, car_n = 0, 0
        for box in results[0].boxes:
            cid = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = CLASS_COLORS.get(cid, (255, 255, 255))
            name = CLASS_NAMES.get(cid, str(cid))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if cid == 0:
                person_n += 1
            elif cid == 2:
                car_n += 1

        # Jonli hisoblagich (HUD).
        cv2.rectangle(frame, (0, 0), (230, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Person: {person_n}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Car: {car_n}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Real-time natijani DATABASE'ga yozish (har 30 kadrda - A.P2).
        # Bu real-time detection'ni analitika quvuriga ulaydi.
        if _DB_OK and frame_count % 5 == 0:
            try:
                database.log_realtime_detection(
                    source="youtube_live", frame_index=frame_count,
                    person_count=person_n, car_count=car_n)
            except Exception as e:
                print(f"[DB xato] {e}")

        cv2.imshow("YOLO - Person & Car Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Foydalanuvchi to'xtatdi.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Tugadi. Jami {frame_count} frame qayta ishlandi.")


if __name__ == "__main__":
    main()