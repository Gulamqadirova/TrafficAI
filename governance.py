from __future__ import annotations
import hashlib
from typing import Dict, List, Tuple

import pandas as pd



#  1. MA'LUMOT SIFATI (Data Quality) - Task 4
REQUIRED_COLUMNS = ["timestamp", "zone", "car_count", "person_count"]


def validate_dataframe(df: pd.DataFrame) -> Dict[str, object]:
    """Detection DataFrame'ini tekshiradi va muammolar hisobotini qaytaradi."""
    issues: List[str] = []

    # 1. Bo'sh emasligini tekshirish
    if df is None or df.empty:
        return {"valid": False, "issues": ["DataFrame bo'sh yoki mavjud emas"],
                "rows": 0, "missing_values": 0}

    # 2. Kerakli ustunlar bormi
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            issues.append(f"Kerakli ustun yo'q: '{col}'")

    # 3. Bo'sh (NaN) qiymatlar
    missing = int(df.isna().sum().sum())
    if missing > 0:
        issues.append(f"{missing} ta bo'sh (NaN) qiymat topildi")

    # 4. Manfiy hisoblar (mantiqsiz)
    for col in ("car_count", "person_count"):
        if col in df.columns and (df[col] < 0).any():
            issues.append(f"'{col}' ustunida manfiy qiymat bor")

    # 5. Takroriy qatorlar
    dups = int(df.duplicated().sum())
    if dups > 0:
        issues.append(f"{dups} ta takroriy qator")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "rows": len(df),
        "missing_values": missing,
        "duplicate_rows": dups,
    }


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Ma'lumotni tozalaydi: NaN to'ldiradi, manfiylarni nolga keltiradi,
    takrorlarni o'chiradi. Tozalangan df va nima qilinganini qaytaradi."""
    if df is None or df.empty:
        return df, {"action": "bo'sh ma'lumot - tozalanmadi"}

    report = {}
    out = df.copy()

    # Raqamli ustunlardagi NaN'larni 0 bilan to'ldiramiz
    num_cols = out.select_dtypes(include="number").columns
    nan_before = int(out[num_cols].isna().sum().sum())
    out[num_cols] = out[num_cols].fillna(0)
    report["nan_filled"] = nan_before

    # Manfiy hisoblarni 0 ga
    for col in ("car_count", "person_count"):
        if col in out.columns:
            neg = int((out[col] < 0).sum())
            out.loc[out[col] < 0, col] = 0
            report[f"{col}_negatives_fixed"] = neg

    # Takrorlarni o'chiramiz
    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = before - len(out)

    return out, report



#  2. ANONIMIZATSIYA (Privacy) - C.P5
def anonymise_id(raw_id: str, salt: str = "tpbi") -> str:
    """Shaxsiy identifikatorni (masalan, raqam belgisi) qaytarib bo'lmaydigan
    xesh bilan almashtiradi. GDPR pseudonymisation tamoyili."""
    return hashlib.sha256(f"{salt}:{raw_id}".encode()).hexdigest()[:12]


def blur_region(frame, bbox) -> None:
    """Kadrdagi belgilangan sohani (yuz/raqam belgisi) xiralashtiradi.
    Joyida (in-place) o'zgartiradi. cv2 mavjud bo'lsa ishlaydi."""
    try:
        import cv2
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return
        # Kuchli Gauss xiralashtirish
        k = max(15, ((x2 - x1) // 5) | 1)   # toq son
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    except Exception:
        pass   # cv2 yo'q bo'lsa, jim o'tadi



#  3. HUJJATLI MATNLAR (Dashboard'da ko'rsatiladi) - C.P5
PRIVACY_POLICY = """
**Maxfiylik va ma'lumotlarni himoya qilish (GDPR)**

Ushbu tizim jamoat joylaridagi trafik va piyodalarni kuzatadi. Quyidagi
tamoyillarga rioya qilinadi:

- **Ma'lumotni minimallashtirish**: faqat agregat hisoblar (mashina/odam soni)
  saqlanadi. Shaxsni aniqlovchi tasvirlar doimiy saqlanmaydi.
- **Anonimizatsiya**: yuzlar va avtomobil raqam belgilari namoyish va
  saqlashdan oldin xiralashtiriladi (`blur_region`). Har qanday identifikator
  qaytarib bo'lmaydigan xesh bilan pseudonimlashtiriladi (`anonymise_id`).
- **Saqlash muddati (retention)**: xom video saqlanmaydi; faqat sonli
  agregatlar bazada qoladi. Agregatlar 30 kundan keyin o'chirilishi belgilangan.
- **Maqsadni cheklash**: ma'lumot faqat trafik boshqaruvi va xavfsizlik
  tahlili uchun ishlatiladi, shaxsiy kuzatuv uchun emas.
"""

ETHICS_STATEMENT = """
**Axloqiy kuzatuv tamoyillari**

- **Shaffoflik**: kuzatuv hududlarida belgilar o'rnatilishi kerak.
- **Adolat (bias)**: model turli yorug'lik/ob-havo sharoitida sinovdan
  o'tkazilishi va kam vakillik qilingan guruhlarga nisbatan xolislik
  tekshirilishi lozim.
- **Inson nazorati**: anomaliya ogohlantirishlari avtomatik jazo emas, balki
  inson operatori ko'rib chiqishi uchun mo'ljallangan.
- **Proporsionallik**: kuzatuv ko'lami kutilayotgan jamoat foydasiga mos
  bo'lishi kerak.
"""

COMPLIANCE_NOTES = """
**Muvofiqlik (Compliance)**

- **GDPR (EU/UK)**: qonuniy asos - jamoat xavfsizligi bo'yicha qonuniy
  manfaat. Ma'lumot subyektining huquqlari hujjatlashtirilgan.
- **DPIA**: yuqori xavfli kuzatuv uchun Ma'lumotlarni himoya qilish ta'sirini
  baholash (Data Protection Impact Assessment) talab qilinadi.
- **Saqlash xavfsizligi**: baza faylga kirish cheklangan; ishlab chiqarishda
  shifrlash tavsiya etiladi.
"""


def governance_summary() -> Dict[str, str]:
    """Dashboard uchun barcha hujjatli matnlarni qaytaradi."""
    return {
        "privacy": PRIVACY_POLICY.strip(),
        "ethics": ETHICS_STATEMENT.strip(),
        "compliance": COMPLIANCE_NOTES.strip(),
    }


if __name__ == "__main__":
    import pandas as pd
    # Data quality demo
    df = pd.DataFrame({
        "timestamp": ["2026-05-20 08:00", "2026-05-20 08:05", "2026-05-20 08:05"],
        "zone": ["A", "B", "B"],
        "car_count": [5, -2, 3],          # manfiy + takror
        "person_count": [10, None, 7],    # NaN
    })
    print("Tekshiruv:", validate_dataframe(df))
    cleaned, rep = clean_dataframe(df)
    print("Tozalash hisoboti:", rep)
    print("Anonim ID misol:", anonymise_id("CAR-12345"))
    print("\nGovernance bo'limlari:", list(governance_summary().keys()))