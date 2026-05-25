from __future__ import annotations
import re
from typing import Optional

import database


class TrafficChatbot:
    """Database'ga ulangan, niyat-tahlilli trafik chatbot."""

    def __init__(self, df=None, kpis=None, anomalies=None, forecasts=None):
        # Eski interfeys bilan moslik uchun argumentlar qoldirilgan, lekin
        # endi chatbot ma'lumotni to'g'ridan-to'g'ri database'dan oladi.
        database.init_db()
        self.forecasts = forecasts or {}
        # Bazadan mavjud zonalar ro'yxatini olamiz (savolda zona aniqlash uchun)
        try:
            rows = database.query("SELECT DISTINCT zone FROM detections")
            self.zones = [r["zone"] for r in rows]
        except Exception:
            self.zones = []


    #  Asosiy kirish nuqtasi
    def ask(self, q: str) -> str:
        if not q or not q.strip():
            return self._help()
        ql = q.lower().strip()

        # Bazada ma'lumot bormi - tekshiramiz
        if not self._has_data():
            return ("Database hali bo'sh. Avval dashboard'da 'Pipeline'ni ishga "
                    "tushiring yoki `python database.py` ni bajaring.")

        zone = self._extract_zone(ql)

        # --- Niyatni aniqlash va mos SQL javobini berish ---
        if self._has(ql, ["anomal", "incident", "unusual", "congest", "jam", "tirband"]):
            return self._anomalies(zone)
        if self._has(ql, ["forecast", "predict", "next hour", "expect", "bashorat"]):
            return self._forecast(zone)
        if self._has(ql, ["peak", "busiest", "most", "eng ko", "eng band"]):
            return self._peak(zone)
        if self._has(ql, ["how many car", "total car", "cars", "mashina", "avtomobil"]):
            return self._cars(zone, ql)
        if self._has(ql, ["how many people", "pedestrian", "people", "footfall",
                          "odam", "piyoda"]):
            return self._people(zone, ql)
        if self._has(ql, ["safe", "capacity", "risk", "xavf", "sig'im"]):
            return self._safety()
        if self._has(ql, ["compare", "vs", "versus", "taqqosla"]):
            return self._compare()
        if self._has(ql, ["real-time", "realtime", "live", "jonli", "youtube"]):
            return self._realtime()
        if self._has(ql, ["help", "what can you", "yordam"]):
            return self._help()

        return ("Tushunmadim. Men quyidagilarni bilaman: mashina/odam soni, "
                "eng band zona, anomaliya/tirbandlik, xavf, zonalar taqqoslash. "
                "Masalan: 'How many cars at JCT_MainRoad?' yoki "
                "'Which zone is busiest?'")


    #  Yordamchi funksiyalar
    @staticmethod
    def _has(q: str, ks: list[str]) -> bool:
        return any(k in q for k in ks)

    def _has_data(self) -> bool:
        try:
            return database.query("SELECT COUNT(*) AS n FROM detections")[0]["n"] > 0
        except Exception:
            return False

    def _extract_zone(self, q: str) -> Optional[str]:
        """Savoldan zona nomini ajratadi (to'liq yoki qisqa nom)."""
        for z in self.zones:
            if z.lower() in q or z.split("_")[-1].lower() in q:
                return z
        return None


    #  SQL'ga asoslangan javoblar
    def _cars(self, zone: Optional[str], q: str) -> str:
        if zone:
            r = database.query(
                "SELECT SUM(car_count) AS total, MAX(car_count) AS peak "
                "FROM detections WHERE zone = ?", (zone,))[0]
            return (f"{zone} zonasida bugun jami {r['total'] or 0:,} mashina aniqlandi "
                    f"(eng yuqori bir vaqtda: {r['peak'] or 0} ta).")
        r = database.query("SELECT SUM(car_count) AS total FROM detections")[0]
        top = database.query(
            "SELECT zone, SUM(car_count) AS c FROM detections "
            "GROUP BY zone ORDER BY c DESC LIMIT 1")[0]
        return (f"Barcha zonalarda jami {r['total'] or 0:,} mashina aniqlandi. "
                f"Eng ko'pi {top['zone']} da ({top['c']:,} ta).")

    def _people(self, zone: Optional[str], q: str) -> str:
        if zone:
            r = database.query(
                "SELECT SUM(person_count) AS total, MAX(person_count) AS peak "
                "FROM detections WHERE zone = ?", (zone,))[0]
            return (f"{zone} zonasida bugun jami {r['total'] or 0:,} odam aniqlandi "
                    f"(eng yuqori bir vaqtda: {r['peak'] or 0} ta).")
        r = database.query("SELECT SUM(person_count) AS total FROM detections")[0]
        top = database.query(
            "SELECT zone, SUM(person_count) AS c FROM detections "
            "GROUP BY zone ORDER BY c DESC LIMIT 1")[0]
        return (f"Barcha zonalarda jami {r['total'] or 0:,} odam aniqlandi. "
                f"Eng ko'pi {top['zone']} da ({top['c']:,} ta).")

    def _peak(self, zone: Optional[str]) -> str:
        # Eng yuqori yo'l bandligi qaysi zona va qachon
        r = database.query(
            "SELECT zone, timestamp, road_occupancy FROM detections "
            "ORDER BY road_occupancy DESC LIMIT 1")[0]
        t = str(r["timestamp"])[11:16] if len(str(r["timestamp"])) > 15 else r["timestamp"]
        return (f"Eng yuqori yo'l bandligi {r['zone']} zonasida, soat {t} da bo'lgan "
                f"({r['road_occupancy']*100:.0f}% sig'imdan).")

    def _anomalies(self, zone: Optional[str]) -> str:
        n = database.query("SELECT COUNT(*) AS n FROM anomalies")[0]["n"]
        if n == 0:
            return "Bugun hech qanday anomaliya yoki tirbandlik aniqlanmadi."
        rows = database.query(
            "SELECT zone, timestamp, car_count, person_count, road_occupancy, "
            "anomaly_reason FROM anomalies ORDER BY road_occupancy DESC LIMIT 1")
        t = rows[0]
        ts = str(t["timestamp"])[11:16] if len(str(t["timestamp"])) > 15 else t["timestamp"]
        return (f"{n} ta anomaliya aniqlandi. Eng jiddiysi {t['zone']} da, soat {ts} da: "
                f"{t['car_count']} mashina ({t['road_occupancy']*100:.0f}% bandlik), "
                f"{t['person_count']} odam. Sabab: {t['anomaly_reason']}.")

    def _safety(self) -> str:
        # 85% dan yuqori bandlikka chiqgan zonalar
        rows = database.query(
            "SELECT DISTINCT zone FROM detections WHERE road_occupancy >= 0.85")
        if rows:
            zs = ", ".join(r["zone"] for r in rows)
            return (f"Xavf aniqlandi. Quyidagi zonalar 85%+ yo'l bandligiga yetgan: {zs}. "
                    f"Svetofor vaqtini ko'rib chiqish va operatorni ogohlantirish tavsiya etiladi.")
        return "Barcha yo'llar bugun 85% tirbandlik chegarasidan past qoldi."

    def _compare(self) -> str:
        rows = database.query(
            "SELECT zone, SUM(car_count) AS cars, SUM(person_count) AS people "
            "FROM detections GROUP BY zone ORDER BY cars DESC")
        lines = [f"{r['zone']}: {r['cars']:,} mashina, {r['people']:,} odam" for r in rows]
        return "Zonalar taqqoslash:\n" + "\n".join(lines)

    def _realtime(self) -> str:
        """Real-time detection log'idan (youtube_detect.py) ma'lumot beradi."""
        try:
            n = database.query("SELECT COUNT(*) AS n FROM detection_log")[0]["n"]
        except Exception:
            n = 0
        if n == 0:
            return ("Real-time jurnalida hali yozuv yo'q. `youtube_detect.py` ni "
                    "ishga tushiring — u jonli detection natijalarini bazaga yozadi.")
        r = database.query(
            "SELECT AVG(car_count) AS ac, AVG(person_count) AS ap, "
            "MAX(car_count) AS mc, MAX(person_count) AS mp FROM detection_log")[0]
        return (f"Real-time jurnalida {n} ta yozuv bor (youtube_detect.py'dan). "
                f"O'rtacha kadrda {r['ac']:.1f} mashina, {r['ap']:.1f} odam; "
                f"eng ko'p {r['mc']} mashina, {r['mp']} odam bir vaqtda.")

    def _forecast(self, zone: Optional[str]) -> str:
        z = zone or (self.zones[0] if self.zones else None)
        f = self.forecasts.get(z) if z else None
        if not f:
            # Forecast ma'lumoti yo'q bo'lsa, bazadagi o'rtachadan oddiy baho
            if not z:
                return "Bashorat uchun zona aniqlanmadi."
            r = database.query(
                "SELECT AVG(car_count) AS avg_c, MAX(car_count) AS max_c "
                "FROM detections WHERE zone = ?", (z,))[0]
            return (f"{z} uchun: o'rtacha {r['avg_c']:.0f} mashina, eng yuqori {r['max_c']} ta. "
                    f"(To'liq bashorat uchun dashboard'da pipeline'ni ishga tushiring.)")
        risk = ("tirbandlik kutilmoqda (>85%)" if f.get("congestion_predicted")
                else "tirbandlik kutilmayapti")
        return (f"{z} bashorati: keyingi soatда ~{f.get('car_predicted_peak')} mashina "
                f"(sig'im {f.get('road_capacity')}), ~{f.get('person_predicted_peak')} odam, "
                f"{risk}.")

    def _help(self) -> str:
        return ("Men database'ga ulangan trafik chatbotman. So'rang: mashina/odam soni "
                "(zona bo'yicha ham), eng band zona va vaqt, anomaliya/tirbandlik, "
                "xavf darajasi, zonalar taqqoslash, bashorat. "
                "Masalan: 'How many cars at MALL_CarParkIn?', 'Which zone is busiest?', "
                "'Compare zones', 'Any congestion?'")


if __name__ == "__main__":
    # Test: avval bazani to'ldiramiz, keyin savollar beramiz
    import synthetic_data
    database.init_db()
    df = synthetic_data.generate_traffic_timeseries()
    database.save_dataframe(df)

    bot = TrafficChatbot()
    for q in ["How many cars in total?",
              "How many cars at JCT_MainRoad?",
              "Which zone is busiest?",
              "Compare zones",
              "Is it safe?",
              "How many people at HUB_StationFwd?"]:
        print(f"USER: {q}\n BOT: {bot.ask(q)}\n")