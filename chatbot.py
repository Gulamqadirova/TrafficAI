from __future__ import annotations
import os
import json
from typing import Optional

import database


#  Claude API sozlash
def _get_api_key() -> Optional[str]:
    """API kalitini Streamlit secrets yoki muhit o'zgaruvchisidan oladi."""
    # 1. Streamlit secrets (deploy qilganda)
    try:
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        pass
    # 2. Muhit o'zgaruvchisi (local)
    return os.environ.get("ANTHROPIC_API_KEY")


def _build_db_context() -> str:
    """Database'dan joriy holat ma'lumotini oladi — LLM'ga kontekst."""
    try:
        det = database.query(
            "SELECT zone, SUM(car_count) AS cars, SUM(person_count) AS people, "
            "MAX(road_occupancy) AS max_occ FROM detections GROUP BY zone")
        an = database.query(
            "SELECT COUNT(*) AS n FROM anomalies")[0]["n"]
        top = max(det, key=lambda r: r["cars"]) if det else {}
        rt = database.query(
            "SELECT COUNT(*) AS n, AVG(car_count) AS ac, AVG(person_count) AS ap "
            "FROM detection_log")[0]

        lines = ["=== TIZIM HOLATI (traffic.db dan) ==="]
        for r in det:
            lines.append(
                f"  {r['zone']}: {r['cars']} ta mashina, {r['people']} ta odam, "
                f"max bandlik {r['max_occ']*100:.0f}%")
        lines.append(f"  Anomaliyalar: {an} ta")
        if top:
            lines.append(f"  Eng band zona: {top['zone']} ({top['cars']} mashina)")
        if rt['n'] > 0:
            lines.append(
                f"  Real-time log: {rt['n']} kadr, o'rtacha "
                f"{rt['ac']:.1f} mashina / {rt['ap']:.1f} odam per kadr")
        return "\n".join(lines)
    except Exception as e:
        return f"(Database ma'lumoti olishda xato: {e})"


SYSTEM_PROMPT = """Siz TPBI (Traffic & Pedestrian Business Intelligence) 
tizimining AI yordamchisisiz. Siz trafik va piyodalar bo'yicha savollarni 
qisqa, aniq va professional tarzda javob berasiz.

Qoidalar:
- Faqat berilgan database ma'lumotlariga asoslaning, o'zingizdan raqam 
  to'qimang.
- Javoblar qisqa bo'lsin (2-4 jumla). Agar savol aniq bo'lmasa, 
  aniqlashtiring.
- Ingliz yoki o'zbek tilida savol bo'lsa, shu tilda javob bering.
- Agar ma'lumot etarli bo'lmasa, shuni aytib, qo'shimcha tavsiya bering."""


class TrafficChatbot:
    """
    AI trafik chatboti — Claude API (LLM) + SQL zaxira.

    Agar ANTHROPIC_API_KEY mavjud bo'lsa:
      savol → DB kontekst → Claude API → tabiiy til javob

    Agar API key yo'q bo'lsa:
      savol → niyat tahlili → SQL so'rov → javob (eski usul)
    """

    def __init__(self, df=None, kpis=None, anomalies=None, forecasts=None):
        database.init_db()
        self.forecasts = forecasts or {}
        self.api_key = _get_api_key()
        self.use_llm = bool(self.api_key)
        self.conversation_history = []   # suhbat xotirasi (LLM uchun)

        try:
            rows = database.query("SELECT DISTINCT zone FROM detections")
            self.zones = [r["zone"] for r in rows]
        except Exception:
            self.zones = []

    def ask(self, q: str) -> str:
        if not q or not q.strip():
            return self._help()

        if self.use_llm:
            return self._ask_llm(q)
        else:
            return self._ask_sql(q)

    # --------------------------------------------------------------------- #
    #  LLM yo'l: Claude API
    # --------------------------------------------------------------------- #
    def _ask_llm(self, q: str) -> str:
        """Claude API orqali javob — database konteksti bilan."""
        try:
            import urllib.request
            db_context = _build_db_context()
            user_content = f"{db_context}\n\nSavol: {q}"

            # Suhbat xotirasiga qo'shamiz
            self.conversation_history.append(
                {"role": "user", "content": user_content})

            payload = json.dumps({
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 400,
                "system": SYSTEM_PROMPT,
                "messages": self.conversation_history[-10:],  # oxirgi 10 xabar
            }).encode()

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                method="POST")

            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())

            answer = data["content"][0]["text"].strip()
            # Faqat javobni (DB kontekstsiz) xotiraga saqlaymiz
            self.conversation_history.append(
                {"role": "assistant", "content": answer})
            return answer

        except Exception as e:
            # API ishlamasa, SQL fallback'ga o'tamiz
            self.use_llm = False
            fallback = self._ask_sql(q)
            return f"{fallback}\n\n*(Claude API: {e} — SQL rejimga o'tildi)*"

    # --------------------------------------------------------------------- #
    #  SQL zaxira yo'l (API keysiz ishlaydi)
    # --------------------------------------------------------------------- #
    def _ask_sql(self, q: str) -> str:
        ql = q.lower().strip()
        if not self._has_data():
            return ("Database bo'sh. Pipeline'ni ishga tushiring.")
        zone = self._extract_zone(ql)
        if self._has(ql, ["anomal", "incident", "congest", "jam", "tirband"]):
            return self._anomalies(zone)
        if self._has(ql, ["forecast", "predict", "bashorat"]):
            return self._forecast(zone)
        if self._has(ql, ["peak", "busiest", "most", "eng ko", "eng band"]):
            return self._peak(zone)
        if self._has(ql, ["car", "mashina", "avtomobil"]):
            return self._cars(zone, ql)
        if self._has(ql, ["people", "person", "pedestrian", "odam", "piyoda"]):
            return self._people(zone, ql)
        if self._has(ql, ["safe", "risk", "xavf"]):
            return self._safety()
        if self._has(ql, ["compare", "vs", "taqqosla"]):
            return self._compare()
        if self._has(ql, ["real-time", "live", "jonli", "youtube"]):
            return self._realtime()
        if self._has(ql, ["help", "yordam"]):
            return self._help()
        return ("Tushunmadim. So'rang: mashina/odam soni, anomaliya, "
                "eng band zona, xavf, taqqoslash. "
                "Masalan: 'How many cars at JCT_MainRoad?'")

    # --------------------------------------------------------------------- #
    #  Yordamchi
    # --------------------------------------------------------------------- #
    @staticmethod
    def _has(q, ks): return any(k in q for k in ks)

    def _has_data(self):
        try:
            return database.query(
                "SELECT COUNT(*) AS n FROM detections")[0]["n"] > 0
        except Exception:
            return False

    def _extract_zone(self, q):
        for z in self.zones:
            if z.lower() in q or z.split("_")[-1].lower() in q:
                return z
        return None

    def _cars(self, zone, q):
        if zone:
            r = database.query(
                "SELECT SUM(car_count) AS t, MAX(car_count) AS p "
                "FROM detections WHERE zone=?", (zone,))[0]
            return (f"{zone}: {r['t'] or 0:,} mashina (peak: {r['p'] or 0}).")
        r = database.query("SELECT SUM(car_count) AS t FROM detections")[0]
        top = database.query(
            "SELECT zone, SUM(car_count) AS c FROM detections "
            "GROUP BY zone ORDER BY c DESC LIMIT 1")[0]
        return (f"Jami {r['t'] or 0:,} mashina. "
                f"Eng ko'p: {top['zone']} ({top['c']:,}).")

    def _people(self, zone, q):
        if zone:
            r = database.query(
                "SELECT SUM(person_count) AS t, MAX(person_count) AS p "
                "FROM detections WHERE zone=?", (zone,))[0]
            return (f"{zone}: {r['t'] or 0:,} odam (peak: {r['p'] or 0}).")
        r = database.query(
            "SELECT SUM(person_count) AS t FROM detections")[0]
        top = database.query(
            "SELECT zone, SUM(person_count) AS c FROM detections "
            "GROUP BY zone ORDER BY c DESC LIMIT 1")[0]
        return (f"Jami {r['t'] or 0:,} odam. "
                f"Eng ko'p: {top['zone']} ({top['c']:,}).")

    def _peak(self, zone):
        r = database.query(
            "SELECT zone, timestamp, road_occupancy FROM detections "
            "ORDER BY road_occupancy DESC LIMIT 1")[0]
        t = str(r["timestamp"])[11:16]
        return (f"Eng yuqori bandlik: {r['zone']}, soat {t} "
                f"({r['road_occupancy']*100:.0f}%).")

    def _anomalies(self, zone):
        n = database.query("SELECT COUNT(*) AS n FROM anomalies")[0]["n"]
        if n == 0:
            return "Anomaliya topilmadi."
        r = database.query(
            "SELECT zone, timestamp, car_count, road_occupancy, anomaly_reason "
            "FROM anomalies ORDER BY road_occupancy DESC LIMIT 1")[0]
        ts = str(r["timestamp"])[11:16]
        return (f"{n} anomaliya. Eng jiddiysi: {r['zone']}, {ts}, "
                f"{r['car_count']} mashina ({r['road_occupancy']*100:.0f}%). "
                f"Sabab: {r['anomaly_reason']}.")

    def _safety(self):
        rows = database.query(
            "SELECT DISTINCT zone FROM detections WHERE road_occupancy>=0.85")
        if rows:
            return (f"XAVF: {', '.join(r['zone'] for r in rows)} "
                    f"85%+ bandlikka yetgan.")
        return "Barcha yo'llar 85% chegarasidan past."

    def _compare(self):
        rows = database.query(
            "SELECT zone, SUM(car_count) AS c, SUM(person_count) AS p "
            "FROM detections GROUP BY zone ORDER BY c DESC")
        return "Taqqoslash:\n" + "\n".join(
            f"  {r['zone']}: {r['c']:,} mashina, {r['p']:,} odam"
            for r in rows)

    def _forecast(self, zone):
        z = zone or (self.zones[0] if self.zones else None)
        if not z:
            return "Zona aniqlanmadi."
        f = self.forecasts.get(z)
        if not f:
            r = database.query(
                "SELECT AVG(car_count) AS a, MAX(car_count) AS m "
                "FROM detections WHERE zone=?", (z,))[0]
            return (f"{z}: o'rtacha {r['a']:.0f}, max {r['m']} mashina.")
        risk = "tirbandlik kutilmoqda" if f.get("congestion_predicted") \
            else "tirbandlik kutilmaydi"
        return (f"{z}: ~{f.get('car_predicted_peak')} mashina, "
                f"~{f.get('person_predicted_peak')} odam. {risk}.")

    def _realtime(self):
        try:
            r = database.query(
                "SELECT COUNT(*) AS n, AVG(car_count) AS ac, "
                "AVG(person_count) AS ap FROM detection_log")[0]
        except Exception:
            return "Real-time jurnal mavjud emas."
        if r["n"] == 0:
            return ("Real-time jurnalida yozuv yo'q. "
                    "youtube_detect.py ni ishga tushiring.")
        return (f"Real-time: {r['n']} kadr, o'rtacha "
                f"{r['ac']:.1f} mashina / {r['ap']:.1f} odam.")

    def _help(self) -> str:
        mode = "Claude API (LLM)" if self.use_llm else "SQL (fallback)"
        return (f"Men TPBI trafik chatbotiman [{mode}]. "
                "So'rang: mashina/odam soni, anomaliya, eng band zona, "
                "xavf, taqqoslash, bashorat, real-time. "
                "Masalan: 'Compare zones', 'Any congestion?'")

    @property
    def mode(self) -> str:
        return "Claude API (LLM)" if self.use_llm else "SQL keyword matching"


if __name__ == "__main__":
    import synthetic_data
    database.init_db()
    df = synthetic_data.generate_traffic_timeseries()
    database.save_dataframe(df)
    bot = TrafficChatbot()
    print(f"Rejim: {bot.mode}")
    for q in ["How many cars in total?", "Compare zones", "Any anomalies?"]:
        print(f"Q: {q}\nA: {bot.ask(q)}\n")