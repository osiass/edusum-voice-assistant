# EduSum Ses Asistana (Voice)

> **Gercek Zamanli Ses Kopyalama ve Egitim Soru-Cevap Asistani**
> _Gemini 1.5 ve Coqui XTTS v2 guclendirilmis Baglam Duyarli Sesli Etkilesim hatti_

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Coqui TTS](https://img.shields.io/badge/Coqui-XTTS_v2-green)
![Gemini](https://img.shields.io/badge/LLM-Gemini_1.5_Flash-purple)

## Genel Bakis

**EduSum Voice**, EduSum ekosisteminin "agzi" ve "kulaklaridir". Ogrencileri dinler, sorularini (bozuk cumlelerle bile olsa) anlar, RAG kutuphanesine danisir ve **ogrencinin kendi sesiyle** (veya belirlenen bir ogretmen personasiyla) Gercek Zamanli Ses Kopyalama (Zero-Shot Voice Cloning) kullanarak cevap verir.

---

## Temel Ozellikler

### 1. Zero-Shot Ses Kopyalama
- **Coqui XTTS v2** temel modelini kullanir.
- 3 saniyelik referans sesi (`kayit.wav`) analiz ederek konusmacinin sesini gercek zamanli olarak kopyalar.
- Fine-tuning (ince ayar) gerektirmez.

### 2. "Lise Asistani" Personasi (Gemini 1.5)
- Lise Ders Asistani olarak davranmasi icin ozel olarak promptlanmis LLM.
- **Kati Kurallar:** Asla halusinasyon gormez (uydurmaz). Sadece saglanan RAG parcalarini kullanir.
- **Pedagojik Format:** "Nedir" sorularini tanimlarla, "Yorumlayiniz" sorularini benzetmelerle aciklar.

### 3. Dusuk Gecikmeli Hat (Pipeline)
- Optimize edilmis `VoiceQAOrchestrator` akisi yonetir: `STT (Sesi Yaziya Dokme) -> Erisim (Retrieval) -> Uretim (Generation) -> TTS (Yaziyi Sese Dokme)`.
- Daha akici etkilesimler icin asenkron isleme.

---

## Kullanim

### 1. API'yi Baslatin
```bash
python run_voice_prod.py
```

### 2. Istek Gonderin
```bash
curl -X POST "http://localhost:8001/ask" \
     -F "audio=@sorum.wav" \
     -F "ref_audio=@ses_ornegim.wav"
```

---

## Yapi

- `src/voice/tts_service.py`: Coqui TTS icin sarmalayici (wrapper) (`api_manual` ve `gpu` cikarimini destekler).
- `src/voice_qa.py`: LLM ve TTS'i birbirine baglayan orkestrator.
- `run_voice_prod.py`: FASTAPI giris noktasi.

---
*Egitimde ses getiren teknoloji.*
