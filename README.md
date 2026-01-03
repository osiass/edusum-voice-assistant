# EduSum Ses Asistanı (The Voice)

> **Yeni Nesil Eğitim Odaklı Sesli Etkileşim Motoru**
> _Gemini 2.5 Flash ve Coqui XTTS v2 ile Güçlendirilmiş Gerçek Zamanlı Pipeline_

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Model](https://img.shields.io/badge/LLM-Gemini_2.5_Flash-purple)
![TTS](https://img.shields.io/badge/TTS-Coqui_XTTS_v2-green)
![Processing](https://img.shields.io/badge/Audio-Librosa-yellow)

## 🏗️ Teknik Mimari ve Modeller

Bu modül, sesi metne, metni bilgiye, bilgiyi tekrar insansı sese dönüştüren 4 aşamalı bir boru hattıdır (pipeline).

### 1. Beyin: Google Gemini 2.5 Flash
- **Model:** `gemini-2.5-flash` (En güncel, düşük gecikmeli sürüm).
- **Rolü:** RAG kütüphanesinden gelen ham ders notlarını, bir "Lise Öğretmeni" edasıyla işler.
- **Konfigürasyon:** `temperature=0.3` (Halüsinasyonu önlemek için düşük yaratıcılık), `max_tokens=500` (Kısa ve öz cevaplar).

### 2. Ses Üretimi: Coqui XTTS v2
- **Model:** `coqui/XTTS-v2`
- **Özellik:** **Zero-Shot Voice Cloning** (Sıfır Atışlı Ses Kopyalama).
- **Çalışma Prensibi:** Sisteme verilen sadece 3 saniyelik bir `wav` dosyasındaki (`speaker_reference`) tınıyı (timbre) ve tonu analiz eder. Üretilen cevabı o kişinin sesiyle okur.

### 3. Ses İşleme: Librosa & SoundFile
- **Kütüphaneler:** `librosa`, `soundfile`, `numpy`.
- **Görev:** Gelen farklı formatlardaki (mp3, m4a, ogg) sesleri standart `22050Hz Mono WAV` formatına dönüştürür ve gürültüden arındırır.

---

## 🚀 Akış Şeması (Workflow)

```mermaid
graph LR
    A[🎤 Kullanıcı Sesi] -- STT (Whisper) --> B(Metin);
    B -- Sorgu --> C[📚 EduSum RAG];
    C -- Bağlam (Context) --> D[🧠 Gemini 2.5 Flash];
    D -- Cevap Metni --> E[🗣️ Coqui XTTS v2];
    E -- Klonlanmış Ses --> F[🎧 Çıktı Audio];
```

---

## ⚙️ Prompt Mühendisliği (System Prompt)
Sistemin kullandığı LLM, "Doğruluk" ve "Pedagoji" odaklı katı kurallarla yönetilir:
- **Asla Uydurma (No Hallucination):** Sadece RAG'dan gelen PDF parçalarını kullanır.
- **Format Kontrolü:** "Nedir?" sorularına tanım cümlesiyle, "Yorumla" sorularına metaforla başlar.
- **Kişilik:** Bir Lise Ders Asistanı gibi resmi ama anlaşılır konuşur.

---

## 💻 Kullanım

```bash
# Servisi Başlat (FastAPI - Uvicorn)
python run_voice_prod.py
```

**Örnek İstek (cURL):**
```bash
curl -X POST "http://localhost:8001/ask" \
     -F "audio=@soru.wav" \
     -F "ref_audio=@ahmet_hoca.wav"
```

---
*Eğitimin yeni sesi.*
