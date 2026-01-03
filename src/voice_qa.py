
import os
import logging
import google.generativeai as genai
from api.rag.system import RAGSystem
from api.voice.tts_service import TTSService

logger = logging.getLogger(__name__)

class VoiceQAOrchestrator:
    def __init__(self):
        self.rag = RAGSystem()
        self.tts = TTSService()
        
        # API Key Config
        # Views.py GEMINI_API_KEY kullanıyor, onu öncelikli alalım
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            logger.warning("[QA] GEMINI_API_KEY veya GOOGLE_API_KEY bulunamadı! LLM çalışmayabilir.")
        else:
            genai.configure(api_key=api_key)

    def ask(self, query, speaker_wav_path, output_audio_path):
        """
        Uçtan uca Soru-Cevap akışı:
        Query -> RAG -> Context -> LLM -> Text -> TTS -> Audio
        """
        
        # 1. RAG: Bilgi Getir
        logger.info(f"[QA] Soru: {query}")
        search_results = self.rag.search(query, top_k=3)
        
        context_str = ""
        if search_results:
            context_str = "\n".join([f"- {r['chunk']} (Skor: {r['score']:.2f})" for r in search_results])
        else:
            context_str = "Veri bulunamadı."
            
        # 2. LLM: Cevap Üret (Lise Ders Asistanı Modu)
        prompt = f"""
Sistem: Sen bir lise ders asistanısın.
Biyoloji, fizik, kimya, matematik, tarih, coğrafya, felsefe, din kültürü, edebiyat gibi
tüm derslerde gelen içerikleri doğru şekilde açıklayabilirsin.

Sana verilen "results" alanındaki metinler:
- Ders kitaplarından birebir alınmıştır
- Güvenilir ve doğrudur
- Birden fazla ders ve birden fazla parça (chunk) içerebilir

GÖREVİN:
1. Sadece verilen chunk içeriklerine dayanarak cevap ver.
2. Kullanıcının sorusuna UYGUN olan ders içeriğini esas al.
3. Alakasız derslerden gelen chunk’ları (soruya uymuyorsa) GÖRMEZDEN GEL.
4. Uygun olan chunk’lardan bilgileri birleştirerek tek, akıcı ve anlaşılır bir cevap üret.

CEVAP ÜRETİM KURALLARI:
- Soru “nedir?” ise:
  • Kısa ve net bir tanım yap
  • Ardından 2–4 cümlelik açıklama ekle
- Soru “açıklayınız / yorumlayınız” ise:
  • Paragraf halinde açıkla
- Soru “örnek veriniz” ise:
  • En az 1 somut örnek ver

ZORUNLU KURALLAR:
- En az 1 uygun chunk varsa CEVAP ÜRETMEK ZORUNDASIN
- “Bilmiyorum”, “yetersiz bilgi” gibi ifadeler YASAK
- Dış bilgi ekleme
- Uydurma yapma
- Markdown kullanma
- Akademik ama sade Türkçe kullan
- Aynı bilgiyi tekrar etme

SORU:
{query}

KAYNAK METİNLER (results):
{context_str}
"""
        
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            answer_text = response.text.strip()
        except Exception as e:
            logger.error(f"[QA] LLM Hatası: {e}")
            answer_text = "Özür dilerim, bir hata oluştu."

        logger.info(f"[QA] Cevap: {answer_text}")

        # 3. TTS: Ses Üret
        # Eğer cevap "Bulunamadı" ise belki ses üretmek istemeyiz? 
        # Ama kullanıcı "Bulunamadı" sesini duymalı.
        
        audio_success = self.tts.generate_audio(
            text=answer_text,
            output_file=output_audio_path,
            speaker_wav=speaker_wav_path,
            language="tr"
        )
        
        return {
            "text_answer": answer_text,
            "audio_path": output_audio_path if audio_success else None,
            "context_used": search_results
        }
