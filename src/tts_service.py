
import os
import torch
import logging
from threading import Lock

logger = logging.getLogger(__name__)

class TTSService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TTSService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        
        self.lock = Lock()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = True
        
        # Model yolları (Varsayılan veya Fine-Tuned)
        # Bu yollar kullanıcı modeli indirip koyduğunda aktif olacak
        self.checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../active_models/xtts_v2")
        self.os_path = os.getcwd()

    def load_model(self):
        """
        Modeli hafızaya yükler (Singleton)
        Otomatik olarak Base XTTS v2 modelini indirir.
        """
        if self.model is not None:
            return True

        logger.info(f"[TTS] Model başlatılıyor... Cihaz: {self.device}")
        try:
            # === MONKEYPATCH BAŞLANGIÇ ===
            # Windows/TorchCodec hatasını aşmak için torchaudio.load'u librosa ile değiştiriyoruz
            import torchaudio
            import librosa
            import torch
            
            def librosa_load_wrapper(filepath, **kwargs):
                """Torchaudio.load yerine Librosa kullanan wrapper"""
                # filepath path-like veya string olabilir
                path = str(filepath)
                # Orijinal wav'ı oku (sr=None ile orijinal hızı koru)
                wav, sr = librosa.load(path, sr=None)
                # Tensor'a çevir: (Channels, Time) formatına getir
                # Librosa (Time,) döner (Mono), Torchaudio (C, T) ister
                tensor = torch.tensor(wav).unsqueeze(0) 
                return tensor, sr

            logger.warning("[TTS] TorchCodec hatasını önlemek için 'torchaudio.load' yamalanıyor...")
            torchaudio.load = librosa_load_wrapper
            # === MONKEYPATCH BİTİŞ ===

            from TTS.api import TTS
            
            # Base Model (Otomatik İndirme)
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            
            logger.info("[TTS] Base XTTS v2 modeli başarıyla yüklendi!")
            return True
            
        except ImportError:
            logger.error("[TTS] 'coqui-tts' yüklü değil. Veri Mühendisiniz ile görüşün.")
            return False
        except Exception as e:
            logger.error(f"[TTS] Yükleme hatası: {e}")
            return False

    def generate_audio(self, text, output_file, speaker_wav=None, language="tr"):
        """
        Metni sese çevirir (Voice Cloning).
        """
        with self.lock:
            if not self.model:
                if not self.load_model():
                    return False
            
            try:
                logger.info(f"[TTS] Ses üretiliyor (Cloning): '{text[:20]}...'")
                
                # Speaker Wav Kontrolü ve Sanitizasyon
                if not speaker_wav or not os.path.exists(speaker_wav):
                    # Fallback
                    fallback_wav = os.path.join(self.os_path, "data/kayit.wav")
                    if os.path.exists(fallback_wav):
                        speaker_wav = fallback_wav
                        logger.info(f"[TTS] Referans ses kullanılıyor: {speaker_wav}")
                    else:
                        logger.error("[TTS] Referans ses dosyası bulunamadı!")
                        return False
                
                # Windows ve Torchaudio Uyumluluğu İçin: Sesi Temizle (Re-encode)
                try:
                    import librosa
                    import soundfile as sf
                    import uuid
                    
                    # Geçici dosya yolu
                    temp_speaker_wav = os.path.join(self.os_path, f"temp_speaker_{uuid.uuid4()}.wav")
                    
                    # Librosa ile yükle (Her şeyi açar)
                    y, sr = librosa.load(speaker_wav, sr=22050) # TTS genelde 22050 sever
                    
                    # Soundfile ile temiz WAV olarak kaydet
                    sf.write(temp_speaker_wav, y, sr)
                    
                    # Artık referansımız bu temiz dosya
                    speaker_wav = temp_speaker_wav
                    logger.info(f"[TTS] Referans ses temizlendi: {speaker_wav}")
                    
                except Exception as e:
                    logger.warning(f"[TTS] Ses temizleme başarısız (Orijinal kullanılacak): {e}")

                # Inference (High-Level API)
                self.model.tts_to_file(
                    text=text,
                    file_path=output_file,
                    speaker_wav=speaker_wav,
                    language=language,
                    split_sentences=True
                )
                
                logger.info(f"[TTS] Dosya kaydedildi: {output_file}")
                return True

            except Exception as e:
                logger.error(f"[TTS] Inference hatası: {e}")
                return False
