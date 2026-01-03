
import os
import argparse
import logging
import csv
import warnings
import numpy as np
import librosa
import soundfile as sf
import whisper

# Python 3.13 Uyumluluk Güncellemesi:
# pydub (audioop bağımlılığı yüzünden) yerine librosa + soundfile kullanıyoruz.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_audio_vad(input_path, output_dir, top_db=30, min_sec=1.5, max_sec=12.0):
    """
    Librosa kullanarak sessizliğe göre sesi böler.
    XTTS v2 için optimum süreler hedeflenir.
    """
    logger.info(f"Ses yükleniyor (Librosa): {input_path}")
    
    try:
        # Sesi yükle (22050Hz - XTTS standardı)
        y, sr = librosa.load(input_path, sr=22050)
    except Exception as e:
        logger.error(f"Dosya okunamadı. Hata: {e}")
        return []

    logger.info(f"Ses yüklendi. Süre: {len(y)/sr:.2f} sn. Sessizlik taranıyor...")
    
    # Sessiz olmayan aralıkları bul (top_db: ne kadar sessizliğe tolerans var)
    intervals = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)
    
    os.makedirs(output_dir, exist_ok=True)
    processed_files = []
    
    logger.info(f"Bulunan parça sayısı: {len(intervals)}")
    
    for i, (start, end) in enumerate(intervals):
        duration = (end - start) / sr
        
        # Çok kısa veya çok uzun parçaları filtrele
        if duration < min_sec: continue
        # Çok uzunsa bölmek gerekir ama şimdilik sadece kaydediyoruz (ileri seviye VAD gerekebilir)
        
        chunk = y[start:end]
        
        # Dosya adı (Düzeltme: wavs/ öneki kaldırıldı, sadece dosya adı)
        filename = f"audio_{i:04d}.wav"
        full_path = os.path.join(output_dir, filename)
        
        # Klasör yapısını oluştur
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Kaydet
        sf.write(full_path, chunk, sr)
        processed_files.append(full_path)
        
    logger.info(f"Filtreleme sonrası toplam {len(processed_files)} parça kaydedildi.")
    return processed_files

def generate_metadata(processed_files, output_csv):
    """
    Ses dosyalarını Whisper ile yazıya döker.
    FFmpeg bağımlılığını aşmak için sesi Librosa ile yükleyip Whisper'a array olarak veriyoruz.
    """
    logger.info("Transkripsiyon başlatılıyor (Whisper Medium)...")
    
    # Modeli Yükle
    try:
        model = whisper.load_model("medium") # GPU varsa cuda kullanır
    except Exception as e:
        logger.error(f"Whisper yüklenemedi: {e}")
        return

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|') # XTTS formatı: path|text|speaker
        
        count = 0
        for wav_file in processed_files:
            try:
                # KRİTİK DÜZELTME: Whisper'ın ffmpeg kullanmasını engellemek için
                # Sesi librosa ile okuyup array olarak veriyoruz.
                audio_array, _ = librosa.load(wav_file, sr=16000) # Whisper 16k ister
                
                result = model.transcribe(audio_array, language="tr")
                text = result["text"].strip()
                
                # Çok kısa transkriptleri atla
                if len(text) < 5: continue
                
                # Göreceli yol hesapla
                # Beklenen: wavs/audio_xxxx.wav
                # wav_file: .../dataset/voice_data/wavs/audio_xxxx.wav
                filename = os.path.basename(wav_file)
                rel_path = f"wavs/{filename}"
                
                writer.writerow([rel_path, text, "TR_Speaker"])
                print(f"[{count+1}/{len(processed_files)}] {rel_path} -> {text[:40]}...")
                count += 1
                
            except Exception as e:
                logger.error(f"Hata ({wav_file}): {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XTTS v2 Dataset Hazırlayıcı (Librosa Version)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Giriş WAV dosyası")
    parser.add_argument("--output", "-o", type=str, default="dataset/voice_data", help="Çıktı klasörü")
    
    args = parser.parse_args()
    
    full_output_dir = os.path.join(os.getcwd(), args.output)
    wavs_dir = os.path.join(full_output_dir, "wavs")
    metadata_path = os.path.join(full_output_dir, "metadata.csv")
    
    print(f"Hedef Klasör: {full_output_dir}")
    
    files = split_audio_vad(args.input, wavs_dir)
    if files:
        generate_metadata(files, metadata_path)
        print(f"\n✅ Dataset hazır!\nVeri: {wavs_dir}\nManifest: {metadata_path}")
    else:
        print("❌ Hiç ses parçası çıkarılamadı.")
