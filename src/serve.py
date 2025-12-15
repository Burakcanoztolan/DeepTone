
import gradio as gr
import librosa
import numpy as np
import tensorflow as tf

# 1. Eğitilmiş Modeli ve Etiketleri Yükle
MODEL_PATH = "duygu_modeli.h5"
LABELS_PATH = "etiketler.npy"

print("Model yükleniyor...")
model = tf.keras.models.load_model(MODEL_PATH)
etiketler = np.load(LABELS_PATH, allow_pickle=True)

# 2. Tahmin Fonksiyonu
def tahmin_et(ses_dosyasi):
    """
    Kullanıcının yüklediği sesi alır, işler ve sonucu söyler.
    """
    if ses_dosyasi is None:
        return "Lütfen ses dosyası yükleyin."

    # Sesi işle (train.py'daki ile aynı mantıkta olmalı)
    audio, sample_rate = librosa.load(ses_dosyasi, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)

    # Modele uygun boyuta getir (1, 40, 1)
    veri = mfccs_scaled.reshape(1, 40, 1)

    # Tahmin yap
    olasiliklar = model.predict(veri)
    tahmin_index = np.argmax(olasiliklar, axis=1)[0]
    sonuc = etiketler[tahmin_index]

    return f"Bu seste algılanan duygu: {sonuc.upper()}"

# 3. Gradio Arayüzü
interface = gr.Interface(
    fn=tahmin_et,
    inputs=gr.Audio(type="filepath", label="Sesinizi Kaydedin veya Yükleyin"),
    outputs="text",
    title="Ses Duygu Analizi Sistemi",
    description="Eğitilen CNN modeli ses tonunuzdan duygu durumunuzu tahmin eder.",
    examples=[] # İstersen buraya örnek dosya yolları ekleyebilirsin
)

if __name__ == "__main__":
    interface.launch(share=True)
