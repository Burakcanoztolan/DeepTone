
import gradio as gr
import librosa
import numpy as np
import tensorflow as tf
import joblib # Scaler'ı yüklemek için

MODEL_PATH = "duygu_modeli.h5"
LABELS_PATH = "etiketler.npy"
SCALER_PATH = "scaler.save" # Scaler dosyamız

print("Sistem yükleniyor...")
model = tf.keras.models.load_model(MODEL_PATH)
etiketler = np.load(LABELS_PATH, allow_pickle=True)
scaler = joblib.load(SCALER_PATH) # Eğittiğimiz matematiği geri çağırıyoruz

def tahmin_et(ses_dosyasi):
    if ses_dosyasi is None:
        return "Lütfen ses dosyası yükleyin."

    # Sesi işle
    audio, sample_rate = librosa.load(ses_dosyasi, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_raw = np.mean(mfccs.T, axis=0)

    # --- KRİTİK ADIM: NORMALİZASYON ---
    # Gelen sesi de aynı şekilde sıkıştırıyoruz
    # (reshape(1, -1) tek bir örnek olduğu için gerekli)
    mfccs_std = scaler.transform(mfccs_scaled_raw.reshape(1, -1))

    # Modele uygun boyuta getir
    veri = mfccs_std.reshape(1, 40, 1)

    # Tahmin
    olasiliklar = model.predict(veri)
    tahmin_index = np.argmax(olasiliklar, axis=1)[0]
    sonuc = etiketler[tahmin_index]

    return f"Sonuç: {sonuc.upper()}"

interface = gr.Interface(
    fn=tahmin_et,
    inputs=gr.Audio(type="filepath", label="Test Et"),
    outputs="text",
    title="Hibrit DeepTone Analizi"
)

if __name__ == "__main__":
    interface.launch(share=True)
