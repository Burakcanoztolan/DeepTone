
import os
import librosa
import numpy as np
import pandas as pd
import opendatasets as od
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from model import create_cnn_model # model.py dosyasından fonksiyonu çektik

# --- AYARLAR ---
# Eğer veriyi daha önce indirdiysen tekrar indirmesin
if not os.path.exists("./toronto-emotional-speech-set-tess"):
    od.download("https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess")

DATA_PATH = "/content/toronto-emotional-speech-set-tess/TESS-Toronto-emotional-speech-set-data"
MODEL_SAVE_PATH = "duygu_modeli.h5"
LABELS_SAVE_PATH = "etiketler.npy"

# --- FONKSİYONLAR ---
def özellik_cikar(dosya_yolu):
    audio, sample_rate = librosa.load(dosya_yolu, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

# --- 1. VERİ YÜKLEME ---
print("Veriler taranıyor...")
data = []
for dizin, alt_dizinler, dosyalar in os.walk(DATA_PATH):
    for dosya in dosyalar:
        if dosya.endswith(".wav"):
            duygu = dosya.split('_')[-1].split('.')[0] # Dosya isminden duygu al
            dosya_tam_yolu = os.path.join(dizin, dosya)
            data_ozellikleri = özellik_cikar(dosya_tam_yolu)
            data.append([data_ozellikleri, duygu])

df = pd.DataFrame(data, columns=['Ozellikler', 'Duygu_Etiketi'])
print(f"Veri seti hazır: {len(df)} adet ses dosyası.")

# --- 2. VERİ ÖN İŞLEME ---
X = np.array(df['Ozellikler'].tolist())
y = np.array(df['Duygu_Etiketi'].tolist())

le = LabelEncoder()
y_encoded = to_categorical(le.fit_transform(y))

# Etiket isimlerini kaydedelim (Serve dosyasında lazım olacak)
np.save(LABELS_SAVE_PATH, le.classes_)
print("Etiketler kaydedildi.")

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# CNN için boyutlandırma (Adet, 40, 1)
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# --- 3. MODEL EĞİTİMİ (CNN) ---
print("Model eğitiliyor...")
model = create_cnn_model(input_shape=(40, 1), num_classes=y_encoded.shape[1])
history = model.fit(X_train_cnn, y_train, batch_size=32, epochs=50, validation_data=(X_test_cnn, y_test), verbose=1)

# Modeli Kaydet
model.save(MODEL_SAVE_PATH)
print(f"Model başarıyla kaydedildi: {MODEL_SAVE_PATH}")

# --- 4. GRAFİKLER VE RAPORLAMA ---
# Başarı Grafiği
plt.figure(figsize=(10,4))
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Test')
plt.title("Model Başarısı")
plt.legend()
plt.savefig("basari_grafigi.png") # Resmi kaydet
plt.show()

# Random Forest Karşılaştırması
X_train_rf = X_train.reshape(X_train.shape[0], -1)
X_test_rf = X_test.reshape(X_test.shape[0], -1)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf, y_train) # Random Forest eğit
rf_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(rf_model.predict(X_test_rf), axis=1))
cnn_acc = model.evaluate(X_test_cnn, y_test, verbose=0)[1]

print(f"Random Forest Başarı: %{rf_acc*100:.2f}")
print(f"CNN Başarı: %{cnn_acc*100:.2f}")

# Confusion Matrix
y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
