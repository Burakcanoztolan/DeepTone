
import os
import librosa
import numpy as np
import pandas as pd
import opendatasets as od
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import create_hybrid_model

# --- 1. VERİ SETLERİNİ İNDİR ---
# TESS İndir
if not os.path.exists("./toronto-emotional-speech-set-tess"):
    od.download("https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess")

# RAVDESS İndir (YENİ!)
if not os.path.exists("./ravdess-emotional-speech-audio"):
    od.download("https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio")

DATA_PATH_TESS = "/content/toronto-emotional-speech-set-tess/TESS Toronto emotional speech set data"
DATA_PATH_RAVDESS = "/content/ravdess-emotional-speech-audio"

MODEL_SAVE_PATH = "duygu_modeli.h5"
LABELS_SAVE_PATH = "etiketler.npy"
SCALER_SAVE_PATH = "scaler.save"

# --- AUGMENTATION ---
def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)

# --- VERİ İŞLEME VE BİRLEŞTİRME ---
data = []
print("Veriler taranıyor (TESS + RAVDESS)...")

# 1. TESS VERİLERİNİ OKU
print("TESS işleniyor...")
for dizin, alt_dizinler, dosyalar in os.walk(DATA_PATH_TESS):
    for dosya in dosyalar:
        if dosya.endswith(".wav"):
            # TESS formatı: OAF_happy.wav -> 'happy'
            duygu = dosya.split('_')[-1].split('.')[0].lower()
            dosya_tam_yolu = os.path.join(dizin, dosya)
            try:
                audio, sr = librosa.load(dosya_tam_yolu, res_type='kaiser_fast')
                # TESS verisini ekle (Augmentation ile)
                data.append([np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0), duygu])
                data.append([np.mean(librosa.feature.mfcc(y=noise(audio), sr=sr, n_mfcc=40).T, axis=0), duygu])
            except: pass

# 2. RAVDESS VERİLERİNİ OKU (YENİ!)
print("RAVDESS işleniyor...")
# RAVDESS Duygu Haritası (Sayı -> Yazı)
ravdess_map = {
    '01': 'neutral', '02': 'neutral', # Calm'ı da Neutral yapıyoruz
    '03': 'happy', '04': 'sad', '05': 'angry',
    '06': 'fear', '07': 'disgust', '08': 'surprise'
}

for dizin, alt_dizinler, dosyalar in os.walk(DATA_PATH_RAVDESS):
    for dosya in dosyalar:
        if dosya.endswith(".wav"):
            try:
                # Dosya ismi örn: 03-01-06-01-01-01-01.wav (3. parça '06' yani fear)
                parcalar = dosya.split('-')
                if len(parcalar) > 2:
                    duygu_kodu = parcalar[2]
                    if duygu_kodu in ravdess_map:
                        duygu = ravdess_map[duygu_kodu]

                        # TESS ile etiket uyumu sağlamak için 'surprise' -> 'pleasant_surprise' düzeltmesi
                        if duygu == 'surprise': duygu = 'pleasant_surprise'
                        if duygu == 'fear': duygu = 'fear' # TESS'te klasör adı neyse o olmalı

                        dosya_tam_yolu = os.path.join(dizin, dosya)
                        audio, sr = librosa.load(dosya_tam_yolu, res_type='kaiser_fast')

                        # RAVDESS verisini ekle (Sadece orjinal ve gürültülü)
                        data.append([np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0), duygu])
                        data.append([np.mean(librosa.feature.mfcc(y=noise(audio), sr=sr, n_mfcc=40).T, axis=0), duygu])
            except: pass

df = pd.DataFrame(data, columns=['Ozellikler', 'Duygu_Etiketi'])
print(f"Toplam Birleştirilmiş Veri Sayısı: {len(df)}")

# --- HAZIRLIK ---
X = np.array(df['Ozellikler'].tolist())
y = np.array(df['Duygu_Etiketi'].tolist())

# Scaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_SAVE_PATH)

# Etiketleme
le = LabelEncoder()
y_encoded = to_categorical(le.fit_transform(y))
np.save(LABELS_SAVE_PATH, le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# --- EĞİTİM ---
print("Model iki veri setiyle eğitiliyor...")
model = create_hybrid_model(input_shape=(40, 1), num_classes=y_encoded.shape[1])

from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

history = model.fit(
    X_train_cnn, y_train,
    batch_size=64, # Veri çok olduğu için batch arttı
    epochs=50,
    validation_data=(X_test_cnn, y_test),
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Grafikler
plt.figure(figsize=(10,4))
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Test')
plt.title(f"Model Başarısı (Veri Sayısı: {len(df)})")
plt.legend()
plt.savefig("basari_grafigi.png")
plt.show()

y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.savefig("confusion_matrix.png")
plt.show()
