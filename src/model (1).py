
import tensorflow as tf
from tensorflow.keras.models import Sequential # <--- Unutulan kısım burasıydı!
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, BatchNormalization

def create_hybrid_model(input_shape, num_classes):
    model = Sequential()

    # CNN Kısmı (Özellik Çıkarıcı)
    # BatchNormalization ekledik, eğitimi hızlandırır
    model.add(Conv1D(128, kernel_size=5, padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(64, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # LSTM Kısmı (Zaman Analizi)
    # CNN çıkışını LSTM'e veriyoruz. Flatten yapmıyoruz!
    # return_sequences=True diyoruz çünkü bir sonraki de LSTM katmanı
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32)) # Son karar için tek vektör

    # Sınıflandırma Kısmı
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
