
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten

def create_cnn_model(input_shape, num_classes):
    """
    CNN Model yapısını oluşturur ve derlenmiş modeli döndürür.
    """
    model = Sequential()

    # 1. Konvolüsyon Bloğu
    model.add(Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    # Düzleştirme
    model.add(Flatten())

    # Tam Bağlantılı Katmanlar (Dense)
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3)) # Ezberlemeyi önler

    # Çıkış Katmanı
    model.add(Dense(num_classes, activation='softmax'))

    # Modeli derle
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
