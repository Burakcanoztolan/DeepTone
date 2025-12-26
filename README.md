🧠 DeepTone: Evrişimli Sinir Ağları (CNN) ile Akustik Duygu Tanıma Sistemi
==========================================================================

**DeepTone**, insan sesi sinyallerindeki (audio signals) gizli öznitelikleri analiz ederek, konuşmacının **duygusal durumunu (Affective State)** sınıflandıran, uçtan uca (end-to-end) bir Derin Öğrenme projesidir. Proje, özellikle **İnsan-Bilgisayar Etkileşimi (HCI)** ve **Duygusal Bilişim (Affective Computing)** alanlarında kullanılmak üzere tasarlanmıştır.

* * * * *

📑 İçindekiler
--------------

1. [Proje Özeti ve Literatür](#proje-özeti-ve-literatür)
2. [Veri Seti Özellikleri](#veri-seti-özellikleri)
3. [Metodoloji ve Teknik Mimari](#metodoloji-ve-teknik-mimari)
    * [Sinyal İşleme ve MFCC](#1-sinyal-işleme-ve-mfcc-mel-frequency-cepstral-coefficients)
    * [Model Topolojisi (1D-CNN)](#2-model-topolojisi-1d-cnn)
4. [Deneysel Kurulum](#deneysel-kurulum)
5. [Performans Analizi ve Sonuçlar](#performans-analizi-ve-sonuçlar)
6. [Kurulum ve Kullanım](#kurulum-ve-kullanım)
7. [Gelecek Çalışmalar](#gelecek-çalışmalar)
8. [Hazırlayan](#hazırlayan)
* * * * *

🎯 Proje Özeti ve Literatür
---------------------------

Duygu analizi genellikle metin tabanlı (NLP) yapılsa da, ses tonu, vurgu ve frekans değişimleri (prosodi) metnin içeremediği hayati sinyaller taşır. Bu çalışmada, ham ses verilerinden **spektral özniteliklerin** çıkarılması ve bu özniteliklerin **Evrişimli Sinir Ağları (CNN)** ile işlenmesi hedeflenmiştir.

Proje, geleneksel makine öğrenmesi yöntemlerinin (SVM, Random Forest) aksine, öznitelik mühendisliğini (feature engineering) minimize ederek, sesin yerel ve zamansal özelliklerini otomatik öğrenen bir mimari sunar.

* * * * *

💾 Veri Seti Özellikleri
------------------------

Çalışmada **Toronto Emotional Speech Set (TESS)** kullanılmıştır.

-   **Kaynak:** Northwestern University

-   **Örneklem Sayısı:** 2800 Adet `.wav` dosyası

-   **Katılımcılar:** 26 ve 64 yaşlarında iki kadın konuşmacı.

-   **Sınıf Dağılımı:** Veri seti, sınıf dengesizliği (imbalance) içermemektedir. Her duygu sınıfı için eşit sayıda (400 adet) veri bulunur.

-   **Sınıflar:** *Anger (Kızgın), Disgust (İğrenme), Fear (Korku), Happiness (Mutlu), Pleasant Surprise (Şaşkın), Sadness (Üzgün), Neutral (Nötr).*

* * * * *

🛠 Metodoloji ve Teknik Mimari
------------------------------

Proje akışı üç ana fazdan oluşur: **Ön İşleme (Preprocessing)**, **Öznitelik Çıkarımı (Feature Extraction)** ve **Sınıflandırma (Classification)**.

### 1\. Sinyal İşleme ve MFCC (Mel-Frequency Cepstral Coefficients)

Ham ses sinyali (Amplitude vs Time), makine öğrenmesi modelleri için doğrudan anlamlı değildir. Bu nedenle sinyaller, insan kulağının işitme algısını modelleyen **Mel Skalasına** dönüştürülmüştür.

-   **Örnekleme Hızı (Sample Rate):** 22.050 Hz

-   **Öznitelik Sayısı:** Her ses karesi için **40 MFCC katsayısı** çıkarılmıştır.

-   **Matematiksel Süreç:**

    1.  **Pre-emphasis:** Yüksek frekansların enerjisini artırma.

    2.  **Framing & Windowing:** Sinyali kısa süreli çerçevelere bölme (Hamming Window).

    3.  **FFT (Fast Fourier Transform):** Zaman alanından frekans alanına geçiş.

    4.  **Mel Filterbank:** İnsan algısına uygun logaritmik frekans ölçekleme.

    5.  **DCT (Discrete Cosine Transform):** Korelasyonu azaltarak MFCC katsayılarını elde etme.

### 2\. Model Topolojisi (1D-CNN)

Ses verisi, görüntüden farklı olarak tek boyutlu (zaman eksenli) bir yapıdadır. Bu nedenle **1D Convolutional Neural Network** mimarisi tercih edilmiştir.

| **Katman (Layer)** | **Yapılandırma** | **Açıklama** |
| --- | --- | --- |
| **Input Layer** | (40, 1) | 40 boyutlu MFCC vektör girişi. |
| **Conv1D** | 64 Filters, Kernel=5, Stride=1 | Yerel frekans desenlerini yakalar. Aktivasyon: `ReLU`. |
| **MaxPooling1D** | Pool Size=2 | Boyut azaltma yaparak işlem yükünü düşürür ve overfitting'i önler. |
| **Flatten** | - | Konvolüsyon haritasını (feature map) tek boyutlu vektöre çevirir. |
| **Dense (FC)** | 128 Neurons | Tam bağlantılı katman. Yüksek seviyeli karar verme birimi. |
| **Dropout** | 0.3 (%30) | Regülarizasyon tekniği (Ezberlemeyi önler). |
| **Output Layer** | 7 Neurons | `Softmax` aktivasyon fonksiyonu ile sınıflara ait olasılık dağılımı üretir. |

* * * * *

🔬 Deneysel Kurulum
-------------------

Modelin eğitimi Google Colab ortamında, GPU hızlandırma (NVIDIA Tesla T4) kullanılarak gerçekleştirilmiştir.

**Hiperparametreler (Hyperparameters):**

-   **Optimizer:** Adam (Adaptive Moment Estimation) - `learning_rate=0.001`

-   **Loss Function:** Categorical Crossentropy (Çok sınıflı sınıflandırma için)

-   **Batch Size:** 32

-   **Epochs:** 60 (Early Stopping mekanizması ile izlenmiştir)

-   **Train/Test Split:** %80 Eğitim, %20 Test

* * * * *

📊 Performans Analizi ve Sonuçlar
---------------------------------

Geliştirilen DeepTone modeli, temel (baseline) model olarak seçilen **Random Forest** ile kıyaslanmıştır.

### Karşılaştırmalı Sonuç Tablosu

| **Algoritma** | **Mimari Türü** | **Doğruluk (Accuracy)** | **Kayıp (Loss)** |
| --- | --- | --- | --- |
| Random Forest | Ensemble Learning | %98.93 | - |
| **DeepTone (Proposed)** | **Deep Learning (CNN)** | **%99.82** 🏆 | **0.0062** |

### Analiz ve Yorumlar

1.  **Doğruluk:** CNN modelinin %99.82'lik başarısı, ses özniteliklerinin hiyerarşik yapısını öğrenmede derin ağların üstünlüğünü kanıtlamıştır.

2.  **Genelleştirme:** Eğitim (%99.20) ve Test (%99.82) başarılarının birbirine yakın olması, modelde **Overfitting (Aşırı Öğrenme)** probleminin başarıyla engellendiğini gösterir.

3.  **Hata Analizi:** Karmaşıklık matrisine (Confusion Matrix) göre, modelin en çok zorlandığı iki sınıfın *Sadness* ve *Neutral* olduğu, bunun sebebinin ise iki duygunun da düşük enerji ve benzer frekans aralığına sahip olması olduğu değerlendirilmiştir.

*(Şekil 1: Test verisi üzerindeki Karmaşıklık Matrisi)*

*(Şekil 2: Eğitim süreci boyunca Accuracy ve Loss değişimi)*

* * * * *

💻 Kurulum ve Kullanım
----------------------

Proje, modüler dosya yapısına sahiptir.

### Dosya Yapısı

-   `model.py`: Model mimarisini tanımlayan sınıf yapısı.

-   `train.py`: Veri işleme pipeline'ı ve eğitim döngüsü.

-   `serve.py`: Gradio tabanlı demo arayüzü.

### Çalıştırma Adımları

**1\. Bağımlılıkları Yükleyin:**

Bash

```
pip install -r requirements.txt

```

**2\. Eğitimi Başlatın:**

Bash

```
python train.py

```

*Bu işlem sonucunda en iyi model ağırlıkları `duygu_modeli.h5` olarak kaydedilir.*

**3\. Arayüzü Başlatın:**

Bash

```
python serve.py

```

* * * * *

🔮 Gelecek Çalışmalar
---------------------

Bu proje kapsamında elde edilen başarıyı daha ileri taşımak için şu adımlar planlanmaktadır:

-   **Veri Çoğaltma (Data Augmentation):** Sese gürültü ekleme, hız değiştirme (Time-stretching) gibi yöntemlerle modelin gürültülü ortamlardaki dayanıklılığının artırılması.

-   **LSTM Entegrasyonu:** CNN katmanlarının çıkışına LSTM (Long Short-Term Memory) eklenerek, sesin uzun vadeli zamansal bağımlılıklarının (temporal dependencies) modellenmesi.

-   **Gerçek Zamanlı Akış:** Sisteme WebSocket entegrasyonu yapılarak canlı telefon görüşmelerinde anlık analiz yeteneği kazandırılması.

* * * * *

👤 Hazırlayan
-------------

Burak Can ÖZTOLAN

Bilgisayar Mühendisliği Bölümü

Proje Teslim Tarihi: 30 Aralık 2024
