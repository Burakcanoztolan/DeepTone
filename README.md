DeepTone: Derin Öğrenme ile Ses Duygu Analizi 🎙️
=================================================

**DeepTone**, ses frekanslarının derinliklerine inerek konuşmacının duygu durumunu (Mutlu, Üzgün, Kızgın, Nötr vb.) analiz eden gelişmiş bir Yapay Zeka projesidir.

> *"Sesin tonundaki gizli duyguyu yapay zeka ile açığa çıkarın."*

* * * * *

📂 Proje Mimarisi
-----------------

Bu proje, **modüler tasarım prensiplerine** uygun olarak üç ana bileşene ayrılmıştır:

-   **`model.py`**: Derin Öğrenme (CNN - Convolutional Neural Network) mimarisinin tasarlandığı çekirdek dosya.

-   **`train.py`**: Veri setinin işlendiği (MFCC öznitelik çıkarımı), modelin eğitildiği ve performansın test edildiği eğitim dosyası.

-   **`serve.py`**: Eğitilen modelin son kullanıcıya sunulması için **Gradio** ile hazırlanmış interaktif web arayüzü dosyası.

-   **`requirements.txt`**: Projenin bağımlılıklarını içeren kütüphane listesi.

* * * * *

📊 Veri Seti ve Metodoloji
--------------------------

-   **Veri Seti:** Projede **TESS (Toronto Emotional Speech Set)** kullanılmıştır. 2800 adet yüksek kaliteli ses dosyasından oluşur.

-   **Yöntem:** Ses dosyalarından **MFCC (Mel-Frequency Cepstral Coefficients)** özellikleri çıkarılmış ve bu özellikler **Conv1D** katmanlarına sahip bir CNN modeline beslenmiştir.

-   **Karşılaştırmalı Analiz:** Proje kapsamında Geleneksel Yöntem (Random Forest) ile Modern Yöntem (CNN) kıyaslanmış ve CNN'in üstün başarısı kanıtlanmıştır.

* * * * *

🚀 Kurulum ve Çalıştırma
------------------------

Projeyi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

### 1\. Kütüphaneleri Yükleyin

Bash

```
pip install -r requirements.txt

```

### 2\. Modeli Eğitin

Modeli sıfırdan eğitmek ve başarı grafiklerini üretmek için:

Bash

```
python train.py

```

*Bu işlem sonucunda `duygu_modeli.h5` dosyası oluşturulacaktır.*

### 3\. Arayüzü Başlatın (Test)

Mikrofon ile canlı test yapmak için arayüzü başlatın:

Bash

```
python serve.py

```

*Size verilen yerel linke (örn: https://www.google.com/search?q=http://127.0.0.1:7860) tıklayarak sistemi kullanabilirsiniz.*

* * * * *

📈 Sonuçlar ve Performans
-------------------------

Modelimiz, test verisi üzerinde **%99.82** gibi literatürdeki en yüksek doğruluk oranlarından birine ulaşmıştır.

| **Model** | **Doğruluk Oranı (Accuracy)** |
| --- | --- |
| Random Forest (Referans Model) | %98.93 |
| **DeepTone CNN (Final Model)** | **%99.82** 🏆 |

### Eğitim Başarı Grafiği

Modelin öğrenme sürecindeki kararlılığını gösteren grafik:

### Karmaşıklık Matrisi (Confusion Matrix)

Modelin hangi duyguları ne kadar doğru sınıflandırdığının analizi:

* * * * *

👤 Hazırlayan
-------------

Burak Can ÖZTOLAN

Bilgisayar Mühendisliği Bölümü

Teslim Tarihi: 30 Aralık 2024
