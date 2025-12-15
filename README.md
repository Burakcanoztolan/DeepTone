# Ses Tonundan Duygu Analizi Projesi 🎤

Bu proje, Derin Öğrenme (CNN) yöntemleri kullanılarak ses kayıtlarından insanların duygu durumunu (Mutlu, Üzgün, Kızgın vb.) tahmin eden bir yapay zeka uygulamasıdır.

## 📂 Proje İçeriği
Proje, modüler bir yapıda tasarlanmış olup 3 ana dosyadan oluşur:
1.  **`model.py`**: CNN model mimarisinin tanımlandığı dosya.
2.  **`train.py`**: Veri setinin işlendiği, modelin eğitildiği ve performans grafiklerinin çizildiği dosya.
3.  **`serve.py`**: Gradio kütüphanesi ile kullanıcı arayüzünün oluşturulduğu dosya.

## 📊 Veri Seti
Projede **TESS (Toronto Emotional Speech Set)** kullanılmıştır.
- Veri seti Kaggle üzerinden otomatik çekilmektedir.
- 7 farklı duygu sınıfı içerir.

## 🚀 Kurulum ve Çalıştırma

Projeyi çalıştırmak için aşağıdaki adımları izleyin:

1. **Gerekli Kütüphaneleri Yükleyin:**
   ```bash
   pip install -r requirements.txt
