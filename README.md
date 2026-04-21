# 🦠 COVID Risk Analizi (Hybrid AI)

Bu proje, kullanıcıların belirttiği semptomlara göre COVID-19 riskini tahmin eden bir **Hybrid Yapay Zeka sistemi**dir.

Sistem, **Makine Öğrenmesi (ML)** ve **Kural Tabanlı (Rule-Based)** yaklaşımı birleştirerek daha gerçekçi ve güvenilir sonuçlar üretir.

---

## 🚀 Özellikler

* 🧠 **Hybrid AI (ML + Rule-Based)**
* 📊 Gerçek zamanlı risk skoru (%)
* ⚠️ Kritik durum analizi (nefes darlığı vb.)
* 📈 ML model karşılaştırması (Logistic Regression, Random Forest, XGBoost)
* 🔍 Confusion Matrix ve model performans analizi
* 🧑‍⚕️ AI destekli yorum sistemi
* 🌙 STREAMLİT ARAYÜZÜ

---

## 🧠 Nasıl Çalışır?

Sistem 3 aşamada çalışır:

### 1. Makine Öğrenmesi (ML)

Model, geçmiş verilerden öğrenerek kullanıcı girdisine göre bir olasılık tahmini yapar.

### 2. Kural Tabanlı Sistem

Semptom sayısı ve kritik belirtiler (örneğin nefes darlığı) değerlendirilir.

### 3. Hybrid Karar Mekanizması

ML sonucu ve kural tabanlı skor birleştirilerek final risk hesaplanır:

```
Final Risk = 0.6 * ML + 0.4 * Rule-Based
```

Ek olarak:

* Çok sayıda semptom → risk artırılır
* Kritik durum → direkt yüksek risk

---

## 📊 Kullanılan Modeller

* Logistic Regression
* Random Forest
* XGBoost

En iyi performans veren model otomatik seçilmiştir.

---

## 📈 Model Performansı

Projede model değerlendirmesi için:

* Accuracy Score
* Confusion Matrix
* Classification Report

kullanılmıştır.

---

## ⚙️ Kurulum

### 1. Gereksinimleri yükle

```
pip install -r requirements.txt
```

### 2. Uygulamayı çalıştır

```
terminale dosya yoluyla girip şunu yazmak
streamlit run app.py
```

---

## 🧪 Kullanım

1. Semptomları seç
2. "Tahmin Et" butonuna bas
3. Risk skorunu ve analiz sonuçlarını incele

---

## 📁 Proje Yapısı

```
ml-covid19-hasta-tahmini/
│
├── app.py              # Streamlit uygulaması
├── model.pkl          # Eğitilmiş model
├── scaler.pkl         # Veri ölçekleyici
├── features.pkl       # Feature listesi
├── Cleaned-Data.csv   # Veri seti
├── requirements.txt   # Gerekli kütüphaneler
└── README.md          # Proje açıklaması
```

---

## ⚠️ Önemli Not

Bu sistem sadece **tahmin amaçlıdır**.
Kesin tanı için bir sağlık kuruluşuna başvurulmalıdır.

---

## 👨‍💻 Geliştirici

Bu proje, makine öğrenmesi ve veri analizi becerilerini geliştirmek amacıyla hazırlanmıştır.

---

## 💡 Not

Projede veri sızıntısı (data leakage) önlenmiş ve model gerçekçi sonuçlar verecek şekilde optimize edilmiştir.

---

🔥 Bu proje, basit bir tahmin sisteminden öte, **gerçek dünya problemlerine yönelik bir AI çözümü** olarak geliştirilmiştir.
