import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# LOAD
# =========================
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="COVID AI", layout="centered")

# =========================
# 👤 KULLANICI BİLGİLERİ
# =========================
st.markdown("## 👤 Kişisel Bilgiler")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Yaş", 0, 100, 25)

with col2:
    gender = st.radio(
        "Cinsiyet",
        ["Erkek", "Kadın", "Diğer"],
        horizontal=True
    )
# =========================
# 🤒 SEMPTOM GİRİŞLERİ (Eksik Olan Kısım)
# =========================
st.markdown("## 🤒 Belirtiler")

fever = st.checkbox("Ateş")
tiredness = st.checkbox("Yorgunluk")
dry_cough = st.checkbox("Kuru Öksürük")
difficulty_breathing = st.checkbox("Nefes Darlığı")
sore_throat = st.checkbox("Boğaz Ağrısı")
pains = st.checkbox("Vücut Ağrıları")
nasal_congestion = st.checkbox("Burun Tıkanıklığı")
runny_nose = st.checkbox("Burun Akıntısı")
diarrhea = st.checkbox("İshal")
# =========================
# 🔄 AGE ONE-HOT
# =========================
age_0_9 = 1 if age <= 9 else 0
age_10_19 = 1 if 10 <= age <= 19 else 0
age_20_24 = 1 if 20 <= age <= 24 else 0
age_25_59 = 1 if 25 <= age <= 59 else 0
age_60_plus = 1 if age >= 60 else 0

# =========================
# 🔄 GENDER ONE-HOT (SAFE)
# =========================
gender_male = 1 if gender == "Erkek" else 0
gender_female = 1 if gender == "Kadın" else 0

# güvenlik (asla 2 tane 1 olmasın)
if gender_male + gender_female > 1:
    st.error("⚠️ Cinsiyet seçiminde hata")
    st.stop()

# =========================
# INPUT KONTROL
# =========================
if not any([fever, tiredness, dry_cough, difficulty_breathing,
            sore_throat, pains, nasal_congestion,
            runny_nose, diarrhea]):
    st.warning("⚠️ Lütfen en az bir semptom seç")
    st.stop()

# =========================
# FEATURE MAP
# =========================
input_dict = {
    "Fever": int(fever),
    "Tiredness": int(tiredness),
    "Dry-Cough": int(dry_cough),
    "Difficulty-in-Breathing": int(difficulty_breathing),
    "Sore-Throat": int(sore_throat),
    "Pains": int(pains),
    "Nasal-Congestion": int(nasal_congestion),
    "Runny-Nose": int(runny_nose),
    "Diarrhea": int(diarrhea),
    "Age_0-9": age_0_9,
    "Age_10-19": age_10_19,
    "Age_20-24": age_20_24,
    "Age_25-59": age_25_59,
    "Age_60_plus": age_60_plus,
    "Gender_Male": gender_male,
    "Gender_Female": gender_female
}

# eksikleri otomatik 0 yap
X = np.array([[input_dict.get(col, 0) for col in features]])

try:
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0][1]
except:
    proba = model.predict_proba(X)[0][1]

ml_risk = proba * 100

# =========================
# RULE BASED
# =========================
symptom_count = sum(input_dict.values())
critical_case = difficulty_breathing and fever

rule_risk = symptom_count * 10

# =========================
# HYBRID
# =========================
final_risk = 0.6 * ml_risk + 0.4 * rule_risk

if symptom_count >= 7:
    final_risk = max(final_risk, 90)
elif symptom_count >= 5:
    final_risk = max(final_risk, 75)

if critical_case:
    final_risk = max(final_risk, 95)

final_risk = min(final_risk, 100)

# =========================
# RISK BAR
# =========================
st.markdown("## 📊 Risk Sonucu")

st.progress(int(final_risk))
st.markdown(f"### 🔥 Risk Skoru: %{final_risk:.1f}")

if final_risk > 85:
    st.error("🚨 Yüksek Risk")
elif final_risk > 60:
    st.warning("⚠️ Orta Risk")
else:
    st.success("✅ Düşük Risk")

# =========================
# GRAFİK
# =========================
st.markdown("## 📈 Risk Dağılımı")

fig, ax = plt.subplots()
ax.bar(["ML", "Rule"], [ml_risk, rule_risk])
ax.set_title("Risk Kaynakları")
st.pyplot(fig)

# =========================
# BREAKDOWN
# =========================
st.markdown("## 🔍 Belirti Etkisi")

for k, v in input_dict.items():
    if v == 1:
        st.write(f"✔ {k}")

# =========================
# ANALİZ
# =========================
st.markdown("## 🧠 Detay")

col1, col2, col3 = st.columns(3)
col1.metric("ML", f"%{ml_risk:.1f}")
col2.metric("Rule", f"%{rule_risk:.1f}")
col3.metric("Semptom", symptom_count)

# =========================
# YORUM
# =========================
st.markdown("## 🤖 AI Doktor Analizi")

messages = []
risk_factors = []
advice_list = []

# =========================
# 🔥 KRİTİK BELİRTİ ANALİZİ
# =========================
if difficulty_breathing:
    messages.append("⚠️ Nefes darlığı ciddi bir solunum problemi göstergesidir.")
    risk_factors.append("Nefes darlığı")

if fever:
    messages.append("🌡️ Ateş, vücudun enfeksiyona verdiği tepkidir.")
    risk_factors.append("Ateş")

if dry_cough:
    messages.append("😷 Kuru öksürük COVID-19 ile yaygın olarak ilişkilidir.")
    risk_factors.append("Kuru öksürük")

if diarrhea:
    messages.append("🤢 Sindirim sistemi belirtileri de COVID ile ilişkili olabilir.")
    risk_factors.append("Sindirim belirtisi")

# =========================
# 🧠 SEMPTOM YOĞUNLUĞU
# =========================
if symptom_count >= 7:
    messages.append("💣 Çok sayıda semptom tespit edildi (yüksek yoğunluk).")
elif symptom_count >= 5:
    messages.append("⚠️ Orta-yüksek seviyede semptom yoğunluğu.")
elif symptom_count >= 3:
    messages.append("ℹ️ Hafif semptom yoğunluğu.")

# =========================
# 🧪 RİSK ANALİZİ
# =========================
if final_risk > 85:
    main_comment = "🚨 YÜKSEK RİSK"
    explanation = "Birden fazla güçlü belirti ve model analizi yüksek olasılığa işaret ediyor."
    advice_list.append("🏥 En yakın sağlık kuruluşuna başvur")
    advice_list.append("🧪 COVID testi yaptır")
    advice_list.append("🚫 Kendini izole et")

elif final_risk > 60:
    main_comment = "⚠️ ORTA RİSK"
    explanation = "Bazı güçlü belirtiler mevcut, dikkatli olunmalı."
    advice_list.append("🧪 Test yaptırman önerilir")
    advice_list.append("😷 Maske kullan ve sosyal mesafeye dikkat et")

else:
    main_comment = "✅ DÜŞÜK RİSK"
    explanation = "Şu an ciddi bir kombinasyon gözükmüyor."
    advice_list.append("👀 Belirtileri takip et")
    advice_list.append("💧 Bol sıvı tüket ve dinlen")

# =========================
# 💣 KRİTİK OVERRIDE
# =========================
if critical_case:
    main_comment = "🚨 KRİTİK DURUM"
    explanation = "Nefes darlığı + ateş birlikte görüldü."
    advice_list = [
        "🚑 ACİL olarak hastaneye git",
        "🫁 Solunum desteği gerekebilir"
    ]

# =========================
# 📊 ML vs RULE YORUM
# =========================
if ml_risk > rule_risk + 20:
    messages.append("🤖 ML modeli bu kombinasyonu riskli olarak değerlendiriyor.")
elif rule_risk > ml_risk + 20:
    messages.append("📏 Semptom sayısı riski artırıyor.")
else:
    messages.append("⚖️ ML ve kural sistemi benzer sonuç verdi.")

# =========================
# 🧾 SONUÇ YAZDIR
# =========================
st.markdown(f"### {main_comment}")
st.write(f"🧠 {explanation}")

# 🔍 nedenler
if risk_factors:
    st.markdown("### 🔍 Öne Çıkan Risk Faktörleri")
    for rf in risk_factors:
        st.write(f"• {rf}")

# 🧠 detaylı analiz
st.markdown("### 🧠 Detaylı Analiz")
for msg in messages:
    st.write(msg)

# 💡 öneriler
st.markdown("### 💡 Öneriler")
for adv in advice_list:
    st.write(f"• {adv}")

# =========================
# 🎯 EXTRA AKILLI YORUM
# =========================
if symptom_count >= 6 and not critical_case:
    st.warning("📌 Çok sayıda semptom var, dikkatli olunmalı.")

if symptom_count <= 2:
    st.info("ℹ️ Semptom sayısı düşük, risk sınırlı olabilir.")

# =========================
# ⚠️ UYARI
# =========================
st.caption("⚠️ Bu analiz tıbbi tanı değildir, yalnızca tahmin amaçlıdır.")

# =========================
# ℹ️ SİSTEM AÇIKLAMASI
# =========================
st.markdown("## ℹ️ Sistem Nasıl Çalışır?")

st.info("""
Bu uygulama, COVID-19 riskini tahmin etmek için geliştirilmiş bir **Hybrid Yapay Zeka sistemidir**.

🔹 Sistem iki ana bileşenden oluşur:

1. 🤖 Makine Öğrenmesi (Machine Learning)
- Geçmiş veriler üzerinden eğitilmiş model kullanılır
- Belirtiler arasındaki ilişkileri analiz eder
- Olasılık bazlı risk tahmini üretir

2. 📏 Kural Tabanlı Sistem (Rule-Based)
- Semptom sayısı ve şiddeti değerlendirilir
- Kritik belirtiler (örneğin nefes darlığı) ekstra ağırlık alır
- Daha mantıklı ve stabil sonuç üretir

🔹 Sonuç:
👉 Bu iki sistem birleştirilerek (Hybrid AI)
👉 Daha gerçekçi ve dengeli bir risk skoru elde edilir

📊 Amaç:
Kullanıcıya hızlı ve anlaşılır bir risk analizi sunmak
""")

# =========================
# 🧠 EK BİLGİ 
# =========================
st.markdown("### 🧠 Teknik Detay")

st.write("""
- Model: Logistic Regression / Random Forest / XGBoost
- Veri: COVID semptom dataseti
- Feature Engineering uygulanmıştır
- Data Leakage engellenmiştir
- Confusion Matrix ile model değerlendirilmiştir
""")

# =========================
# 👨‍💻 GELİŞTİRİCİ
# =========================
st.markdown("---")
st.markdown("## 👨‍💻 Geliştirici")

st.write("Bu proje **Muharrem Ünal** tarafından geliştirilmiştir.")

st.markdown("""
🔗 GitHub: https://github.com/mugoexe  
📸 Instagram: https://instagram.com/print.mugo  
""")

# =========================
# ⚠️ UYARI
# =========================
st.caption("⚠️ Bu sistem tıbbi tanı koymaz, sadece tahmin yapar.")
