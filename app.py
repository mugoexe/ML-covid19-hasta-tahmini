import streamlit as st
import numpy as np
import joblib


# YÜKLE

model = joblib.load("model.pkl")
feature_names = joblib.load("features.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="COVID Risk Sistemi", layout="centered")

st.title("🦠 COVID Risk Analizi (Hybrid AI)")
st.markdown("Belirtileri seç, sistem riskini hesaplasın")


# INPUT

st.markdown("## 🧾 Belirtiler")

fever = st.checkbox("Ateş")
tiredness = st.checkbox("Yorgunluk")
dry_cough = st.checkbox("Kuru öksürük")
difficulty_breathing = st.checkbox("Nefes darlığı")
sore_throat = st.checkbox("Boğaz ağrısı")
pains = st.checkbox("Ağrılar")
nasal_congestion = st.checkbox("Burun tıkanıklığı")
runny_nose = st.checkbox("Burun akması")
diarrhea = st.checkbox("İshal")

st.markdown("## 👤 Ek Bilgiler")

age_0_9 = st.checkbox("Yaş 0-9")
age_10_19 = st.checkbox("Yaş 10-19")
age_20_24 = st.checkbox("Yaş 20-24")
age_25_59 = st.checkbox("Yaş 25-59")
age_60_plus = st.checkbox("60+ yaş")

gender_female = st.checkbox("Kadın")
gender_male = st.checkbox("Erkek")

contact_yes = st.checkbox("COVID temas var")
contact_unknown = st.checkbox("Temas bilinmiyor")


# FEATURE DICT 

input_dict = {
    "Fever": fever,
    "Tiredness": tiredness,
    "Dry-Cough": dry_cough,
    "Difficulty-in-Breathing": difficulty_breathing,
    "Sore-Throat": sore_throat,
    "Pains": pains,
    "Nasal-Congestion": nasal_congestion,
    "Runny-Nose": runny_nose,
    "Diarrhea": diarrhea,

    "Age_0-9": age_0_9,
    "Age_10-19": age_10_19,
    "Age_20-24": age_20_24,
    "Age_25-59": age_25_59,
    "Age_60+": age_60_plus,

    "Gender_Female": gender_female,
    "Gender_Male": gender_male,

    "Contact_Yes": contact_yes,
    "Contact_DontKnow": contact_unknown,
    "Contact_No": 0
}


# BUTON

if st.button("🚀 Tahmin Et"):

    # FEATURE ARRAY (SIRALI)
    X = np.array([[input_dict.get(col, 0) for col in feature_names]])

    # SCALER (eğer LogisticRegression seçilmişse işe yarar)
    try:
        X_scaled = scaler.transform(X)
    except:
        X_scaled = X

    # MODEL
    try:
        proba = model.predict_proba(X_scaled)[0][1]
    except:
        proba = model.predict_proba(X)[0][1]

    ml_risk = proba * 100

    # =========================
    # RULE BASED
    # =========================
    symptom_count = sum([
        fever, tiredness, dry_cough, difficulty_breathing,
        sore_throat, pains, nasal_congestion,
        runny_nose, diarrhea
    ])

    critical_case = difficulty_breathing and fever
    rule_risk = symptom_count * 10

    
    # HYBRID
    
    final_risk = 0.6 * ml_risk + 0.4 * rule_risk

    # HARD RULES
    if symptom_count >= 7:
        final_risk = max(final_risk, 90)
    elif symptom_count >= 5:
        final_risk = max(final_risk, 75)

    if critical_case:
        final_risk = max(final_risk, 95)

    final_risk = min(final_risk, 100)

   
    # UI
   
    st.markdown("## 📊 Sonuç")

    st.progress(int(final_risk))
    st.markdown(f"### Risk Skoru: %{final_risk:.1f}")

    if final_risk > 85:
        st.error("🚨 Yüksek Risk! COVID ihtimali çok yüksek.")
    elif final_risk > 60:
        st.warning("⚠️ Orta Risk")
    else:
        st.success("✅ Düşük Risk")

   
    # DETAY
   
    st.markdown("## 🔍 Analiz")

    col1, col2, col3 = st.columns(3)
    col1.metric("ML", f"%{ml_risk:.1f}")
    col2.metric("Kural", f"%{rule_risk:.1f}")
    col3.metric("Semptom", symptom_count)

   
    # UYARILAR
  
    if symptom_count >= 6:
        st.warning("📌 Çok fazla semptom var")

    if critical_case:
        st.error("⚠️ Kritik durum: Nefes darlığı")

  
    # AI YORUM
  
    st.markdown("## 🧠 AI Doktor Yorumu")

    if final_risk > 85:
        st.info("🚨 ACİL: Hastaneye git")
    elif critical_case:
        st.info("⚠️ Doktora görün")
    elif final_risk > 65:
        st.info("🧪 Test yaptır")
    elif final_risk > 40:
        st.info("😷 Dikkatli ol")
    else:
        st.info("✅ Sorun yok gibi")

    st.markdown("---")
    st.caption("⚠️ Bu sistem sadece tahmin amaçlıdır.")
