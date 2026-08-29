
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
 
# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")
 
# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        color: #FF4B4B;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #999;
        margin-bottom: 1.5rem;
        font-size: 1rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        margin-top: 1rem;
    }
    .high-risk {
        background-color: rgba(255, 75, 75, 0.1);
        border: 2px solid #FF4B4B;
    }
    .low-risk {
        background-color: rgba(46, 204, 113, 0.1);
        border: 2px solid #2ECC71;
    }
    .factor-box {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background-color: rgba(255,255,255,0.04);
        margin-bottom: 0.5rem;
        border-left: 4px solid #555;
    }
    .factor-bad { border-left: 4px solid #FF4B4B; }
    .factor-good { border-left: 4px solid #2ECC71; }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
 
# ---------------- LOAD MODEL ----------------
model = joblib.load("LR_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")
 
# ---------------- HEADER ----------------
st.markdown('<p class="main-title">❤️ Heart Disease Risk Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">by SUMAN BISWAS — Machine Learning based risk estimation tool</p>', unsafe_allow_html=True)
 
tab1, tab2 = st.tabs(["🔍 Prediction", "ℹ️ About"])
 
# ================= TAB 1: PREDICTION =================
with tab1:
 
    st.sidebar.header("📋 Patient Details")
 
    age = st.sidebar.slider("Age", 18, 100, 40)
    sex = st.sidebar.selectbox("Sex", ["M", "F"])
    chest_pain = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.sidebar.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])
 
    predict_btn = st.sidebar.button("🔍 Predict Risk", use_container_width=True, type="primary")
 
    if predict_btn:
 
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }
 
        input_df = pd.DataFrame([raw_input])
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]
        scaled_input = scaler.transform(input_df)
 
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0]
        risk_pct = probability[1] * 100
 
        col1, col2 = st.columns([1, 1])
 
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                title={'text': "Heart Disease Risk (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FF4B4B" if prediction == 1 else "#2ECC71"},
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(46, 204, 113, 0.2)"},
                        {'range': [40, 70], 'color': "rgba(241, 196, 15, 0.2)"},
                        {'range': [70, 100], 'color': "rgba(255, 75, 75, 0.2)"}
                    ],
                }
            ))
            fig.update_layout(height=300, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
 
        with col2:
            st.write("")
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card high-risk">
                    <h2>⚠️ High Risk</h2>
                    <p style="font-size:1.05rem;">Model predicts elevated heart disease risk.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card low-risk">
                    <h2>✅ Low Risk</h2>
                    <p style="font-size:1.05rem;">Model predicts low heart disease risk.</p>
                </div>
                """, unsafe_allow_html=True)
            st.caption("⚠️ This is a machine learning estimate, not a medical diagnosis. Please consult a doctor for actual health concerns.")
 
        st.divider()
        st.subheader("🩺 Key Risk Factors Spotted")
 
        factors = []
        if chest_pain == "ASY":
            factors.append(("Asymptomatic chest pain — often linked with higher risk", "bad"))
        if st_slope == "Flat":
            factors.append(("Flat ST slope — commonly associated with heart disease", "bad"))
        if exercise_angina == "Y":
            factors.append(("Exercise-induced angina present", "bad"))
        if oldpeak >= 2.0:
            factors.append((f"High ST depression (Oldpeak = {oldpeak})", "bad"))
        if cholesterol >= 300:
            factors.append((f"High cholesterol ({cholesterol} mg/dL)", "bad"))
        if fasting_bs == 1:
            factors.append(("Elevated fasting blood sugar", "bad"))
        if max_hr < 120:
            factors.append((f"Lower max heart rate achieved ({max_hr} bpm)", "bad"))
 
        if st_slope == "Up":
            factors.append(("Upward ST slope — generally a healthy sign", "good"))
        if exercise_angina == "N":
            factors.append(("No exercise-induced angina", "good"))
        if max_hr >= 150:
            factors.append((f"Good max heart rate achieved ({max_hr} bpm)", "good"))
 
        if not factors:
            st.info("No strong individual risk factors detected — result is based on overall pattern.")
        else:
            for text, kind in factors:
                icon = "🔴" if kind == "bad" else "🟢"
                css_class = "factor-bad" if kind == "bad" else "factor-good"
                st.markdown(f'<div class="factor-box {css_class}">{icon} {text}</div>', unsafe_allow_html=True)
 
        st.divider()
        with st.expander("📊 View Full Input Summary"):
            summary_df = pd.DataFrame({
                "Field": ["Age", "Sex", "Chest Pain", "Resting BP", "Cholesterol", "Fasting BS",
                          "Resting ECG", "Max HR", "Exercise Angina", "Oldpeak", "ST Slope"],
                "Value": [age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                          resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]
            })
            st.table(summary_df)
 
            csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Report (CSV)", csv, "heart_risk_report.csv", "text/csv")
 
    else:
        st.info("👈 Fill in the patient details in the sidebar and click **Predict Risk** to see results.")
 
# ================= TAB 2: ABOUT =================
with tab2:
    st.subheader("About this App")
    st.markdown("""
    This tool estimates the likelihood of heart disease using a **Logistic Regression** model
    trained on the UCI Heart Failure Prediction dataset (918 patient records, 11 clinical features).
 
    **Features used:** Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol,
    Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise-Induced Angina,
    ST Depression (Oldpeak), and ST Slope.
 
    **Disclaimer:** This app is built for learning purposes and provides a statistical estimate only.
    It is **not** a substitute for professional medical advice, diagnosis, or treatment.
    Always consult a qualified doctor for any heart-related concerns.
    """)
    st.divider()
    st.caption("Built with Streamlit · Model: Logistic Regression · Author: Akarsh")
 
