import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Crop & Soil Insight", layout="centered")

st.title("🌾Smart Crop Recommendation & Soil Insight System")

# =========================
# LOAD MODEL & LABELS
# =========================
model = pickle.load(open('crop_model.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))

# =========================
# LOAD DATASET (for safe input range)
# =========================
df = pd.read_csv('Crop_recommendation (1).csv')

N_min, N_max = df['N'].min(), df['N'].max()
P_min, P_max = df['P'].min(), df['P'].max()
K_min, K_max = df['K'].min(), df['K'].max()

# =========================
# INPUT SECTION (MANUAL ENTRY)
# =========================
st.header("🌱 Enter Soil & Environmental Parameters")

st.info("Enter values based on your soil test report")

N = st.number_input("Nitrogen (N)", float(N_min), float(N_max), float(N_min), step=1.0)
P = st.number_input("Phosphorus (P)", float(P_min), float(P_max), float(P_min), step=1.0)
K = st.number_input("Potassium (K)", float(K_min), float(K_max), float(K_min), step=1.0)

temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0, step=0.1)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0, step=0.1)
ph = st.number_input("pH", 0.0, 14.0, 6.5, step=0.1)
rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 100.0, step=0.1)

# Create input dataframe
input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                        columns=['N','P','K','temperature','humidity','ph','rainfall'])

st.subheader("📊 Input Data")
st.write(input_df)

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict"):

    probs = model.predict_proba(input_df)[0]

    # Top 3 crops
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3_crops = [labels[i] for i in top3_idx]

    # =========================
    # OUTPUT 1: TOP-3 CROPS
    # =========================
    st.subheader("🌾 Top-3 Recommended Crops")

    for i, crop in enumerate(top3_crops):
        st.write(f"{i+1}. {crop}")

    # =========================
    # OUTPUT 2: PROBABILITIES
    # =========================
    st.subheader("📊 Prediction Probabilities")

    prob_df = pd.DataFrame({
        "Crop": top3_crops,
        "Probability": probs[top3_idx]
    })

    st.table(prob_df)

    # Chart
    fig, ax = plt.subplots()
    ax.bar(prob_df["Crop"], prob_df["Probability"])
    plt.xticks(rotation=30)
    st.pyplot(fig)

    # =========================
    # OUTPUT 3: SOIL CONDITION
    # =========================
    st.subheader("🌱 Soil Condition & Suggestions")

    issues = []
    suggestions = []

    # Nitrogen
    if N < 40:
        issues.append("Low Nitrogen")
        suggestions.append("Apply urea or nitrogen-rich fertilizer")
    elif N > 100:
        issues.append("Excess Nitrogen")
        suggestions.append("Reduce nitrogen fertilizer")

    # Phosphorus
    if P < 30:
        issues.append("Low Phosphorus")
        suggestions.append("Apply DAP")
    elif P > 80:
        issues.append("Excess Phosphorus")
        suggestions.append("Reduce phosphorus usage")

    # Potassium
    if K < 30:
        issues.append("Low Potassium")
        suggestions.append("Apply potash fertilizer")
    elif K > 70:
        issues.append("Excess Potassium")
        suggestions.append("Reduce potassium fertilizer")

    # pH
    if ph < 5.5:
        issues.append("Acidic Soil")
        suggestions.append("Apply lime to increase pH")
    elif ph > 7:
        issues.append("Alkaline Soil")
        suggestions.append("Add organic matter")

    # Display results
    if issues:
        st.write("### ⚠️ Issues Detected")
        for i in issues:
            st.write("-", i)

        st.write("### ✅ Suggestions")
        for s in suggestions:
            st.write("-", s)
    else:
        st.success("Soil conditions are optimal. No improvements needed.")