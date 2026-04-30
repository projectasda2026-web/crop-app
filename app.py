import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open('crop_model.pkl', 'rb'))

# Title
st.title("🌾 Smart Crop Recommendation System")
st.write("Enter soil and environmental conditions")

# Sidebar inputs
st.sidebar.header("Input Parameters")

def user_input():
    N = st.sidebar.slider("Nitrogen", 0, 150, 50)
    P = st.sidebar.slider("Phosphorus", 0, 150, 50)
    K = st.sidebar.slider("Potassium", 0, 150, 50)
    temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
    ph = st.sidebar.slider("pH", 0.0, 14.0, 6.5)
    rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

    data = {
        'N': N,
        'P': P,
        'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }

    return pd.DataFrame([data])

input_df = user_input()

st.subheader("📊 Input Data")
st.write(input_df)

# Prediction
if st.button("Predict Crop"):

    probs = model.predict_proba(input_df)[0]

    # Load original labels
    df = pd.read_csv('Crop_recommendation (1).csv')
    labels = sorted(df['label'].unique())

    # Top 3
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3_crops = [labels[i] for i in top3_idx]

    result_df = pd.DataFrame({
        "Crop": top3_crops,
        "Probability": probs[top3_idx]
    })

    st.subheader("🌾 Top 3 Recommended Crops")
    st.table(result_df)

    # Plot
    fig, ax = plt.subplots()
    sns.barplot(x="Crop", y="Probability", data=result_df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Soil analysis
    issues, actions = [], []

    N = input_df['N'][0]
    P = input_df['P'][0]
    K = input_df['K'][0]
    ph_val = input_df['ph'][0]

    if N < 50:
        issues.append("Nitrogen is low")
        actions.append("Apply urea")
    elif N > 120:
        issues.append("Nitrogen is high")
        actions.append("Reduce nitrogen fertilizer")

    if P < 40:
        issues.append("Phosphorus is low")
        actions.append("Add DAP")
    elif P > 100:
        issues.append("Phosphorus is high")
        actions.append("Reduce phosphorus")

    if K < 40:
        issues.append("Potassium is low")
        actions.append("Apply potash")
    elif K > 80:
        issues.append("Potassium is high")
        actions.append("Reduce potassium")

    if ph_val < 5.5:
        issues.append("Soil acidic")
        actions.append("Apply lime")
    elif ph_val > 7:
        issues.append("Soil alkaline")
        actions.append("Add organic matter")

    st.subheader("🌱 Soil Analysis")

    if issues:
        for i in issues:
            st.warning(i)
    else:
        st.success("Soil conditions are good")

    st.subheader("✅ Recommendations")
    for a in actions:
        st.write("- ", a)