# ============================================================
# PCOS CLINICAL DECISION SUPPORT SYSTEM (CDSS)
# Streamlit Application
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd

from model_loader import load_model_artifacts
from preprocessing import preprocess_input
from prediction import predict_pcos
from explainability import generate_lime_explanation

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="PCOS AI-CDSS",
    page_icon="🩺",
    layout="wide"
)

# ------------------------------------------------------------
# PCOS THEME (NO WARNINGS, MODERN UI)
# ------------------------------------------------------------

st.markdown("""
<style>

/* MAIN APP BACKGROUND */
.stApp {
    background: linear-gradient(
        135deg,
        #F3E5F5 0%,
        #EDE7F6 40%,
        #FCE4EC 100%
    );
    color: #2E2A4A;
    font-family: "Segoe UI", sans-serif;
}

/* SIDEBAR BACKGROUND */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #E1BEE7 0%,
        #F8BBD0 100%
    );
}

/* HEADINGS */
h1 {
    color: #6A1B9A;
    font-weight: 800;
}
h2 {
    color: #7E57C2;
    font-weight: 700;
}
h3 {
    color: #8E24AA;
}

/* TEXT */
p, li, label {
    color: #3E3A5F;
    font-size: 16px;
}

/* CARD CONTAINERS */
.card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(6px);
    padding: 28px;
    border-radius: 20px;
    box-shadow: 0px 10px 26px rgba(106, 27, 154, 0.15);
    margin-bottom: 30px;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(90deg, #B388EB, #F48FB1);
    color: white;
    border-radius: 16px;
    padding: 0.7em 1.8em;
    font-weight: 700;
    border: none;
    box-shadow: 0px 6px 18px rgba(180, 136, 235, 0.45);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #9575CD, #EC407A);
    transform: scale(1.05);
}

/* INPUT FIELDS */
input, select {
    border-radius: 10px !important;
}

/* LIME OUTPUT FRAME */
iframe {
    border-radius: 18px;
    box-shadow: 0px 8px 22px rgba(126, 87, 194, 0.35);
    background: white;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

model, scaler, FEATURES = load_model_artifacts()

# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------

st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 PCOS Risk Prediction",
        "🧠 Explainability (LIME)",
        "ℹ️ About Project"
    ]
)

# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.image("pcos_3d_reproductive_system.png", width=1200)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("PCOS Risk Analysis & Early Detection using AI")

    st.subheader("👩‍⚕️ What is PCOS?")
    st.write("""
    Polycystic Ovary Syndrome (PCOS) is a hormonal disorder affecting women of
    reproductive age. It is associated with ovarian dysfunction, metabolic
    imbalance, and long-term health complications.
    """)

    st.image(
        "pcos_ovary_follicles_3d.png",
        caption="3D Visualization of Ovarian Follicles in PCOS",
        width=1000
    )

    st.subheader("⚠️ Effects of PCOS")
    st.write("""
    - Irregular menstrual cycles  
    - Weight gain and insulin resistance  
    - Acne, hair loss, excess facial hair  
    - Infertility and increased metabolic risk  
    """)

    st.subheader("🌍 Global & Indian Prevalence")
    st.write("""
    - **Globally:** ~8–13% of women  
    - **India:** ~10–22% (often underdiagnosed)  
    """)

    st.subheader("🎯 Project Objectives")
    st.write("""
    - AI-based early PCOS risk prediction  
    - Explainable ML for clinical transparency  
    - Privacy-aware learning architecture  
    - Real-time Clinical Decision Support System  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "📊 PCOS Risk Prediction":

    st.title("📊 PCOS Risk Prediction")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (yrs)", 12, 55, 25)
        weight = st.number_input("Weight (Kg)", 30.0, 120.0, 55.0)
        height = st.number_input("Height (Cm)", 130.0, 190.0, 160.0)

    with col2:
        pimples = st.selectbox("Pimples", ["No", "Yes"])
        hair_loss = st.selectbox("Hair Loss", ["No", "Yes"])
        hair_growth = st.selectbox("Excess Facial Hair Growth", ["No", "Yes"])
        skin_darkening = st.selectbox("Skin Darkening", ["No", "Yes"])

    with col3:
        cycle = st.selectbox("Cycle Regularity", ["Regular", "Irregular"])
        fast_food = st.selectbox("Fast Food Intake", [0, 1, 2])
        stress = st.selectbox("Stress Level", [0, 1, 2])
        sleep = st.selectbox("Sleep Quality", [0, 1, 2])
        sugar = st.selectbox("Sugar Intake", [0, 1, 2])
        family_history = st.selectbox("Family History of PCOS", [0, 1])

    if st.button("🔮 Predict PCOS Risk"):

        input_dict = {
            "Age (yrs)": age,
            "Weight (Kg)": weight,
            "Height(Cm)": height,
            "Pimples(Y/N)": pimples,
            "Hair loss(Y/N)": hair_loss,
            "hair growth(Y/N)": hair_growth,
            "Skin darkening (Y/N)": skin_darkening,
            "Cycle(R/I)": cycle,
            "Fast_Food_Intake": fast_food,
            "Stress_Level": stress,
            "Sleep_Quality": sleep,
            "Sugar_Intake": sugar,
            "Family_History_PCOS": family_history
        }

        X_processed = preprocess_input(input_dict, FEATURES, scaler)
        prediction, probability = predict_pcos(model, X_processed)

        st.subheader("🩺 Prediction Result")

        if prediction == 1:
            st.error(f"⚠️ High Risk of PCOS (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Low Risk of PCOS (Probability: {probability:.2f})")

        st.session_state["last_input"] = X_processed

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# EXPLAINABILITY PAGE
# ============================================================

elif page == "🧠 Explainability (LIME)":

    st.title("🧠 Explainable AI – LIME")

    if "last_input" not in st.session_state:
        st.warning("Please perform a prediction first.")
    else:
        explanation = generate_lime_explanation(
            model,
            scaler.transform(
                pd.DataFrame(np.zeros((1, len(FEATURES))), columns=FEATURES)
            ),
            FEATURES,
            st.session_state["last_input"][0]
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.components.v1.html(
            explanation.as_html(),
            height=520,
            scrolling=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About This Project")

    st.image(
        "pcos_ai_health_assistant.png",
        caption="AI-powered Clinical Decision Support System for PCOS",
        width=1000
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("""
    **Project Title:**  
    PCOS Risk Analysis and Early Detection using Personalized Integrated AI  

    **Developed by:**  
    7th Semester B.Tech Students 
    - Sarojini Vaishnavi Peri
    - G Santosh Kumar
    - C Akshatha Shivani    
    - Kruthika

    **Technologies Used:**  
    - Machine Learning (KNN, LR, RF, XGBoost)  
    - Genetic Algorithm + Hill Climbing  
    - Federated Learning (FedAvg – conceptual)  
    - Explainable AI (LIME & SHAP)  
    - Streamlit Deployment  

    **Purpose:**  
    To assist clinicians and patients with transparent, interpretable,
    and real-time PCOS risk assessment.
    """)
    st.markdown('</div>', unsafe_allow_html=True)


