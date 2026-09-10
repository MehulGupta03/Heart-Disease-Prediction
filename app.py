import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CardioAI | Heart Disease Prediction",
    # page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "KNN_heart.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
COLUMNS_PATH = BASE_DIR / "columns.pkl"


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak",
    "Sex_M",
    "ChestPainType_ATA",
    "ChestPainType_NAP",
    "ChestPainType_TA",
    "RestingECG_Normal",
    "RestingECG_ST",
    "ExerciseAngina_Y",
    "ST_Slope_Flat",
    "ST_Slope_Up"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    try:
        columns = joblib.load(COLUMNS_PATH)
    except Exception:
        columns = FEATURES

    return model, scaler, columns


# ============================================================
# CHECK FILES
# ============================================================

missing_files = []

if not MODEL_PATH.exists():
    missing_files.append("KNN_heart.pkl")

if not SCALER_PATH.exists():
    missing_files.append("scaler.pkl")

if not COLUMNS_PATH.exists():
    missing_files.append("columns.pkl")


if missing_files:

    st.error("❌ Required PKL file(s) were not found.")

    st.markdown(
        "### Make sure these files are in the same folder as `app.py`:"
    )

    st.code(
        """app.py
KNN_heart.pkl
scaler.pkl
columns.pkl"""
    )

    st.markdown("### Missing files:")

    for file in missing_files:
        st.write(f"❌ `{file}`")

    st.stop()


# ============================================================
# LOAD MODEL FILES
# ============================================================

try:

    model, scaler, saved_columns = load_model()

except Exception as e:

    st.error("❌ Error while loading the PKL files.")

    st.exception(e)

    st.stop()


# ============================================================
# CUSTOM CSS
# ============================================================

st.html(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {

        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(239, 68, 68, 0.08),
                transparent 28%
            ),

            radial-gradient(
                circle at 90% 20%,
                rgba(59, 130, 246, 0.08),
                transparent 28%
            ),

            #0b0f17;

        color: #e2e8f0;
    }


    /* ========================================================
       FIX TOP CLIPPING
       ======================================================== */

    .block-container {

        padding-top: 5rem !important;

        padding-bottom: 3rem !important;

        max-width: 1450px;

        width: 100%;

        box-sizing: border-box;
    }


    /* ========================================================
       HIDE SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {

        display: none;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {

        position: relative;

        overflow: hidden;

        width: 100%;

        margin-top: 10px;

        min-height: 300px;

        box-sizing: border-box;

        padding: 32px 45px 24px 45px;

        border-radius: 25px;

        background:
            linear-gradient(
                135deg,
                rgba(239,68,68,0.14),
                rgba(15,23,42,0.94)
            );

        border:
            1px solid rgba(239,68,68,0.20);

        box-shadow:
            0 25px 70px rgba(0,0,0,0.28);

        animation:
            heroAppear 0.7s ease;
    }


    .hero::before {

        content: "";

        position: absolute;

        width: 350px;

        height: 350px;

        right: -120px;

        top: -190px;

        border-radius: 50%;

        background:
            rgba(239,68,68,0.13);

        filter: blur(55px);

        animation:
            floatGlow 5s ease-in-out infinite;
    }


    .hero-small {

        position: relative;

        z-index: 2;

        color: #f87171;

        font-size: 13px;

        font-weight: 800;

        text-transform: uppercase;

        letter-spacing: 2px;

        margin-bottom: 12px;
    }


    .hero-title {

        position: relative;

        z-index: 2;

        font-size: clamp(38px, 5vw, 62px);

        line-height: 1.05;

        font-weight: 900;

        color: #f8fafc;

        margin: 0;

        padding: 0;
    }


    .hero-title span {

        color: #ef4444;
    }


    .hero-description {

        position: relative;

        z-index: 2;

        max-width: 820px;

        color: #94a3b8;

        font-size: 16px;

        line-height: 1.7;

        margin-top: 18px;
    }


    /* ========================================================
       ECG
       ======================================================== */

    .ecg-wrapper {

        position: relative;

        z-index: 2;

        width: 100%;

        height: 62px;

        margin-top: 18px;

        overflow: hidden;

        opacity: 0.9;
    }


    .ecg {

        display: block;

        width: 100%;

        height: 62px;
    }


    .ecg-line {

        fill: none;

        stroke: #ef4444;

        stroke-width: 3;

        stroke-linecap: round;

        stroke-linejoin: round;

        stroke-dasharray: 1000;

        stroke-dashoffset: 1000;

        animation:
            ecgMove 4s linear infinite;

        filter:
            drop-shadow(
                0 0 5px rgba(239,68,68,0.8)
            );
    }


    /* ========================================================
       SECTION TITLES
       ======================================================== */

    .section-title {

        margin-top: 32px;

        margin-bottom: 5px;

        font-size: 25px;

        font-weight: 800;

        color: #f8fafc;
    }


    .section-subtitle {

        color: #64748b;

        font-size: 13px;

        margin-bottom: 22px;
    }


    /* ========================================================
       INPUT LABELS
       ======================================================== */

    label {

        color: #cbd5e1 !important;

        font-weight: 600 !important;
    }


    /* ========================================================
       INPUT BOXES
       ======================================================== */

    div[data-baseweb="input"] {

        background:
            rgba(255,255,255,0.035);

        border-radius: 12px;
    }


    div[data-baseweb="select"] {

        background:
            rgba(255,255,255,0.035);

        border-radius: 12px;
    }


    /* ========================================================
       PREDICT BUTTON
       ======================================================== */

    .stButton > button {

        height: 58px;

        border-radius: 15px;

        border:
            1px solid rgba(239,68,68,0.35);

        background:
            linear-gradient(
                135deg,
                #dc2626,
                #ef4444
            );

        color: white;

        font-size: 16px;

        font-weight: 800;

        box-shadow:
            0 10px 30px rgba(239,68,68,0.18);

        transition:
            all 0.2s ease;
    }


    .stButton > button:hover {

        transform:
            translateY(-3px);

        box-shadow:
            0 15px 40px rgba(239,68,68,0.35);
    }


    /* ========================================================
       RESULT CARD
       ======================================================== */

    .result-card {

        margin-top: 30px;

        padding: 32px;

        border-radius: 22px;

        text-align: center;

        animation:
            resultAppear 0.7s ease;
    }


    .result-positive {

        background:
            linear-gradient(
                135deg,
                rgba(239,68,68,0.17),
                rgba(127,29,29,0.12)
            );

        border:
            1px solid rgba(239,68,68,0.35);

        box-shadow:
            0 0 45px rgba(239,68,68,0.08);
    }


    .result-negative {

        background:
            linear-gradient(
                135deg,
                rgba(34,197,94,0.13),
                rgba(20,83,45,0.10)
            );

        border:
            1px solid rgba(34,197,94,0.30);

        box-shadow:
            0 0 45px rgba(34,197,94,0.06);
    }


    .result-icon {

        font-size: 48px;

        margin-bottom: 10px;

        animation:
            iconPop 0.7s ease;
    }


    .result-title {

        font-size: 30px;

        font-weight: 900;

        color: #f8fafc;
    }


    .result-confidence {

        margin-top: 10px;

        color: #94a3b8;

        font-size: 15px;
    }


    /* ========================================================
       PROBABILITY CARDS
       ======================================================== */

    .prob-card {

        margin-top: 20px;

        padding: 25px;

        border-radius: 18px;

        background:
            rgba(255,255,255,0.035);

        border:
            1px solid rgba(255,255,255,0.07);

        transition:
            transform 0.2s ease,
            border-color 0.2s ease;
    }


    .prob-card:hover {

        transform:
            translateY(-3px);

        border-color:
            rgba(255,255,255,0.16);
    }


    .prob-label {

        color: #94a3b8;

        font-size: 13px;

        font-weight: 700;

        text-transform: uppercase;

        letter-spacing: 1px;
    }


    .prob-value {

        margin-top: 8px;

        font-size: 34px;

        font-weight: 900;

        color: #f8fafc;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {

        background:
            rgba(255,255,255,0.025);

        border:
            1px solid rgba(255,255,255,0.06);

        border-radius: 16px;

        padding: 18px;

        margin-bottom: 15px;
    }


    /* ========================================================
       EXPANDER
       ======================================================== */

    div[data-testid="stExpander"] {

        background:
            rgba(255,255,255,0.025);

        border:
            1px solid rgba(255,255,255,0.07);

        border-radius: 16px;
    }


    /* ========================================================
       DISCLAIMER
       ======================================================== */

    .disclaimer {

        margin-top: 35px;

        padding: 22px;

        border-radius: 16px;

        background:
            rgba(245,158,11,0.06);

        border:
            1px solid rgba(245,158,11,0.16);

        color: #94a3b8;

        font-size: 13px;

        line-height: 1.6;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {

        margin-top: 40px;

        padding-top: 25px;

        border-top:
            1px solid rgba(255,255,255,0.06);

        text-align: center;

        color: #475569;

        font-size: 11px;

        letter-spacing: 2px;
    }


    /* ========================================================
       ANIMATIONS
       ======================================================== */

    @keyframes heroAppear {

        from {

            opacity: 0;

            transform:
                translateY(-15px);
        }

        to {

            opacity: 1;

            transform:
                translateY(0);
        }
    }


    @keyframes floatGlow {

        0%, 100% {

            transform:
                translateY(0);
        }

        50% {

            transform:
                translateY(25px);
        }
    }


    @keyframes ecgMove {

        0% {

            stroke-dashoffset: 1000;
        }

        100% {

            stroke-dashoffset: 0;
        }
    }


    @keyframes resultAppear {

        0% {

            opacity: 0;

            transform:
                translateY(20px)
                scale(0.97);
        }

        100% {

            opacity: 1;

            transform:
                translateY(0)
                scale(1);
        }
    }


    @keyframes iconPop {

        0% {

            transform:
                scale(0);
        }

        70% {

            transform:
                scale(1.15);
        }

        100% {

            transform:
                scale(1);
        }
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 768px) {

        .block-container {

            padding-top: 4rem !important;

            padding-left: 1rem !important;

            padding-right: 1rem !important;
        }


        .hero {

            padding:
                28px 24px 20px 24px;

            min-height: 270px;

            border-radius: 20px;
        }


        .hero-title {

            font-size: 40px;
        }


        .hero-description {

            font-size: 14px;
        }


        .section-title {

            font-size: 22px;
        }

    }

    </style>
    """
)


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">

        <div class="hero-small">
            AI-POWERED HEALTH ASSESSMENT
        </div>


        <h1 class="hero-title">
            Heart Disease
            <span>Assessment</span>
        </h1>


        <div class="hero-description">

            Enter the patient's clinical information below.
            CardioAI uses a trained machine learning model
            to estimate the likelihood of heart disease.

        </div>


        <div class="ecg-wrapper">

            <svg
                class="ecg"
                viewBox="0 0 1000 100"
                preserveAspectRatio="none"
            >

                <path
                    class="ecg-line"
                    d="
                    M0,50
                    L100,50
                    L125,50
                    L135,25
                    L145,75
                    L155,50
                    L230,50
                    L260,50
                    L275,15
                    L290,85
                    L305,50
                    L380,50
                    L420,50
                    L435,30
                    L445,70
                    L455,50
                    L530,50
                    L560,50
                    L575,10
                    L590,90
                    L605,50
                    L680,50
                    L720,50
                    L735,25
                    L745,75
                    L755,50
                    L830,50
                    L860,50
                    L875,15
                    L890,85
                    L905,50
                    L1000,50
                    "
                />

            </svg>

        </div>

    </div>
    """
)


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.html(
    """
    <div class="section-title">
        Patient Information
    </div>

    <div class="section-subtitle">
        Enter the clinical parameters required by the prediction model.
    </div>
    """
)


# ============================================================
# ROW 1
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50,
        step=1
    )


with col2:

    sex = st.selectbox(
        "Sex",
        [
            "Male",
            "Female"
        ]
    )


with col3:

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120,
        step=1
    )


# ============================================================
# ROW 2
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=0,
        max_value=700,
        value=200,
        step=1
    )


with col2:

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        [
            "No (≤ 120 mg/dL)",
            "Yes (> 120 mg/dL)"
        ]
    )


with col3:

    max_hr = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150,
        step=1
    )


# ============================================================
# ROW 3
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=-5.0,
        max_value=10.0,
        value=0.0,
        step=0.1
    )


with col2:

    chest_pain = st.selectbox(
        "Chest Pain Type",
        [
            "ATA - Atypical Angina",
            "NAP - Non-Anginal Pain",
            "TA - Typical Angina",
            "ASY - Asymptomatic"
        ]
    )


with col3:

    resting_ecg = st.selectbox(
        "Resting ECG",
        [
            "Normal",
            "ST - ST-T Wave Abnormality",
            "LVH - Left Ventricular Hypertrophy"
        ]
    )


# ============================================================
# ROW 4
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        [
            "No",
            "Yes"
        ]
    )


with col2:

    st_slope = st.selectbox(
        "ST Slope",
        [
            "Up",
            "Flat",
            "Down"
        ]
    )


with col3:

    st.write("")


# ============================================================
# CREATE MODEL INPUT
# ============================================================

input_data = {

    "Age":
        age,

    "RestingBP":
        resting_bp,

    "Cholesterol":
        cholesterol,

    "FastingBS":
        1 if fasting_bs.startswith("Yes") else 0,

    "MaxHR":
        max_hr,

    "Oldpeak":
        oldpeak,

    "Sex_M":
        1 if sex == "Male" else 0,

    "ChestPainType_ATA":
        1 if chest_pain.startswith("ATA") else 0,

    "ChestPainType_NAP":
        1 if chest_pain.startswith("NAP") else 0,

    "ChestPainType_TA":
        1 if chest_pain.startswith("TA") else 0,

    "RestingECG_Normal":
        1 if resting_ecg == "Normal" else 0,

    "RestingECG_ST":
        1 if resting_ecg.startswith("ST") else 0,

    "ExerciseAngina_Y":
        1 if exercise_angina == "Yes" else 0,

    "ST_Slope_Flat":
        1 if st_slope == "Flat" else 0,

    "ST_Slope_Up":
        1 if st_slope == "Up" else 0
}


input_df = pd.DataFrame(
    [input_data]
)


# ============================================================
# FORCE EXACT FEATURE ORDER
# ============================================================

input_df = input_df.reindex(
    columns=FEATURES
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.write("")

predict_button = st.button(
    "🔍  Assess Heart Disease Risk",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ====================================================
        # PROGRESS ANIMATION
        # ====================================================

        progress = st.progress(
            0,
            text="Preparing patient data..."
        )


        for i in range(100):

            time.sleep(0.006)


            if i < 30:

                text = "Preparing patient data..."

            elif i < 60:

                text = "Scaling clinical features..."

            elif i < 90:

                text = "Running prediction..."

            else:

                text = "Finalizing assessment..."


            progress.progress(
                i + 1,
                text=text
            )


        progress.empty()


        # ====================================================
        # SCALE INPUT
        # ====================================================

        scaled_input = scaler.transform(
            input_df
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        prediction = model.predict(
            scaled_input
        )[0]


        # ====================================================
        # PROBABILITIES
        # ====================================================

        probabilities = model.predict_proba(
            scaled_input
        )[0]


        classes = list(
            model.classes_
        )


        if 1 in classes:

            probability_yes = probabilities[
                classes.index(1)
            ]

        else:

            probability_yes = 0.0


        if 0 in classes:

            probability_no = probabilities[
                classes.index(0)
            ]

        else:

            probability_no = 0.0


        # ====================================================
        # RESULT CARD
        # ====================================================

        if prediction == 1:

            st.html(
                f"""
                <div class="result-card result-positive">

                    <div class="result-icon">
                        ❤️
                    </div>

                    <div class="result-title">
                        Higher Likelihood of Heart Disease
                    </div>

                    <div class="result-confidence">

                        Model confidence:

                        <strong>
                            {probability_yes * 100:.2f}%
                        </strong>

                    </div>

                </div>
                """
            )

        else:

            st.html(
                f"""
                <div class="result-card result-negative">

                    <div class="result-icon">
                        💚
                    </div>

                    <div class="result-title">
                        Lower Likelihood of Heart Disease
                    </div>

                    <div class="result-confidence">

                        Model confidence:

                        <strong>
                            {probability_no * 100:.2f}%
                        </strong>

                    </div>

                </div>
                """
            )


        # ====================================================
        # PROBABILITY CARDS
        # ====================================================

        prob1, prob2 = st.columns(2)


        with prob1:

            st.html(
                f"""
                <div class="prob-card">

                    <div class="prob-label">
                        ❤️ Heart Disease
                    </div>

                    <div class="prob-value">
                        {probability_yes * 100:.2f}%
                    </div>

                </div>
                """
            )


        with prob2:

            st.html(
                f"""
                <div class="prob-card">

                    <div class="prob-label">
                        ✓ No Heart Disease
                    </div>

                    <div class="prob-value">
                        {probability_no * 100:.2f}%
                    </div>

                </div>
                """
            )


        # ====================================================
        # GRAPH
        # ====================================================

        st.html(
            """
            <div class="section-title">
                Prediction Probability
            </div>

            <div class="section-subtitle">
                Probability distribution generated by the model.
            </div>
            """
        )


        fig = go.Figure()


        fig.add_trace(
            go.Bar(

                x=[
                    "Heart Disease",
                    "No Heart Disease"
                ],

                y=[
                    probability_yes * 100,
                    probability_no * 100
                ],

                text=[
                    f"{probability_yes * 100:.1f}%",
                    f"{probability_no * 100:.1f}%"
                ],

                textposition="outside",

                textfont=dict(
                    size=14,
                    color="#e2e8f0"
                ),

                marker=dict(

                    color=[
                        "#ef4444",
                        "#22c55e"
                    ],

                    line=dict(
                        width=0
                    )
                ),

                hovertemplate=
                    "<b>%{x}</b>"
                    "<br>Probability: %{y:.2f}%"
                    "<extra></extra>"
            )
        )


        fig.update_layout(

            height=390,

            margin=dict(
                l=45,
                r=40,
                t=35,
                b=75
            ),

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            showlegend=False,

            font=dict(
                family="Inter, sans-serif",
                color="#cbd5e1"
            ),

            xaxis=dict(

                title="",

                tickangle=0,

                tickfont=dict(
                    size=13,
                    color="#cbd5e1"
                ),

                showgrid=False,

                zeroline=False,

                fixedrange=True
            ),

            yaxis=dict(

                title=dict(

                    text="Probability (%)",

                    font=dict(
                        size=12,
                        color="#94a3b8"
                    )
                ),

                range=[
                    0,
                    105
                ],

                ticksuffix="%",

                tickfont=dict(
                    size=11,
                    color="#94a3b8"
                ),

                gridcolor=
                    "rgba(255,255,255,0.08)",

                zeroline=False,

                fixedrange=True
            )
        )


        st.plotly_chart(
            fig,

            use_container_width=True,

            config={
                "displayModeBar": False,
                "responsive": True
            }
        )


        # ====================================================
        # PATIENT SUMMARY
        # ====================================================

        st.html(
            """
            <div class="section-title">
                Patient Summary
            </div>

            <div class="section-subtitle">
                Clinical information submitted for this assessment.
            </div>
            """
        )


        summary_col1, summary_col2 = st.columns(2)


        with summary_col1:

            st.metric(
                "Age",
                f"{age} years"
            )

            st.metric(
                "Resting BP",
                f"{resting_bp} mm Hg"
            )

            st.metric(
                "Cholesterol",
                f"{cholesterol} mg/dL"
            )

            st.metric(
                "Maximum Heart Rate",
                f"{max_hr} bpm"
            )


        with summary_col2:

            st.metric(
                "Chest Pain",
                chest_pain.split(" - ")[0]
            )

            st.metric(
                "Resting ECG",
                resting_ecg.split(" - ")[0]
            )

            st.metric(
                "Exercise Angina",
                exercise_angina
            )

            st.metric(
                "ST Slope",
                st_slope
            )


        # ====================================================
        # MODEL INPUT FEATURES
        # ====================================================

        with st.expander(
            "📋 View Model Input Features"
        ):

            display_df = pd.DataFrame(
                {
                    "Feature":
                        input_df.columns,

                    "Value":
                        input_df.iloc[0].values
                }
            )


            st.dataframe(
                display_df,

                use_container_width=True,

                hide_index=True
            )


        # ====================================================
        # DOWNLOAD REPORT
        # ====================================================

        prediction_text = (

            "Higher Likelihood of Heart Disease"

            if prediction == 1

            else

            "Lower Likelihood of Heart Disease"
        )


        report = f"""
CARDIOAI
HEART DISEASE ASSESSMENT
==========================================

PATIENT INFORMATION
==========================================

Age:
{age} years

Sex:
{sex}

Resting Blood Pressure:
{resting_bp} mm Hg

Cholesterol:
{cholesterol} mg/dL

Fasting Blood Sugar:
{fasting_bs}

Maximum Heart Rate:
{max_hr} bpm

Oldpeak:
{oldpeak}

Chest Pain Type:
{chest_pain}

Resting ECG:
{resting_ecg}

Exercise-Induced Angina:
{exercise_angina}

ST Slope:
{st_slope}


==========================================
MODEL RESULT
==========================================

Prediction:
{prediction_text}

Heart Disease Probability:
{probability_yes * 100:.2f}%

No Heart Disease Probability:
{probability_no * 100:.2f}%


==========================================
MODEL INFORMATION
==========================================

Algorithm:
K-Nearest Neighbors

Number of Neighbors:
5

Preprocessing:
StandardScaler

Number of Features:
15


==========================================
DISCLAIMER
==========================================

This prediction is generated by a machine
learning model for educational and
demonstration purposes only.

It should not be considered a medical diagnosis.

Please consult a qualified healthcare
professional for medical evaluation.
"""


        st.download_button(

            "📄  Download Assessment Report",

            data=report,

            file_name="cardioai_assessment.txt",

            mime="text/plain",

            use_container_width=True
        )


    except Exception as e:

        st.error(
            "❌ An error occurred during prediction."
        )

        st.exception(e)


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.html(
    """
    <div class="disclaimer">

        <strong>
            ⚕️ Medical Disclaimer
        </strong>

        <br><br>

        CardioAI is an educational machine learning
        application. The predictions generated by
        this model should not be interpreted as a
        clinical diagnosis, medical advice, or a
        substitute for professional medical evaluation.

    </div>


    <div class="footer">

        CARDIOAI
        &nbsp;•&nbsp;
        HEART DISEASE ASSESSMENT

    </div>
    """
)