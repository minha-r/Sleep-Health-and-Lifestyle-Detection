import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SleepWell AI",
    page_icon="🌙",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: bold;
    color: #6558D3;
    text-align: center;
}

.subtitle {
    font-size: 17px;
    color: #777777;
    text-align: center;
}

.result-title {
    font-size: 24px;
    font-weight: bold;
    margin-top: 15px;
}

.result-card {
    padding: 20px;
    border-radius: 15px;
    background-color: #F4F2FF;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MACHINE LEARNING MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/svm_sleep_disorder_model.pkl"
    )

    feature_columns = list(model.feature_names_in_)

    return model, feature_columns


try:
    model, feature_columns = load_model()

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🌙 SleepWell AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Sleep Disorder Prediction using Lifestyle Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()


# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------

st.sidebar.title("👤 Your Sleep Profile")

st.sidebar.write(
    "Enter your health and lifestyle information below."
)

st.sidebar.subheader("Personal Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.sidebar.slider(
    "Age",
    18, 80, 30
)

occupation = st.sidebar.selectbox(
    "Occupation",
    [
        "Software Engineer",
        "Doctor",
        "Sales Representative",
        "Teacher",
        "Nurse",
        "Engineer",
        "Accountant",
        "Scientist",
        "Lawyer",
        "Manager",
        "Salesperson"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("😴 Sleep Information")

sleep_duration = st.sidebar.slider(
    "Sleep Duration (hours)",
    3.0, 12.0, 7.0, 0.1
)

quality_sleep = st.sidebar.slider(
    "Quality of Sleep",
    1, 10, 7
)

st.sidebar.divider()

st.sidebar.subheader("🏃 Lifestyle Information")

physical_activity = st.sidebar.slider(
    "Physical Activity Level",
    0, 100, 50
)

stress_level = st.sidebar.slider(
    "Stress Level",
    1, 10, 5
)

bmi_category = st.sidebar.selectbox(
    "BMI Category",
    [
        "Normal",
        "Normal Weight",
        "Overweight",
        "Obese"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("❤️ Health Information")

systolic_bp = st.sidebar.number_input(
    "Systolic Blood Pressure",
    min_value=80,
    max_value=200,
    value=120
)

diastolic_bp = st.sidebar.number_input(
    "Diastolic Blood Pressure",
    min_value=40,
    max_value=130,
    value=80
)

heart_rate = st.sidebar.slider(
    "Heart Rate (BPM)",
    40, 150, 75
)

daily_steps = st.sidebar.number_input(
    "Daily Steps",
    min_value=0,
    max_value=50000,
    value=7000
)


# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "🔍 Prediction",
    "📊 Lifestyle Overview",
    "ℹ️ About Project"
])

# --------------------------------------------------
# TAB 1 - PREDICTION
# --------------------------------------------------

with tab1:

    st.subheader("Sleep Disorder Analysis")

    st.write(
        "Analyze your sleep patterns and lifestyle "
        "to predict your possible sleep disorder."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Sleep Duration",
            f"{sleep_duration} hrs"
        )

    with col2:
        st.metric(
            "Sleep Quality",
            f"{quality_sleep}/10"
        )

    with col3:
        st.metric(
            "Daily Activity",
            f"{daily_steps:,} steps"
        )

    st.write("")

    predict_button = st.button(
        "✨ Predict My Sleep Condition",
        use_container_width=True
    )

    if predict_button:

        try:

            # --------------------------------------
            # CREATE INPUT DATA
            # --------------------------------------

            input_data = pd.DataFrame([{
                "Gender": gender,
                "Age": age,
                "Occupation": occupation,
                "Sleep Duration": sleep_duration,
                "Quality of Sleep": quality_sleep,
                "Physical Activity Level": physical_activity,
                "Stress Level": stress_level,
                "BMI Category": bmi_category,
                "Blood Pressure": f"{systolic_bp}/{diastolic_bp}",
                "Heart Rate": heart_rate,
                "Daily Steps": daily_steps
            }])

            # --------------------------------------
            # ONE-HOT ENCODING
            # --------------------------------------

            categorical_columns = [
                "Gender",
                "Occupation",
                "BMI Category",
                "Blood Pressure"
            ]

            input_encoded = pd.get_dummies(
                input_data,
                columns=categorical_columns,
                dtype=int
            )

            # --------------------------------------
            # MATCH TRAINING FEATURES
            # --------------------------------------

            input_encoded = input_encoded.reindex(
                columns=feature_columns,
                fill_value=0
            )

            # --------------------------------------
            # PREDICTION
            # --------------------------------------

            predicted_label = model.predict(input_encoded)[0]

            # --------------------------------------
            # DISPLAY RESULT
            # --------------------------------------

            st.divider()

            st.subheader("Prediction Result")

            if predicted_label == "None":

                st.success("Normal Sleep Pattern")

                st.markdown(
                    "### No Sleep Disorder Detected"
                )

            elif predicted_label == "Insomnia":

                st.warning("Insomnia Prediction")

                st.markdown(
                    "### Possible Insomnia"
                )

            elif predicted_label == "Sleep Apnea":

                st.error("Sleep Apnea Prediction")

                st.markdown(
                    "### Possible Sleep Apnea"
                )

            st.info(
                "This prediction is based on patterns learned "
                "from the training dataset. It is not a medical diagnosis."
            )

        except Exception as e:

            st.error(f"Prediction error: {e}")
# --------------------------------------------------
# TAB 2 - LIFESTYLE OVERVIEW
# --------------------------------------------------

with tab2:

    st.subheader("Your Lifestyle Overview")

    col1, col2 = st.columns(2)

    with col1:

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=sleep_duration,
                title={"text": "Sleep Duration (Hours)"},
                gauge={
                    "axis": {"range": [0, 12]},
                    "bar": {"color": "#6558D3"},
                    "steps": [
                        {"range": [0, 6], "color": "#FEE2E2"},
                        {"range": [6, 8], "color": "#DCFCE7"},
                        {"range": [8, 12], "color": "#E0E7FF"}
                    ]
                }
            )
        )

        fig.update_layout(height=300)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=quality_sleep,
                title={"text": "Sleep Quality"},
                gauge={
                    "axis": {"range": [0, 10]},
                    "bar": {"color": "#16A085"},
                    "steps": [
                        {"range": [0, 4], "color": "#FEE2E2"},
                        {"range": [4, 7], "color": "#FEF3C7"},
                        {"range": [7, 10], "color": "#DCFCE7"}
                    ]
                }
            )
        )

        fig.update_layout(height=300)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("Lifestyle Details")

    lifestyle_data = pd.DataFrame({
        "Parameter": [
            "Physical Activity",
            "Stress Level",
            "Heart Rate",
            "Daily Steps",
            "BMI Category",
            "Blood Pressure"
        ],

        "Value": [
            f"{physical_activity}",
            f"{stress_level}/10",
            f"{heart_rate} BPM",
            f"{daily_steps:,}",
            bmi_category,
            f"{systolic_bp}/{diastolic_bp}"
        ]
    })

    st.dataframe(
        lifestyle_data,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# TAB 3 - ABOUT PROJECT
# --------------------------------------------------

with tab3:

    st.subheader("About SleepWell AI")

    st.write("""
    SleepWell AI is a Machine Learning-based classification
    application designed to predict possible sleep disorders
    using sleep health and lifestyle information.
    """)

    st.markdown("### 🎯 Project Objectives")

    st.write("""
    - Analyze sleep patterns and lifestyle factors.
    - Identify possible sleep disorders.
    - Apply Machine Learning classification algorithms.
    - Compare model performance.
    - Provide predictions through an interactive dashboard.
    """)

    st.markdown("### 🧠 Prediction Categories")

    st.write("""
    **1. None:** No sleep disorder is recorded.

    **2. Insomnia:** A condition associated with difficulty
    falling asleep or staying asleep.

    **3. Sleep Apnea:** A condition involving repeated
    interruptions in breathing during sleep.
    """)

    st.markdown("### ⚙️ Technologies Used")

    st.write("""
    - Python
    - Pandas & NumPy
    - Scikit-learn
    - Support Vector Machine (SVM)
    - Streamlit
    - Plotly
    """)

    st.caption(
        "SleepWell AI | Machine Learning Project"
    )