import streamlit as st
import requests
import pandas as pd

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Healthcare Prediction System",
    page_icon="🏥",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a,
            #1e293b,
            #0f766e
        );
        color: white;
    }

    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 18px;
        color: #d1d5db;
        text-align: center;
        margin-bottom: 40px;
    }

    .prediction-box {
        background-color: rgba(255,255,255,0.08);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        margin-top: 20px;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.08);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# LABEL MAPPINGS
# =====================================================

binary_labels = {
    0: "Less than 1 Month Stay",
    1: "More than 1 Month Stay"
}

multi_labels = {
    0: "0-10 Days",
    1: "11-20 Days",
    2: "21-30 Days",
    3: "31-40 Days",
    4: "41-50 Days",
    5: "51-60 Days",
    6: "61-70 Days",
    7: "71-80 Days",
    8: "81-90 Days",
    9: "91-100 Days",
    10: "More than 100 Days"
}

# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
    <div class="main-title">
        🏥 Healthcare Stay Prediction System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        AI Powered Hospital Stay Duration Prediction using Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Prediction Settings")

model_choice = st.sidebar.selectbox(
    "Select Prediction Type",
    [
        "Binary Classification",
        "Multi-Class Classification"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    This system predicts:
    
    ✅ Patient Stay Duration
    
    ✅ Hospitalization Risk
    
    ✅ Probability Scores
    
    ✅ Confidence Levels
    """
)

# =====================================================
# INPUT SECTION
# =====================================================

st.subheader("📝 Patient Information")

col1, col2 = st.columns(2)

# =====================================================
# LEFT COLUMN
# =====================================================

with col1:

    Hospital_code = st.number_input(
        "Hospital Code",
        min_value=1,
        value=10
    )

    Hospital_type_code = st.selectbox(
        "Hospital Type Code",
        ["a", "b", "c", "d", "e", "f", "g"]
    )

    City_Code_Hospital = st.number_input(
        "City Code Hospital",
        min_value=1,
        value=1
    )

    Hospital_region_code = st.selectbox(
        "Hospital Region Code",
        ["X", "Y", "Z"]
    )

    Available_Extra_Rooms_in_Hospital = st.number_input(
        "Available Extra Rooms",
        min_value=0,
        value=2
    )

    Department = st.selectbox(
        "Department",
        [
            "radiotherapy",
            "anesthesia",
            "gynecology",
            "surgery",
            "TB & Chest disease"
        ]
    )

    Ward_Type = st.selectbox(
        "Ward Type",
        ["P", "Q", "R", "S", "T", "U"]
    )

# =====================================================
# RIGHT COLUMN
# =====================================================

with col2:

    Ward_Facility_Code = st.selectbox(
        "Ward Facility Code",
        ["A", "B", "C", "D", "E", "F"]
    )

    Bed_Grade = st.selectbox(
        "Bed Grade",
        [1, 2, 3, 4]
    )

    City_Code_Patient = st.number_input(
        "City Code Patient",
        min_value=0,
        value=7
    )

    Type_of_Admission = st.selectbox(
        "Type of Admission",
        ["Emergency", "Trauma", "Urgent"]
    )

    Severity_of_Illness = st.selectbox(
        "Severity of Illness",
        ["Minor", "Moderate", "Extreme"]
    )

    Visitors_with_Patient = st.number_input(
        "Visitors with Patient",
        min_value=0,
        value=3
    )

    Age = st.selectbox(
        "Age",
        [
            "0-10",
            "11-20",
            "21-30",
            "31-40",
            "41-50",
            "51-60",
            "61-70",
            "71-80",
            "81-90",
            "91-100"
        ]
    )

    Admission_Deposit = st.number_input(
        "Admission Deposit",
        min_value=0,
        value=5000
    )

# =====================================================
# PREDICT BUTTON
# =====================================================

st.markdown("---")

if st.button("🔍 Generate Prediction", use_container_width=True):

    input_data = {

        "Hospital_code": Hospital_code,
        "Hospital_type_code": Hospital_type_code,
        "City_Code_Hospital": City_Code_Hospital,
        "Hospital_region_code": Hospital_region_code,
        "Available_Extra_Rooms_in_Hospital": Available_Extra_Rooms_in_Hospital,
        "Department": Department,
        "Ward_Type": Ward_Type,
        "Ward_Facility_Code": Ward_Facility_Code,
        "Bed_Grade": Bed_Grade,
        "City_Code_Patient": City_Code_Patient,
        "Type_of_Admission": Type_of_Admission,
        "Severity_of_Illness": Severity_of_Illness,
        "Visitors_with_Patient": Visitors_with_Patient,
        "Age": Age,
        "Admission_Deposit": Admission_Deposit
    }

    if model_choice == "Binary Classification":
        endpoint = "binary"
    else:
        endpoint = "multiclass"

    try:

        with st.spinner("Analyzing Patient Data..."):

            response = requests.post(
                f"http://127.0.0.1:8000/predict/{endpoint}",
                json=input_data
            )

            result = response.json()

        st.success("Prediction Generated Successfully")

        # =================================================
        # PREDICTION RESULT
        # =================================================

        raw_prediction = result["prediction"]

        if endpoint == "binary":

            final_prediction = binary_labels.get(
                int(raw_prediction),
                raw_prediction
            )

        else:

            final_prediction = multi_labels.get(
                int(raw_prediction),
                raw_prediction
            )

        st.markdown("---")

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:

            st.metric(
                "Predicted Stay Duration",
                final_prediction
            )

        with metric_col2:

            if "confidence_score" in result:

                st.metric(
                    "Confidence Score",
                    f"{result['confidence_score']}%"
                )

        # =================================================
        # BINARY PROBABILITIES
        # =================================================

        if "probabilities" in result:

            st.subheader("📊 Prediction Chances")

            binary_probs = {
                binary_labels[int(k.split('_')[1])]: v
                for k, v in result["probabilities"].items()
            }

            prob_df = pd.DataFrame(
                binary_probs.items(),
                columns=["Stay Type", "Chance (%)"]
            )

            st.dataframe(
                prob_df,
                use_container_width=True
            )

            st.bar_chart(
                prob_df.set_index("Stay Type")
            )

        # =================================================
        # MULTI CLASS PROBABILITIES
        # =================================================

        if "all_class_probabilities" in result:

            st.subheader("📈 All Stay Duration Chances")

            multi_probs = {
                multi_labels[int(k)]: v
                for k, v in result["all_class_probabilities"].items()
            }

            multi_df = pd.DataFrame(
                multi_probs.items(),
                columns=["Stay Duration", "Chance (%)"]
            )

            st.dataframe(
                multi_df,
                use_container_width=True
            )

            st.bar_chart(
                multi_df.set_index("Stay Duration")
            )

    except Exception as e:

        st.error(f"Error: {e}")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <h5 style='color:lightgray;'>
        Developed using Streamlit + FastAPI + Machine Learning
    </h5>
    </center>
    """,
    unsafe_allow_html=True
)