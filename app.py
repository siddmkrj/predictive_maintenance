"""
Predictive Maintenance — Streamlit Application
Loads the best trained model (Decision Tree, Random Forest, Logistic Regression, or Gradient Boosting)
from local path or Hugging Face Model Hub and provides a web interface for engine failure prediction.
"""

import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# ─── Config ───
MODEL_REPO_ID = "mukherjee78/predictive-maintenance-random-forest"
MODEL_FILENAME = "best_model.pkl"
LOCAL_MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pkl")

# Feature order must match training data (data_preparation.py / model_training.py)
FEATURE_COLUMNS = [
    "engine_rpm",
    "lub_oil_pressure",
    "fuel_pressure",
    "coolant_pressure",
    "lub_oil_temp",
    "coolant_temp",
]

# ─── Page Configuration ───
st.set_page_config(
    page_title="Engine Predictive Maintenance",
    page_icon="⚙️",
    layout="centered",
)

# ─── Load Model (local file or Hugging Face) ───
# Loaded lazily on first prediction so the Space UI appears immediately (avoids startup hang).
@st.cache_resource
def load_model():
    """Load model from LOCAL_MODEL_PATH if present, else from Hugging Face Model Hub."""
    try:
        if os.path.isfile(LOCAL_MODEL_PATH):
            return joblib.load(LOCAL_MODEL_PATH)
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model",
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Ensure the model exists at Hugging Face or set MODEL_PATH to a local file.")
        raise


# ─── App Header ───
st.title("⚙️ Engine Predictive Maintenance")
st.markdown(
    "Enter engine sensor readings below to predict whether the engine "
    "requires **maintenance** or is operating under **normal** conditions."
)

st.divider()

# ─── Input Form ───
st.subheader("Sensor Readings")

col1, col2 = st.columns(2)

with col1:
    engine_rpm = st.number_input(
        "Engine RPM",
        min_value=0, max_value=3000, value=700, step=10,
        help="Rotational speed of the engine in revolutions per minute"
    )
    lub_oil_pressure = st.number_input(
        "Lub Oil Pressure (bar)",
        min_value=0.0, max_value=10.0, value=3.0, step=0.1,
        help="Lubricating oil pressure in bar"
    )
    fuel_pressure = st.number_input(
        "Fuel Pressure (bar)",
        min_value=0.0, max_value=50.0, value=12.0, step=0.5,
        help="Fuel supply pressure in bar"
    )

with col2:
    coolant_pressure = st.number_input(
        "Coolant Pressure (bar)",
        min_value=0.0, max_value=10.0, value=2.5, step=0.1,
        help="Coolant system pressure in bar"
    )
    lub_oil_temp = st.number_input(
        "Lub Oil Temperature (°C)",
        min_value=0.0, max_value=200.0, value=80.0, step=1.0,
        help="Lubricating oil temperature in degrees Celsius"
    )
    coolant_temp = st.number_input(
        "Coolant Temperature (°C)",
        min_value=0.0, max_value=200.0, value=82.0, step=1.0,
        help="Coolant temperature in degrees Celsius"
    )

st.divider()

# ─── Prediction ───
if st.button("🔍 Predict Engine Condition", type="primary", use_container_width=True):

    # Save inputs into a dataframe (column order must match training)
    input_data = pd.DataFrame(
        [[engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp]],
        columns=FEATURE_COLUMNS,
    )

    # Display input summary
    st.markdown("**Input Summary:**")
    st.dataframe(input_data, use_container_width=True, hide_index=True)

    # Load model on first use (so Space shows UI immediately; download happens here with spinner)
    with st.spinner("Loading model..."):
        try:
            artifact = load_model()
        except Exception:
            st.stop()

    # Unwrap artifact: either raw estimator (legacy) or dict with model + optional scaler
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        scaler = artifact.get("scaler")
    else:
        model = artifact
        scaler = None

    # Apply scaler for Logistic Regression (features were scaled at training time)
    X = input_data
    if scaler is not None:
        X = scaler.transform(input_data)

    # Make prediction
    try:
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.divider()

    # Display result (1 = needs maintenance, 0 = normal)
    if prediction == 1:
        st.error("🔴 **Prediction: Engine Needs Maintenance**")
        st.metric("Failure Probability", f"{probability[1]:.1%}")
        st.warning(
            "⚠️ The sensor readings indicate potential engine degradation. "
            "Schedule maintenance to prevent unplanned downtime."
        )
    else:
        st.success("🟢 **Prediction: Engine Operating Normally**")
        st.metric("Normal Probability", f"{probability[0]:.1%}")
        st.info(
            "✅ The engine appears to be operating within normal parameters. "
            "Continue routine monitoring."
        )

# ─── Footer ───
st.divider()
st.caption(
    "Model: Best of Decision Tree / Random Forest / Logistic Regression / Gradient Boosting / XGBoost | "
    "Source: [Hugging Face Model Hub](https://huggingface.co/mukherjee78/predictive-maintenance-random-forest)"
)
