"""
Step 6b: Streamlit app - Turbofan Engine RUL Predictor

SETUP (run locally, not in Colab):
1. Rename this file to app.py (or run it as-is with its current name).
2. Put these files in the SAME folder:
   - turbofan_app.py (this file)
   - rul_model.pkl
   - feature_cols.pkl
   - sensor_defaults.pkl
   (all 3 .pkl files come from running Step 6a in Colab, then downloading them)
3. In CMD, cd into that folder, then:
   pip install streamlit joblib pandas scikit-learn
   streamlit run turbofan_app.py
4. It opens automatically in your browser at http://localhost:8501
"""

import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Engine RUL Predictor", page_icon="🔧")

model = joblib.load("rul_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")
sensor_defaults = joblib.load("sensor_defaults.pkl")

st.title("🔧 Turbofan Engine — Remaining Useful Life Predictor")
st.write(
    "Enter sensor readings for an engine cycle and get a predicted "
    "Remaining Useful Life (RUL), in cycles, before maintenance is needed."
)

# The 4 sensors that matter most (from Step 5's feature importance) get
# their own sliders. Everything else uses the training-data median as a
# sensible default, editable in the "Advanced" expander below.
key_sensors = ["sensor_11", "sensor_9", "sensor_4", "sensor_3"]

st.subheader("Key sensor readings")
user_inputs = {}
for sensor in key_sensors:
    default_val = float(sensor_defaults[sensor])
    user_inputs[sensor] = st.slider(
        sensor,
        min_value=float(default_val * 0.7),
        max_value=float(default_val * 1.3),
        value=default_val,
    )

with st.expander("Advanced: adjust remaining sensors (optional)"):
    for sensor in feature_cols:
        if sensor in key_sensors:
            continue
        default_val = float(sensor_defaults[sensor])
        user_inputs[sensor] = st.number_input(sensor, value=default_val)

if st.button("Predict RUL"):
    input_df = pd.DataFrame([user_inputs])[feature_cols]  # enforce correct order
    prediction = model.predict(input_df)[0]

    st.metric("Predicted Remaining Useful Life", f"{prediction:.0f} cycles")

    if prediction < 20:
        st.error("⚠️ Critical: schedule maintenance immediately.")
    elif prediction < 50:
        st.warning("⚠️ Warning: plan maintenance soon.")
    else:
        st.success("✅ Healthy: no immediate action needed.")

    st.caption(
        "Note: this model's average error (MAE) was ~12 cycles on test data — "
        "treat this prediction as a decision-support estimate, not an exact countdown."
    )
