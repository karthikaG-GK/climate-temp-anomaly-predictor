import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

model = load_model('model/bestmodel.h5', custom_objects={'mse': MeanSquaredError()})
scaler = joblib.load("model/scaler.pkl")

# Load your full global temperature dataset
df = pd.read_csv("preprocessed_files/global_temp_df.csv")
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time")

# Scale the temperature data
temp_data = df[["Global_Temp"]].values
scaled_temp = scaler.transform(temp_data)

# Sequence creation function

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/premium-photo/thermometer-spotlight-isolated_172429-1566.jpg");
             background-attachment: fixed;
             background-size: cover;
             background-repeat: no-repeat;
             background-position: center;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
def align_right():
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 100% !important;
            margin-left: auto;
            margin-top: 30;
            margin-right: 0;  /* Pushes content to the right */
            padding-right: 80px;  /* Add space from the right edge */
        }

        .stTitle, .stMarkdown, .stNumberInput, .stSelectbox, .stButton {
            white-space: nowrap !important;  /* Prevent text wrap */
            
        }
        </style>
        """,
        unsafe_allow_html=True
    )
def create_input_sequence(data, window_size):
    return data[-window_size:].reshape(1, window_size, 1)

add_bg_from_url()
align_right()

left_col, center_col, right_col = st.columns([1, 1, 2]) 

def predict_for_future_date(model, scaler, scaled_data, future_date, window_size):
  
    last_date = df["time"].max()
    months_diff = (future_date.year - last_date.year) * 12 + (future_date.month - last_date.month)

    input_seq = scaled_data[-window_size:].tolist()

    for i in range(months_diff):
        seq_input = np.array(input_seq[-window_size:]).reshape(1, window_size, 1)
        next_pred = model.predict(seq_input)[0][0]
        input_seq.append([next_pred])  # Append the predicted value for next step

    final_prediction = scaler.inverse_transform([[input_seq[-1][0]]])[0][0]
    return final_prediction

with right_col:
    st.title("Global Temperature Anomaly Predictor")
    st.write("Enter a future date to predict the temperature anomaly.")
    month = st.selectbox("Select Month", list(range(1, 13)))
    year = st.number_input("Enter Year", min_value=df["time"].dt.year.max() + 1, step=1)

    if st.button("Predict"):
        try:
            future_date = datetime(year=int(year), month=int(month), day=1)
            predicted_temp = predict_for_future_date(model, scaler, scaled_temp, future_date, window_size=30)
            if predicted_temp > 2:
                st.error(f"Predicted anomaly for {future_date.strftime('%B %Y')}: {predicted_temp:.4f} ℃ \n Temperature is high 🥵")
            else:
                st.success(f"Predicted anomaly for {future_date.strftime('%B %Y')}: {predicted_temp:.4f} ℃")
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
