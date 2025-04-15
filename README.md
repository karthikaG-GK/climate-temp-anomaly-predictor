# 🌍 Climate Temp Anomaly Predictor
Predict the future, one degree at a time.
This project is a machine learning-powered web app that predicts global temperature anomalies for any given future month and year. It leverages deep learning (LSTM) models trained on historic global climate data and is wrapped in an interactive Streamlit interface.

# 📊 Demo
🌐 Live App (if deployed): https://your-deployment-url.com
🖼️ Sample Prediction: "Predicted anomaly for March 2040: +1.8253 ℃"

# 🧠 Project Motivation

* Climate change is real — and knowing what lies ahead is crucial. This tool aims to help scientists, students, and the public:
* Understand global temperature trends.
* Predict possible future anomalies using historical data.
* Visualize the long-term impact of climate change.

# 🧰 Tech Stack

| Steps         | Tools Used                       |
|---------------|----------------------------------|
| Data          |	NOAA, .csv files, .nc files      |
| Preprocessing	| Pandas, NumPy, Scikit-learn, EM  |
| Model         |	LSTM (TensorFlow / Keras)        |
| Frontend    	| Streamlit                        |
| Deployment    | GitHub                           |

# 🚀 Features
📅 Choose any future month/year to predict anomalies
🔁 Automatically scales and processes input data
💡 Uses a windowed time series approach (30-day sequence)
🌐 Fully interactive Streamlit web UI
🎨 Custom background image and layout
✅ Trained model loaded using .h5, with MinMaxScaler support
⚙️ Easy to deploy and extend

# 📂 Project Structure
```
climate-temp-anomaly-predictor/
│
├── app.py                     # Streamlit web app
├── model/
│   ├── bestmodel.h5           # Trained LSTM model
│   ├── scaler.pkl             # MinMaxScaler used during training
├── preprocessed_files/
│   └── global_temp_df.csv     # Cleaned and imputed dataset
├── requirements.txt           # Required Python packages
├── README.md                  # Project overview and instructions
└── .gitignore
```
# 💻 Usage

Launch the app with streamlit run app.py
Select a future month and year
Click Predict
View the predicted temperature anomaly in Celsius

# 📈 Model Overview

Model Type: LSTM (Long Short-Term Memory)
Input: 30-step historical temperature sequence
Output: Single anomaly value
Loss Function: Mean Squared Error (MSE)
Scaling: MinMaxScaler used during preprocessing

# ⚠️ Limitations

Only trained on global average temperatures — doesn't account for region-specific patterns
Assumes historical patterns continue into the future (no external forcing variables)
Extrapolation too far into the future may reduce accuracy

# 📌 Future Improvements

Region-based anomaly prediction
Add confidence intervals or uncertainty estimates
Deploy as an API + frontend

# 📜 License

This project is licensed under the MIT License.

# 🙋‍♀️ Acknowledgments

NOAA for historical temperature datasets
Streamlit for making ML apps easy to deploy
TensorFlow/Keras for the modeling framework

