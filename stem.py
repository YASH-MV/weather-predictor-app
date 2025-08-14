import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Load dataset
data = pd.read_csv("data.csv")

# Prepare features and labels
X = data[["Temperature", "Humidity", "WindSpeed", "Pressure"]]
y = data["Weather"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🌦 Weather Prediction App")
st.write("Enter the weather details below to predict if the day will be Sunny, Rainy, or Overcast.")

# Input fields
temp = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
wind = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=150.0, step=0.1)
pressure = st.number_input("Pressure (hPa)", min_value=800.0, max_value=1100.0, step=0.1)

# Predict button
if st.button("Predict Weather"):
    user_data = pd.DataFrame({
        "Temperature": [temp],
        "Humidity": [humidity],
        "WindSpeed": [wind],
        "Pressure": [pressure]
    })

    prediction = model.predict(user_data)[0]
    st.subheader(f"Predicted Weather: **{prediction}**")