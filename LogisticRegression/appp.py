import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("logistic_model.pkl")

st.title("🚢 Titanic Survival Prediction App")

st.write("""
Enter the passenger information below to predict survival on the Titanic.
""")

# User input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.slider("Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert inputs to numerical format
sex = 1 if sex == "male" else 0
embarked_dict = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_dict[embarked]

input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# Predict button
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    result = "Survived 🟢" if prediction == 1 else "Did Not Survive 🔴"
    st.subheader(f"Prediction: {result}")
