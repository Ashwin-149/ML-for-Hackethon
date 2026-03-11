import joblib
import numpy as np

# load trained model
model = joblib.load("models/energy_model.pkl")

# example input
input_data = np.array([[234, 18, 0, 1, 17, 14, 2, 6]])

prediction = model.predict(input_data)

print("Predicted Energy Consumption:", prediction[0], "kW")