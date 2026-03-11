import joblib

model = joblib.load("models/month_energy_model.pkl")

forecast = model.forecast(steps=1)

print("Predicted Next Month Energy Consumption:", forecast[0])

forecast_yearly = model.forecast(steps=12)

print("Predicted Next Year Energy Consumption:", forecast_yearly[0])