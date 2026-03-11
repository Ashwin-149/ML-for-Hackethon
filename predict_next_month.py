import joblib

model = joblib.load("models/month_energy_model.pkl")

# next month
forecast_month = model.forecast(steps=1)

print("Predicted Next Month Energy:", forecast_month.iloc[0], "KW")

# next year
forecast_year = model.forecast(steps=12)

print("\nNext Year Monthly Forecast:")
for i, val in enumerate(forecast_year):
    print(f"Month {i+1}:", val)

print("\nTotal Next Year Energy:", forecast_year.sum(), "KW")