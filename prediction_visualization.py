import joblib
import pandas as pd
import matplotlib.pyplot as plt

# load trained model
model = joblib.load("models/energy_model.pkl")

predictions = []
hours = []

# assume fixed electrical parameters
voltage = 234
intensity = 18
sub1 = 0
sub2 = 1
sub3 = 17
day_of_week = 2
month = 6

for hour in range(24):

    input_data = pd.DataFrame([{
        "Voltage": voltage,
        "Global_intensity": intensity,
        "Sub_metering_1": sub1,
        "Sub_metering_2": sub2,
        "Sub_metering_3": sub3,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month
    }])

    prediction = model.predict(input_data)

    predictions.append(prediction[0])
    hours.append(hour)

# plot prediction
plt.figure(figsize=(10,5))

plt.plot(hours, predictions, marker='o')

plt.xlabel("Hour of Day")
plt.ylabel("Predicted Energy Consumption (kW)")
plt.title("24 Hour Energy Demand Forecast")

plt.grid(True)

plt.show()