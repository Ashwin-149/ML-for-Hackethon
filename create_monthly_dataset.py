import pandas as pd

data = pd.read_csv("dataset/energy_dataset.csv")

data.replace("?", None, inplace=True)
data = data.dropna()

data['Global_active_power'] = data['Global_active_power'].astype(float)

data['datetime'] = pd.to_datetime(
    data['Date'] + " " + data['Time'],
    format="%d/%m/%Y %H:%M:%S"
)

# Monthly energy consumption
monthly = data.resample('M', on='datetime')['Global_active_power'].sum()

monthly = monthly.reset_index()

monthly.to_csv("monthly_energy.csv", index=False)

print("Monthly dataset created")