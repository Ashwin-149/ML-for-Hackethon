import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib


# load dataset
data = pd.read_csv("dataset/energy_dataset.csv", nrows=50000)

# clean data
data.replace("?", None, inplace=True)
data = data.dropna()

# convert numeric columns
data = data.astype({
    'Global_active_power': float,
    'Voltage': float,
    'Global_intensity': float,
    'Sub_metering_1': float,
    'Sub_metering_2': float,
    'Sub_metering_3': float
})

# create datetime features
data['datetime'] = pd.to_datetime(data['Date'] + " " + data['Time'],dayfirst=True)

data['hour'] = data['datetime'].dt.hour
data['day_of_week'] = data['datetime'].dt.dayofweek
data['month'] = data['datetime'].dt.month

features = [
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3',
    'hour',
    'day_of_week',
    'month'
]

target = 'Global_active_power'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))

# save model
joblib.dump(model, "models/energy_model.pkl")

print("Model trained and saved.")