import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib

data = pd.read_csv("monthly_energy.csv")

series = data['Global_active_power']

model = ARIMA(series, order=(5,1,0))

model_fit = model.fit()

joblib.dump(model_fit, "models/month_energy_model.pkl")

print("Monthly forecasting model saved")