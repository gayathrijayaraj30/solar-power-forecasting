import pandas as pd
import numpy as np
import joblib

# Load saved model and data
model = joblib.load('/Users/gayathrijayaraj/Desktop/energy-forecasting-project/backend/models/xgboost_model.pkl')
data = pd.read_csv('data/merged_featured_data.csv', parse_dates=['date'])

features = [
    'Plant Name',
    'Time Zone',
    'Production-Today(kWh)',
    'Consumption-Today(kWh)',
    'Feed-in Power-Today(kWh)',
    'Energy Purchased-Today(kWh)',
    'Self-used Ratio(%)',
    'Anticipated Yield(INR)',
    'Consumption(kWh)',
    'Grid Feed-in(kWh)',
    'Electricity Purchasing(kWh)',
    'temp_max_C',
    'temp_min_C',
    'precipitation_mm',
    'temp_range',
    'rain_flag',
    'days_since_rain',
    'day_of_week',
    'month',
    'day_of_week_sin',
    'day_of_week_cos',
    'month_sin',
    'month_cos',
    'prod_lag_1',
    'cons_lag_1',
    'temp_max_lag_1',
    'temp_min_lag_1',
    'precip_lag_1',
    'rain_flag_lag_1',
    'prod_lag_2',
    'cons_lag_2',
    'temp_max_lag_2',
    'temp_min_lag_2',
    'precip_lag_2',
    'rain_flag_lag_2',
    'prod_lag_3',
    'cons_lag_3',
    'temp_max_lag_3',
    'temp_min_lag_3',
    'precip_lag_3',
    'rain_flag_lag_3',
    'prod_lag_7',
    'cons_lag_7',
    'temp_max_lag_7',
    'temp_min_lag_7',
    'precip_lag_7',
    'rain_flag_lag_7',
    'prod_rollmean_3',
    'prod_rollstd_3',
    'tempmax_rollmean_3',
    'tempmax_rollstd_3',
    'prod_rollmean_7',
    'prod_rollstd_7',
    'tempmax_rollmean_7',
    'tempmax_rollstd_7'
]

X = data[features]

# Convert categorical columns to category dtype
categorical_cols = ['Plant Name', 'Time Zone', 'rain_flag', 'rain_flag_lag_1', 'rain_flag_lag_2', 'rain_flag_lag_3', 'rain_flag_lag_7']
for col in categorical_cols:
    X[col] = X[col].astype('category')

y_true = data['Production(kWh)']

# Predict
y_pred = model.predict(X)

# Compute residuals
residuals = y_true - y_pred
std_dev = np.std(residuals)

# Flag anomalies: threshold = 2 standard deviations
threshold = 2 * std_dev
data['residual'] = residuals
data['anomaly'] = abs(residuals) > threshold

# Save results
data[['date', 'Production(kWh)', 'residual', 'anomaly']].to_csv('data/anomaly_results.csv', index=False)

print(f"Anomalies detected: {data['anomaly'].sum()}")
