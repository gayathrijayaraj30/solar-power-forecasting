import pandas as pd
import numpy as np

def load_raw_data(inverter_path, plant_path, weather_path):
    inverter = pd.read_csv(inverter_path, parse_dates=['Updated Time'])
    plant = pd.read_csv(plant_path, parse_dates=['Updated Time'])
    weather = pd.read_csv(weather_path, parse_dates=['date'])

    inverter.rename(columns={'Updated Time': 'date'}, inplace=True)
    plant.rename(columns={'Updated Time': 'date'}, inplace=True)
    weather.rename(columns={'date': 'date'}, inplace=True)

    return inverter, plant, weather

def preprocess_and_merge(inverter, plant, weather):
    # Aggregate inverter by date (sum if multiple records)
    inv_agg = inverter.groupby('date').agg({
        'Production(kWh)': 'sum',
        'Consumption(kWh)': 'sum',
        'Grid Feed-in(kWh)': 'sum',
        'Electricity Purchasing(kWh)': 'sum'
    }).reset_index()

    # Merge plant and inverter data on date
    df = pd.merge(plant, inv_agg, on='date', how='inner', suffixes=('_plant', '_inverter'))

    # Merge weather data on date
    df = pd.merge(df, weather, on='date', how='inner')

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    # --- FEATURE ENGINEERING ---

    # Temperature range
    df['temp_range'] = df['temp_max_C'] - df['temp_min_C']

    # Binary precipitation flag (rain or no rain)
    df['rain_flag'] = (df['precipitation_mm'] > 0).astype(int)

    # Time since last rain (in days)
    days_since_rain = []
    count = 1000  # large number for initial days
    for rain in df['rain_flag']:
        if rain == 1:
            count = 0
        else:
            count += 1
        days_since_rain.append(count)
    df['days_since_rain'] = days_since_rain

    # Cyclical encoding for day of week and month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lag features for key variables (lags of 1, 2, 3, 7 days)
    lags = [1, 2, 3, 7]
    for lag in lags:
        df[f'prod_lag_{lag}'] = df['Production(kWh)'].shift(lag)
        df[f'cons_lag_{lag}'] = df['Consumption(kWh)'].shift(lag)
        df[f'temp_max_lag_{lag}'] = df['temp_max_C'].shift(lag)
        df[f'temp_min_lag_{lag}'] = df['temp_min_C'].shift(lag)
        df[f'precip_lag_{lag}'] = df['precipitation_mm'].shift(lag)
        df[f'rain_flag_lag_{lag}'] = df['rain_flag'].shift(lag)

    # Rolling window features: 3-day and 7-day rolling mean and std for production and temperature max
    windows = [3, 7]
    for window in windows:
        df[f'prod_rollmean_{window}'] = df['Production(kWh)'].rolling(window).mean()
        df[f'prod_rollstd_{window}'] = df['Production(kWh)'].rolling(window).std()
        df[f'tempmax_rollmean_{window}'] = df['temp_max_C'].rolling(window).mean()
        df[f'tempmax_rollstd_{window}'] = df['temp_max_C'].rolling(window).std()

    # Forward fill missing values (for lags/rolling), then backfill as fallback
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # Drop any remaining NaNs just in case
    df.dropna(inplace=True)

    return df

def save_processed_data(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    inverter_path = 'data/inverter.csv'
    plant_path = 'data/plant.csv'
    weather_path = 'data/weather.csv'
    output_path = 'data/merged_featured_data.csv'


    print("Loading raw data...")
    inverter, plant, weather = load_raw_data(inverter_path, plant_path, weather_path)

    print("Preprocessing and feature engineering...")
    df = preprocess_and_merge(inverter, plant, weather)

    print(f"Saving processed data to {output_path}...")
    save_processed_data(df, output_path)

    print("Data preprocessing complete!")
