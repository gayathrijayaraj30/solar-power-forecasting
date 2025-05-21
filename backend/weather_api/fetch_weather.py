import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_weather_data_chunk(start_date, end_date, latitude, longitude):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&timezone=Asia/Kolkata"
    )
    response = requests.get(url)
    data = response.json()
    
    if 'daily' not in data:
        raise ValueError(f"'daily' key not found in API response: {data}")
    
    daily = data['daily']
    df = pd.DataFrame({
        'date': daily['time'],
        'temp_max_C': daily['temperature_2m_max'],
        'temp_min_C': daily['temperature_2m_min'],
        'precipitation_mm': daily['precipitation_sum'],
    })
    return df

def fetch_weather_data(start_date, end_date, latitude=10.52385, longitude=76.21313):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # We'll fetch data in 90-day chunks to be safe
    delta = timedelta(days=90)
    
    dfs = []
    current_start = start
    
    while current_start <= end:
        current_end = min(current_start + delta - timedelta(days=1), end)
        print(f"Fetching data from {current_start.date()} to {current_end.date()}...")
        
        try:
            df_chunk = fetch_weather_data_chunk(current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d"), latitude, longitude)
            dfs.append(df_chunk)
        except Exception as e:
            print(f"Failed to fetch data for {current_start.date()} to {current_end.date()}: {e}")
        
        current_start = current_end + timedelta(days=1)
    
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        return full_df
    else:
        print("No data fetched.")
        return pd.DataFrame()

if __name__ == "__main__":
    start_date = "2024-04-01"
    end_date = "2025-03-30"
    df = fetch_weather_data(start_date, end_date)
    print(df)
    df.to_csv('data/weather.csv', index=False)
