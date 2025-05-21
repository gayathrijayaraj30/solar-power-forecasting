import pandas as pd
import matplotlib.pyplot as plt

# Load the weather data CSV
df = pd.read_csv('data/weather.csv')

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Show first 5 rows to understand data
print(df.head())

# Summary info (data types, non-null counts)
print(df.info())

# Basic statistics summary
print(df.describe())

# Plot Max and Min Temperature over time
plt.figure(figsize=(12,6))
plt.plot(df['date'], df['temp_max_C'], label='Max Temp (°C)', color='red')
plt.plot(df['date'], df['temp_min_C'], label='Min Temp (°C)', color='blue')
plt.title('Daily Max and Min Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

# Plot Daily Precipitation over time
plt.figure(figsize=(12,4))
plt.bar(df['date'], df['precipitation_mm'], color='skyblue')
plt.title('Daily Precipitation Over Time')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.grid(True)
plt.show()
