import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# File paths
DATA_PATH = 'data/merged_featured_data.csv'
ANOMALY_RESULTS_PATH = 'data/anomaly_results_with_shap.csv'
SHAP_VALUES_PATH = 'data/shap_anomaly_values.npy'

# Your original feature list used in the model
FEATURE_NAMES = [
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

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, parse_dates=['date'])
    anomalies = pd.read_csv(ANOMALY_RESULTS_PATH, parse_dates=['date'])
    shap_values = np.load(SHAP_VALUES_PATH, allow_pickle=True)  # Correct load for .npy
    return data, anomalies, shap_values

def plot_shap_bar(shap_vals, feature_names):
    plt.figure(figsize=(10, 6))
    shap_vals_df = pd.DataFrame({'feature': feature_names, 'shap_value': shap_vals})
    shap_vals_df = shap_vals_df.reindex(shap_vals_df.shap_value.abs().sort_values(ascending=False).index)
    shap_vals_df = shap_vals_df.head(15)
    plt.barh(shap_vals_df['feature'], shap_vals_df['shap_value'])
    plt.xlabel("SHAP value")
    plt.title("Top SHAP feature contributions")
    plt.gca().invert_yaxis()
    st.pyplot(plt)
    plt.clf()

def main():
    st.title("Solar Production Anomaly Detection & SHAP Explanation")

    data, anomalies, shap_values = load_data()

    st.sidebar.header("Filter anomalies")
    min_date = anomalies['date'].min()
    max_date = anomalies['date'].max()
    start_date = st.sidebar.date_input("Start date", min_date)
    end_date = st.sidebar.date_input("End date", max_date)

    filtered_anomalies = anomalies[
        (anomalies['date'] >= pd.to_datetime(start_date)) &
        (anomalies['date'] <= pd.to_datetime(end_date))
    ]

    st.write(f"### Anomalies between {start_date} and {end_date}")
    st.write(f"Total anomalies: {len(filtered_anomalies)}")

    if len(filtered_anomalies) == 0:
        st.info("No anomalies found in the selected date range.")
        return

    anomaly_index = st.selectbox("Select an anomaly by index", filtered_anomalies.index)

    anomaly = filtered_anomalies.loc[anomaly_index]
    st.write("#### Anomaly details")
    st.write(anomaly)

    st.write(f"**Date:** {anomaly['date'].date()}")
    st.write(f"**Actual Production:** {anomaly['Production(kWh)']:.2f} kWh")
    st.write(f"**Residual (Error):** {anomaly['residual']:.2f} kWh")

    # SHAP explanation for this anomaly
    # The shap_values array is aligned with anomalies DataFrame rows
    idx_in_shap = filtered_anomalies.index.get_loc(anomaly_index)
    shap_val = shap_values[idx_in_shap]

    st.write("#### SHAP Feature Importance for this Anomaly")
    shap.initjs()
    plot_shap_bar(shap_val, FEATURE_NAMES)

if __name__ == "__main__":
    main()
