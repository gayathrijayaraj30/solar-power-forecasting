import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

DATA_PATH = 'data/merged_featured_data.csv'

def load_and_prepare_data(filepath, sequence_length=14):
    df = pd.read_csv(filepath, parse_dates=['date'])
    df.sort_values('date', inplace=True)

    # Use the correct column name from your CSV
    df = df[['date', 'Production-Today(kWh)']].dropna()
    df = df.rename(columns={'Production-Today(kWh)': 'target'})

    # Normalize
    scaler = MinMaxScaler()
    df['target_scaled'] = scaler.fit_transform(df[['target']])

    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(df['target_scaled'].values[i-sequence_length:i])
        y.append(df['target_scaled'].values[i])

    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    return X[:split], y[:split], X[split:], y[split:], scaler, df

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def plot_predictions(actual, predicted, title='Actual vs Predicted'):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Power (kWh)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    print("Loading and preparing data...")
    X_train, y_train, X_test, y_test, scaler, df = load_and_prepare_data(DATA_PATH)

    print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
    model = build_lstm_model((X_train.shape[1], 1))

    print("Training model...")
    history = model.fit(
        X_train[..., np.newaxis], y_train,
        epochs=30,
        batch_size=16,
        validation_data=(X_test[..., np.newaxis], y_test),
        verbose=1
    )

    print("Evaluating model...")
    y_pred_scaled = model.predict(X_test[..., np.newaxis])
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    print(f"Test RMSE: {rmse:.4f} kWh")

    plot_predictions(y_test_actual, y_pred)
