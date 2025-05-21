import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np

DATA_PATH = '/Users/gayathrijayaraj/Desktop/energy-forecasting-project/data/merged_featured_data.csv'
MODEL_SAVE_PATH = '/Users/gayathrijayaraj/Desktop/energy-forecasting-project/backend/models/xgboost_model.pkl'

def load_data(filepath):
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath, parse_dates=['date'])
    return df

def prepare_data(df, target_col='Production(kWh)'):
    df = df.sort_values('date').reset_index(drop=True)
    y = df[target_col]
    dates = df['date']
    X = df.drop(columns=['date', target_col])
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category')
    return X, y, dates

def train_xgboost(X_train, y_train, X_valid, y_valid):
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        verbosity=1,
        enable_categorical=True
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=50
    )
    return model

def main():
    df = load_data(DATA_PATH)
    X, y, dates = prepare_data(df)

    tscv = TimeSeriesSplit(n_splits=5)

    fold = 1
    rmses = []
    final_model = None  # store last fold's model to save

    for train_index, val_index in tscv.split(X):
        print(f"\nTraining fold {fold}...")
        X_train, X_valid = X.iloc[train_index], X.iloc[val_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[val_index]

        model = train_xgboost(X_train, y_train, X_valid, y_valid)

        preds = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        print(f"Fold {fold} RMSE: {rmse:.4f}")
        rmses.append(rmse)

        final_model = model  # overwrite, so last fold model is saved
        fold += 1

    print(f"\nAverage RMSE over folds: {np.mean(rmses):.4f}")

    # Create the models directory if it doesn't exist
    model_dir = os.path.dirname(MODEL_SAVE_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the final model
    joblib.dump(final_model, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
