import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential, load_model
from keras.layers import LSTM, GRU, Dense, Dropout, Input
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

def prepare_data(df, feature_col='Close', time_steps=60):
    """Prepare data for deep learning models."""
    data = df[[feature_col]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler, data

def build_lstm(time_steps):
    model = Sequential([
        Input(shape=(time_steps, 1)),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_gru(time_steps):
    model = Sequential([
        Input(shape=(time_steps, 1)),
        GRU(units=50, return_sequences=True),
        Dropout(0.2),
        GRU(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_dl_model(model, X, y, epochs=10, batch_size=32, progress_bar=None):
    """Train the model and update Streamlit progress bar."""
    if progress_bar is not None:
        for epoch in range(epochs):
            model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0)
            progress_bar.progress((epoch + 1) / epochs)
    else:
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def predict_future_dl(model, last_sequence, scaler, days=7):
    """Predict future days iteratively."""
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(days):
        pred = model.predict(current_seq.reshape(1, current_seq.shape[0], 1), verbose=0)
        predictions.append(pred[0, 0])
        # Update sequence
        current_seq = np.roll(current_seq, -1)
        current_seq[-1] = pred[0, 0]
        
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

def train_and_predict_arima(data, forecast_steps=7):
    """Train ARIMA and forecast."""
    try:
        model = ARIMA(data, order=(5, 1, 0))
        fitted = model.fit()
        forecast = fitted.forecast(steps=forecast_steps)
        return forecast
    except Exception as e:
        print(f"ARIMA error: {e}")
        return np.array([np.nan]*forecast_steps)

def calculate_metrics(y_true, y_pred):
    """Calculate RMSE, MAE, R2."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def get_model_path(ticker, model_type):
    return os.path.join(MODELS_DIR, f"{ticker}_{model_type}.keras")

def load_or_train_dl_model(ticker, model_type, df, force_train=False, progress_bar=None):
    time_steps = 60
    X, y, scaler, raw_data = prepare_data(df, time_steps=time_steps)
    model_path = get_model_path(ticker, model_type)
    
    if os.path.exists(model_path) and not force_train:
        model = load_model(model_path)
    else:
        if model_type == 'LSTM':
            model = build_lstm(time_steps)
        else:
            model = build_gru(time_steps)
        # Fast training for Streamlit responsiveness
        model = train_dl_model(model, X, y, epochs=5, progress_bar=progress_bar)
        model.save(model_path)
        
    # Validation predictions for metrics
    train_size = int(len(X) * 0.8)
    X_test, y_test = X[train_size:], y[train_size:]
    
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    metrics = calculate_metrics(y_true, y_pred)
    
    # Predict future
    last_seq_scaled = scaler.transform(raw_data[-time_steps:]).flatten()
    future_7 = predict_future_dl(model, last_seq_scaled, scaler, days=7)
    future_30 = predict_future_dl(model, last_seq_scaled, scaler, days=30)
    
    return {
        "model": model,
        "metrics": metrics,
        "future_7": future_7,
        "future_30": future_30,
        "y_true": y_true,
        "y_pred": y_pred
    }

def generate_ensemble_forecast(ticker, df):
    """Generate ensemble predictions using LSTM, GRU, and ARIMA."""
    lstm_res = load_or_train_dl_model(ticker, 'LSTM', df)
    gru_res = load_or_train_dl_model(ticker, 'GRU', df)
    
    arima_data = df['Close'].values
    arima_7 = train_and_predict_arima(arima_data, 7)
    arima_30 = train_and_predict_arima(arima_data, 30)
    
    ensemble_7 = (lstm_res['future_7'] + gru_res['future_7'] + arima_7) / 3
    ensemble_30 = (lstm_res['future_30'] + gru_res['future_30'] + arima_30) / 3
    
    return {
        "LSTM": lstm_res,
        "GRU": gru_res,
        "ARIMA_7": arima_7,
        "ARIMA_30": arima_30,
        "Ensemble_7": ensemble_7,
        "Ensemble_30": ensemble_30
    }
