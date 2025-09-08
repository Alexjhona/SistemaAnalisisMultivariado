
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Tentativa de importar pmdarima
try:
    import pmdarima as pm
    HAS_PM = True
except ImportError:
    HAS_PM = False

st.set_page_config(page_title="Benchmark Avançado Completo", layout="centered")
st.title("📊 Benchmark Avançado: Auto ARIMA, Auto SARIMA, LSTM e Prophet")

st.sidebar.header("Parâmetros")
forecast_horizon = st.sidebar.slider("Horizonte de previsão (períodos)", 1, 60, 12, 1)
methods = {
    "arima": st.sidebar.checkbox("ARIMA", value=True),
    "sarima": st.sidebar.checkbox("SARIMA", value=True),
    "lstm": st.sidebar.checkbox("LSTM", value=True),
    "prophet": st.sidebar.checkbox("Prophet", value=True),
}

uploaded_file = st.file_uploader("Envie o CSV da série", type=["csv"])

# ---------------- Função de pré-processamento -----------------
def preprocess_series(file):
    df = pd.read_csv(file)
    if df.shape[1] > 1:
        col = st.selectbox("Selecione a coluna da série", df.columns)
        series = df[col]
    else:
        series = df.iloc[:, 0]
    series = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(series.values.reshape(-1,1))
    return series, scaled, scaler

# ---------------- Modelos -----------------
def forecast_arima(train, h):
    if HAS_PM:
        model = pm.auto_arima(train, seasonal=False, suppress_warnings=True)
        return model.predict(h)
    else:
        st.warning("pmdarima não encontrado. Usando ARIMA(1,1,1) como fallback.")
        model = ARIMA(train, order=(1,1,1)).fit()
        return model.forecast(steps=h)

def forecast_sarima(train, h, m=12):
    if HAS_PM:
        model = pm.auto_arima(train, seasonal=True, m=m, suppress_warnings=True)
        return model.predict(h)
    else:
        st.warning("pmdarima não encontrado. Usando SARIMA(1,1,1)(1,1,1,12) como fallback.")
        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
        return model.forecast(steps=h)

def forecast_prophet(train, h):
    df = pd.DataFrame({"ds": pd.date_range(start="2000-01-01", periods=len(train), freq="D"),
                       "y": train.values})
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=h)
    fc = m.predict(future)
    return fc["yhat"].iloc[-h:].values

def forecast_lstm(scaled, h, look_back=5):
    X, y = [], []
    for i in range(len(scaled)-look_back):
        X.append(scaled[i:i+look_back, 0])
        y.append(scaled[i+look_back, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(look_back,1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, verbose=0)

    input_seq = scaled[-look_back:].reshape((1, look_back, 1))
    preds = []
    for _ in range(h):
        yhat = model.predict(input_seq, verbose=0)
        preds.append(yhat[0,0])
        input_seq = np.append(input_seq[:,1:,:], [[[yhat]]], axis=1)
    return np.array(preds)

# ---------------- Métricas -----------------
def compute_metrics(true, pred):
    true, pred = np.array(true), np.array(pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100 if np.all(true != 0) else np.nan
    return rmse, mae, mape

# ---------------- Decomposição -----------------
def decompose_series(series, period=12):
    result = seasonal_decompose(series, period=period, model="additive")
    fig, axes = plt.subplots(4, 1, figsize=(10,8), sharex=True)
    result.observed.plot(ax=axes[0], title="Original")
    result.trend.plot(ax=axes[1], title="Tendência")
    result.seasonal.plot(ax=axes[2], title="Sazonalidade")
    result.resid.plot(ax=axes[3], title="Resíduos")
    st.pyplot(fig)

# ---------------- Plot -----------------
def plot_forecasts(actual, forecasts, titles):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(np.arange(len(actual)), actual, label="Treino")
    for fc, tt in zip(forecasts, titles):
        x = np.arange(len(actual), len(actual)+len(fc))
        ax.plot(x, fc, label=tt)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# ---------------- Execução -----------------
if uploaded_file:
    series, scaled, scaler = preprocess_series(uploaded_file)
    st.subheader("Série tratada")
    st.line_chart(series)

    st.subheader("🔎 Decomposição da Série")
    try:
        decompose_series(series, period=12)
    except Exception as e:
        st.warning(f"Não foi possível decompor a série: {e}")

    forecasts, titles, metrics = [], [], []

    test_size = min(forecast_horizon, len(series)//3)
    train, test = series[:-test_size], series[-test_size:]

    if methods["arima"]:
        try:
            fc = forecast_arima(train, forecast_horizon)
            forecasts.append(fc)
            titles.append("ARIMA")
            rmse, mae, mape = compute_metrics(test[:len(fc)], fc[:len(test)])
            metrics.append((rmse, mae, mape))
        except Exception as e:
            st.warning(f"Erro no ARIMA: {e}")

    if methods["sarima"]:
        try:
            fc = forecast_sarima(train, forecast_horizon, m=12)
            forecasts.append(fc)
            titles.append("SARIMA")
            rmse, mae, mape = compute_metrics(test[:len(fc)], fc[:len(test)])
            metrics.append((rmse, mae, mape))
        except Exception as e:
            st.warning(f"Erro no SARIMA: {e}")

    if methods["prophet"]:
        try:
            fc = forecast_prophet(train, forecast_horizon)
            forecasts.append(fc)
            titles.append("Prophet")
            rmse, mae, mape = compute_metrics(test[:len(fc)], fc[:len(test)])
            metrics.append((rmse, mae, mape))
        except Exception as e:
            st.warning(f"Erro no Prophet: {e}")

    if methods["lstm"]:
        try:
            fc_scaled = forecast_lstm(scaled[:-test_size], forecast_horizon)
            fc = scaler.inverse_transform(fc_scaled.reshape(-1,1)).flatten()
            forecasts.append(fc)
            titles.append("LSTM")
            rmse, mae, mape = compute_metrics(test[:len(fc)], fc[:len(test)])
            metrics.append((rmse, mae, mape))
        except Exception as e:
            st.warning(f"Erro no LSTM: {e}")

    if forecasts:
        st.subheader("Previsões")
        plot_forecasts(series, forecasts, titles)

        st.subheader("Métricas de Avaliação")
        df_metrics = pd.DataFrame(metrics, columns=["RMSE", "MAE", "MAPE"], index=titles)
        st.write(df_metrics)

        best_model = df_metrics["RMSE"].idxmin()
        st.subheader("📌 Recomendação")
        st.success(f"O modelo **{best_model}** apresentou o menor RMSE e deve ser priorizado para previsões futuras.")
    else:
        st.info("Selecione pelo menos um modelo.")
else:
    st.info("Envie um arquivo CSV para começar.")
