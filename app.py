
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.api import Holt, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Time Series Forecasting Benchmark", layout="centered")

st.title("📈 Benchmark de Séries Temporais")
st.write(
    "Carregue um CSV **de uma coluna** com a série temporal e compare métodos de previsão: "
    "Naive, Média, Drift, Holt, Holt-Winters (Aditivo) e ARIMA (1,1,1)."
)

# Sidebar controls
st.sidebar.header("Parâmetros")
forecast_horizon = st.sidebar.slider("Horizonte de previsão (períodos)", 1, 60, 24, 1)

methods = {
    "naive": st.sidebar.checkbox("Naive", value=True),
    "mean": st.sidebar.checkbox("Média", value=True),
    "drift": st.sidebar.checkbox("Drift", value=True),
    "holt": st.sidebar.checkbox("Holt", value=True),
    "hw": st.sidebar.checkbox("Holt-Winters (Aditivo)", value=True),
    "arima": st.sidebar.checkbox("ARIMA (1,1,1)", value=True),
}

uploaded_file = st.file_uploader("Envie o CSV da série (uma coluna). Se tiver mais colunas, selecione qual usar.", type=["csv"])

def read_series(file) -> pd.Series:
    df = pd.read_csv(file, header=None)
    # If multiple columns, let user choose
    if df.shape[1] > 1:
        col = st.selectbox("Selecione a coluna com a série", options=list(range(df.shape[1])), index=0, format_func=lambda i: f"Coluna {i}")
        s = df.iloc[:, col]
    else:
        s = df.iloc[:, 0]
    # Try to coerce to numeric
    s = pd.to_numeric(s, errors="coerce")
    s = s.dropna().reset_index(drop=True)
    return s

def forecast_methods(train: pd.Series, h: int, msel: dict):
    forecasts = []
    titles = []

    # Naive: last value repeated
    if msel.get("naive"):
        last_val = train.iloc[-1]
        naive_forecast = np.full(h, last_val)
        forecasts.append(naive_forecast)
        titles.append("Naive")

    # Mean: sample mean repeated
    if msel.get("mean"):
        mu = train.mean()
        mean_forecast = np.full(h, mu)
        forecasts.append(mean_forecast)
        titles.append("Mean")

    # Drift: random walk with drift (line from first to last, extrapolated)
    if msel.get("drift"):
        n = len(train)
        drift = (train.iloc[-1] - train.iloc[0]) / (n - 1) if n > 1 else 0.0
        # Forecast h steps ahead: last + k*drift
        k = np.arange(1, h + 1)
        drift_forecast = train.iloc[-1] + k * drift
        forecasts.append(drift_forecast)
        titles.append("Drift")

    # Holt's linear trend
    if msel.get("holt"):
        try:
            holt_fit = Holt(train).fit()
            holt_forecast = holt_fit.forecast(h)
            forecasts.append(np.asarray(holt_forecast))
            titles.append("Holt")
        except Exception as e:
            st.warning(f"Falha no Holt: {e}")

    # Holt-Winters additive (seasonal_periods=12)
    if msel.get("hw"):
        try:
            hw_fit = ExponentialSmoothing(train, seasonal="add", trend="add", seasonal_periods=12).fit()
            hw_forecast = hw_fit.forecast(h)
            forecasts.append(np.asarray(hw_forecast))
            titles.append("HW Additive")
        except Exception as e:
            st.warning(f"Falha no Holt-Winters: {e}")

    # ARIMA(1,1,1)
    if msel.get("arima"):
        try:
            arima_fit = ARIMA(train, order=(1, 1, 1)).fit()
            arima_forecast = arima_fit.forecast(steps=h)
            forecasts.append(np.asarray(arima_forecast))
            titles.append("ARIMA(1,1,1)")
        except Exception as e:
            st.warning(f"Falha no ARIMA: {e}")

    return forecasts, titles

def plot_forecasts(actual: pd.Series, forecasts, titles):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(len(actual)), actual.values, label="Dados (treino)")
    for fc, tt in zip(forecasts, titles):
        x = np.arange(len(actual), len(actual) + len(fc))
        ax.plot(x, fc, label=tt)
    ax.set_title("Benchmark de Séries Temporais")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

if uploaded_file is not None:
    series = read_series(uploaded_file)
    st.subheader("Prévia dos dados")
    st.write(series.head(10).to_frame("valor"))

    forecasts, titles = forecast_methods(series, forecast_horizon, methods)

    if forecasts:
        st.subheader("Gráfico de previsões")
        plot_forecasts(series, forecasts, titles)

        # Optional: show a simple table of last/first values
        st.subheader("Resumo")
        st.write(pd.DataFrame({
            "método": titles,
            "primeiro_valor_previsto": [float(fc[0]) for fc in forecasts],
            "último_valor_previsto": [float(fc[-1]) for fc in forecasts],
        }))
    else:
        st.info("Selecione pelo menos um método na barra lateral.")
else:
    st.info("Envie um arquivo CSV para começar.")
