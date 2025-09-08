
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# time series models
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Benchmark Completo de Séries Temporais", layout="centered")
st.title("📈 Benchmark Completo — Auto ARIMA / Auto SARIMA, Prophet, LSTM + Decomposição e Métricas")

st.markdown("""
**O que o app faz (automático):**
- pré-processamento da série (conversão numérica, remoção de NaNs, escalonamento para LSTM)
- inferência de sazonalidade simples (métrica baseada no tamanho da série)
- decomposição em tendência/sazonal/resíduo
- seleção automática de ordens para ARIMA/SARIMA via `pmdarima.auto_arima`
- treino e previsão com Prophet e LSTM
- cálculo de RMSE, MAE, MAPE e recomendação automática pelo menor RMSE
""")

# ---------------- Sidebar params ----------------
st.sidebar.header("Parâmetros")
forecast_horizon = st.sidebar.slider("Horizonte de previsão (períodos)", 1, 60, 12, 1)
run_arima = st.sidebar.checkbox("Auto ARIMA", value=True)
run_sarima = st.sidebar.checkbox("Auto SARIMA", value=True)
run_prophet = st.sidebar.checkbox("Prophet", value=True)
run_lstm = st.sidebar.checkbox("LSTM", value=True)
lstm_epochs = st.sidebar.number_input("LSTM epochs", min_value=1, max_value=200, value=20, step=1)
lstm_lookback = st.sidebar.number_input("LSTM look_back", min_value=1, max_value=30, value=5, step=1)

uploaded_file = st.file_uploader("Envie o CSV da série (caso haja coluna de datas, o app tentará detectá-la).", type=["csv"])

# ---------------- Helpers / Preprocessing ----------------
def infer_date_column(df):
    # try to detect a datetime-like column
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=False)
            non_null = parsed.notna().sum()
            if non_null / len(parsed) > 0.8:  # majority parseable -> likely date
                return col
        except Exception:
            continue
    return None

def preprocess_series(file):
    df = pd.read_csv(file)
    original_df = df.copy()

    # detect date column
    date_col = infer_date_column(df)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        # drop rows with NaT in date col
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        # choose numeric column if more than one column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            # maybe the numeric column is string formatted; try convert others
            candidates = [c for c in df.columns if c != date_col]
            for c in candidates:
                try:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                    if df[c].notna().sum() > 0:
                        numeric_cols.append(c)
                except Exception:
                    pass
        if len(numeric_cols) == 0:
            raise ValueError("Não foi possível identificar uma coluna numérica na tabela.")
        # if multiple numeric, let user choose
        if len(numeric_cols) > 1:
            sel = st.selectbox("Mais de uma coluna numérica detectada. Selecione a série:", numeric_cols)
        else:
            sel = numeric_cols[0]
        series = pd.Series(df[sel].values, index=pd.DatetimeIndex(df[date_col].values))
    else:
        # no date column detected: choose a column (or first)
        if df.shape[1] > 1:
            sel = st.selectbox("Selecione a coluna com a série (nenhuma coluna de datas detectada)", df.columns)
            series = pd.to_numeric(df[sel], errors="coerce")
        else:
            series = pd.to_numeric(df.iloc[:,0], errors="coerce")
        series = series.dropna().reset_index(drop=True)
        # create integer index for plotting
        series.index = pd.RangeIndex(start=0, stop=len(series), step=1)
        series = pd.Series(series.values, index=series.index)

    # final cleaning: dropna and ensure floats
    series = series.dropna().astype(float)

    # infer seasonal period m
    m = 1
    n = len(series)
    if n >= 48:
        m = 12  # monthly-ish seasonality default
    elif n >= 28:
        m = 7   # weekly-ish
    else:
        m = 1
    # scale for LSTM
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(series.values.reshape(-1,1))

    return series, scaled, scaler, m

# ---------------- Decomposition ----------------
def decompose_and_plot(series, m):
    st.subheader("Decomposição da série temporal")
    if m <= 1:
        st.info("Sazonalidade não inferida (série curta). Pulando decomposição sazonal.")
        return None
    try:
        res = seasonal_decompose(series, period=m, model="additive", extrapolate_trend='freq')
        fig, axes = plt.subplots(4,1, figsize=(8,9), sharex=True)
        axes[0].plot(series.index, series.values); axes[0].set_ylabel("Original")
        axes[1].plot(res.trend.index, res.trend.values); axes[1].set_ylabel("Trend")
        axes[2].plot(res.seasonal.index, res.seasonal.values); axes[2].set_ylabel("Seasonal")
        axes[3].plot(res.resid.index, res.resid.values); axes[3].set_ylabel("Residual")
        plt.tight_layout()
        st.pyplot(fig)
        return res
    except Exception as e:
        st.warning(f"Decomposição falhou: {e}")
        return None

# ---------------- Forecast models ----------------
def auto_arima_forecast(train, h, seasonal=False, m=1):
    # use pmdarima auto_arima to find model; then forecast h steps
    ar = pm.auto_arima(train, seasonal=seasonal, m=m, suppress_warnings=True, error_action="ignore", stepwise=True)
    # get forecast
    fc = ar.predict(n_periods=h)
    return fc, ar

def prophet_forecast(train, h):
    # train must be a pandas Series with DatetimeIndex ideally
    if isinstance(train.index, pd.DatetimeIndex):
        df = pd.DataFrame({"ds": train.index, "y": train.values})
    else:
        # fabricate dates starting at 2000-01-01 daily
        df = pd.DataFrame({"ds": pd.date_range(start="2000-01-01", periods=len(train), freq="D"),
                           "y": train.values})
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=h)
    fc = m.predict(future)
    return fc["yhat"].iloc[-h:].values, m

def lstm_forecast(scaled_train, h, scaler, look_back=5, epochs=20):
    X, y = [], []
    for i in range(len(scaled_train)-look_back):
        X.append(scaled_train[i:i+look_back, 0])
        y.append(scaled_train[i+look_back, 0])
    X, y = np.array(X), np.array(y)
    if len(X) == 0:
        raise ValueError("Série curta demais para o look_back definido para LSTM.")
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, activation="tanh", input_shape=(X.shape[1],1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, verbose=0)
    # forecast iteratively
    input_seq = scaled_train[-look_back:].reshape((1, look_back, 1))
    preds = []
    for _ in range(h):
        yhat = model.predict(input_seq, verbose=0)
        preds.append(yhat[0,0])
        input_seq = np.append(input_seq[:,1:,:], [[[yhat]]], axis=1)
    preds = np.array(preds)
    preds = scaler.inverse_transform(preds.reshape(-1,1)).flatten()
    return preds

# ---------------- Metrics ----------------
def compute_metrics(true, pred):
    true = np.array(true); pred = np.array(pred)
    # align lengths
    L = min(len(true), len(pred))
    true = true[:L]; pred = pred[:L]
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100 if np.all(true != 0) else np.nan
    return rmse, mae, mape

# ---------------- Plot helper ----------------
def plot_forecasts(series, forecasts, titles):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(series.index, series.values, label="Histórico (treino+teste)")
    for fc, tt in zip(forecasts, titles):
        # if series index is datetime, create future datetimes; else use integer range
        if isinstance(series.index, pd.DatetimeIndex):
            last = series.index[-1]
            # assume daily frequency for future points
            future_idx = pd.date_range(start=last + pd.Timedelta(days=1), periods=len(fc), freq='D')
            ax.plot(future_idx, fc, marker='o', label=tt)
        else:
            x = np.arange(len(series), len(series)+len(fc))
            ax.plot(x, fc, marker='o', label=tt)
    ax.set_xlabel("Index")
    ax.set_ylabel("Valor")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# ---------------- Main ----------------
if uploaded_file is None:
    st.info("Envie um CSV para começar. O arquivo pode ter uma coluna de datas (reconhecida automaticamente) e uma coluna numérica com a série.")
else:
    try:
        series, scaled, scaler, m = preprocess_series(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao preprocessar: {e}")
        st.stop()

    st.subheader("Série tratada (preview)")
    st.write(series.head(20))
    st.line_chart(series)

    # decomposition
    decomposition = decompose_and_plot(series, m)

    # prepare train/test split for evaluation
    test_size = min(forecast_horizon, max(1, len(series)//6))
    train = series[:-test_size] if test_size < len(series) else series[:-1]
    test = series[-test_size:]

    st.write(f"Tamanho total: {len(series)} - Treino: {len(train)} - Teste (para avaliação): {len(test)} - m (sazonalidade inferida) = {m}")

    forecasts = []
    titles = []
    metrics_list = []
    models_info = {}

    # Auto ARIMA (non-seasonal if m==1)
    if run_arima:
        try:
            seasonal_flag = True if (m and m>1) else False
            fc_arima, ar_model = auto_arima_forecast(train, forecast_horizon, seasonal=seasonal_flag, m=m)
            forecasts.append(fc_arima)
            titles.append("Auto ARIMA")
            models_info["Auto ARIMA"] = ar_model.summary().as_text() if hasattr(ar_model, "summary") else str(ar_model)
            # compute metrics comparing with test
            rmse, mae, mape = compute_metrics(test.values, fc_arima[:len(test)])
            metrics_list.append((rmse, mae, mape))
        except Exception as e:
            st.warning(f"Auto ARIMA falhou: {e}")

    # Auto SARIMA (force seasonal if m>1)
    if run_sarima and m>1:
        try:
            fc_sarima, sar_model = auto_arima_forecast(train, forecast_horizon, seasonal=True, m=m)
            forecasts.append(fc_sarima)
            titles.append("Auto SARIMA")
            models_info["Auto SARIMA"] = sar_model.summary().as_text() if hasattr(sar_model, "summary") else str(sar_model)
            rmse, mae, mape = compute_metrics(test.values, fc_sarima[:len(test)])
            metrics_list.append((rmse, mae, mape))
        except Exception as e:
            st.warning(f"Auto SARIMA falhou: {e}")
    elif run_sarima and m<=1:
        st.info("Sazonalidade não detectada; pulando Auto SARIMA (defina m manualmente em uma versão futura se desejar).")

    # Prophet
    if run_prophet:
        try:
            fc_prophet, prop_model = prophet_forecast(train, forecast_horizon)
            forecasts.append(fc_prophet)
            titles.append("Prophet")
            models_info["Prophet"] = "Prophet model trained"
            rmse, mae, mape = compute_metrics(test.values, fc_prophet[:len(test)])
            metrics_list.append((rmse, mae, mape))
        except Exception as e:
            st.warning(f"Prophet falhou: {e}")

    # LSTM
    if run_lstm:
        try:
            # train LSTM on train portion scaled
            scaled_train = scaler.transform(train.values.reshape(-1,1))
            fc_lstm = lstm_forecast(scaled_train, forecast_horizon, scaler, look_back=int(lstm_lookback), epochs=int(lstm_epochs))
            forecasts.append(fc_lstm)
            titles.append("LSTM")
            models_info["LSTM"] = f"LSTM epochs={lstm_epochs}, look_back={lstm_lookback}"
            rmse, mae, mape = compute_metrics(test.values, fc_lstm[:len(test)])
            metrics_list.append((rmse, mae, mape))
        except Exception as e:
            st.warning(f"LSTM falhou: {e}")

    # show forecasts plot
    if len(forecasts) > 0:
        st.subheader("Previsões (os pontos futuros são do horizonte solicitado)")
        plot_forecasts(series, forecasts, titles)

        # metrics table
        st.subheader("Métricas de Avaliação por Modelo (comparadas com o conjunto de teste)")
        df_metrics = pd.DataFrame(metrics_list, columns=["RMSE","MAE","MAPE"], index=titles)
        st.write(df_metrics.style.format({"RMSE":"{:.4f}","MAE":"{:.4f}","MAPE":"{:.2f}"}))

        # recommendation - choose by RMSE
        best = df_metrics["RMSE"].idxmin()
        st.subheader("📌 Recomendação Automática")
        st.success(f"Modelo recomendado: **{best}**, pois apresentou o menor RMSE ({df_metrics['RMSE'].min():.4f}).")

        # decision rules (simple automatic guidance)
        st.subheader("🔎 Instruções de tomada de decisão (automáticas)")
        st.markdown("- Se o modelo recomendado for **LSTM**: considere aumentar epochs ou look_back se houver overfitting/baixo desempenho; LSTM demanda mais dados e tuning.")
        st.markdown("- Se modelos ARIMA/SARIMA estiverem competindo de perto: prefira **SARIMA** se houver sazonalidade clara (m > 1) e interpretação é importante; prefira ARIMA para séries curtas sem sazonalidade.")
        st.markdown("- Se **Prophet** estiver melhor: é uma boa escolha quando há feriados, fortes sazonalidades e quando você tem um índice de tempo robusto.")
        st.markdown("- Se o erro (RMSE) for alto em todos os modelos: investigue outliers, mudanças estruturais, ou use features exógenas e enrichimento de dados.")

        # optional: show model summaries for ARIMA/SARIMA (if available)
        for k,v in models_info.items():
            if "ARIMA" in k or "SARIMA" in k:
                st.subheader(f"Resumo do modelo: {k}")
                st.text(v if isinstance(v, str) else str(v))
    else:
        st.info("Nenhum modelo gerou previsões. Verifique os dados e parâmetros.")
