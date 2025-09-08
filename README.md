# 📈 Time Series Benchmark

This Streamlit application was created to **solve business problems involving time series forecasting**. Whether you need to predict demand, production, sales, consumption, or any other indicator over time, this tool allows you to quickly compare different classic forecasting models and identify the best approach for your context.

## ✨ What does the app deliver for your business?

- **Fast analysis:** Upload your own dataset (in CSV format) and get forecast comparisons in seconds.
- **Data-driven decisions:** Compare traditional methods (Naive, Mean, Drift, Holt, Holt-Winters, ARIMA) and see which one delivers the most realistic forecasts for your challenge.
- **Flexibility:** Use any dataset relevant to your business — for example, milk production, monthly sales, cash flow, website visits, and more.
- **Visual and objective results:** The app displays comparative charts and easy-to-understand summaries to support quick decision-making.

> **Note:** This app was developed using a **milk production** dataset, but **you can use any time series dataset** you want. The only requirement is that the file is in CSV format, with a single column of numeric values (or select the relevant column after uploading).

## 🚀 How to use

1. **Clone the repository and install dependencies:**
    ```bash
    git clone https://github.com/your-user/your-repo.git
    cd your-repo
    pip install -r requirements.txt
    ```

2. **Run the app:**
    ```bash
    streamlit run app.py
    ```

3. **In your browser:**
    - Upload a CSV file with **just one column** containing the time series (numeric values).
    - If the CSV has more than one column, select which one to use in the interface.
    - Choose which methods you want to apply in the sidebar.
    - Adjust the forecast horizon as needed for your business.

## 🗂️ Input file format

- **CSV with a single column** of numeric values.
- No header, by default. Example:
    ```
    120
    130
    128
    135
    142
    ```
- If there is more than one column, select the desired column after upload.

## ⚙️ Implemented methods

- **Naive:** Repeats the last observed value.
- **Mean:** Repeats the mean value of the series.
- **Drift:** Extrapolates a line between the first and last value.
- **Holt:** Exponential smoothing with trend.
- **Holt-Winters (Additive):** Exponential smoothing with trend and additive seasonality (period = 12).
- **ARIMA (1,1,1):** Basic ARIMA model, without automatic parameter search.

# 📈 Time Series Benchmark

This Streamlit application was created to **solve business problems involving time series forecasting**. Whether you need to predict demand, production, sales, consumption, or any other indicator over time, this tool allows you to quickly compare different classic forecasting models and identify the best approach for your context.

## ✨ What does the app deliver for your business?

- **Fast analysis:** Upload your own dataset (in CSV format) and get forecast comparisons in seconds.
- **Data-driven decisions:** Compare traditional methods (Naive, Mean, Drift, Holt, Holt-Winters, ARIMA) and see which one delivers the most realistic forecasts for your challenge.
- **Flexibility:** Use any dataset relevant to your business — for example, milk production, monthly sales, cash flow, website visits, and more.
- **Visual and objective results:** The app displays comparative charts and easy-to-understand summaries to support quick decision-making.

> **Note:** This app was developed using a **milk production** dataset, but **you can use any time series dataset** you want. The only requirement is that the file is in CSV format, with a single column of numeric values (or select the relevant column after uploading).

## 🚀 How to use

1. **Clone the repository and install dependencies:**
    ```bash
    git clone https://github.com/higorfct/Time-Series-Benchmark.git
    cd Time-Series-Benchmark
    pip install -r requirements.txt
    ```

2. **Run the app:**
    ```bash
    streamlit run app.py
    ```

3. **In your browser:**
    - Upload a CSV file with **just one column** containing the time series (numeric values).
    - If the CSV has more than one column, select which one to use in the interface.
    - Choose which methods you want to apply in the sidebar.
    - Adjust the forecast horizon as needed for your business.

## 🗂️ Input file format

- **CSV with a single column** of numeric values.
- No header, by default. Example:
    ```
    120
    130
    128
    135
    142
    ```
- If there is more than one column, select the desired column after upload.

## ⚙️ Implemented methods

- **Naive:** Repeats the last observed value.
- **Mean:** Repeats the mean value of the series.
- **Drift:** Extrapolates a line between the first and last value.
- **Holt:** Exponential smoothing with trend.
- **Holt-Winters (Additive):** Exponential smoothing with trend and additive seasonality (period = 12).
- **ARIMA (1,1,1):** Basic ARIMA model, without automatic parameter search.

## 📊 What the app shows

- Chart comparing the forecasts of the selected methods.
- Summary of forecasted values (first and last in the horizon).
- Preview of the first values in the loaded series.

## 🧩 Main dependencies

- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `statsmodels`

Install them with:
```bash
pip install -r requirements.txt
```

## 📄 Notes

- The app is for **didactic benchmarking** and only explores basic model configurations.
- For series with seasonality different from 12, adjust the code in the Holt-Winters function as needed.
- ARIMA does not perform automatic order selection.

---
