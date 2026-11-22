# Retail-Sales-forecasting
Time series sales forecasting with ARIMA, SARIMA, Prophet and Linear Regression
# Sales Forecasting: Linear Regression, ARIMA, SARIMA & Prophet

A course assignment (Module on Sales Forecasting) comparing four approaches to forecasting daily retail sales: Linear Regression on date features, ARIMA, SARIMA, and Facebook/Meta Prophet.

[Open in Google Colab](https://colab.research.google.com/drive/15j55BXCXnjahSlVXjLDbHmcDSmzhXW54?usp=sharing)

## Overview

1. Load the retail sales data and keep only `Order Date` and `Sales`.
2. Convert dates to datetime and aggregate to total sales per day.
3. Extract features (`year`, `month`, `day`, `dayofweek`) and plot the sales trend.
4. Split chronologically: first 80% train, last 20% test (no shuffling).
5. Fit and evaluate four models on the held-out test period.

## Models

| Model | Details |
|---|---|
| Linear Regression | Calendar features (year, month, day, day of week) |
| ARIMA | `SARIMAX` with order `(0, 1, 1)` |
| SARIMA | `SARIMAX` with order `(0, 1, 1)` and seasonal order `(0, 0, 2, 7)` (weekly seasonality) |
| Prophet | Daily and yearly seasonality enabled |

For ARIMA/SARIMA the daily series is reindexed to a continuous daily frequency, with days that have no orders filled with 0.

## Results (test set)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 1030.95 | 777.68 | 0.04 |
| ARIMA | 1059.68 | 614.81 | -0.21 |
| SARIMA | 1043.56 | 609.45 | -0.18 |
| Prophet | 1026.22 | 660.98 | -0.00 |

### Observations

- All four models perform poorly: RMSE is around 1,030 to 1,060 for every model, and R² is near zero or negative. None of them explains much of the variation in daily sales.
- SARIMA and ARIMA have the lowest MAE (about 610), but their negative R² means they do worse than simply predicting the mean on the test set.
- Linear Regression has the highest R² (0.04) but also the highest MAE (about 778). Its R² is only marginally above zero, so calling it the "best" model is a weak claim.
- Daily sales are noisy and spiky, so daily-level forecasting is hard with these methods. Aggregating to weekly or monthly sales, adding lag features, or trying tree-based models would be sensible next steps.

## Dataset

`stores_sales_forecasting.csv`, a retail store sales dataset with order dates and sales values, sourced from Kaggle. Add the dataset link here: `<https://www.kaggle.com/datasets/tanayatipre/store-sales-forecasting-dataset>`. The file is read with `encoding='cp1252'`.

## Setup

Requires Python 3.9+.

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels prophet jupyter
```

Place `stores_sales_forecasting.csv` in the same folder as the notebook, then:

```bash
jupyter notebook Assignment_6_1.ipynb
```



