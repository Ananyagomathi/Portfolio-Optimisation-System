# Multi-Stock Price Prediction & Portfolio Optimization Pipeline

## Overview

This repository provides an end-to-end workflow for:

1. **Data Collection** — Download historical OHLC data for one or many tickers.
2. **Data Cleaning & Preprocessing** — Load, clean, normalize and reshape time-series data for modeling.
3. **Modeling**  
   - **Univariate LSTM** for single-asset forecasts.  
   - **Multivariate LSTM** for simultaneous forecasting of multiple stocks.  
   - **Simple RNN** baseline on synthetic sequences.  
4. **Portfolio Construction & Evaluation** —  
   - Compute predicted returns, select top-N assets per forecast horizon, calculate alpha & risk metrics.  
   - Visualize actual vs. predicted prices and portfolio performance.  

## Features & Scripts

### Data Collection & Preparation

- **collect.py**  
  Fetches historical data for a single ticker via `yfinance`.  
- **collect1final.py**  
  Batch-fetch multiple tickers listed in `try1.txt` into `stock_prices.csv`.  
- **msft_historical_data.csv**  
  Sample CSV of MSFT price history for quick testing.  
- **stock_prices.csv**  
  Combined OHLC data for all tickers fetched by `collect1final.py`.  
- **stocks500.csv**  
  A larger universe of 500 tickers used for batch experiments.  
- **fiftystocks.csv**, **tenstocks.csv**  
  Smaller lists (50 and 10 tickers) for rapid prototyping of multivariate models.  

### Cleaning & Utility

- **data.py**  
  Cleans & indexes raw CSVs, normalizes price columns.  
- **remove_mostly_empty_columns.py**  
  Drops columns with predominantly missing data from a cleaned dataset.  

### Modeling

- **hyoerwitherrors.py** & **hyperparamersforA.py**  
  Single-asset LSTM experiments with varying look-back windows and train/test splits.  
- **for10stocks.py**  
  Multivariate LSTM forecasting on a 10-stock universe — builds rolling portfolios of top 4 predicted return assets.  
- **mlproject.py**  
  SimpleRNN baseline on synthetic sequence data.  
- **project.py**  
  LSTM on Quandl AAPL “Close” data (2008–2018); trains 2-layer LSTM, reports MAE/MSE/RMSE/MAPE and plots predictions. :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}  
- **project1.py**  
  LSTM on S&P 500 CSV (“Price” column); computes per-day returns, alpha vs. market, volatility, and visualization. :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}  

### Portfolio Analytics

- **creatingportfoliopy**  
  (If present) Custom scripts for advanced portfolio back-testing and visualization.  
- **creatingportfoliopy.py** *(renamed from creatingportfoliopy)*  
  Generates portfolio weightings from forecast outputs and charts cumulative returns.  

## Requirements

- Python 3.7+  
- Core: `numpy`, `pandas`, `matplotlib`, `scikit-learn`  
- Modeling: `tensorflow` or `keras`, `yfinance`, `quandl` (for `project.py`)  
- HTTP: `requests` (if extending to REST-based data sources)  
