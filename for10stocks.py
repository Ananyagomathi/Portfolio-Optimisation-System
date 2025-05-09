import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('fiftystocks.csv')  # Columns are 'Stock1', 'Stock2', ..., 'Stock50'

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Function to create a dataset for LSTM
def create_dataset(data, look_back=1, forecast_horizon=10):
    X, Y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        a = data[i:(i + look_back), :]
        X.append(a)
        Y.append(data[i + look_back:i + look_back + forecast_horizon, :])
    return np.array(X), np.array(Y)

# Prepare the data
look_back = 60
forecast_horizon = 10  # Forecasting 1% to 10% ahead
X, Y = create_dataset(scaled_data, look_back, forecast_horizon)

# Split data into train and test sets
train_size = int(len(X) * 0.72)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Build the LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(look_back, df.shape[1])),
    LSTM(100),
    Dropout(0.3),
    Dense(df.shape[1] * forecast_horizon)  # Output layer nodes = number of stocks * forecast horizon
])

# Reshape Y_train and Y_test for training and testing
Y_train = Y_train.reshape(Y_train.shape[0], df.shape[1] * forecast_horizon)
Y_test = Y_test.reshape(Y_test.shape[0], df.shape[1] * forecast_horizon)

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=5, batch_size=72, verbose=1)

# Making predictions
test_predict = model.predict(X_test)
test_predict = test_predict.reshape(test_predict.shape[0], forecast_horizon, df.shape[1])
test_predict = scaler.inverse_transform(test_predict.reshape(-1, df.shape[1])).reshape(test_predict.shape)

# Compute returns for each horizon and plot results
portfolios = {}
alpha_list = []
risk_list = []

for horizon in range(forecast_horizon):
    predicted_prices = test_predict[:, horizon, :]
    last_prices = df.values[train_size + look_back + horizon - 1:-forecast_horizon + horizon]
    predicted_returns = (predicted_prices - last_prices) / last_prices
    top_indices = np.argsort(-predicted_returns.mean(axis=0))[:4]  # Selects top 4 indices with highest average returns
    portfolios[f'P{horizon+1}'] = df.columns[top_indices].tolist()
    average_returns = predicted_returns.mean(axis=0)[top_indices]

    # Calculate actual returns
    actual_returns = (last_prices[1:] - last_prices[:-1]) / last_prices[:-1]

    # Adjust lengths for beta calculation
    min_length = min(len(actual_returns), len(predicted_returns) - 1)
    adjusted_market_returns = actual_returns[:min_length].mean(axis=1)
    adjusted_predicted_returns = predicted_returns[:min_length].mean(axis=1)

    # Compute Alpha
    risk_free_rate = 0.01 / 252  # Assuming a constant risk-free rate per trading day
    beta = np.cov(adjusted_market_returns, adjusted_predicted_returns)[0, 1] / np.var(adjusted_market_returns)
    alpha = adjusted_predicted_returns - (risk_free_rate + beta * (adjusted_market_returns - risk_free_rate))
    alpha = alpha.mean()
    alpha_list.append(alpha)

    # Compute Risk (Standard Deviation of Returns)
    risk = np.std(adjusted_predicted_returns)
    risk_list.append(risk)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in top_indices:
        ax.plot(predicted_prices[:, i], label=f'Predicted {df.columns[i]}')
        ax.plot(last_prices[:, i], label=f'Actual {df.columns[i]}', alpha=0.7)
    ax.set_title(f'Time Horizon {horizon+1}%: Portfolio {portfolios[f"P{horizon+1}"]}\nReturns: {average_returns}\nAlpha: {alpha:.4f} Risk: {risk:.4f}')
    ax.legend()
    plt.show()

# Print portfolio returns, alpha, and risk
for i, (key, value) in enumerate(portfolios.items()):
    print(f"{key}: {value} | Alpha: {alpha_list[i]:.4f} | Risk: {risk_list[i]:.4f}")
