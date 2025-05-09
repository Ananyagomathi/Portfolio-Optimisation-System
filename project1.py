import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Load data from CSV
df = pd.read_csv('S&P 500 Historical Data (1).csv')

# Clean 'Close' column if needed (remove commas and convert to float)
df['Price'] = df['Price'].str.replace(',', '').astype(float)

# Sort by date if not already sorted
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# Extract the 'Close' price for prediction
data = df[['Price']]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Convert the dataset into sequences and labels
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Define the lookback period (number of previous days to use for prediction)
look_back = 30
X, Y = create_dataset(scaled_data, look_back)

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
train_X, test_X = X[0:train_size], X[train_size:len(X)]
train_Y, test_Y = Y[0:train_size], Y[train_size:len(Y)]

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=256, return_sequences=True, input_shape=(1, look_back)))
model.add(Dropout(0.2))
model.add(LSTM(units=256))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print summary of the model architecture
print(model.summary())

# Train the model
model.fit(train_X, train_Y, epochs=20, batch_size=32, verbose=1)

# Predict on test data
predictions = model.predict(test_X)
predictions = scaler.inverse_transform(predictions)

# Calculate evaluation metrics
mae = mean_absolute_error(test_Y, predictions)
mse = mean_squared_error(test_Y, predictions)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test_Y - predictions) / test_Y)) * 100

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Calculate returns for actual and predicted prices
actual_returns = data.pct_change().dropna()
predicted_returns = pd.DataFrame(predictions).pct_change().dropna()

# Compute Alpha
risk_free_rate = 0.01  # Assuming a constant risk-free rate
market_returns = actual_returns.mean()  # Assuming market returns as average of actual returns
beta = np.cov(actual_returns.T, predicted_returns.T)[0, 1] / np.var(market_returns)

alpha = actual_returns - (risk_free_rate + beta * (market_returns - risk_free_rate))
alpha = alpha.mean()

print(f"Alpha: {alpha:.4f}")

# Compute Risk (Standard Deviation of Returns)
risk = np.std(predicted_returns)
print(f"Risk (Standard Deviation of Returns): {risk:.4f}")

# Plotting
plt.figure(figsize=(14, 8))
plt.plot(data.index, data['Price'], label='Actual Stock Price')
plt.plot(data.index[look_back:], predictions, label='Predicted Stock Price')
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
