import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('Hyperparametersn.csv')

# Selecting the column to predict
data = df[['Adj Close']].values  # Assuming you are predicting 'Close' prices

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Creating a function to process the data for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Adjusted split ratio
train_size = int(len(data_scaled) * 0.72)
test_size = len(data_scaled) - train_size
train, test = data_scaled[0:train_size,:], data_scaled[train_size:len(data_scaled),:]

# Reshape into X=t and Y=t+1
look_back = 60
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, epochs=5, batch_size=72, verbose=1)

# Making predictions
test_predict = model.predict(X_test)

# Calculate mean squared error
test_predict = scaler.inverse_transform(test_predict)  # Inverse transform to get actual value predictions
Y_test_inv = scaler.inverse_transform([Y_test])        # Inverse transform the actual test values
mse = mean_squared_error(Y_test_inv[0], test_predict[:,0])
mae = mean_absolute_error(Y_test_inv[0], test_predict[:,0])
print(f"Mean Squared Error on the Test Data: {mse}")
print(f"Mean Absolute Error on the Test Data: {mae}")

# Plotting
plt.figure(figsize=(10,6))
actual_prices = df['Adj Close'].iloc[train_size+look_back:len(data_scaled)-1]
plt.plot(actual_prices, label='Actual Prices', marker='o', markevery=1, drawstyle='steps-post')

# Predicted prices plotting
predicted_prices = test_predict[:,0]
index_values = df.index[train_size+look_back:len(data_scaled)-1]
plt.plot(index_values, predicted_prices, label='Predicted Prices', marker='x', markevery=1, drawstyle='steps-post')

# Adding annotations for actual prices
for i, txt in enumerate(actual_prices):
    plt.annotate(f"{txt:.2f}", (index_values[i], actual_prices.iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Adding annotations for predicted prices
for i, txt in enumerate(predicted_prices):
    plt.annotate(f"{txt:.2f}", (index_values[i], predicted_prices[i]), textcoords="offset points", xytext=(0,-15), ha='center')

# Displaying the number of points
num_points_actual = len(actual_prices)
num_points_predicted = len(predicted_prices)
plt.title(f"Stock Prices: {num_points_actual} Actual Points, {num_points_predicted} Predicted Points")

actual_prices = df['Adj Close'].iloc[train_size+look_back:len(data_scaled)-1]
plt.plot(actual_prices, label='Actual Prices', marker='o', markevery=1, drawstyle='steps-post')

# Predicted prices plotting
predicted_prices = test_predict[:,0]
index_values = df.index[train_size+look_back:len(data_scaled)-1]
plt.plot(index_values, predicted_prices, label='Predicted Prices', marker='x', markevery=1, drawstyle='steps-post')

# Adding annotations for actual prices
for i, txt in enumerate(actual_prices):
    plt.annotate(f"{txt:.2f}", (index_values[i], actual_prices.iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Adding annotations for predicted prices
for i, txt in enumerate(predicted_prices):
    plt.annotate(f"{txt:.2f}", (index_values[i], predicted_prices[i]), textcoords="offset points", xytext=(0,-15), ha='center')

# Displaying the number of points
num_points_actual = len(actual_prices)
num_points_predicted = len(predicted_prices)
plt.title(f"Stock Prices: {num_points_actual} Actual Points, {num_points_predicted} Predicted Points")

plt.grid(True)
plt.legend()
plt.show()
