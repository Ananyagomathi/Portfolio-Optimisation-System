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
for index, row in data.iterrows():
    print(row['Price'])

# Alternatively, you can access the values directly and print them
# Using .values to get a numpy array of values
prices = data['Price'].values
for price in prices:
    print(price)
