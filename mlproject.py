import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

# Create some sample data
# Let's say we have sequences of 10 timesteps with 1 feature each
X = np.random.rand(1000, 10, 1)
y = np.random.rand(1000, 1)

# Define the RNN model
model = Sequential()

# Add a SimpleRNN layer with 50 units (neurons)
model.add(SimpleRNN(50, input_shape=(10, 1)))

# Add a Dense layer with 1 unit (for the output)
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Make predictions
predictions = model.predict(X)

print(predictions)
