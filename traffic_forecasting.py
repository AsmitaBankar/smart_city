import pandas as pd

file_path = "E:/Desktop/Traffic_Forecasting_Project/test_BdBKkAj.csv"  # Use forward slashes `/`
df = pd.read_csv(file_path)

print(df.head())  # Print first few rows
# Check column names, data types, and missing values
print(df.info())

df.ffill(inplace=True)  # Forward fill missing values

df['DateTime'] = pd.to_datetime(df['DateTime'])

# Extract time components
df['hour'] = df['DateTime'].dt.hour
df['day_of_week'] = df['DateTime'].dt.dayofweek
df['month'] = df['DateTime'].dt.month
df['weekend'] = (df['day_of_week'] >= 5).astype(int)  # 1 if Saturday/Sunday, else 0

print(df.head())  # Verify changes

df['traffic_count'] = df.groupby(['Junction', 'hour'])['ID'].transform('count')

# Drop the 'ID' column (not needed anymore)
df.drop(columns=['ID'], inplace=True)

print(df.head())  # Check traffic count

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['traffic_count', 'hour', 'day_of_week', 'month']] = scaler.fit_transform(df[['traffic_count', 'hour', 'day_of_week', 'month']])

print(df.describe())  # Verify normalization

import numpy as np

# Convert traffic count data to a NumPy array
traffic_data = df['traffic_count'].values

# Define sequence length (how many past hours to consider)
seq_length = 10  

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Create input (X) and output (y) sequences
X, y = create_sequences(traffic_data, seq_length)

# Reshape for LSTM input: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

print("X shape:", X.shape)  # (num_samples, seq_length, 1)
print("y shape:", y.shape)  # (num_samples,)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),  # First LSTM layer
    LSTM(50),  # Second LSTM layer
    Dense(1)  # Output layer (predicts traffic count)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=20, batch_size=16)

# Save trained model
model.save("traffic_forecast_model.h5")
print("Model saved successfully!")

# Make predictions on the last sequence of data
last_sequence = traffic_data[-seq_length:].reshape(1, seq_length, 1)  # Reshape for LSTM
predicted_traffic = model.predict(last_sequence)[0][0]

print(f"Predicted Traffic Count for Next Hour: {predicted_traffic}")
