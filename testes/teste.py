import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Activation, MaxPooling1D, Dropout, Input
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load and preprocess data
file = pd.read_csv("dsV4.csv", header=0, index_col=0, sep=";", decimal=",")
data = file.iloc[:, 0]
index = file.index.values
index = pd.to_datetime(index, format='%d/%m/%Y %H:%M')

# Create sliding windows
window = 48
horizon = 24
refresh_period = 24

x = np.array([data[i:i+window] for i in range(0, len(data) - window - horizon + 1, refresh_period)])
y = np.array([data[i+window:i+window+horizon] for i in range(0, len(data) - window - horizon + 1, refresh_period)])
x_index = np.array([index[i:i+window] for i in range(0, len(index) - window - horizon + 1, refresh_period)])
y_index = np.array([index[i+window:i+window+horizon] for i in range(0, len(index) - window - horizon + 1, refresh_period)])


x = x.reshape((x.shape[0], window, 1))

# Split the data
x_train, x_test, y_train, y_test, x_train_index, x_test_index, y_train_index, y_test_index = train_test_split(
    x, y, x_index, y_index, test_size=0.25, shuffle=False
)

# Build the model
model = Sequential([
    Input(shape=(window, 1)),
    Conv1D(filters=32, kernel_size=2, padding='causal', strides=2),
    MaxPooling1D(pool_size=2, strides=2),
    Activation('relu'),
    LSTM(32, return_sequences=False),
    Activation('tanh'),
    Dense(128, activation='relu'),
    Dense(horizon)
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=200)
y_pred = model.predict(x_test)

df = pd.DataFrame({'real':y_test.flatten(), 'previsto':y_pred.flatten()}, index=y_test_index.flatten())
df.index.name = 'timestamp'
df.to_csv('previsao.csv')

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Calculate the metrics
r2 = r2_score(df['real'], df['previsto'])
mae = mean_absolute_error(df['real'], df['previsto'])
mse = mean_squared_error(df['real'], df['previsto'])
rmse = np.sqrt(mse)

df['hour'] = df.index.hour
df['date'] = df.index.date

# Formatacao do horario
df['hour'] = df['hour'].apply(lambda x: f'{x:02d}:00:00')

# Pivot the DataFrame
pivot_df = df.pivot(index='hour', columns='date', values='real')
pivot_df.columns.name = None
pivot_df.to_csv('real_dias.csv')

pivot_df = df.pivot(index='hour', columns='date', values='previsto')
pivot_df.columns.name = None
pivot_df.to_csv('previsto_dias.csv')

# Load the pivoted data from the CSVs
real_pivot = pd.read_csv('real_dias.csv', index_col=0)
previsto_pivot = pd.read_csv('previsto_dias.csv', index_col=0)

# Initialize an empty DataFrame to store the metrics

metrics_df = pd.DataFrame(columns=['R2', 'MAE', 'MSE', 'RMSE'], index=real_pivot.columns)
metrics_df.index.name = 'date'

# Calculate metrics for each day
for col in real_pivot.columns:
    y_true = real_pivot[col].dropna()  # Remove NaN values, if any
    y_prev = previsto_pivot[col].dropna()
    
    # Ensure the lengths match after dropping NaNs
    if len(y_true) == len(y_prev):
        r2 = r2_score(y_true, y_prev)
        mae = mean_absolute_error(y_true, y_prev)
        mse = mean_squared_error(y_true, y_prev)
        rmse = np.sqrt(mse)
        
        metrics_df.loc[col] = [r2, mae, mse, rmse]

# Save the metrics DataFrame to a CSV file
metrics_df.to_csv('metrics_per_day.csv')

n_plots = 2
fig, ax = plt.subplots(n_plots, 1, sharex=True)

for i in range(n_plots):
    ax[i].plot(x_test_index[i], x_test[i], label='Input')
    ax[i].plot(y_test_index[i], y_pred[i], label='Previsao')
    ax[i].plot(y_test_index[i], y_test[i], label='Valor Real')

    ax[i].set_xlabel('Data')
    ax[i].set_ylabel('Consumo')

    ax[i].xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax[i].xaxis.set_minor_locator(mdates.HourLocator())

    ax[i].grid()
    ax[i].legend()

fig.savefig('previsao.png')
