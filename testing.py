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

# Define constants for windowing
window = 24             # Full day as input (00:00 to 00:00 of the next day)
horizon = 24            # Predict one day after the end of the input sequence
refresh_period = 1     # Refresh every 24 hours

# Prepare sliding windows of data
x = np.array([data[i:i+window] for i in range(0, len(data) - window - horizon + 1, refresh_period)])
y = np.array([data[i+window+horizon-1] for i in range(0, len(data) - window - horizon + 1, refresh_period)])

# Prepare corresponding indices
x_index = np.array([index[i:i+window] for i in range(0, len(index) - window - horizon + 1, refresh_period)])
y_index = np.array([index[i+window+horizon-1] for i in range(0, len(index) - window - horizon + 1, refresh_period)])

x_df = pd.DataFrame(x)
y_df = pd.DataFrame(y)

#Split dos dados de treino e teste
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, x_train_index, x_test_index, y_train_index, y_test_index = train_test_split(
    x, y, x_index, y_index, test_size=0.25, shuffle=False
)


df_previsao = pd.read_csv('./previsoes/previsao_client9.csv')

prev_index = df_previsao['timestamp'].values
prev = df_previsao['previsto'].values

n_plots = 1
fig, ax = plt.subplots(n_plots, 1, sharex=True)


for i in range(24):
    ax.plot(index[:72], data[:72], label='Valor Real', linestyle='--', color='black')
    ax.plot(x_index[i], x[i], label='Input', color='blue')
    ax.plot(prev_index[0:i+1], prev[0:i+1], label='Output', color='orange', marker='o')

    ax.set_xlabel('Data')
    ax.set_ylabel('Consumo')

    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.xaxis.set_minor_locator(mdates.HourLocator())

    ax.grid()
    ax.legend()
    print('savingfig')
    fig.savefig(f'previsao{i}.png')


