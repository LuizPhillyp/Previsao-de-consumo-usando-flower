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


print(y_test.size)


# print(x_df)

# n_plots = 2
# fig, ax = plt.subplots(n_plots, 1, sharex=True)

# for i in range(n_plots):

#     ax[i].plot(index[:48], data[:48], label='Actual', linestyle='--', color='black')
#     ax[i].plot(x_index[i], x[i], label='Input', color='blue')
#     ax[i].scatter(y_index[i], y[i], label='Output', color='orange')
    
    

#     ax[i].set_xlabel('Data')
#     ax[i].set_ylabel('Consumo')

#     ax[i].xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
#     ax[i].xaxis.set_minor_locator(mdates.HourLocator())

#     ax[i].grid()
#     ax[i].legend()

# fig.savefig('previsao.png')


