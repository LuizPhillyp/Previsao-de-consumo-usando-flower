import tensorflow as tf
import argparse
import pandas as pd
import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from filelock import FileLock
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Propriedades do cliente usadas pelo flower
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model, x_train, y_train, x_test, y_test, cid):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.client_id = cid
        self.round = 0

    def get_parameters(self, config):

        return self.model.get_weights()
    
    def fit(self, parameters, config):

        self.model.set_weights(parameters)
        history = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=1, batch_size=200)

        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):

        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return float(loss), len(self.x_test), {"accuracy": float(accuracy)}

def main() -> None:

    #o programa requer o id do cliente como argumento de entrada

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        choices=range(0, 20),
        required=True,
    )
    args = parser.parse_args()

    file = pd.read_csv("dsV4.csv", header=0, index_col=0, parse_dates=True, sep=";", decimal=",")
    data = file.iloc[:, args.client_id-1]

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
        #Prepara x para a entrada da ANN
    x = x.reshape((x.shape[0], window, 1))

    #Split dos dados de treino e teste
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test, x_train_index, x_test_index, y_train_index, y_test_index = train_test_split(
        x, y, x_index, y_index, test_size=0.25, shuffle=False
    )

    # Rede neural baseada em LSTM's e CNN's
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, LSTM, Dense, Activation, MaxPooling1D, Dropout, Input

    model = Sequential([
        Input(shape=(window, 1)),
        Conv1D(filters=32, kernel_size=2, padding='causal', strides=2),
        MaxPooling1D(pool_size=2, strides=2),
        Activation('relu'),
        LSTM(32, return_sequences=False),
        Activation('tanh'),
        Dense(128, activation='relu'),
        Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])

    #inicializacao do cliente
    client = FlowerClient(model, x_train, y_train, x_test, y_test, args.client_id).to_client()
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client,
    )

    #o restante do codigo ira executar apenas quando todos os rounds acabaram

    #previsao dos valores teste
    y_pred = model.predict(x_test)

    df = pd.DataFrame({'real':y_test, 'previsto':y_pred.flatten()}, index=y_test_index)
    df.index.name = 'timestamp'

    #arquivo com previsao de todos os valores teste de cada cliente
    df.to_csv(f'./previsoes/previsao_client{args.client_id}.csv')

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    # calculo das metricas
    r2 = r2_score(df['real'], df['previsto'])
    mae = mean_absolute_error(df['real'], df['previsto'])
    mse = mean_squared_error(df['real'], df['previsto'])
    rmse = np.sqrt(mse)

    file = open(f'./metricas/metricas_cliente{args.client_id}.csv', 'w')
    file.write('r2,mae,mse,rmse')
    file.write(f'{r2},{mae},{mse},{rmse}')

    # plot de n_plots instancias de previsao
    n_plots = 2
    fig, ax = plt.subplots(n_plots, 1, sharex=True)

    for i in range(n_plots):
        ax[i].plot(y_test_index[:24*(i+1)], y_pred[:24*(i+1)], label='Previsao')
        ax[i].plot(y_test_index[:24*(i+1)], y_test[:24*(i+1)], label='Valor Real')

        ax[i].set_xlabel('Tempo')
        ax[i].set_ylabel('Consumo')

        ax[i].xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        ax[i].xaxis.set_minor_locator(mdates.HourLocator())

        ax[i].grid()
        ax[i].legend()

    fig.savefig(f'./plots/previsao_client{args.client_id}.png')

if __name__ == "__main__":
    main()
