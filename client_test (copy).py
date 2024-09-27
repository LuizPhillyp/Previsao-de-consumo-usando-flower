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
        history = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=100, batch_size=200)

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

    #Preparacao dos features e targets

    window = 24             #Quantas horas passadas serao usadas de input
    horizon = 24            #Quantas horas futuras serao previstas

    refresh_period = 24     #Quantas horas ate pegar um novo input

    #Aplicacao mais eficiente de janelas deslizantes
    x = np.array([data[i:i+window] for i in range(0, len(data) - window - horizon + 1, refresh_period)])
    y = np.array([data[i+window:i+window+horizon] for i in range(0, len(data) - window - horizon + 1, refresh_period)])
    x_index = np.array([index[i:i+window] for i in range(0, len(index) - window - horizon + 1, refresh_period)])
    y_index = np.array([index[i+window:i+window+horizon] for i in range(0, len(index) - window - horizon + 1, refresh_period)])

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
        Dense(horizon)
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

    df = pd.DataFrame({'real':y_test.flatten(), 'previsto':y_pred.flatten()}, index=y_test_index.flatten())
    df.index.name = 'timestamp'

    #arquivo com previsao de todos os valores teste de cada cliente
    df.to_csv(f'./previsoes/client_{args.client_id}/previsao_client{args.client_id}.csv')

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    # calculo das metricas
    r2 = r2_score(df['real'], df['previsto'])
    mae = mean_absolute_error(df['real'], df['previsto'])
    mse = mean_squared_error(df['real'], df['previsto'])
    rmse = np.sqrt(mse)

    metrics_file = './metricas/metricas_clientes.csv'
    lock = FileLock(metrics_file + ".lock")

    # concatenando as metricas no arquivo das metricas de cada cliente
    with lock:
        file_exists = os.path.isfile(metrics_file)
        with open(metrics_file, mode='a') as f:
            if not file_exists:
                f.write("Client ID,R2,MAE,MSE,RMSE\n")
            f.write(f"{args.client_id},{r2},{mae},{mse},{rmse}\n")


    #Separando as previsoes por dia
    df['hour'] = df.index.hour
    df['date'] = df.index.date

    # Formatacao do horario
    df['hour'] = df['hour'].apply(lambda x: f'{x:02d}:00:00')

    pivot_df = df.pivot(index='hour', columns='date', values='real')
    pivot_df.columns.name = None
    pivot_df.to_csv(f'./previsoes/client_{args.client_id}/real_dias_client{args.client_id}.csv')

    pivot_df = df.pivot(index='hour', columns='date', values='previsto')
    pivot_df.columns.name = None
    pivot_df.to_csv(f'./previsoes/client_{args.client_id}/previsto_dias_client{args.client_id}.csv')

    # calculo das metricas para cada dia
    real_pivot = pd.read_csv(f'./previsoes/client_{args.client_id}/real_dias_client{args.client_id}.csv', index_col=0)
    previsto_pivot = pd.read_csv(f'./previsoes/client_{args.client_id}/previsto_dias_client{args.client_id}.csv', index_col=0)

    metrics_df = pd.DataFrame(columns=['R2', 'MAE', 'MSE', 'RMSE'], index=real_pivot.columns)
    metrics_df.index.name = 'date'

    for col in real_pivot.columns:
        y_true = real_pivot[col].dropna()
        y_prev = previsto_pivot[col].dropna()
        
        if len(y_true) == len(y_prev):
            r2 = r2_score(y_true, y_prev)
            mae = mean_absolute_error(y_true, y_prev)
            mse = mean_squared_error(y_true, y_prev)
            rmse = np.sqrt(mse)
            
            metrics_df.loc[col] = [r2, mae, mse, rmse]

    metrics_df.to_csv(f'./metricas/metricas_por_dia_client{args.client_id}.csv')


    # plot de duas instancias de previsao
    n_plots = 5
    fig, ax = plt.subplots(n_plots, 1, sharex=True)

    for i in range(n_plots):
        ax[i].plot(x_test_index[i], x_test[i], label='Input')
        ax[i].plot(y_test_index[i], y_pred[i], label='Previsao')
        ax[i].plot(y_test_index[i], y_test[i], label='Valor Real')

        ax[i].set_xlabel('Tempo')
        ax[i].set_ylabel('Consumo')

        ax[i].xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        ax[i].xaxis.set_minor_locator(mdates.HourLocator())

        ax[i].grid()
        ax[i].legend()

    fig.savefig(f'./plots/previsao_client{args.client_id}.png')

if __name__ == "__main__":
    main()
