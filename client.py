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
        history = self.model.fit(
            self.x_train,
            self.y_train,
            validation_split=0.1,
        )

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
    series = file.iloc[:, args.client_id-1]


    horizon = 24 # o numero de steps para serem previstos no futuro (so faz sentido para previsao em multistep)
    window_size = 64 # o numero de steps no passado que seram usados como features


    n_samples = len(series) - window_size
    target_size = 1

    x = np.zeros(shape=(n_samples, window_size))
    y = np.zeros(shape=(n_samples, target_size))

    for t in range(n_samples):
        x[t] = series[t:window_size + t]
        y[t] = series[window_size + t]

    x = x.reshape(-1, window_size, 1)

    
    # identifica qual fracao do dataset sera usado para testes (o restante sera para o treinamento)
    split = n_samples // 4


    x_train = x[:n_samples - split]
    x_test = x[-split:]

    y_train = y[:n_samples - split]
    y_test = y[-split:]


    # determinacao do modelo (pode ser alterado)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(window_size, kernel_size=3, padding='causal', strides=1, input_shape=(x_train.shape[1], x_train.shape[2])),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling1D(strides=2),
        tf.keras.layers.Conv1D(64, kernel_size=3, padding='causal', strides=1),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Activation('tanh'),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(1)
    ], name="lstm_cnn")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])


    #inicializacao do cliente
    client = FlowerClient(model, x_train, y_train, x_test, y_test, args.client_id).to_client()
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client,
    )

    #o restante do codigo ira executar apenas quando todos os rounds acabaram


    #determina se o programa ira prever mais de um step no futuro
    multistep = False
    if multistep:

        predictions = np.zeros(shape=y_test.shape)
        last_input = x_test[0]

        for i in range(horizon):
            p = model.predict(last_input.reshape(1, -1, 1))[0, 0]
            predictions[i] = p
            last_input = np.roll(last_input, -1)
            last_input[-1] = p
    
    else:
         
         predictions = model.predict(x_test)

    # definicao das metricas (pode ser alterado)
    r2 = r2_score(y_test[:horizon], predictions[:horizon])
    mae = mean_absolute_error(y_test[:horizon], predictions[:horizon])
    mse = mean_squared_error(y_test[:horizon], predictions[:horizon])
    rmse = np.sqrt(mse)


    #o segmento a seguir cria o arquivo csv com as metricas
    #ele garante que os clientes nao vao conseguir escrever
    #no arquivo simultaneamente

    metrics_file = 'metrics.csv'
    lock = FileLock(metrics_file + ".lock")

    with lock:
        file_exists = os.path.isfile(metrics_file)
        with open(metrics_file, mode='a') as f:
            if not file_exists:
                f.write("Client ID,R2,MAE,MSE,RMSE\n")
            f.write(f"{args.client_id},{r2},{mae},{mse},{rmse}\n")

    #segmento para plotar os graficos. Ele apenas plota o target do segmento de teste
    #contra a previsao feita pelo modelo

    timestamps = file.index[-split:]
    timestamps = pd.to_datetime(timestamps, dayfirst=True)
    
    plt.figure()
    plt.plot(timestamps[:horizon], y_test[:horizon], label='Actual', marker='.')
    plt.plot(timestamps[:horizon], predictions[:horizon], label='Predicted', marker='.')

    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))

    for ts in timestamps[:horizon]:
        if ts.hour == 0:
            plt.axvline(x=ts, color='gray', linestyle='--', linewidth=0.5)
            plt.text(ts, plt.ylim()[1]*0.75, ts.strftime('%Y-%m-%d'), rotation=90, verticalalignment='bottom')

    plt.xlabel('Hour')
    plt.ylabel('Value')
    plt.legend()
    plt.title(f'Client {args.client_id} Predictions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'./plots/previsao_cliente_{args.client_id}.png')

if __name__ == "__main__":
    main()
