# client.py
import flwr as fl
import numpy as np
import utils
from sklearn.model_selection import train_test_split

def client(data):
    # Create an LSTM model
    model = utils.create_lstm_model()

    # Load client data
    X, Y = utils.load_har_data(data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)

    print(X_train)
    print(Y_train)

    # Set initial LSTM model parameters
    utils.set_initial_lstm_params(model)


    fl.client.start_numpy_client(server_address="40.113.153.115:8008", client=HARClient())



    class HARClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return utils.get_lstm_model_parameters(model)

        def fit(self, parameters, config):
            utils.set_lstm_model_params(model, parameters)
            model.fit(X_train, Y_train, epochs=10, verbose=0)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_lstm_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):
            utils.set_lstm_model_params(model, parameters)
            loss, accuracy = model.evaluate(X_test, Y_test)
            return loss, len(X_test), {"accuracy": accuracy}