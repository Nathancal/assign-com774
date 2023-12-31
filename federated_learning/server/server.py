# server.py
import flwr as fl
from typing import Dict
import utils 
import pandas as pd
import argparse
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
from azureml.core import Workspace, Dataset

# Replace with your actual values
tenant_id = "6f0b9487-4fa8-42a8-aeb4-bf2e2c22d4e8"
client_id = "1bee10b2-17dd-4a50-b8aa-488d27bdd5a1"
client_secret = "MZK8Q~M5oNATdagyRKMUs-V-2dNggq3aAlRRdb8W"
subscription_id = "092da66a-c312-4a87-8859-56031bb22656"

# Create a credentials object using the service principal details
credentials = ClientSecretCredential(
    tenant_id=tenant_id,
    client_id=client_id,
    client_secret=client_secret
)

ml_client = MLClient.from_config(credential=credentials)

# Load the Azure ML workspace
ws = Workspace.from_config()

# Get the arugments we need to avoid fixing the dataset path in code
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Training data for model server')
args = parser.parse_args()

data_name = args.trainingdata

# Load and preprocess combined HAR data
X, Y = utils.load_har_data(data_name)

print(X)
print(Y)

model = utils.create_lstm_model()

def fit_round(server_round: int) -> Dict:
    """Send round number to client"""
    return {"server_round": server_round}

def get_evaluate_fn(model):
    """Build an evaluation function for Flower to use to assess performance"""
    def evaluate(server_round: int, parameters:fl.server.history, config: Dict[str, fl.common.Scalar]):
        """Update the model to use the given parameters and return its score"""
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X, Y)
        return loss, {"accuracy": accuracy}

    return evaluate

# Start Flower server for 10 rounds of federated learning
if __name__ == "__main__":
    utils.set_initial_lstm_params(model)
    # Set up a FedAvg strategy using the functions above expecting 2 clients
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=9,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    # Start the Flower server with the strategy for 10 runs
    fl.server.start_server(server_address="40.113.153.115:8008",
                           strategy=strategy,
                           config=fl.server.ServerConfig(num_rounds=25))
