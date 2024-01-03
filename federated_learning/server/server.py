import flwr as fl
from typing import Dict
import utils 
import pandas as pd
import argparse
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace, Model, Environment
import mlflow
import datetime
from keras.models import save_model
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig


# Replace with your actual values
tenant_id = "6f0b9487-4fa8-42a8-aeb4-bf2e2c22d4e8"
client_id = "1bee10b2-17dd-4a50-b8aa-488d27bdd5a1"
client_secret = "MZK8Q~M5oNATdagyRKMUs-V-2dNggq3aAlRRdb8W"
subscription_id = "092da66a-c312-4a87-8859-56031bb22656"


# Load Azure Machine Learning workspace from configuration file
ws = Workspace.from_config(path='./config.json')
environment = Environment.get(workspace=ws, name="development")

# Get the arguments we need to avoid fixing the dataset path in code
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Training data for model server')
args = parser.parse_args()

data_name = args.trainingdata

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
data_asset = ml_client.data.get("combined_data_sub1-9", version="1")

df = pd.read_csv(data_asset.path)

print(df)
# Load and preprocess combined HAR data
X, Y = utils.load_har_data(data_asset.path)

print(X)
print(Y)

# Start MLflow run
mlflow.start_run()

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
        # Log metrics to MLflow
        mlflow.log_metric("loss", loss)
        mlflow.log_metric("accuracy", accuracy)
        return loss, {"accuracy": accuracy}

    return evaluate

# Set initial LSTM parameters
utils.set_initial_lstm_params(model)

# Set up a FedAvg strategy using the functions above expecting 2 clients
strategy = fl.server.strategy.FedAvg(
    min_available_clients=9,
    evaluate_fn=get_evaluate_fn(model),
    on_fit_config_fn=fit_round,
)

# Start the Flower server with the strategy for 10 runs
fl.server.start_server(server_address="0.0.0.0:8008",
                       strategy=strategy,
                       config=fl.server.ServerConfig(num_rounds=25))

# Get the current date and time
current_datetime = datetime.now()

# Format as a string
formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")

# Save the federated model after training
save_model(model, "federated_model_{formatted_datetime}.h5")



# After saving the individual models
mlflow.log_artifact(f"federated_model_{formatted_datetime}.h5")
model_path = mlflow.get_artifact_uri(f"federated_model_{formatted_datetime}.h5")
mlflow.register_model(model_path, f"federated_model_{formatted_datetime}")

model = Model.register(workspace=ws, model_path=model_path, model_name=f"federated_model_{formatted_datetime}")


# Retrieve accuracy from MLflow
run = mlflow.get_run()
accuracy = run.data.metrics.get("accuracy", 0.0)

accuracy_threshold=0.8
        # Deploy only if accuracy is greater than the threshold
if accuracy > accuracy_threshold:
    # Define inference configuration
    inference_config = InferenceConfig(entry_script="score.py", runtime="python", conda_file=environment)

            # Deploy the model as a web service
    aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)


    # Deploy the model as a web service
    service_name = f"{model.name.lower()}-service"
    service = Model.deploy(workspace=ws,
                                    name=service_name,
                                    models=[model],
                                    inference_config=inference_config,
                                    deployment_config=aciconfig)
    service.wait_for_deployment(show_output=True)

# End MLflow run
mlflow.end_run()