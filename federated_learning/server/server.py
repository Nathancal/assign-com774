import flwr as fl
from typing import Dict
import utils
import pandas as pd
import argparse
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace,  Environment, Run
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig, Model
from keras.models import save_model
import datetime

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

# Set up Azure ML run
run = Run.get_context()

# Log parameters
run.log("trainingdata", data_name)

# Load and preprocess combined HAR data
X, Y = utils.load_har_data(data_asset.path)

model = utils.create_lstm_model()

def fit_round(server_round: int) -> Dict:
    """Send round number to client"""
    run.log("server_round", server_round)
    return {"server_round": server_round}

def get_evaluate_fn(model):
    """Build an evaluation function for Flower to use to assess performance"""
    def evaluate(server_round: int, parameters:fl.server.history, config: Dict[str, fl.common.Scalar]):
        """Update the model to use the given parameters and return its score"""
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X, Y)

        # Log metrics to Azure ML
        run.log("loss", loss)
        run.log("accuracy", accuracy)

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
current_datetime = datetime.datetime.now()

# Format as a string
formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")

# Save the federated model after training
save_model(model, "federated_model_{formatted_datetime}.h5")

# Log the saved model as an artifact
run.upload_file(name="outputs/federated_model_{formatted_datetime}.h5", path_or_stream="federated_model_{formatted_datetime}.h5")

# Log the federated model to Azure ML
run.log_artifact("federated_model_{formatted_datetime}.h5")
model_path = run.get_file_names()["outputs/federated_model_{formatted_datetime}.h5"]

# Register the model in Azure ML
azure_model = Model.register(workspace=ws,
                                  model_name=f"federated_model_{formatted_datetime}",
                                  model_path=model_path,
                                  tags={"run_id": run.id, "experiment_name": run.experiment.name},
                                  description="Federated Model registered from Flower training")

# Retrieve accuracy from Azure ML
accuracy = run.get_metrics().get("accuracy", 0.0)

accuracy_threshold = 0.8
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

# End Azure ML run
run.complete()