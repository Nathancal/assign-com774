import flwr as fl
from typing import Dict
import utils
import pandas as pd
import argparse
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace, Environment
from keras.models import save_model
import datetime
import threading
import mlflow 
from sklearn.model_selection import train_test_split
import logging
from azureml.core import Experiment
from azureml.core.authentication import ServicePrincipalAuthentication
import argparse
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.core.webservice import AciWebservice, AksWebservice, Webservice
from azureml.core import Workspace, Model
from azureml.core.run import Run
import mlflow
import time

import pandas as pd
from azureml.core.run import Run
from datetime import datetime, timedelta, timezone
from azure.ai.ml.entities import (
    AzureMLOnlineInferencingServer,
    ModelPackage,
    CodeConfiguration,
    BaseEnvironment,
)
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    Model,
)
from azureml.core import Environment
from azureml.core.model import  Model
import random
import string
import uuid
import os

# Load Azure Machine Learning workspace from configuration file

# Get the arguments we need to avoid fixing the dataset path in code
parser = argparse.ArgumentParser()

parser.add_argument("--training_data", type=str, required=True, help='Path to the dataset')
parser.add_argument("--minimum_clients", type=int, required=True, help='experiment name')

# Convert minimum_clients to an integer
args = parser.parse_args()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
client_id = "3ce68579-31fd-417f-9037-97a114f15e9d"
client_secret = "MZK8Q~M5oNATdagyRKMUs-V-2dNggq3aAlRRdb8W"
# Replace with your actual values
tenant_id = "6f0b9487-4fa8-42a8-aeb4-bf2e2c22d4e8"
resource_group = "assignment2-b00903995"
workspace_name = "assignment2-ML-workspace"
mlflow_location = "westeurope"
subscription_id = "092da66a-c312-4a87-8859-56031bb22656"



# Authenticate using service principal credentials
# Service principal authentication configuration
svc_pr_password = "MZK8Q~M5oNATdagyRKMUs-V-2dNggq3aAlRRdb8W"
svc_pr = ServicePrincipalAuthentication(
    tenant_id="6f0b9487-4fa8-42a8-aeb4-bf2e2c22d4e8",
    service_principal_id="1bee10b2-17dd-4a50-b8aa-488d27bdd5a1",
    service_principal_password=svc_pr_password
)

# Connect to Azure ML workspace
ws = Workspace.from_config(auth=svc_pr, path='./config.json')


mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

mlflow.autolog()     

environment = Environment.get(workspace=ws, name="development")

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
data_asset = ml_client.data._get_latest_version(args.training_data)

# Load and preprocess combined HAR data
X, Y = utils.load_har_data(data_asset.path)

run = Run.get_context()
last_round_flag = threading.Event()

num_rounds = 4
model = utils.create_lstm_model()

# Define experiment name
experiment_name = "Fed-Learning-Server-Staging-Env"

# Get an existing experiment or create a new one
experiment = Experiment(workspace=ws, name=experiment_name)

mlflow.set_experiment(experiment)
def generate_8_digit_uuid():
    # Generate a UUID and take the last 8 characters
        
    generated_uuid = str(uuid.uuid4().hex)[-8:]

    return generated_uuid

        
def fit_round(server_round: int) -> Dict:

    # Define a fixed factor to decrease the learning rate
    learning_rate_decay_factor = 0.98

    # Update the learning rate based on the round
    new_learning_rate = model.optimizer.lr * learning_rate_decay_factor
    model.optimizer.lr = new_learning_rate

    # Send round number to client
    config = {"learning_rate": new_learning_rate}
    return {"server_round": server_round, "config": config}

def get_evaluate_fn(model, experiment):

    """Build an evaluation function for Flower to use to assess performance"""
    def evaluate(server_round: int, parameters:fl.server.history, config: Dict[str, fl.common.Scalar]):
        
        """Update the model to use the given parameters and return its score"""
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X, Y)

        with mlflow.start_run():
            mlflow.log_param("server_round", server_round)

            logger.info("server_round" + str(server_round))
            # Log accuracy and loss as metrics
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('loss', loss)

            current_datetime = datetime.now()

                    # Format as a string
            formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
                    
                    # Save the model after training
            model.save(f"outputs/Federated_Model_{formatted_datetime}_model.h5")

                    # Construct the model path based on the relative path within the outputs directory
            model_path_relative = f"outputs/Federated_Model_{formatted_datetime}_model.h5"
            
            accuracy_publish_threshold = 0.4
                    # Deploy only if accuracy is greater than the threshold
            if accuracy > accuracy_publish_threshold:
                        # Log the saved model as an artifact
                run.upload_file(name=model_path_relative, path_or_stream=model_path_relative)
                exp_name = "Federated-server-mdl-" + "acc-"+"{:.2f}".format(accuracy).split(".")[1]
                formatted_datetime

                logger.info("RUN-ID: " +run.id)
                        
                model_name=exp_name+"-"+formatted_datetime
                        # Register the model in Azure ML
                azure_model = Model.register(workspace=ws,
                    model_name=model_name,
                    model_path=model_path_relative,
                    tags={"run_id": run.id, "experiment_name": "Federated-Server-Model", "accuracy": accuracy},
                    description=f"Federated Model registered from Flower training, Accuracy: {accuracy}"
                )   

                if server_round == num_rounds:
                    deploy_models()   
        mlflow.end_run()
        return loss, {"accuracy": accuracy}
    
    return evaluate

def start_flower_server(experiment):
    # Set up a FedAvg strategy using the functions above expecting 2 clients
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=args.minimum_clients,
        evaluate_fn=get_evaluate_fn(model, experiment),
        on_fit_config_fn=fit_round,
    )
    

  # Define a lambda function to start the Flower server
    server = fl.server.start_server(
        server_address="0.0.0.0:8008",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
    )
    
def deploy_models():

        # Get a list of all registered models
    all_models = Model.list(workspace=ws)

    # Set the threshold for accuracy
    accuracy_threshold = 0.5
    name ="Federated"
    # Filter models based on the accuracy threshold and timestamp
    filtered_models = [
        model
        for model in all_models
        if (
            "accuracy" in model.tags
            and float(model.tags["accuracy"]) > accuracy_threshold
            and model.created_time >= (datetime.now(timezone.utc) - timedelta(minutes=15))  # Change the time window as needed
            and name in model.name 
        )
    ]

    # Get the model with the highest accuracy among the filtered models
    recent_high_accuracy_models = sorted(
        filtered_models, key=lambda x: float(x.tags["accuracy"]), reverse=True
    )

    if recent_high_accuracy_models:

        highest_accuracy_model = recent_high_accuracy_models[0]

        environment = Environment.get(workspace=ws, name="infer-env")

        logger.info("ENVIRONMENT: " + environment.name)
        modelLoad = ml_client.models.get(name=highest_accuracy_model.name, label="latest")

        logger.info("WSURI: " + ws.get_mlflow_tracking_uri())

        logger.info("MODEL NAME: " + highest_accuracy_model.name)
        
      

        pakage_config = ModelPackage(
            target_environment=f"{highest_accuracy_model.name}-env-pkg",
            base_environment_source=BaseEnvironment(
                type="asset",
                resource_id="azureml:infer-env:4"
            ),
            inferencing_server=AzureMLOnlineInferencingServer(
                code_configuration=CodeConfiguration(code="./", scoring_script="score.py")
            ),
            tags={"run_id": highest_accuracy_model.tags["run_id"], "model": highest_accuracy_model.name},
        )

        model_package = ml_client.models.package(highest_accuracy_model.name, modelLoad.version, pakage_config)
        
        eight_digit_uuid = generate_8_digit_uuid()

        endpoint = ManagedOnlineEndpoint(name="fed-model-endp"+eight_digit_uuid)
        endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        logger.info("ENDPOINT:" + endpoint.name)

        deployment_package = ManagedOnlineDeployment(
            name=endpoint.name,
            endpoint_name=endpoint.name,
            environment=model_package,
            instance_type="Standard_D4as_v4",
            instance_count=1,
            environment_variables={'MODEL_NAME': highest_accuracy_model.name},
            # pass environment variables to the score.py
            tags={"run_id": highest_accuracy_model.tags["run_id"], "experiment_name": highest_accuracy_model.tags["experiment_name"], "model": highest_accuracy_model.name},
        )

        logger.info("MODEL DEPLOYMENT: " + deployment_package.endpoint_name)

        max_retries = 3
        retries = 0

        while retries < max_retries:
            try:
                deploy_result = ml_client.online_deployments.begin_create_or_update(deployment_package).result()
                logger.info(deploy_result)
                break
            except Exception as e:
                logger.error(f"Error during deployment: {str(e)}")
                retries += 1
                time.sleep(5)  # Add a delay before retrying

        if retries == max_retries:
            logger.error("Maximum retries reached. Deployment failed.")


# Start Flower server for 10 rounds of federated learning
if __name__ == "__main__":
    utils.set_initial_lstm_params(model)
    # Set up a FedAvg strategy using the functions above expecting 2 clients
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=args.minimum_clients,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    # Start the Flower server with the strategy for 10 runs
    start_flower_server()


   
    