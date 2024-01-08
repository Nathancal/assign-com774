import flwr as fl
import utils
from sklearn.model_selection import train_test_split
import logging
from azureml.core.authentication import ServicePrincipalAuthentication
import argparse
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.core.webservice import AciWebservice, AksWebservice, Webservice
from azureml.core import Workspace, Model
from azureml.core.run import Run
import mlflow
import os
import numpy as np
import pandas as pd
from azureml.core.run import Run
from datetime import datetime, timedelta, timezone
from azureml.core.compute import AmlCompute
import threading
from azure.ai.ml.entities import (
    AzureMLOnlineInferencingServer,
    ModelPackage,
    CodeConfiguration,
    BaseEnvironment,
    ModelConfiguration,
)
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    Model,
)
from azureml.core import Environment, Webservice
from azureml.core.model import InferenceConfig, Model
import random
import string
from mlflow.deployments import get_deploy_client
import json
from azureml.core.compute import ComputeTarget
import uuid
from sklearn.preprocessing import LabelEncoder


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace with your actual values
tenant_id = "6f0b9487-4fa8-42a8-aeb4-bf2e2c22d4e8"
client_id = "3ce68579-31fd-417f-9037-97a114f15e9d"
client_secret = "MZK8Q~M5oNATdagyRKMUs-V-2dNggq3aAlRRdb8W"
resource_group = "assignment2-b00903995"
workspace_name = "assignment2-ML-workspace"

# Authenticate using service principal credentials
# Service principal authentication configuration
svc_pr_password = "MZK8Q~M5oNATdagyRKMUs-V-2dNggq3aAlRRdb8W"
svc_pr = ServicePrincipalAuthentication(
    tenant_id="6f0b9487-4fa8-42a8-aeb4-bf2e2c22d4e8",
    service_principal_id="1bee10b2-17dd-4a50-b8aa-488d27bdd5a1",
    service_principal_password=svc_pr_password
)
# Load your Azure ML workspace
ws = Workspace.from_config(auth=svc_pr, path='./config.json')

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.autolog()


# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help='Path to the dataset')
parser.add_argument("--experiment_name", type=str, required=True, help='experiment name')
args = parser.parse_args()
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

client_completed_event = threading.Event()
run = Run.get_context()

def generate_8_digit_uuid():
    # Generate a UUID and take the last 8 characters
    generated_uuid = str(uuid.uuid4().hex)[-8:]
    return generated_uuid


def deploy_models():

        # Get a list of all registered models
    all_models = Model.list(workspace=ws)

    # Set the threshold for accuracy
    accuracy_threshold = 0.6
    name = args.experiment_name
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
    logger.info(recent_high_accuracy_models[0])
    logger.error("GETS TO CLIENT DEPLOY!")

    if recent_high_accuracy_models:

        highest_accuracy_model = recent_high_accuracy_models[0]

        environment = Environment.get(workspace=ws, name="infer-env")

        logger.info("ENVIRONMENT: " + environment.name)
        modelLoad = ml_client.models.get(name=highest_accuracy_model.name, label="latest")

        logger.info("WSURI: " + ws.get_mlflow_tracking_uri())
        current_directory = os.path.dirname(os.path.abspath(__file__))
        logger.info("MODEL NAME: " + highest_accuracy_model.name)
        pakage_config = ModelPackage(
            target_environment=f"{highest_accuracy_model.name}-pkg",
            base_environment_source=BaseEnvironment(
                type="asset",
                resource_id="azureml:infer-env:4"
            ),
            inferencing_server=AzureMLOnlineInferencingServer(
                code_configuration=CodeConfiguration(code=current_directory, scoring_script="score.py")
            ),
            tags={"run_id": highest_accuracy_model.tags["run_id"], "model": highest_accuracy_model.name},
        )

        model_package = ml_client.models.package(highest_accuracy_model.name, modelLoad.version, pakage_config)
        eight_digit_uuid = generate_8_digit_uuid()
        original_string = args.experiment_name
        endpoint_safe_string = original_string.replace("_", "-")
        endpoint = ManagedOnlineEndpoint(name=endpoint_safe_string)
        endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        logger.info("ENDPOINT_CLIENT: " + endpoint.name)
        
        deployment_package = ManagedOnlineDeployment(
            name=endpoint.name,
            endpoint_name=endpoint.name,
            environment=model_package,
            instance_count=1,
            instance_type="Standard_E4s_v3",
            environment_variables={'MODEL_NAME': highest_accuracy_model.name},
            # pass environment variables to the score.py
            tags={"run_id": highest_accuracy_model.tags["run_id"], "experiment_name": highest_accuracy_model.tags["experiment_name"], "model": highest_accuracy_model.name},
        )

        logger.info("MODEL DEPLOYMENT: " + deployment_package.endpoint_name)

        deploy_result = ml_client.online_deployments.begin_create_or_update(deployment_package).result()

        logger.info("DEPLOY_RESULT: " + deploy_result)

        mlflow.end_run()

def start_fl_client():

    # Create an LSTM model
    model = utils.create_lstm_model()
    print("TRACKING URI: " + ws.get_mlflow_tracking_uri())
    # Load client data
    X, Y = utils.load_har_data(args.data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)


    logger.info(f"shape: {Y_train.shape}")
    # Get the current run context in an Azure ML job

    # Set initial LSTM model parameters
    utils.set_initial_lstm_params(model)
    
    class HARClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return utils.get_lstm_model_parameters(model)

        def fit(self, parameters, config):
            with mlflow.start_run():

                utils.set_lstm_model_params(model, parameters)
                history = model.fit(X_train, Y_train, epochs=10, verbose=0)


                # Log accuracy and loss as metrics
                accuracy = history.history['accuracy'][-1]
                loss = history.history['loss'][-1]
                
                # Log metrics using MLflow
                mlflow.log_metric('accuracy', accuracy)
                mlflow.log_metric('loss', loss)

                print(f"Training finished for round {config['server_round']}")

                if config['server_round'] == 4:
                    deploy_models()
                return utils.get_lstm_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):
            round_num = config.get("server_round", 1)  # Assuming a default value of 1 if not present

            utils.set_lstm_model_params(model, parameters)
            loss, accuracy = model.evaluate(X_test, Y_test)
            
            # Your list of classes
            class_names = ['Running', 'Sitting', 'Standing', 'Walking', 'downstairs', 'upstairs']

            # Initialize the LabelEncoder
            encoder = LabelEncoder()

            # Fit the encoder to your classes and transform them to numerical labels
            encoded_labels = encoder.fit_transform(class_names)


            predictions2 = np.argmax(model.predict(X_test), axis=1)
            predicted_class_names = encoder.inverse_transform(predictions2)

            logger.info("Predict2" + str(predicted_class_names))

            # Log accuracy and loss as metrics
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('loss', loss)

            current_datetime = datetime.now()

            # Format as a string
            formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")

            # Save the model after training
            model.save(f"outputs/{args.experiment_name}_round{round_num}_{formatted_datetime}_model.h5")

            # Construct the model path based on the relative path within the outputs directory
            model_path_relative = f"outputs/{args.experiment_name}_round{round_num}_{formatted_datetime}_model.h5"

            # Log the saved model as an artifact
            run.upload_file(name=model_path_relative, path_or_stream=model_path_relative)

            exp_name = f"{args.experiment_name}_round{round_num}" + "-" + "acc-"+"{:.2f}".format(accuracy).split(".")[1]

            logger.info("RUN-ID: " + run.id)
        
            accuracy_publish_threshold = 0.6
                    # Deploy only if accuracy is greater than the threshold
            if accuracy > accuracy_publish_threshold:

                model_name = exp_name + "-" + formatted_datetime
                # Register the model in Azure ML
                azure_model = Model.register(
                    workspace=ws,
                    model_name=model_name,
                    model_path=model_path_relative,
                    tags={"run_id": run.id, "experiment_name": exp_name, "accuracy": accuracy},
                    description=f"{exp_name} registered from Flower training, Accuracy: {accuracy}"
                )
            
            mlflow.end_run()
            return loss, len(X_test), {"accuracy": accuracy}
    # Start the FL client
    fl.client.start_numpy_client(server_address="0.0.0.0:8008", client=HARClient())

if __name__ == "__main__":
    start_fl_client()


