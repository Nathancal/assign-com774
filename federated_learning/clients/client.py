import flwr as fl
import numpy as np
import utils
from sklearn.model_selection import train_test_split
import os
import logging
from azureml.core.authentication import ServicePrincipalAuthentication
import argparse
import datetime
from azureml.core import Workspace, Model
from azureml.core.run import Run
from deploy_model import deploy_azure_model
import mlflow

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

# Get the current run context in an Azure ML job
run = Run.get_context()

def client():
    try:
        # Initialize MLflow
        mlflow_location = "westeurope"
        subscription_id = "092da66a-c312-4a87-8859-56031bb22656"
        mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
        mlflow.start_run()
        mlflow.log_param("total_subjects", args.total_subjects)

        # Define command-line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--data", type=str, required=True, help='Path to the dataset')
        parser.add_argument("--experiment_name", type=str, required=True, help='experiment name')
        args = parser.parse_args()

        # Create an LSTM model
        model = utils.create_lstm_model()

        if args.data is None:
            raise ValueError("Invalid file path: 'data' environment variable is not set....")

        # Load client data
        X, Y = utils.load_har_data(args.data)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)

        logger.info(f"Loaded client data. X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")

        # Set initial LSTM model parameters
        utils.set_initial_lstm_params(model)

        # Define the HARClient class before using it
        class HARClient(fl.client.NumPyClient):
            def get_parameters(self, config):
                return utils.get_lstm_model_parameters(model)

            def fit(self, parameters, config):
                utils.set_lstm_model_params(model, parameters)
                model.fit(X_train, Y_train, epochs=10, verbose=0)

                # Get the current date and time
                current_datetime = datetime.datetime.now()

                # Format as a string
                formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")

                # Save the model after training
                model_path = f"client_model_subject{config['subject_id']}_{formatted_datetime}.h5"
                model.save(model_path)

                # Log the model file as an artifact to Azure ML
                run.upload_file(model_path)

                # Register the model with Azure ML
                model = Model.register(workspace=ws, model_path=model_path, model_name=f"client_model_subject{config['subject_id']}_{formatted_datetime}")

                # Deploy the model to Azure
                # Note: Implement the deploy_azure_model function to deploy the model
                deploy_azure_model(f"client_model_subject{config['subject_id']}_{formatted_datetime}", model_path)

                logger.info(f"Training finished for round {config['server_round']}")
                return utils.get_lstm_model_parameters(model), len(X_train), {}

            def evaluate(self, parameters, config):
                utils.set_lstm_model_params(model, parameters)
                loss, accuracy = model.evaluate(X_test, Y_test)
                logger.info(f"Evaluation completed. Loss: {loss}, Accuracy: {accuracy}")
                return loss, len(X_test), {"accuracy": accuracy}

        # Connect to the server
        fl.client.start_client("40.68.31.180:8008", client=HARClient())

    except Exception as e:
        logger.error(f"Error in client script: {str(e)}")
        # Log exception to Azure ML
        run.log("error_message", str(e))

if __name__ == "__main__":
    client()