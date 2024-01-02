import flwr as fl
import numpy as np
import utils
from sklearn.model_selection import train_test_split
import os
import logging
import argparse
import mlflow
from keras.models import save_model
from deploy_model import deploy_azure_model
import datetime
from azureml.core import  Model, Workspace

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load your Azure ML workspace
ws = Workspace.from_config()


def client():
    try:
        # Define command-line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--data", type=str, required=True, help='Path to the dataset')
        args = parser.parse_args()
        logger.info(f"Client Started..")

        logger.info(f"client data: {args.data}")

        # Create an LSTM model
        model = utils.create_lstm_model()

        if args.data is None:
            raise ValueError("Invalid file path: 'data' environment variable is not set.")

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
                current_datetime = datetime.now()

                # Format as a string
                formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
                                
                # Save the model after training
                save_model(model, f"client_model_subject{config['subject_id']}_{formatted_datetime}.h5")
                # After saving the individual models
                mlflow.log_artifact(f"client_model_subject{config['subject_id']}_{formatted_datetime}.h5")
                model_path = mlflow.get_artifact_uri(f"client_model_subject{config['subject_id']}_{formatted_datetime}.h5")
                mlflow.register_model(model_path, f"client_model_subject{config['subject_id']}_{formatted_datetime}")

                model = Model.register(workspace=ws, model_path=model_path, model_name=f"client_model_subject{config['subject_id']}_{formatted_datetime}")

                # Deploy the model to Azure
                deploy_azure_model(f"client_model_subject{config['subject_id']}_{formatted_datetime}", model_path)


                logger.info(f"Training finished for round {config['server_round']}")
                return utils.get_lstm_model_parameters(model), len(X_train), {}

            def evaluate(self, parameters, config):
                utils.set_lstm_model_params(model, parameters)
                loss, accuracy = model.evaluate(X_test, Y_test)
                logger.info(f"Evaluation completed. Loss: {loss}, Accuracy: {accuracy}")
                return loss, len(X_test), {"accuracy": accuracy}

        # ... (rest of the code)

    except Exception as e:
        logger.error(f"Error in client script: {str(e)}")
        # Log exception to MLflow
        mlflow.log_param("error_message", str(e))

if __name__ == "__main__":
    # Start MLflow run
    mlflow.start_run()
    client()
    # End MLflow run
    mlflow.end_run()