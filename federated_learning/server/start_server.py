import argparse
import mlflow
from azureml.core import Workspace, Experiment, Run, Environment
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from concurrent.futures import ProcessPoolExecutor
import logging
from azureml.core.authentication import ServicePrincipalAuthentication
import psutil
import os
# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
    

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

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
     

environment = Environment.get(workspace=ws, name="development")
# Get the environment name and version
environment_name = environment.name
environment_version = str(environment.version)

run = Run.get_context()

def start_server():


        try:
                # Specify the name of the dataset
            dataset_name = f"subject2-4"
            # Check if the dataset follows the naming convention
            if dataset_name.startswith("subject"):
                data_asset = ml_client.data._get_latest_version(dataset_name)

                # Create a unique experiment name with timestamp
                experiment_name = f"Fed-Learning-Server-Staging-Env"

                # Check if the experiment already exists
                if experiment_name not in ws.experiments:
                    # If not, create a new experiment
                    experiment = Experiment(workspace=ws, name=experiment_name)
                    logger.info(f"Experiment {experiment_name} does not exist, will be created now.")
                else:
                    # If it exists, get the existing experiment
                    experiment = ws.experiments[experiment_name]
                    logger.info(f"Experiment {experiment_name} already exists, job being added there for client {data_asset}")

                    inputs = {
                        "input_data": Input(type=AssetTypes.URI_FILE, path=data_asset.path),
                        "experiment_name": experiment.name
                    }

                    # Define your job with the correct environment name and version
                    job = command(
                        code="./",  # local path where the code is stored
                        command="python server.py --training_data ${{inputs.input_data}} --minimum_clients 3",
                        inputs=inputs,
                        environment=f"azureml:{environment_name}:{environment_version}",
                        compute="job-run-compute",
                        experiment_name=experiment_name,  # Pass the experiment name to your job
                    )

                    # Assuming ml_client is your MLClient instance
                    returned_job = ml_client.jobs.create_or_update(job)
                    # Log memory usage
                    memory_usage = psutil.virtual_memory().percent
                    mlflow.log_metric("memory_usage_percent", memory_usage)

                    # Log CPU usage
                    cpu_usage = psutil.cpu_percent()
                    mlflow.log_metric("cpu_usage_percent", cpu_usage)

                    # Wait for the job to complete
                    returned_job.wait_for_completion()
        
                #logger.info(f"Job for subject {subject_num + 1} completed. Run ID: {run_id}")

        except Exception as e:
            logger.error(f"Error submitting job for subject {subject_num + 1}: {str(e)}")
                # Log exception to Azure ML
            mlflow.log_param("error_message", str(e))

# Start Flower server for 10 rounds of federated learning
if __name__ == "__main__":
    start_server()