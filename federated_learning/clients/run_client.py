import argparse
from azureml.core import Workspace, Environment, ScriptRunConfig, Experiment, Run
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from concurrent.futures import ProcessPoolExecutor
import logging
import mlflow
import time


# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
mlflow.set_tracking_uri("http://81.108.194.8:5000/")


parser = argparse.ArgumentParser()
parser.add_argument("--total_subjects", type=int, required=True, help='Subject number')
args = parser.parse_args()

# Get the current run
run = Run.get_context()

# Get the job name from the run's properties
job_name = run.get_metrics().get("AzureML.JobName")

subject_num = args.total_subjects

# Replace with your actual values
tenant_id = "6f0b9487-4fa8-42a8-aeb4-bf2e2c22d4e8"
client_id = "1bee10b2-17dd-4a50-b8aa-488d27bdd5a1"
client_secret = "MZK8Q~M5oNATdagyRKMUs-V-2dNggq3aAlRRdb8W"
subscription_id = "092da66a-c312-4a87-8859-56031bb22656"

credentials = ClientSecretCredential(tenant_id=tenant_id, client_id=client_id, client_secret=client_secret)

ml_client = MLClient.from_config(credential=DefaultAzureCredential(credentials=credentials))

# Load your Azure ML workspace
ws = Workspace.from_config()

# Get the latest version of the Azure ML environment
environment = Environment.get(workspace=ws, name="development")

# Function to submit a job
def submit_job(subject_num):
    try:
        
        # Specify the name of the dataset
        dataset_name = f"subject{subject_num + 1}"

        data_asset = ml_client.data._get_latest_version(dataset_name)

        # Create a unique experiment name with timestamp
        experiment_name = f"client_experiment_{subject_num + 1}"

        # Check if the experiment already exists
        if experiment_name not in ws.experiments:
            # If not, create a new experiment
            experiment = Experiment(workspace=ws, name=experiment_name)
            logger.info(f"Experiment {experiment_name} does not exist, will be created now.")
        else:
            # If it exists, get the existing experiment
            experiment = ws.experiments[experiment_name]
            logger.info(f"Experiment {experiment_name} already exists, job being added there for client {data_asset}")

        # Define a ScriptRunConfig
        script_config = ScriptRunConfig(source_directory=".",
                                        script="client.py",
                                        compute_target="compute-resources",  # Specify your compute target
                                        environment=environment,
                                        arguments=["--data", data_asset.path, "--experiment_name", experiment_name])
        
        with mlflow.start_run(experiment_name="Fed-Learning-Client-Staging-Env"):
            # Log parameters to MLflow
            mlflow.log_param("subject_num", subject_num + 1)
            mlflow.log_param("experiment_name", experiment_name)

            # Submit the job
            run = experiment.submit(script_config, tags={"Subject": subject_num + 1})
            logger.info(f"Job for subject {subject_num + 1} submitted.")
            
            # Log run ID and experiment ID to MLflow
            mlflow.log_param("run_id", run.id)
            mlflow.log_param("experiment_id", experiment.id)

    except Exception as e:
        logger.error(f"Error submitting job for subject {subject_num + 1}: {str(e)}")
        # Log exception to MLflow
        mlflow.log_param("error_message", str(e))

# Submit jobs in parallel
with ProcessPoolExecutor() as executor:
    executor.map(submit_job, range(args.total_subjects))
    # End MLflow run
    mlflow.end_run()
    
