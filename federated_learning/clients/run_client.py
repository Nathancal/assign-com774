import argparse
from azureml.core import Workspace, Experiment, Run
from azure.ai.ml.entities._assets.environment import Environment
from azure.ai.ml import Input
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from concurrent.futures import ProcessPoolExecutor
import logging
from azure.ai.ml import command
from azure.ai.ml import UserIdentityConfiguration
from azureml.core.authentication import ServicePrincipalAuthentication

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--total_subjects", type=int, required=True, help='Subject number')
args = parser.parse_args()

subject_num = args.total_subjects

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

# Connect to Azure ML workspace
ws = Workspace.from_config(auth=svc_pr, path='./config.json')


ml_client = MLClient.from_config(credential=DefaultAzureCredential())
     

environment = Environment.get(workspace=ws, name="development")


# Get the current run context in an Azure ML job
run = Run.get_context()
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
            logger.info(f"Experiment {experiment_name} already exists.., job being added there for client {data_asset}")
        
        inputs = {
            "input_data": Input(type=AssetTypes.URI_FILE, path=data_asset.path), 
            "experiment_name": experiment.name
        }


        job = command(
            code="./src",  # local path where the code is stored
            command="python client.py --data ${{inputs.input_data}} --experiment_name ${{inputs.experiment_name}}",
            inputs=inputs,
            environment="azureml:development:12",
            compute="compute-resources",
            identity=UserIdentityConfiguration(),
        )
        # # Start the Azure ML run
        # run = experiment.submit(script_config, tags={"Subject": subject_num + 1})
        run_id = run.id
        logger.info(f"Job for subject {subject_num + 1} submitted. Run ID: {run_id}")
        # submit the command
        returned_job = ml_client.jobs.create_or_update(job)
        # get a URL for the status of the job
        run.log("Studio_rul", returned_job.studio_url)
        # Log parameters to Azure ML
        run.log("subject_num", subject_num + 1)
        run.log("experiment_name", experiment_name)

        # Log run ID and experiment ID to Azure ML
        run.log("run_id", run.id)
        run.log("experiment_id", experiment.id)

        # Wait for the run to complete
        run.wait_for_completion()

        # Log additional metrics
        run.log("run_duration", run.get_metrics().get("DurationInSeconds"))

   
        logger.info(f"Job for subject {subject_num + 1} completed. Run ID: {run_id}")

    except Exception as e:
        logger.error(f"Error submitting job for subject {subject_num + 1}: {str(e)}")
        # Log exception to Azure ML
        run.log("error_message", str(e))

# Submit jobs in parallel
with ProcessPoolExecutor() as executor:
    executor.map(submit_job, range(args.total_subjects))

# End Azure ML run
run.complete()