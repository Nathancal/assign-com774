import argparse
import mlflow
from azureml.core import Workspace, Experiment, Run, Environment
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from concurrent.futures import ProcessPoolExecutor
import logging
from azureml.core.authentication import ServicePrincipalAuthentication
import psutil
import mlflow.azureml

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--total_subjects", type=int, required=True, help='Subject number')
args = parser.parse_args()

subject_num = args.total_subjects

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

# If using service principal authentication, set the credentials
mlflow.azureml.init(
    workspace_name=workspace_name,
    subscription_id=subscription_id,
    resource_group=resource_group,
    location=mlflow_location,
    client_id=client_id,
    client_secret=client_secret,
    tenant_id=tenant_id,
)

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
     

environment = Environment.get(workspace=ws, name="development")
# Get the environment name and version
environment_name = environment.name
environment_version = str(environment.version)

run = Run.get_context()

# Function to submit a job
def submit_job(subject_num):

    with mlflow.start_run():
        mlflow.log_param("total_subjects", args.total_subjects)

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
            
            # inputs = {
            #     "input_data": Input(type=AssetTypes.URI_FILE, path=data_asset.path), 
            #     "experiment_name": experiment.name
            # }

            # # Define your job with the correct environment name and version
            # job = command(
            #     code="./",  # local path where the code is stored
            #     command="python client.py --data ${{inputs.input_data}} --experiment_name ${{inputs.experiment_name}}",
            #     inputs=inputs,
            #     environment=f"azureml:{environment_name}:{environment_version}",
            #     compute="compute-resources",
            #     identity=svc_pr,
            #     experiment_name=experiment_name,  # Pass the experiment name to your job
            # )

            mlflow.azureml.run(
                script_name="client.py",
                arguments=[
                    "--data", f"azureml://{data_asset.name}/{data_asset.version}",
                    "--experiment_name", experiment_name
                ],
                backend="azureml",
                backend_config={
                    "compute": "compute-resources",
                    "environment": f"azureml://{environment_name}:{environment_version}",
                },
            )

               # Log memory usage
            memory_usage = psutil.virtual_memory().percent
            mlflow.log_metric("memory_usage_percent", memory_usage)

            # Log CPU usage
            cpu_usage = psutil.cpu_percent()
            mlflow.log_metric("cpu_usage_percent", cpu_usage)
            # # Start the Azure ML run
            # run = experiment.submit(script_config, tags={"Subject": subject_num + 1})
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Job for subject {subject_num + 1} submitted. Run ID: {run_id}")
            # submit the command

            mlflow.log_param("subject_num", subject_num + 1)
            mlflow.log_param("experiment_name", experiment_name)
            mlflow.log_param("run_id", run_id)
            mlflow.log_param("experiment_id", experiment.id)

            # Wait for the run to complete
            mlflow.wait_for_completion()

            # Log additional metrics
            mlflow.log_metric("run_duration", run.get_metrics().get("DurationInSeconds"))

    
            logger.info(f"Job for subject {subject_num + 1} completed. Run ID: {run_id}")

        except Exception as e:
            logger.error(f"Error submitting job for subject {subject_num + 1}: {str(e)}")
            # Log exception to Azure ML
            mlflow.log_param("error_message", str(e))

# Submit jobs in parallel
with ProcessPoolExecutor() as executor:
    executor.map(submit_job, range(args.total_subjects))

# End Azure ML run
run.complete()