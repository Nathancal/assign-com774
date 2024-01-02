# run_client.py
import argparse
from azureml.core import Workspace, Environment, ScriptRunConfig, Experiment
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from concurrent.futures import ThreadPoolExecutor


parser = argparse.ArgumentParser()
parser.add_argument("--total_subjects", type=int, required=True, help='Subject number')
args = parser.parse_args()

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
    # Specify the name of the dataset
    dataset_name = f"subject{subject_num + 1}"

    data_asset = ml_client.data._get_latest_version(dataset_name)

    # Create a unique experiment name with timestamp
    experiment_name = f"client_experiment_{subject_num + 1}"

    # Check if the experiment already exists
    if experiment_name not in ws.experiments:
        # If not, create a new experiment
        experiment = Experiment(workspace=ws, name=experiment_name)
    else:
        # If it exists, get the existing experiment
        experiment = ws.experiments[experiment_name]

    # Define a ScriptRunConfig
    script_config = ScriptRunConfig(source_directory=".",
                                    script="client.py",
                                    compute_target="your_compute_name",  # Specify your compute target
                                    environment=environment,
                                    arguments=["--data", data_asset.path])

    # Submit the job
    run = experiment.submit(script_config, tags={"Subject": subject_num + 1})
    print(f"Job for subject {subject_num + 1} submitted.")

# Submit jobs in parallel
with ThreadPoolExecutor() as executor:
    executor.map(submit_job, range(args.total_subjects))