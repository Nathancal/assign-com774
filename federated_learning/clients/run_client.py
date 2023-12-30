# run_client.py
import argparse
from azureml.core import Workspace, Run, Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


parser = argparse.ArgumentParser()
parser.add_argument("--total_subjects", type=int, required=True, help='Subject number')
args = parser.parse_args()

subject_num = args.subject_num

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# Load your Azure ML workspace
ws = Workspace.from_config()

# Get the latest version of the Azure ML environment
environment = Environment.get(workspace=ws, name="development")

# Create an InferenceConfig
inference_config = InferenceConfig(entry_script="client.py",
                                   environment=environment)

# Deploy multiple clients
for subject_num in range(args.total_subjects):
    # Specify the name of the dataset
    dataset_name = f"subject{subject_num + 1}"

    data_asset = ml_client.data._get_latest_version(dataset_name)

    # Deploy the client as an Azure AI job
    service_name = f"fl-client-service-subject-{subject_num + 1}"
    
     # Pass dataset path as an environment variable
    deployment_config = AciWebservice.deploy_configuration(environment_variables={"data": data_asset.path})

    deployed_service = Model.deploy(workspace=ws,
                                    name=service_name,
                                    models=[],
                                    inference_config=inference_config,
                                    deployment_config=deployment_config,
                                    overwrite=True)

    deployed_service.wait_for_deployment(show_output=True)