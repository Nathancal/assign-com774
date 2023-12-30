from client import client
import argparse
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import pandas as pd
from azureml.core.image import ContainerImage
from azureml.core import Workspace, Environment

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# Load your Azure ML workspace
ws = Workspace.from_config()

# Get the arugments we need to avoid fixing the dataset path in code
parser = argparse.ArgumentParser()
parser.add_argument("--totalsubjects", type=int, required=True, help='Total number of subjects in model.')
args = parser.parse_args()

# Get the latest version of the Azure ML environment
environment = Environment.get(workspace=ws, name="development")


for subject_num in range(args.totalsubjects):

    # Specify the name of the dataset
    dataset_name = "subject"+ str(subject_num + 1)

    data_asset = ml_client.data._get_latest_version(dataset_name)

    individual_df = pd.read_csv(data_asset.path)

      # Build and register Docker image for client
    client_image_config = ContainerImage.image_configuration(execution_script="client.py",
                                                             runtime="python",
                                                             environment=environment,
                                                             dependencies=["../utils/utils.py"],
                                                             description=f"Federated Learning Client {subject_num + 1}")
    
    client_image = ContainerImage.create(name=f"fl-client-image-{subject_num + 1}",
                                         models=[],
                                         image_config=client_image_config,
                                         workspace=ws)

    # Deploy the client image as a web service
    client_service = client_image.deploy(workspace=ws, name=f"fl-client-service-{subject_num + 1}")
