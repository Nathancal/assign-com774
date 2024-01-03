import logging
from azureml.core import Workspace, Model, Environment, ScriptRunConfig
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from datetime import datetime
from azureml.core.run import Run
from azureml.core.authentication import ServicePrincipalAuthentication


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Service principal authentication configuration
svc_pr_password = "MZK8Q~M5oNATdagyRKMUs-V-2dNggq3aAlRRdb8W"
svc_pr = ServicePrincipalAuthentication(
    tenant_id="6f0b9487-4fa8-42a8-aeb4-bf2e2c22d4e8",
    service_principal_id="9da84d5d-c745-4ddc-bb1b-ff3574f5b530",
    service_principal_password=svc_pr_password
)
# Connect to your Azure ML workspace
ws = Workspace.from_config(auth=svc_pr, path='./config.json')

environment = Environment.get(workspace=ws, name="testing")

# Start Azure ML run
run = Run.get_context()

def deploy_azure_model(model_name, model_path, accuracy_threshold=0.8):
    try:
        # Get the registered model
        model = Model.register(workspace=ws, model_path=model_path, model_name=model_name)

        # Retrieve accuracy from Azure ML
        accuracy = run.get_metrics().get("accuracy", 0.0)

        # Deploy only if accuracy is greater than the threshold
        if accuracy > accuracy_threshold:
            # Define inference configuration
            inference_config = InferenceConfig(entry_script="score.py", runtime="python", conda_file=environment)

            # Deploy the model as a web service
            aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
            service_name = f"{model_name.lower()}-{formatted_datetime}-service"

            # Pass a parameter to the entry script using the arguments parameter
            inference_config.arguments = ["--modelname", model_name]

            # Create a ScriptRunConfig
            script_run_config = ScriptRunConfig(source_directory=".",
                                               script="score.py",
                                               arguments=inference_config.arguments,  # Pass arguments here
                                               compute_target="compute-instance",  # Specify your compute target
                                               environment=inference_config.environment)

            service = Model.deploy(workspace=ws,
                                   name=service_name,
                                   models=[model],
                                   inference_config=inference_config,
                                   deployment_config=aciconfig)
            service.wait_for_deployment(show_output=True)

            # Log deployment details to Azure ML
            run.log("model_name", model_name)
            run.log("service_name", service_name)
            run.log("service_url", service.scoring_uri)

            logger.info(f"Model {model_name} deployed successfully.")

        else:
            logger.info(f"Model {model_name} skipped deployment as accuracy is below the threshold.")

    except Exception as e:
        logger.error(f"Error deploying model {model_name}: {str(e)}")
        # Log exception to Azure ML
        run.log("error_message", str(e))

# Get the current date and time as a formatted string
formatted_datetime = datetime.now().strftime("%Y%m%d%H%M%S")