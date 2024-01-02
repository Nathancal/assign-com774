
import logging
from azureml.core import Workspace, Model, Environment
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
import mlflow
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to your Azure ML workspace
ws = Workspace.from_config()

environment = Environment.get(workspace=ws, name="testing")


def deploy_azure_model(model_name, model_path, accuracy_threshold=0.8):
    try:
        # Get the registered model
        model = Model.register(workspace=ws, model_path=model_path, model_name=model_name)

        # Retrieve accuracy from MLflow
        run = mlflow.get_run()
        accuracy = run.data.metrics.get("accuracy", 0.0)

        # Deploy only if accuracy is greater than the threshold
        if accuracy > accuracy_threshold:
            # Define inference configuration
            inference_config = InferenceConfig(entry_script="score.py", runtime="python", conda_file=environment)

            # Deploy the model as a web service
            aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
            service_name = f"{model_name.lower()}-{formatted_datetime}-service"
            service = Model.deploy(workspace=ws,
                                   name=service_name,
                                   models=[model],
                                   inference_config=inference_config,
                                   deployment_config=aciconfig)
            service.wait_for_deployment(show_output=True)

            # Log deployment details to MLflow
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("service_name", service_name)
            mlflow.log_param("service_url", service.scoring_uri)

            logger.info(f"Model {model_name} deployed successfully.")

        else:
            logger.info(f"Model {model_name} skipped deployment as accuracy is below the threshold.")

    except Exception as e:
        logger.error(f"Error deploying model {model_name}: {str(e)}")
        # Log exception to MLflow
        mlflow.log_param("error_message", str(e))

# Get the current date and time as a formatted string
formatted_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
