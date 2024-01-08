import os
import numpy as np
from keras.models import load_model
from azureml.core.model import Model
from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
import json
from sklearn.preprocessing import LabelEncoder


def init():
    global model
    
    model_name = os.getenv("MODEL_NAME")
    # Load the model from the Azure ML workspace
    model_path = Model.get_model_path(model_name=model_name)
    model = load_model(model_path)

@rawhttp
def run(request):
    try:
        data = json.loads(request.get_data())
        input_data = data["input_data"]
        # Your list of classes
        class_names = ['Running', 'Sitting', 'Standing', 'Walking', 'downstairs', 'upstairs']

        # Initialize the LabelEncoder
        encoder = LabelEncoder()

        # Fit the encoder to your classes and transform them to numerical labels
        encoded_labels = encoder.fit_transform(class_names)

        prediction = np.argmax(model.predict(input_data), axis=1)
        predicted_class_names = encoder.inverse_transform(prediction)


        # Convert the prediction to a string (you can modify this based on your model output)
        result = predicted_class_names

        # Return the result
        return AMLResponse(result, 200)

    except Exception as e:
        error_msg = f"Error during inference: {str(e)}"
        return AMLResponse(error_msg, 500)