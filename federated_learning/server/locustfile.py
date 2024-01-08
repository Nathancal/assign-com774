from locust import HttpUser, task, between
import pandas as pd
import utils
from sklearn.model_selection import train_test_split

# Load CSV data using pandas
csv_file_path = 'Sub10_imran.csv'
df = pd.read_csv(csv_file_path)

X, Y = utils.load_har_data(csv_file_path)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)


X_test_list = X_test.tolist()
class MyUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def make_prediction(self):
        data = {"input_data": X_test_list}
        api_key = 'aCFNuuGlEI60uk4ISzOvgdQ8UZMFVaWW'
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'client-experiment-4' }
        self.client.post("https://client-experiment-4.westeurope.inference.ml.azure.com/score", json=data, headers=headers)

# Configuring the number of users and spawn rate
class MyUser(HttpUser):
    wait_time = between(1, 3)
    host = "https://client-experiment-4.westeurope.inference.ml.azure.com/score"  # Replace with your actual API host

    # Number of users
    min_wait = 5000
    max_wait = 15000

    @task
    def make_prediction(self):
        data = {"input_data": X_test_list}
        api_key = 'aCFNuuGlEI60uk4ISzOvgdQ8UZMFVaWW'
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'client-experiment-4' }
        self.client.post("https://client-experiment-4.westeurope.inference.ml.azure.com/score", json=data, headers=headers)
