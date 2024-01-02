import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import mlflow

def load_har_data(file):
    mlflow.log_param("file_path", file)

    df = pd.read_csv(file)   
    df = df.dropna()

    # Extract features and labels
    X = df.drop('activity', axis=1).values
    Y = df['activity'].values
    mlflow.log_param("num_features", X.shape[1])

    # Encode activity labels
    encoder = LabelEncoder()
    encoder.fit(['Running', 'Sitting', 'Standing', 'Walking', 'downstaires', 'upstaires'])
    Y = encoder.transform(Y)
    Y = to_categorical(Y)  # One-hot encode the labels

    return X, Y

# Define the LSTM model
def create_lstm_model(num_classes=6, num_features=16):
    mlflow.log_param("num_classes", num_classes)

    model = Sequential()
    model.add(LSTM(45, activation="LeakyReLU", kernel_initializer="he_normal", input_shape=(num_features, 1)))
    model.add(Dense(90, activation="relu", kernel_initializer="he_normal", input_shape=(num_features, 1)))
    model.add(Dense(180, activation="softmax", kernel_initializer="he_normal", input_shape=(num_features, 1)))
    model.add(Dense(360, activation="relu", kernel_initializer="he_normal", input_shape=(num_features,1 )))
    model.add(Dense(180, activation="swish", kernel_initializer="he_normal", input_shape=(num_features,1)))
    model.add(Dense(num_classes, activation='softmax'))  # Assuming 6 classes for classification

    return model

def set_initial_lstm_params(model: Sequential):
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But the server asks for initial parameters from clients at launch. Refer to
    tensorflow.keras.models.Sequential documentation for more information.
    """
    n_features = 16  # Number of features in dataset

    model.build((None, 128, n_features))  # Input shape for LSTM
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.reset_states()

def get_lstm_model_parameters(model: Sequential):
    """Returns the parameters of an LSTM model."""
    return model.get_weights()

def set_lstm_model_params(model: Sequential, params) -> Sequential:
    """Set the parameters of an LSTM model."""
    model.set_weights(params)
    return model

def log_metrics(loss, accuracy):
    mlflow.log_metric("loss", loss)
    mlflow.log_metric("accuracy", accuracy)

def log_model(model, artifact_path="model"):
    mlflow.keras.log_model(model, artifact_path)

def log_artifacts(local_path, artifact_path="artifacts"):
    mlflow.log_artifacts(local_path, artifact_path)