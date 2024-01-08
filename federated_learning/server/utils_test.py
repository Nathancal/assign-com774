import unittest
import numpy as np
import pandas as pd
from utils import load_har_data, create_lstm_model, set_initial_lstm_params, get_lstm_model_parameters, set_lstm_model_params

class TestUtils(unittest.TestCase):

    def setUp(self):
        # Add any setup code that needs to be run before each test
        pass

    def tearDown(self):
        # Add any cleanup code that needs to be run after each test
        pass

    def test_load_har_data(self):
        # Provide a test CSV file for load_har_data
        file_path = "test_data.csv"
        
        # Create a dummy CSV file for testing
        dummy_data = {"feature_1": [1, 2, 3], "feature_2": [4, 5, 6], "activity": ["Walking", "Sitting", "Running"]}
        pd.DataFrame(dummy_data).to_csv(file_path, index=False)

        # Test the function
        X, Y = load_har_data(file_path)

        self.assertEqual(X.shape, (3, 2))
        self.assertEqual(Y.shape, (3, 4))  # Assuming 6 classes

    def test_create_lstm_model(self):
        # Test the creation of an LSTM model
        model = create_lstm_model()

        # Add assertions based on your expectations
        self.assertIsNotNone(model)
        # Add more assertions as needed

    def test_set_initial_lstm_params(self):
        # Test setting initial LSTM parameters
        model = create_lstm_model()
        set_initial_lstm_params(model)

        # Add assertions based on your expectations
        # For example, check if the model parameters are set correctly
        self.assertEqual(model.layers[0].get_config()["units"], 45)
        # Add more assertions as needed

    def test_get_and_set_lstm_model_params(self):
        # Test getting and setting LSTM model parameters
        model = create_lstm_model()
        set_initial_lstm_params(model)
        original_params = get_lstm_model_parameters(model)

        # Modify some parameters (for testing purposes)
        modified_params = original_params.copy()
        modified_params[0] = np.zeros_like(modified_params[0])

        # Set modified parameters
        set_lstm_model_params(model, modified_params)

        # Get the modified parameters
        new_params = get_lstm_model_parameters(model)

        # Add assertions based on your expectations
        self.assertTrue(np.array_equal(new_params[0], np.zeros_like(original_params[0])))
        # Add more assertions as needed

if __name__ == '__main__':
    unittest.main()