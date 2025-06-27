import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

# Test 1: Data Validation
def test_iris_data_shape():
    """
    Tests that the loaded Iris dataset has the correct shape.
    150 samples, 4 features.
    """
    iris = load_iris()
    X = iris.data
    # Assert that there are 150 rows and 4 columns
    assert X.shape == (150, 4), "Iris dataset shape is not (150, 4)"

# Test 2: Evaluation (Sanity Check on Model Prediction)
def test_model_prediction():
    """
    This is a sanity check. It trains a dummy model on dummy data
    and ensures the predict function returns an output of the correct shape.
    """
    # Create a dummy model and data
    X_dummy = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]) # 2 samples, 4 features
    y_dummy = np.array([0, 1])
    
    model = LogisticRegression()
    model.fit(X_dummy, y_dummy)
    
    # Make a prediction on one sample
    prediction = model.predict(X_dummy[0].reshape(1, -1))
    
    # Assert that the prediction output is a single value (in an array)
    assert prediction.shape == (1,), "Model prediction output shape is incorrect"
