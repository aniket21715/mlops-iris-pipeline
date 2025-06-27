import pytest
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# This test requires the model to be trained first.
# Our GitHub Action will run train.py before running this test.

def test_model_can_be_loaded():
    """
    Unit Test for Model Evaluation:
    Checks if the 'iris_model.joblib' file can be loaded
    and if it's a valid scikit-learn model.
    """
    try:
        model = joblib.load('iris_model.joblib')
    except FileNotFoundError:
        pytest.fail("Evaluation test failed: 'iris_model.joblib' not found. Training step might have failed.")
    
    assert isinstance(model, LogisticRegression), "Loaded object is not a LogisticRegression model"

def test_model_prediction_output():
    """
    Unit Test for Model Evaluation:
    Checks if the model's predict() method returns an output of the correct shape.
    """
    model = joblib.load('iris_model.joblib')
    iris = load_iris()
    X_sample = iris.data[0].reshape(1, -1) # Get one sample
    
    prediction = model.predict(X_sample)
    assert prediction.shape == (1,), "Model prediction output shape is incorrect"
