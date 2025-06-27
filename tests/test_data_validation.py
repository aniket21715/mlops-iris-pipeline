import pytest
from sklearn.datasets import load_iris

def test_iris_dataset_has_expected_shape():
    """
    Unit Test for Data Validation:
    Checks if the Iris dataset features have the shape (150, 4).
    """
    iris = load_iris()
    X = iris.data
    assert X.shape == (150, 4), "Data validation failed: Expected shape (150, 4)"

def test_iris_dataset_target_is_not_empty():
    """
    Unit Test for Data Validation:
    Checks if the Iris dataset labels (target) are not empty.
    """
    iris = load_iris()
    y = iris.target
    assert y.size > 0, "Data validation failed: Target array is empty"
