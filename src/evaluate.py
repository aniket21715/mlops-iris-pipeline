import pandas as pd
from sklearn.datasets import load_iris
import joblib

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Load the trained model from the file
try:
    model = joblib.load('iris_model.joblib')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file 'iris_model.joblib' not found. Please run train.py first.")
    exit()

# Evaluate the model
accuracy = model.score(X, y)

print(f"Model Accuracy: {accuracy:.4f}")

# Save the score to a file for the CML report
with open("evaluation_report.txt", "w") as f:
    f.write(f"## Model Evaluation Report\n\n")
    f.write(f"*   **Accuracy:** {accuracy:.4f}\n")
