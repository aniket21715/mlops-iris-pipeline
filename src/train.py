import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a DataFrame for better visualization (optional)
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

print("Training a model on the Iris dataset...")

# Initialize and train the model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

print("Model training complete.")

# Save the trained model to a file
joblib.dump(model, 'iris_model.joblib')

print("Model saved as iris_model.joblib")
