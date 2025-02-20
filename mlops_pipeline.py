import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib

# Set random seed for reproducibility
RANDOM_SEED = 42

# Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets with a 75-25 ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_SEED)

# Define hyperparameter grid with at least three parameters
param_grid = {
    'C': [0.1, 1, 10],                    # Regularization parameter
    'gamma': ['scale', 'auto'],           # Kernel coefficient
    'kernel': ['linear', 'rbf', 'poly']   # Kernel type
}

# Set up MLflow experiment
mlflow.set_experiment("Breast Cancer SVM Hyperparameter Tuning")

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=SVC(random_state=RANDOM_SEED),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,      # 5-fold cross-validation
    n_jobs=-1  # Use all available cores
)

# Perform grid search within an MLflow run
with mlflow.start_run():
    # Fit the grid search
    grid_search.fit(X_train, y_train)

    # Get the best estimator
    best_model = grid_search.best_estimator_

    # Predict on the test set
    y_pred = best_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.sklearn.log_model(best_model, artifact_path="model")

# Print the best parameters and accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy: {:.4f}".format(accuracy))

# Serialize the best model using joblib
joblib.dump(best_model, "svm_model.joblib")
print("Best model saved as svm_model.joblib")

# Load the model and make sample predictions to verify
loaded_model = joblib.load("svm_model.joblib")
sample_data = X_test[:5]
sample_predictions = loaded_model.predict(sample_data)
print("Sample Predictions:", sample_predictions)
print("Actual Labels:", y_test[:5])
