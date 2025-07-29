import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# set experiment name
mlflow.set_experiment("iris classification")

# Load dataset
df = pd.read_csv("data/iris.csv")
X = df.drop("species", axis=1)
y = df["species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model definition
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Start MLflow run
with mlflow.start_run():

    # Train
    clf.fit(X_train, y_train)

    # Predict
    preds = clf.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)

    # Save model
    os.makedirs("src", exist_ok=True)
    joblib.dump(clf, "src/model.pkl")

    # Log model
    mlflow.sklearn.log_model(clf, "model")
