import pandas as pd
import joblib
import os
import json
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



path = os.path.join(os.getcwd(), "some-folder")

# Load hyperparameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

n_estimators = params["n_estimators"]
test_size = params["test_size"]
random_state = params["random_state"]

# ✅ Set MLflow experiment and tracking URI
mlflow.set_tracking_uri("file:./mlruns")  # prevents CI/CD errors
mlflow.set_experiment("iris_classification")

# Load data
df = pd.read_csv("data/iris.csv")
X = df.drop("species", axis=1)
y = df["species"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# Train
clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

print("Tracking URI:", mlflow.get_tracking_uri())

with mlflow.start_run():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("test_size", test_size)
    mlflow.log_metric("accuracy", acc)

    os.makedirs("src", exist_ok=True)
    joblib.dump(clf, "src/model.pkl")

    with open("metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)

    mlflow.sklearn.log_model(clf, "model")
