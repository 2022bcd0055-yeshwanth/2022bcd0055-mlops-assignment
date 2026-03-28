import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import joblib
import sys

# args: dataset, model_type, feature_mode
dataset = sys.argv[1]       # v1 or v2
model_type = sys.argv[2]    # rf or lr
feature_mode = sys.argv[3]  # full or reduced

mlflow.set_experiment("2022bcd0056_experiment")

# Load data
df = pd.read_csv(f"data/diabetes_{dataset}.csv")

# Target column (adjust if needed)
target = "diabetes"

# Feature selection
if feature_mode == "reduced":
    features = df.columns[:5]  # take first few features
    X = df[features]
else:
    X = df.drop(columns=[target])

y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model selection
if model_type == "rf":
    model = RandomForestClassifier(n_estimators=50)
else:
    model = LogisticRegression(max_iter=200)

# MLflow logging
with mlflow.start_run():

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.log_param("dataset", dataset)
    mlflow.log_param("model", model_type)
    mlflow.log_param("features", feature_mode)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    joblib.dump(model, "app/model.pkl")
    mlflow.log_artifact("app/model.pkl")

    print(f"Accuracy: {acc}, F1: {f1}")