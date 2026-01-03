import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Paths
X_TRAIN_PATH = "data/processed/X_train.csv"
X_TEST_PATH = "data/processed/X_test.csv"
Y_TRAIN_PATH = "data/processed/y_train.csv"
Y_TEST_PATH = "data/processed/y_test.csv"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_model.pkl")


def load_data():
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()
    y_test = pd.read_csv(Y_TEST_PATH).values.ravel()
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
    }


def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)


def main():
    mlflow.set_experiment("Diabetes_Detection")

    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run():
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

        # Log model
        save_model(model)
        mlflow.log_artifact(MODEL_PATH)

        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)

        print("Training completed successfully.")
        print(metrics)


if __name__ == "__main__":
    main()

