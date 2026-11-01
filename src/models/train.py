"""Entrenamiento: entrena varios modelos y registra cada uno en MLflow."""

import logging

import mlflow
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from config.config import (
    EXPERIMENT_NAME,
    FEATURES,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


def get_models() -> dict:
    """Devuelve los modelos a comparar (los mismos del notebook)."""
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def load_and_split():
    """Lee los datos limpios y los divide en entrenamiento y prueba."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def train_and_log(name, model, X_train, X_test, y_train, y_test) -> None:
    """Entrena un modelo y guarda en MLflow sus parámetros, métricas y el modelo."""
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "rmse": root_mean_squared_error(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            "r2": r2_score(y_test, preds),
        }

        mlflow.set_tag("model_type", name)
        mlflow.log_params(model.get_params())
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_train.head(5),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        mlflow.set_tag("model_uri", model_info.model_uri)

    logger.info("%s -> RMSE: %.2f | MAE: %.2f | R2: %.3f",
                name, metrics["rmse"], metrics["mae"], metrics["r2"])


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_and_split()
    for name, model in get_models().items():
        train_and_log(name, model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()