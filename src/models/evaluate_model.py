"""Evaluación: aplica el quality gate a la última versión y, si lo pasa, la nombra champion."""

import logging

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from config.config import (
    FEATURES,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_NAME,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RMSE_THRESHOLD,
    TARGET,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


def get_test_data():
    """Recupera el mismo conjunto de prueba que usó el entrenamiento."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    _, X_test, _, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_test, y_test


def get_latest_version(client: MlflowClient) -> str:
    """Devuelve el número de la versión más reciente del modelo registrado."""
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"No hay versiones registradas de '{MODEL_NAME}'.")
    return str(max(int(v.version) for v in versions))


def get_champion_rmse(client: MlflowClient):
    """Devuelve el RMSE del champion actual, o None si todavía no hay champion."""
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    except MlflowException:
        return None
    return float(champion.tags["rmse"])


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    version = get_latest_version(client)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")

    X_test, y_test = get_test_data()
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    client.set_model_version_tag(MODEL_NAME, version, "rmse", f"{rmse:.4f}")
    client.set_model_version_tag(MODEL_NAME, version, "r2", f"{r2:.4f}")
    logger.info("Versión %s -> RMSE: %.2f | R2: %.3f", version, rmse, r2)

    if rmse >= RMSE_THRESHOLD:
        client.set_model_version_tag(MODEL_NAME, version, "quality_gate", "rechazado")
        logger.warning("Quality gate RECHAZADO: RMSE %.2f >= umbral %.2f", rmse, RMSE_THRESHOLD)
        return

    client.set_model_version_tag(MODEL_NAME, version, "quality_gate", "aprobado")
    champion_rmse = get_champion_rmse(client)
    if champion_rmse is not None and rmse > champion_rmse:
        logger.info("Aprobado, pero no mejora al champion actual (RMSE %.2f). Se mantiene el champion.", champion_rmse)
        return

    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)
    logger.info("Quality gate APROBADO: versión %s es ahora '%s'", version, MODEL_ALIAS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()