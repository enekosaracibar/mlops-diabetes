"""Registro: busca el mejor run del experimento y lo registra en el Model Registry."""

import logging

import mlflow
from mlflow.tracking import MlflowClient

from config.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_NAME

logger = logging.getLogger(__name__)


def get_best_run(client: MlflowClient):
    """Devuelve el run terminado con menor RMSE del experimento."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"No existe el experimento '{EXPERIMENT_NAME}'. Ejecuta antes el entrenamiento.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.rmse ASC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No hay runs terminados en el experimento.")
    return runs[0]


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    best_run = get_best_run(client)
    run_id = best_run.info.run_id
    rmse = best_run.data.metrics["rmse"]
    model_type = best_run.data.tags.get("model_type", "desconocido")

    model_uri = best_run.data.tags["model_uri"]
    model_version = mlflow.register_model(model_uri, MODEL_NAME)

    client.update_model_version(
        name=MODEL_NAME,
        version=model_version.version,
        description=f"{model_type} del run {run_id}. RMSE en prueba: {rmse:.2f}",
    )
    logger.info("Registrado '%s' versión %s (%s, RMSE %.2f)",
                MODEL_NAME, model_version.version, model_type, rmse)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()