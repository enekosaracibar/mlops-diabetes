"""Extracción: carga el dataset Diabetes y lo guarda en data/raw."""

import logging

import pandas as pd
from sklearn.datasets import load_diabetes

from config.config import RAW_DATA_PATH

logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    """Carga el dataset Diabetes con los valores reales (sin escalar)."""
    diabetes = load_diabetes(as_frame=True, scaled=False)
    return diabetes.frame


def save_raw_data(df: pd.DataFrame) -> None:
    """Guarda los datos en bruto, creando la carpeta si no existe."""
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)


def main() -> None:
    df = load_raw_data()
    save_raw_data(df)
    logger.info("Datos en bruto guardados en %s (%d filas)", RAW_DATA_PATH, len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()