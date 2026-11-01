"""Validación: comprueba que los datos en bruto tienen el esquema esperado."""

import logging

import pandas as pd

from config.config import FEATURES, RAW_DATA_PATH, TARGET

logger = logging.getLogger(__name__)


def validate_data(df: pd.DataFrame) -> None:
    """Comprueba el esquema de los datos. Si algo falla, lanza un error."""
    # 1. El dataset no está vacío
    if df.empty:
        raise ValueError("El dataset está vacío.")

    # 2. Existe la columna objetivo
    if TARGET not in df.columns:
        raise ValueError(f"Falta la columna objetivo '{TARGET}'.")

    # 3. Existen todas las columnas de entrada
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    # 4. Todas las columnas son numéricas
    non_numeric = [col for col in FEATURES + [TARGET]
                   if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise ValueError(f"Columnas con tipo no numérico: {non_numeric}")

    # 5. Informa de los nulos (no para: la limpieza se encarga de ellos)
    n_nulls = int(df[FEATURES + [TARGET]].isnull().sum().sum())
    logger.info("Valores nulos encontrados: %d", n_nulls)


def main() -> None:
    df = pd.read_csv(RAW_DATA_PATH)
    validate_data(df)
    logger.info("Validación superada (%d filas)", len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()