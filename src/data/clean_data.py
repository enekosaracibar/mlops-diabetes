"""Limpieza: trata tipos, duplicados, valores imposibles y nulos, y guarda en data/processed."""

import logging

import numpy as np
import pandas as pd

from config.config import FEATURES, PROCESSED_DATA_PATH, RAW_DATA_PATH, TARGET

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia los datos: tipos, duplicados, valores imposibles y nulos."""
    df = df.copy()
    cols = FEATURES + [TARGET]

    # 1. Asegura que todo es numérico (lo que no se pueda convertir pasa a nulo)
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Elimina filas duplicadas
    df = df.drop_duplicates()

    # 3. Marca como nulos los valores imposibles
    df.loc[~df["sex"].isin([1, 2]), "sex"] = np.nan
    for col in cols:
        if col != "sex":
            df.loc[df[col] <= 0, col] = np.nan

    # 4. Sin valor objetivo no se puede entrenar: se elimina la fila
    df = df.dropna(subset=[TARGET])

    # 5. Rellena los nulos de las variables de entrada
    for col in FEATURES:
        if col == "sex":
            df[col] = df[col].fillna(df[col].mode()[0])   # valor más frecuente
        else:
            df[col] = df[col].fillna(df[col].median())    # mediana

    return df


def save_clean_data(df: pd.DataFrame) -> None:
    """Guarda los datos limpios, creando la carpeta si no existe."""
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)


def main() -> None:
    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_clean = clean_data(df_raw)
    save_clean_data(df_clean)
    logger.info("Filas antes: %d | después: %d", len(df_raw), len(df_clean))
    logger.info("Datos limpios guardados en %s", PROCESSED_DATA_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()